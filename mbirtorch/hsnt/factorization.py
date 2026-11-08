"""This module computes the maximum-likelihood NNAL factorization that ``dehydrate`` uses.

NNAL stands for non-negative attenuation loss.
"""

import torch

from ._linalg import _attenuation_for_start, _nonneg_least_squares_start, nndsvda
from ._newton import _X_MAX, _resolve_compile, joint_newton_optimize


def _initial_factors(T, num_materials):
    """This function computes the NNDSVDa starting factors from the attenuation X = -log(T).

    Before the log, the function replaces each zero count by half the smallest positive
    transmission.
    """
    return nndsvda(_attenuation_for_start(T), n_components=num_materials)


def _nnal_factorization(T, num_materials, max_steps=1000, rel_tol=1e-8, compile_mode='auto', W_init=None,
                        H_init=None, quick=False, stats=None, weights=None):
    """This function factorizes the transmission ratio as T ~= exp(-W @ H) with W, H >= 0.

    The function minimizes the non-negative attenuation loss.  This loss equals the Poisson
    negative log-likelihood of the counts, up to a constant and the dose.  The solver first runs
    a few block Newton steps, and then it runs a joint truncated Newton solve on (W, H).  The
    problem is not convex, so a different starting point can reach a different local optimum.

    Args:
        T (torch.Tensor): The transmission ratio, with shape (pixels, bins).  T may contain zero
            counts.
        num_materials (int): The rank of the factorization.  It is at most min(pixels, bins).
        max_steps (int, optional): The maximum number of iterations.  Defaults to 1000.
        rel_tol (float, optional): The solver stops after five steps in a row whose relative
            change in loss is at most ``rel_tol``.  Defaults to 1e-8.
        compile_mode (str, optional): 'on' always compiles with torch.compile, and 'off' never
            compiles.  'auto' compiles only on CUDA with a working Triton and only when T has
            at least 5e8 entries.  Defaults to 'auto'.
        W_init, H_init (torch.Tensor, optional): The starting factors.  If only one factor is
            given, the function fits the other one by non-negative least squares.  If neither is
            given, the function uses an NNDSVDa start.  Defaults to None.
        quick (bool, optional): If True, the solver also stops when its step is small compared
            with the noise (see ``_newton._joint_newton_pcg``).  Defaults to False.
        stats (dict, optional): If given, the solver records the effective dose and whether the
            quick stop ended the solve.  Defaults to None.
        weights (torch.Tensor, optional): Nonnegative weights that broadcast against T, such as
            the open beam count of each bin as a (1, bins) tensor.  Each entry's loss is multiplied
            by its weight.  With the open beam counts as weights, the loss is the Poisson negative
            log-likelihood of the counts when the open beam differs between bins.  Defaults to
            None, which gives every entry the weight 1.

    Returns:
        (W, H, steps): W has shape (pixels, num_materials), and H has shape (num_materials,
            bins).  Both are on the device of T and have the dtype of T.  steps is the number
            of steps the solver took.
    """
    if not 1 <= int(num_materials) <= min(T.shape):
        raise ValueError(f"num_materials must be between 1 and min(pixels, bins) = {min(T.shape)}, got {num_materials}")
    compile_mode = _resolve_compile(compile_mode, T)
    if W_init is None and H_init is None:
        W_init, H_init = _initial_factors(T, num_materials)
    elif W_init is None:
        W_init = _nonneg_least_squares_start(_attenuation_for_start(T), H_init)
    elif H_init is None:
        H_init = _nonneg_least_squares_start(_attenuation_for_start(T).T, W_init.T).T
    return joint_newton_optimize(T, num_materials, max_steps, rel_tol, W_init=W_init, H_init=H_init,
                                 compile_mode=compile_mode, quick=quick, stats=stats, weights=weights)


# A zero-count entry whose attenuation is at least this fraction of _newton._X_MAX is treated as
# being at the attenuation bound.
_AT_BOUND = 0.999
# A component that puts at least this fraction of its attenuation on zero-count entries models
# the zero counts rather than a material.
_CAPTURED_SHARE = 0.99


def _zero_count_divergence(W, H, T, chunk=2 ** 23):
    """This function returns the largest fitted attenuation on any zero-count entry.

    The function also returns the number of zero-count entries that are at the attenuation bound
    ``_newton._X_MAX``.
    """
    rows = max(1, chunk // max(T.shape[1], 1))
    x_max, n_above = 0.0, 0
    for i in range(0, T.shape[0], rows):
        zero = T[i:i + rows] <= 1e-12
        if not bool(zero.any()):
            continue
        X = W[i:i + rows] @ H
        if bool((zero & ~torch.isfinite(X)).any()):          # A non-finite fit counts as diverged.
            x_max = float('inf')
        else:
            x_max = max(x_max, float(torch.where(zero, X, torch.zeros_like(X)).max()))
        n_above += int((zero & ~(X < _AT_BOUND * _X_MAX)).sum())
    return x_max, n_above


def _components_at_bound(W, H, T, chunk=2 ** 23):
    """This function counts, for each component, the zero-count entries at which that component
    alone reaches the attenuation bound."""
    rows = max(1, chunk // max(T.shape[1], 1))
    out = torch.zeros(W.shape[1], dtype=torch.int64, device=W.device)
    for k in range(W.shape[1]):
        near = W[:, k] * H[k].amax() >= _AT_BOUND * _X_MAX            # Only these pixels can reach the bound.
        idx = near.nonzero().flatten()
        for i in range(0, idx.numel(), rows):
            p = idx[i:i + rows]
            Xk = W[p, k:k + 1] @ H[k:k + 1]
            out[k] += int(((T[p] <= 1e-12) & (Xk >= _AT_BOUND * _X_MAX)).sum())
    return out


def _zero_count_mass(W, H, T, chunk=2 ** 23):
    """This function returns, for each component, the sum of its attenuation over the zero-count
    entries.

    The result has shape (rank,) and dtype float64.
    """
    rows = max(1, chunk // max(T.shape[1], 1))
    out = torch.zeros(W.shape[1], dtype=torch.float64, device=W.device)
    Hd = H.to(torch.float64)
    for i in range(0, T.shape[0], rows):
        zero = (T[i:i + rows] <= 1e-12).to(torch.float64)
        if bool(zero.any()):
            out += (W[i:i + rows].to(torch.float64) * (zero @ Hd.T)).sum(0)
    return out

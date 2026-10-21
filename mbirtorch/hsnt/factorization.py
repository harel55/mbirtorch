"""This module computes the maximum-likelihood NNAL factorization that ``dehydrate`` uses.

NNAL stands for non-negative attenuation loss.
"""

import torch

from ._loss import BACKGROUND_MAX, _nnal_prep, stable_nnal
from ._linalg import _attenuation_for_start, _nonneg_least_squares_start, nndsvda
from ._newton import _X_MAX, _resolve_compile, joint_newton_optimize


def _initial_factors(T, num_materials):
    """This function computes the NNDSVDa starting factors from the attenuation X = -log(T).

    Before the log, the function replaces each zero count by half the smallest positive
    transmission.
    """
    return nndsvda(_attenuation_for_start(T), n_components=num_materials)


def _nnal_factorization(T, num_materials, max_steps=1000, rel_tol=1e-8, compile_mode='auto', W_init=None,
                        H_init=None):
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
                                 compile_mode=compile_mode)


# Fisher-scoring steps on the spectra and background per update, and the most halvings of each step.
BACKGROUND_NEWTON_STEPS = 20
BACKGROUND_HALVINGS = 30
# The most alternations between the factors and the background.
BACKGROUND_ROUNDS = 50


def _bin_terms(W, H, b, T, with_derivatives=True, chunk=2 ** 23):
    """This function returns, for each bin k, the loss sum_p (m - T log m) of the model
    m = (1 - b) exp(-W @ H) + b.  It also returns, unless ``with_derivatives`` is False, the
    gradient and the Fisher information of that loss with respect to (H[:, k], b[k]).  The loss
    has shape (bins,), the gradient (bins, rank + 1), and the Fisher information
    (bins, rank + 1, rank + 1).  All are float64."""
    rows = max(1, chunk // max(T.shape[1], 1))
    Hd, bd = H.to(torch.float64), b.to(torch.float64)
    K, R = T.shape[1], H.shape[0]
    loss = torch.zeros(K, dtype=torch.float64, device=T.device)
    grad = torch.zeros(K, R + 1, dtype=torch.float64, device=T.device)
    fisher = torch.zeros(K, R + 1, R + 1, dtype=torch.float64, device=T.device)
    for i in range(0, T.shape[0], rows):
        Wc, Tc = W[i:i + rows].to(torch.float64), T[i:i + rows].to(torch.float64)
        e = torch.exp(-(Wc @ Hd))
        m = (1 - bd) * e + bd
        pos = Tc > 0
        loss += (m - torch.where(pos, Tc * torch.log(m), torch.zeros_like(m))).sum(0)
        if not with_derivatives:
            continue
        w = 1 - torch.where(pos, Tc / m, torch.zeros_like(m))           # dloss/dm
        dH = -(1 - bd) * e                                              # dm/dX; dm/dH[r, k] = dH * W[:, r]
        db = 1 - e                                                      # dm/db
        grad[:, :R] += (w * dH).T @ Wc
        grad[:, R] += (w * db).sum(0)
        fisher[:, :R, :R] += torch.einsum('pk,pr,ps->krs', dH * dH / m, Wc, Wc)
        cross = (dH * db / m).T @ Wc
        fisher[:, :R, R] += cross
        fisher[:, R, :R] += cross
        fisher[:, R, R] += (db * db / m).sum(0)
    return loss, grad, fisher


def _spectra_background_step(W, H, b, T, steps=BACKGROUND_NEWTON_STEPS, jitter_rel=1e-9):
    """This function minimizes the loss over each bin's spectrum column H[:, k] and background
    b[k] together, with W fixed, by projected Fisher-scoring steps with step halving.  A change
    of b in a bin of low attenuation is nearly the same as a rescaling of that bin's spectra, so
    the two move together.  It returns (H, b) with the dtypes of the inputs."""
    R = H.shape[0]
    theta = torch.cat([H.T.to(torch.float64), b.to(torch.float64)[:, None]], 1)       # (bins, rank + 1)
    upper = torch.full_like(theta, float('inf'))
    upper[:, R] = BACKGROUND_MAX
    split = lambda th: (th[:, :R].T.contiguous().to(H.dtype), th[:, R].contiguous().to(b.dtype))
    for _ in range(steps):
        loss, grad, fisher = _bin_terms(W, *split(theta), T)
        # Two-metric projection: a variable at a bound whose gradient points outward stays fixed.
        fixed = ((theta <= 0) & (grad > 0)) | ((theta >= upper) & (grad < 0))
        free = (~fixed).to(torch.float64)
        F = fisher * free[:, :, None] * free[:, None, :]
        diag = torch.diagonal(F, dim1=1, dim2=2)
        F = F + torch.diag_embed(torch.where(fixed, torch.ones_like(diag), jitter_rel * diag.clamp_min(0)
                                             + torch.finfo(torch.float64).tiny))
        direction = torch.linalg.solve(F, (grad * free)[..., None])[..., 0]
        new, t = theta.clone(), torch.ones(theta.shape[0], dtype=torch.float64, device=theta.device)
        todo = direction.abs().amax(1) > 0
        for _ in range(BACKGROUND_HALVINGS):
            cand = torch.minimum((theta - t[:, None] * direction).clamp_min(0.0), upper)
            ok = todo & (_bin_terms(W, *split(cand), T, with_derivatives=False)[0] <= loss)
            new = torch.where(ok[:, None], cand, new)
            todo = todo & ~ok
            if not bool(todo.any()):
                break
            t = torch.where(todo, t / 2, t)
        if torch.equal(new, theta):
            break
        theta = new
    return split(theta)


def _nnal_factorization_background(T, num_materials, max_steps=1000, rel_tol=1e-8, compile_mode='auto',
                                   max_rounds=BACKGROUND_ROUNDS):
    """This function factorizes the transmission ratio as T ~= (1 - b) exp(-W @ H) + b, with
    W, H >= 0 and an additive background 0 <= b <= BACKGROUND_MAX in each bin, shared by all
    pixels.  The background is a fraction of the open beam that passes the sample without
    attenuation.

    The function first fits W and H without a background.  Each round then updates H and the
    background together with W fixed, and then solves for W, H and the background together.  It stops
    when the relative change in loss over a round is at most ``rel_tol``, or after
    ``max_rounds`` rounds.  The other arguments are those of ``_nnal_factorization``.

    Returns:
        (W, H, b, rounds, steps): the factors, the background with shape (bins,), the number of
            rounds, and the total number of solver steps.
    """
    W, H, steps = _nnal_factorization(T, num_materials, max_steps=max_steps, rel_tol=rel_tol,
                                      compile_mode=compile_mode)
    compile_mode = _resolve_compile(compile_mode, T)
    b = torch.zeros(T.shape[1], dtype=T.dtype, device=T.device)
    prev = stable_nnal(W @ H, T, _nnal_prep(T), dtype=torch.float64)
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        H, b = _spectra_background_step(W, H, b, T)
        W, H, taken, b = joint_newton_optimize(T, num_materials, max_steps, rel_tol, W_init=W, H_init=H,
                                               compile_mode=compile_mode, background=b, fit_background=True)
        steps += taken
        loss = stable_nnal(W @ H, T, _nnal_prep(T, b), dtype=torch.float64)
        if bool(torch.abs(prev - loss) <= rel_tol * torch.abs(loss)):
            break
        prev = loss
    return W, H, b, rounds, steps


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

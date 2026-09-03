import math
import numpy as np
import h5py
import warnings
import matplotlib.pyplot as plt
from functools import partial
from sklearn.decomposition import non_negative_factorization as nmf
from sklearn.utils.extmath import randomized_svd

import torch


# -----------------------------------------------------------------------
# Hyperspectral Neutron Radiographic/Tomographic Data Denoising Functions
# -----------------------------------------------------------------------
def _nnal_prep(T):
    """
    Precompute the quantities that depend only on T.

    These are constant for a whole solve, so hoisting them out of the iteration
    removes a full log over T from every loss and derivative evaluation.

    Returns:
        (log_T, positive, all_positive, taylor_cutoff)
    """
    positive = T > 0
    Tsafe = torch.where(positive, T, torch.ones((), dtype=T.dtype, device=T.device))
    log_T = torch.log(Tsafe)
    all_positive = bool(positive.all())
    # Crossover for the phi series: truncation ~ |Xp|^3/24 against cancellation
    # ~ eps/|Xp|^2. The old hard-coded 1e-3 is ~40x too small in float32.
    taylor_cutoff = (24.0 * torch.finfo(T.dtype).eps) ** 0.25
    return log_T, positive, all_positive, taylor_cutoff


def stable_nnal(X, T, prep=None, dtype=None):
    """
    Compute a shifted form of the non-negative attentuation loss
    that is much more numerically stable

    Args:
        X: Attenuation estimate, broadcastable against T.
        T: Measured transmission ratio (counts / open beam).
        prep: Optional tuple from _nnal_prep(T). Pass it inside an iteration to
            avoid recomputing log(T) on every call.
        dtype: Accumulation dtype for the final sum. Defaults to X's dtype. Pass
            torch.float64 when the value drives a convergence test: in float32 a
            loss near 3e6 has a resolution of 0.25, so two consecutive losses that
            differ by less than that compare equal and a relative-change test
            fires spuriously.
    """
    log_T, positive, all_positive, taylor_cutoff = _nnal_prep(T) if prep is None else prep

    Xp = X + log_T

    phi = torch.where(
        torch.abs(Xp) < taylor_cutoff,
        Xp * Xp * (0.5 + Xp * (-1.0 / 6.0 + Xp / 24.0)),
        torch.expm1(-Xp) + Xp,
    )

    loss = T * phi

    if not all_positive:
        # T == 0 means Xp == X, so the zero-count term is exp(-Xp). It must be a
        # real exp: expm1(-Xp) saturates at exactly -1 for Xp above ~37 in
        # float64, so reconstructing it as expm1(-Xp) + 1 underflows to zero.
        loss = torch.where(positive, loss, torch.exp(-Xp))

    return torch.sum(loss, dim=(-2, -1), dtype=dtype)


def stable_nnal_derivatives(X: torch.Tensor, T: torch.Tensor, prep=None):
    """
    Given X = W @ H, compute

        G = dL/dX = T - exp(-X)
        Z = d^2L/dX^2 = exp(-X)

    where L is the non-negative attenuation loss in a
    numerically stable way that handles T = 0 appropriately.
    """
    log_T, positive, all_positive, _ = _nnal_prep(T) if prep is None else prep

    Xp = X + log_T

    # Two transcendentals, not four: torch.where evaluates both of its branches,
    # so the original form paid for four exponentials per call. expm1 is the
    # accurate one near Xp = 0, where T - exp(-X) cancels; exp is the accurate
    # one at large Xp, where expm1 saturates at -1 and expm1 + 1 underflows.
    E = torch.expm1(-Xp)
    eXp = torch.exp(-Xp)

    G = -T * E
    Z = T * eXp

    if not all_positive:
        G = torch.where(positive, G, -eXp)
        Z = torch.where(positive, Z, eXp)

    return G, Z


def _randomized_svd(X, n_components, n_oversamples=10, n_iter=4, seed=0):
    """Truncated SVD via a randomized range finder.

    A full SVD of an (n_samples, n_features) matrix is wasted work when only a
    handful of components are wanted, and here n_components is the material count
    -- rarely above 20 against a thousand or more wavelength bins. Seeded so the
    result is reproducible.
    """
    n_rows, n_cols = X.shape
    rank = min(n_components + n_oversamples, n_rows, n_cols)
    generator = torch.Generator(device=X.device).manual_seed(seed)
    Q, _ = torch.linalg.qr(X @ torch.randn(n_cols, rank, generator=generator,
                                           dtype=X.dtype, device=X.device))
    for _ in range(n_iter):                      # power iterations sharpen the range
        Q, _ = torch.linalg.qr(X.T @ Q)
        Q, _ = torch.linalg.qr(X @ Q)
    U, singular_values, Vh = torch.linalg.svd(Q.T @ X, full_matrices=False)
    return Q @ U, singular_values, Vh


def nndsvda(X, n_components, fill='sqrt', fill_scale=1.0):
    """NNDSVD initialization for X ~= W @ H, with zeros filled.

    Every component after the first is one sign-half of a singular vector pair,
    so roughly half its entries are zero. Zeros are poison for multiplicative
    updates, which can never move an entry off zero, so they are filled.

    The fill value has to be chosen at the scale of a FACTOR entry, not of X.
    Each factor carries sqrt(s_k), so a typical W or H entry is of order
    sqrt(mean X) and their product is of order mean X. A fill of f in both
    factors therefore contributes f^2 to the product where both were filled.

      fill = mean X       (classic NNDSVDA)  -> f^2 = (mean X)^2, which only
                          matches the data when mean X ~ 1. At mean X = 34.5
                          it put 1190 into every noise component.
      fill = mean X / 100                    -> f^2 = (mean X)^2 / 1e4. Safe
                          against overshoot but so small that block_newton's
                          two-metric projection froze the filled entries at
                          zero and converged in a reduced subspace, 0.6% worse.
      fill = c * sqrt(mean X)  (default)     -> f^2 = c^2 * mean X: a fixed
                          fraction c^2 of a typical entry, whatever the scale of
                          X. Dimensionally right and scale-free.

    c = 1 was chosen by measurement, not by taste. block_newton's two-metric
    projection freezes any entry it drives to zero, and a fill it can push there
    in one step costs a component: at dosage_rate=3 the loss it reaches improves
    monotonically with the fill, 904,026 at mean/100, 902,547 at c=0.1, 900,323
    at c=0.3, 898,850 at classic mean, 898,514 at c=1 -- the last within 0.001% of
    the joint solver. The multiplicative and joint solvers are indifferent to c
    across that whole range. The price is an initial X that can exceed the data
    by up to 2x where fills coincide; that is bounded and scale-free, unlike the
    35x of classic NNDSVDA on a badly floored matrix, and no solver minds it.

    Args:
        X: Nonnegative array of shape (n_samples, n_features).
        n_components: Factorization rank.
        fill: 'sqrt' (default) fills zeros with fill_scale * sqrt(mean X);
            'mean' is classic NNDSVDA; 'small' is mean X / 100.
        fill_scale: multiplier for the 'sqrt' fill. Defaults to 1.0.

    Returns:
        W: Shape (n_samples, n_components).
        H: Shape (n_components, n_features).
    """
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    if n_components + 10 < min(X.shape):
        U, singular_values, Vh = _randomized_svd(X, n_components)
    else:
        U, singular_values, Vh = torch.linalg.svd(X, full_matrices=False)

    n_components = min(
        n_components,
        U.shape[1],
        Vh.shape[0],
    )

    W = torch.zeros((X.shape[0], n_components), dtype=X.dtype, device=X.device)
    H = torch.zeros((n_components, X.shape[1]), dtype=X.dtype, device=X.device)

    # First singular triplet.
    scale = torch.sqrt(singular_values[0])
    W[:, 0] = scale * torch.abs(U[:, 0])
    H[0, :] = scale * torch.abs(Vh[0, :])

    # Remaining components.
    for component in range(1, n_components):
        u = U[:, component]
        v = Vh[component, :]

        u_pos = torch.clip(u, min=0)
        u_neg = torch.clip(-u, min=0)
        v_pos = torch.clip(v, min=0)
        v_neg = torch.clip(-v, min=0)

        positive_strength = (
            torch.linalg.norm(u_pos) * torch.linalg.norm(v_pos)
        )
        negative_strength = (
            torch.linalg.norm(u_neg) * torch.linalg.norm(v_neg)
        )

        use_positive = positive_strength > negative_strength

        selected_u = torch.where(use_positive, u_pos, u_neg)
        selected_v = torch.where(use_positive, v_pos, v_neg)

        selected_u /= torch.linalg.norm(selected_u) + torch.finfo(X.dtype).eps
        selected_v /= torch.linalg.norm(selected_v) + torch.finfo(X.dtype).eps

        scale = torch.sqrt(singular_values[component])
        W[:, component] = scale * selected_u
        H[component, :] = scale * selected_v

    mean = torch.mean(X)
    if fill == 'mean':
        fill_value = mean
    elif fill == 'small':
        fill_value = mean * 1e-2
    else:
        fill_value = fill_scale * torch.sqrt(mean.clamp_min(0))
    W = torch.where(W == 0, fill_value, W)
    H = torch.where(H == 0, fill_value, H)

    return W, H


def _shifted(V, ratio, shift, mode, mean_dim):
    """V <- max((V + d) * ratio - d, 0), the shifted multiplicative step.

    A plain multiplicative update cannot revive an entry that reaches exactly
    zero, since 0 * anything = 0. On the demo problem 5.5% of W starts at zero
    and that fraction never moves over 6000 iterations, even though 1.77% of all
    entries are zero with a NEGATIVE gradient -- not KKT points, just frozen.

    The offset does not bias the answer. Expanding gives V*ratio + d*(ratio - 1),
    so an interior fixed point needs (ratio - 1)(V + d) = 0, and V + d > 0 forces
    ratio = 1; a zero entry stays zero only while d*(ratio - 1) <= 0, i.e.
    ratio <= 1. Both updates are built so ratio > 1 exactly when the gradient is
    negative, so the fixed-point set is {V >= 0, grad >= 0, V*grad = 0} -- the KKT
    set -- for every d > 0. No decay schedule is needed for correctness.

    mode='boundary' offsets only entries currently at zero, leaving the interior
    step bit-for-bit the original update. mode='const' offsets everything, which
    converges faster but rides on the raw ratio, so an undamped ratio needs a cap.
    """
    if shift <= 0:
        return V * ratio
    # The offset is sized from the per-component mean, but a component that has
    # collapsed has a mean of zero -- so the offset that exists to revive it would
    # vanish exactly when it is needed. Floor each component's scale at a fraction
    # of the whole factor's mean so a dead row or column still gets a live offset.
    per_component = V.mean(dim=mean_dim, keepdim=True)
    floor = 1e-2 * V.mean()
    scale = shift * torch.maximum(per_component, floor).clamp_min(torch.finfo(V.dtype).tiny)
    if mode == 'boundary':
        scale = torch.where(V <= 0, scale.expand_as(V), torch.zeros_like(V))
    return (V * ratio + scale * (ratio - 1.0)).clamp_(min=0)


def quadratic_update(W, H, T, unused=None, update_H=True, prep=None,
                     shift=1e-1, shift_mode='boundary', ratio_max=2.0):
    """Iteratively reweighted multiplicative update for the NNAL.

    The second-order model of the NNAL at the current iterate X is

        L(X') ~ L(X) + G (X' - X) + (1/2) Z (X' - X)^2 = (1/2) Z (X' - V)^2 + c

    with G = T - exp(-X), Z = exp(-X) and target V = X - G/Z. Reweighted least
    squares on that model, relinearized every step, has the NNAL stationary point
    as its fixed point.

    Z V = Z X - G, which needs no exp(+X) and so stays bounded however large the
    attenuation gets. Z V is signed, so the multiplicative update splits it into
    positive and negative parts and puts each on the side of the ratio where it
    keeps W and H nonnegative; at the fixed point the ratio is one, which gives
    (exp(-X) - T) H^T = 0, exactly the NNAL stationarity condition.

    This replaces an earlier update that minimized (1/2) sum T (X + log T)^2 --
    the second-order model expanded about the noiseless solution rather than the
    current iterate, and never relinearized. That objective weights every
    zero-count measurement by exactly zero, so at low dose it discards a large
    fraction of the data (17.7% at dosage_rate=3) and converges somewhere other
    than the NNAL minimum. Here the weight is exp(-X) > 0 everywhere, so no
    measurement is dropped.

    Note this is a Gauss-Newton style surrogate, not a majorizer, so individual
    steps are not guaranteed to decrease the loss.

    Args:
        W: Feature matrix (spatial pixels × num_materials), PyTorch tensor
        H: Spectral basis matrix (num_materials × spectral channels), PyTorch tensor
        T: Data term matrix (spatial pixels × spectral channels), PyTorch tensor
        unused: Placeholder for auxiliary state (not used by this method).
        update_H: If False, keep H fixed and only update W.
        prep: Optional tuple from _nnal_prep(T).

    Returns:
        Updated (W, H) pair as PyTorch tensors
    """
    prep = _nnal_prep(T) if prep is None else prep
    tiny = torch.finfo(T.dtype).tiny

    X = W @ H
    G, Z = stable_nnal_derivatives(X, T, prep)
    ZV = Z * X - G
    # This ratio is undamped and can be very large. shift_mode='const' rides on it
    # for every entry and diverges to NaN within a few hundred iterations without a
    # cap; 'boundary' only offsets entries already at zero and is safe uncapped, so
    # the cap is applied only where it is needed and never alters the plain update.
    cap = ratio_max if (shift > 0 and shift_mode == 'const') else float('inf')
    ratio = ((ZV.clamp(min=0) @ H.T)
             / ((Z * X) @ H.T + (-ZV).clamp(min=0) @ H.T).clamp(min=tiny))
    W = _shifted(W, ratio.clamp(max=cap), shift, shift_mode, 0)

    if update_H:
        # Relinearize before the H step: the model is only second order accurate
        # at the iterate it was expanded about, and W has just moved.
        X = W @ H
        G, Z = stable_nnal_derivatives(X, T, prep)
        ZV = Z * X - G
        ratio = ((W.T @ ZV.clamp(min=0))
                 / (W.T @ (Z * X) + W.T @ (-ZV).clamp(min=0)).clamp(min=tiny))
        H = _shifted(H, ratio.clamp(max=cap), shift, shift_mode, 1)

    return W, H, 0.0


def newton_update(W, H, T, lr_init, update_H=True, prep=None):
    """PyTorch-optimized Newton update with automatic differentiation and line search.

    Args:
        W: Feature matrix (spatial pixels × num_materials), PyTorch tensor
        H: Spectral basis matrix (num_materials × spectral channels), PyTorch tensor
        T: Data term matrix (spatial pixels × spectral channels), PyTorch tensor
        lr_init: Initial learning rate for line search
        update_H: If False, keep H fixed and only update W.

    Returns:
        Updated (W, H) pair as PyTorch tensors
    """
    X = W @ H
    G, Z = stable_nnal_derivatives(X, T, prep)
    init_loss = stable_nnal(X, T, prep)

    # Compute gradients
    grad_W = G @ H.T
    grad_H = W.T @ G

    # Compute Hessian diagonal approximation manually (kept explicit for numerical stability)
    d2L_dW2 = Z @ H.T.pow(2)
    d2L_dH2 = W.T.pow(2) @ Z

    dW = grad_W / (d2L_dW2 + 1e-30)
    dH = grad_H / (d2L_dH2 + 1e-30) if update_H else torch.zeros_like(H)

    learning_rates = lr_init * torch.logspace(
        -10, 1, steps=13, base=2, dtype=T.dtype, device=T.device
    )
    W_candidates = torch.clip(W[None] - learning_rates[:, None, None] * dW[None], min=1e-30)
    if update_H:
        H_candidates = torch.clip(H[None] - learning_rates[:, None, None] * dH[None], min=1e-30)
    else:
        H_candidates = H.expand(learning_rates.shape[0], -1, -1)

    X_candidates = torch.bmm(W_candidates, H_candidates)
    candidate_losses = stable_nnal(X_candidates, T, prep)
    directional_derivatives = (
        torch.sum(grad_W[None] * (W_candidates - W[None]), dim=(1, 2))
        + torch.sum(grad_H[None] * (H_candidates - H[None]), dim=(1, 2))
    )
    armijo = candidate_losses <= init_loss + 1e-4 * directional_derivatives
    valid_losses = torch.where(armijo, candidate_losses, torch.full_like(candidate_losses, torch.inf))
    best_index = torch.argmin(valid_losses)
    best_index = best_index.reshape(1)
    return (
        W_candidates.index_select(0, best_index).squeeze(0),
        H_candidates.index_select(0, best_index).squeeze(0),
        learning_rates.index_select(0, best_index).squeeze(0),
    )

def _rebalance(W, H):
    """Equalize the scale of each component between W and H, leaving W @ H fixed.

    The factorization is invariant to W -> W D, H -> D^-1 H for any positive
    diagonal D, and nothing in a multiplicative update pins that gauge down. At
    low dosage it drifts to extremes -- H rows near 1e6 against W columns near
    1e-3 -- which starves the shifted step's offset (it scales with the mean of
    the factor being updated) and pushes float32 toward its limits. Setting
    ||W[:, r]|| = ||H[r, :]|| each step costs two norms and changes nothing else.
    """
    w = W.norm(dim=0); h = H.norm(dim=1)
    tiny = torch.finfo(W.dtype).tiny
    d = torch.sqrt((h.clamp_min(tiny) / w.clamp_min(tiny)))
    d = torch.where((w > tiny) & (h > tiny), d, torch.ones_like(d))
    return W * d, H / d[:, None]


def _reseed_dead(W, H, rel_tol=1e-6, mode='random'):
    """Re-seed any component that has died in BOTH factors at once.

    The shifted step revives an entry at zero whenever its ratio exceeds one. But a
    component whose W column and H row are both zero has a ratio of exactly zero in
    either update -- the numerator is a product with the dead factor -- so the
    offset term d*(ratio - 1) is negative and the clamp holds it at zero forever.
    No multiplicative step can reach it.

    What it is re-seeded WITH matters as much as whether it is. A constant seed
    keeps the loss honest but leaves the component flat, and a flat spectrum makes
    the gauge fit against the true spectra ill-conditioned: at dosage_rate=1 the
    constant seed reaches the same loss as joint_newton to 0.3% yet scores -84 dB
    on the recovered spectra against joint_newton's 21. A random positive seed
    breaks that symmetry and hands the dynamics something to differentiate; it
    scores 21.4 dB on the same case. Seeded, so runs are reproducible.
    """
    w = W.norm(dim=0); h = H.norm(dim=1)
    dead = (w <= rel_tol * w.max()) & (h <= rel_tol * h.max())
    n_dead = int(dead.sum())
    if n_dead == 0:
        return W, H
    live = ~dead
    W = W.clone(); H = H.clone()
    w_ref = W[:, live].mean() if bool(live.any()) else W.new_tensor(1.0)
    h_ref = H[live].mean() if bool(live.any()) else H.new_tensor(1.0)
    if mode == 'const':
        W[:, dead] = 1e-2 * w_ref
        H[dead] = 1e-2 * h_ref
        return W, H
    g = torch.Generator(device=W.device).manual_seed(0)
    W[:, dead] = 1e-2 * w_ref * torch.rand(W.shape[0], n_dead, generator=g, dtype=W.dtype, device=W.device)
    H[dead] = 1e-2 * h_ref * torch.rand(n_dead, H.shape[1], generator=g, dtype=H.dtype, device=H.device)
    return W, H


def multiplicative_update(W: torch.Tensor, H: torch.Tensor, T: torch.Tensor, unused: torch.Tensor = None,
                          update_H: bool = True, prep=None, shift=1e-2, shift_mode='const',
                          ratio_max=2.0, rebalance=True, reseed='random'):
    """PyTorch-optimized multiplicative update for non-negative factorization.

    Args:
        W: Feature matrix (spatial pixels × num_materials), PyTorch tensor
        H: Spectral basis matrix (num_materials × spectral channels), PyTorch tensor
        T: Data term matrix (spatial pixels × spectral channels), PyTorch tensor
        unused: Placeholder for auxiliary variables (not used in this implementation)
        update_H: If False, keep H fixed and only update W.

    Returns:
        Updated (W, H) pair as PyTorch tensors
    """
    damping_factor = 0.5
    Z = torch.exp(-W @ H)

    # Damping alone does not bound this ratio. At very low dosage T is almost all
    # zeros, so T @ H.T reaches the 1e-30 clip and the ratio reaches ~1e15; the
    # shifted step's offset term d*(ratio - 1) then overflows to NaN within one
    # iteration. Cap it wherever the shifted step rides on the raw ratio. shift=0
    # is left uncapped so it stays bit-for-bit the original update.
    cap = ratio_max if (shift > 0 and shift_mode == 'const') else float('inf')
    W = _shifted(W, (((Z @ H.T) / torch.clip(T @ H.T, min=1e-30)) ** damping_factor).clamp(max=cap),
                 shift, shift_mode, 0)
    if update_H:
        H = _shifted(H, (((W.T @ Z) / torch.clip(W.T @ T, min=1e-30)) ** damping_factor).clamp(max=cap),
                     shift, shift_mode, 1)
        # Only when the shift is on, so shift=0 stays bit-for-bit the original.
        if shift > 0 and reseed:
            W, H = _reseed_dead(W, H, mode=reseed)
        if shift > 0 and rebalance:
            W, H = _rebalance(W, H)

    return W, H, 0.0

def _nnal_rowwise(X, T, prep, dim, dtype=None):
    """NNAL summed over `dim` only: per-pixel (dim=1) or per-wavelength (dim=0).

    dtype is the accumulation dtype of the sum. A per-bin sum over P pixels in
    float32 has an ulp of 0.06 near 1e6 and 8 near 1e8, so an H step that improves
    a bin's loss by less than that is invisible to a line search -- a ceiling on H
    accuracy that worsens linearly with P. Summing in float64 removes it. What
    remains is the float32 truncation of the elementwise terms, which the line
    search's noise floor (_ARMIJO_FLOOR) accounts for.
    """
    log_T, positive, all_positive, taylor_cutoff = prep
    Xp = X + log_T
    phi = torch.where(
        torch.abs(Xp) < taylor_cutoff,
        Xp * Xp * (0.5 + Xp * (-1.0 / 6.0 + Xp / 24.0)),
        torch.expm1(-Xp) + Xp,
    )
    loss = T * phi
    if not all_positive:
        loss = torch.where(positive, loss, torch.exp(-Xp))
    return loss.sum(dim=dim, dtype=dtype)


def _batched_spd_solve(M, g, jitter_rel=1e-9):
    """Solve M[b] d[b] = g[b] for a batch of small SPD matrices.

    Tikhonov damping scaled to each matrix keeps the factorization well posed when
    Z is near zero (heavily attenuated pixels), and the fallback is branch-free so
    no host synchronization is introduced inside the iteration.
    """
    rank = M.shape[-1]
    eye = torch.eye(rank, dtype=M.dtype, device=M.device)
    diag = torch.diagonal(M, dim1=-2, dim2=-1)
    lam = jitter_rel * diag.amax(-1).clamp_min(torch.finfo(M.dtype).tiny)
    A = M + lam[:, None, None] * eye
    L, info = torch.linalg.cholesky_ex(A)
    failed = (info > 0)[:, None, None]
    L = torch.where(failed, eye.expand_as(L), L)
    d = torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
    diag_A = torch.diagonal(A, dim1=-2, dim2=-1).clamp_min(torch.finfo(M.dtype).tiny)
    d = torch.where(failed[:, :, 0], g / diag_A, d)
    # A row carrying no curvature at all -- Z underflowed to zero across the whole
    # row, which happens in float32 once the attenuation exceeds ~88 -- leaves the
    # damping underflowed too, so the fallback divides by `tiny` and overflows.
    # There is no second-order information to act on there, so take no step and
    # let the gradient-driven steps of later iterations bring the iterate back
    # into range.
    return torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


_COMPILED_KERNELS = {}
# Armijo noise floor, as a multiple of eps32 * |row loss|. A decrease smaller than
# this is not trusted, because elements whose step alpha*B falls below ulp(X) do
# not move at all in float32, so the measured decrease is a biased truncation that
# scales linearly with the row length (NOT a random walk: dividing by sqrt(n)
# re-created the spurious-backtracking pathology, 757 backtracks per 60 steps).
# With the row sums accumulated in float64 the constant can drop from 8 to 4:
# backtracks per 60 block steps at c = 0.5/1/2/4/8 were 650/468/160/27/15 at
# P=4096 and 543/473/236/48/15 at P=16384, with the converged loss identical to
# the last digit in every case. Halving the floor halves the loss slop the H-step
# is allowed at large P -- the noise ball in which H wandered at 43.8 dB on 9.4M
# pixels -- for about half an extra loss evaluation per step.
_ARMIJO_FLOOR = 4.0
# Trust-region floor as a fraction of the mean row scale (0 -> machine epsilon, the old behaviour).
_TRUST_FLOOR = 1e-3
# epsilon-active set (Bertsekas): a component within this fraction of the mean row
# scale of zero, with a gradient pushing it out, is treated as AT the bound and
# snapped to exactly zero. Without it a component that hit the feasibility limit
# lands at V - (V/d)*d, a residue of ~1e-17 V in float64 (1e-112 later) that is
# formally free: the row's shared step length min(V/d) is then ~0 and the whole
# row freezes at a non-stationary point. float32 rounds the same residue to 0.
_ACTIVE_TOL = 1e-6


def _kernels(compile_mode):
    """The four hot kernels, eager or compiled: (nnal, derivatives, rowwise, block step).

    compile_mode=None returns the plain functions. Any other value compiles them
    once and caches the result. The Newton solvers spend their time in
    elementwise passes over P x K -- the loss, its derivatives, the per-row loss
    of the line search -- and torch.compile fuses each of those into one kernel:
    measured 2.2x on a whole joint_newton solve at P=4096, 2.9x at P=16384, and
    3.8x on a whole block_newton solve. The GEMM-bound CG inner iteration does
    not benefit (1.07x) and is left eager.

    Compiled and eager agree bit-for-bit over a joint_newton solve (28 steps) and
    over the first 40 block_newton steps; over a 729-step block_newton run the
    fused reductions' different rounding eventually flips one active-set decision
    and the paths separate (max |W,H| difference 1.6e-2), but they end at the
    same loss to 2e-8 relative with identical spectra. Do not expect long
    block_newton runs to be reproducible across the compiled/eager boundary.

    The value of compile_mode is otherwise ignored here: 'max-autotune' measured
    no steady-state gain over the default and 2.4x the compile time, and inductor
    reports too few SMs on this class of card for its GEMM autotuning to apply.

    Compiling costs 4.5 s with a warm inductor cache and ~18 s cold at P=4096, and
    recompiles whenever P, K, R, dtype or the presence of zero counts changes, so
    it pays for repeated solves -- the batched path, many datasets, a service --
    and not for one 0.5 s solve. Default is therefore off.
    """
    if compile_mode is None:
        return stable_nnal, stable_nnal_derivatives, _nnal_rowwise, block_newton_step
    if compile_mode not in _COMPILED_KERNELS:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        compiled = lambda f: torch.compile(f, mode=None, dynamic=False)
        _COMPILED_KERNELS[compile_mode] = (compiled(stable_nnal), compiled(stable_nnal_derivatives),
                                           compiled(_nnal_rowwise), compiled(block_newton_step))
    return _COMPILED_KERNELS[compile_mode]


def block_newton_step(V, other, X, T, prep, axis, ls_max=8, jitter_rel=1e-9,
                      rowwise=None, deriv=None):
    """One exact projected-Newton step on a single factor.

    The NNAL is convex in X and X = W @ H is linear in each factor, so each block
    subproblem is convex. It is also separable: with H fixed the problem splits
    into one independent rank-dimensional problem per pixel, and with W fixed into
    one per wavelength bin. That makes the full (not diagonal) Hessian affordable:
    it is a batch of rank x rank matrices assembled by a single matmul against the
    Khatri-Rao product of the fixed factor with itself.

    Args:
        V: Factor being updated. axis=0 -> W (pixels x rank); axis=1 -> H (rank x bins).
        other: The fixed factor.
        X: Current W @ H, kept incrementally so no extra matmul is needed.
        T: Transmission ratio.
        prep: Tuple from _nnal_prep(T).
        axis: 0 to update W, 1 to update H.

    Returns:
        (V_new, X_new, (num_backtracks, projected_gradient_norm_squared)). The
        gradient norm is measured at the incoming iterate, before the step.
    """
    rowwise = _nnal_rowwise if rowwise is None else rowwise
    deriv = stable_nnal_derivatives if deriv is None else deriv
    log_T, positive, all_positive, taylor_cutoff = prep
    G, Z = deriv(X, T, prep)

    if axis == 0:
        rank = V.shape[1]
        grad = G @ other.T
        rows, cols = torch.triu_indices(rank, rank, device=V.device)
        flat = Z @ (other[rows] * other[cols]).T
    else:
        rank = V.shape[0]
        rows, cols = torch.triu_indices(rank, rank, device=V.device)
        flat = ((other[:, rows] * other[:, cols]).T @ Z).T
        grad = (other.T @ G).T
        V = V.T

    M = flat.new_zeros(flat.shape[0], rank, rank)
    M[:, rows, cols] = flat
    M[:, cols, rows] = flat

    # Two-metric projection: freeze the variables sitting on the V >= 0 boundary
    # whose gradient pushes them further out, and Newton-step the rest.
    eps_active = _ACTIVE_TOL * V.abs().amax(-1, keepdim=True).mean()
    bound = (V <= eps_active) & (grad > 0)
    free = ~bound
    # The projected gradient is the KKT residual for this block, and it is already
    # in hand here; returning it lets the driver test convergence without paying
    # for a second derivative evaluation.
    projected_gnorm2 = ((grad * free) ** 2).sum()
    eye = torch.eye(rank, dtype=V.dtype, device=V.device)
    M = torch.where(free[:, :, None] & free[:, None, :], M, eye.expand_as(M))
    rhs = torch.where(free, grad, torch.zeros_like(grad))

    d = _batched_spd_solve(M, rhs, jitter_rel)
    d = torch.where(free, d, torch.zeros_like(d))

    # An entry sitting at zero with a NEGATIVE gradient violates KKT and must move
    # inward. It is in the free set, but the coupled Newton direction for it can
    # still point outward (d > 0 here means V decreases), and the max-feasible-step
    # rule below would then zero the step for the whole row -- one such entry
    # pins all R of its neighbours, and a component that lands there stays dead.
    # Two-metric projection proper uses the scaled gradient for bound-adjacent
    # variables, so do that: d = grad / diag(M) < 0 moves the entry off the
    # bound. Interior entries keep the full Newton direction.
    diag_M = torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(torch.finfo(V.dtype).tiny)
    inward = (V <= eps_active) & (grad < 0)
    d = torch.where(inward, grad / diag_M, d)

    # Trust region, per row. The feasibility clamp below only bounds directions
    # that drive a factor toward zero; nothing bounds one that grows it. A row
    # whose Z has underflowed to zero (float32 loses exp(-X) past X ~ 88) carries
    # no curvature, so the damped solve returns a direction of order 1/tiny --
    # large but finite, so it survives every non-finite check and only overflows
    # on the next matmul. Keeping the step within the scale of the row itself
    # costs nothing when the Hessian is healthy.
    # The floor is at the problem's scale, not machine epsilon. A row pinned
    # near zero -- values of 1e-112 occur in float64 -- can only grow 16x per
    # step, and the loss-based stopping rule fires during that crawl: a float64
    # W solve stopped with a projected gradient 3e5 times the float32 one. In
    # float32 eps (1.2e-7 against rows of 0.08) happened to be a workable floor.
    # From 1e-3 of the mean row scale any row recovers in two or three steps.
    row_max = V.abs().amax(-1, keepdim=True)
    floor = torch.clamp(_TRUST_FLOOR * row_max.mean(), min=torch.finfo(V.dtype).eps)
    limit = 16.0 * torch.maximum(row_max, floor)
    d = torch.clamp(d, min=-limit, max=limit)

    slope = (grad * d).sum(-1)
    not_descent = (slope <= 0)[:, None]
    diag = torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(torch.finfo(V.dtype).tiny)
    d = torch.where(not_descent, torch.clamp(rhs / diag, min=-limit, max=limit), d)
    slope = (grad * d).sum(-1)

    # Largest step that keeps V >= 0 exactly, so no projection is needed afterwards.
    ratio = torch.where(d > 0, V / d.clamp_min(torch.finfo(V.dtype).tiny),
                        torch.full_like(d, float('inf')))
    alpha = torch.clamp(ratio.amin(-1), max=1.0)

    # X(alpha) along the step is exactly X - alpha * B, so the line search is
    # elementwise: no candidate factors and no extra matmuls are materialised.
    if axis == 0:
        B = d @ other
        base = rowwise(X, T, prep, 1, dtype=torch.float64)
        expand = lambda a: a[:, None]
    else:
        B = other @ d.T
        base = rowwise(X, T, prep, 0, dtype=torch.float64)
        expand = lambda a: a[None, :]

    accepted = torch.zeros_like(alpha)
    done = torch.zeros_like(alpha, dtype=torch.bool)
    num_backtracks = 0
    for _ in range(ls_max):
        trial = torch.where(done, torch.zeros_like(alpha), alpha)
        dim = 1 if axis == 0 else 0
        # The Armijo decrease can fall below the float32 resolution of the row
        # loss, which makes the test fail spuriously and backtrack to the cap.
        # Accept anything that is not measurably worse than the target; the
        # size of "measurably" is discussed at _ARMIJO_FLOOR.
        noise = _ARMIJO_FLOOR * torch.finfo(V.dtype).eps * base.abs()
        ok = (rowwise(X - expand(trial) * B, T, prep, dim, dtype=torch.float64)
              <= base - 1e-4 * trial * slope + noise) | (trial == 0)
        accepted = torch.where(ok & ~done, trial, accepted)
        done = done | ok
        if bool(done.all()):
            break
        alpha = alpha * 0.5
        num_backtracks += 1

    V_new = (V - accepted[:, None] * d).clamp_(min=0.0)
    V_new = torch.where(bound, torch.zeros_like(V_new), V_new)
    X_new = X - expand(accepted) * B
    if axis == 1:
        V_new = V_new.T.contiguous()
    return V_new, X_new, (num_backtracks, projected_gnorm2)


def block_newton_optimize(T, num_materials, max_steps, rel_tol, update_H=True,
                          convergence_check_interval=1, W_init=None, H_init=None,
                          jitter_rel=1e-9, compile_mode=None):
    """Alternating exact projected-Newton minimization of the NNAL."""
    _, _, rowwise, step_fn = _kernels(compile_mode)
    prep = _nnal_prep(T)
    W, H = W_init, H_init
    X = W @ H
    prev_loss = rowwise(X, T, prep, 1, dtype=torch.float64).sum()
    gnorm0 = None
    num_steps = 0
    for step in range(max_steps):
        X = W @ H                      # resynchronize against incremental drift
        W, X, info_W = step_fn(W, H, X, T, prep, 0, jitter_rel=jitter_rel)
        gnorm2 = info_W[1]
        if update_H:
            H, X, info_H = step_fn(H, W, X, T, prep, 1, jitter_rel=jitter_rel)
            gnorm2 = gnorm2 + info_H[1]
            # A component dead in BOTH factors has a zero Hessian block and zero
            # gradient in either step -- no Newton or gradient move reaches it.
            # Same remedy as the multiplicative update: re-seed it.
            W, H = _reseed_dead(W, H)
        num_steps = step + 1
        if rel_tol > 0 and num_steps % convergence_check_interval == 0:
            # rel_tol is the relative change in the loss between checks, the same
            # meaning it has for every other method. The sum is accumulated in
            # float64: in float32 a loss near 9e5 is quantized at ~0.06, coarser
            # than the per-step progress here, and the test would fire on noise.
            # The projected-gradient (KKT) test is kept as a fallback for data a
            # rank-`num_materials` model fits exactly -- the shifted loss then goes
            # to zero and its relative change stays O(1) forever, while the
            # gradient still vanishes. A KKT test alone is not enough because it is
            # relative to the gradient at the START, so a better initialization
            # makes the same rel_tol a stricter target: at dosage 100 the default
            # init ran 4x longer than a poor one for an identical loss.
            loss = rowwise(X, T, prep, 1, dtype=torch.float64).sum()
            gnorm = gnorm2.sqrt()
            if gnorm0 is None:
                gnorm0 = gnorm
            if (bool(torch.abs(loss - prev_loss) <= rel_tol * torch.abs(loss))
                    or bool(gnorm <= max(rel_tol ** 2, 100 * torch.finfo(T.dtype).eps) * gnorm0)):
                break
            prev_loss = loss
    return W, H, num_steps


def _joint_dot(a, b, c, d):
    return (a * b).sum() + (c * d).sum()


def _joint_blocks(flat, rows, cols, rank, free, jitter):
    """(B,Q) upper triangle -> damped SPD (B,R,R) with frozen vars set to identity."""
    M = flat.new_zeros(flat.shape[0], rank, rank)
    M[:, rows, cols] = flat
    M[:, cols, rows] = flat
    eye = torch.eye(rank, dtype=M.dtype, device=M.device)
    M = torch.where(free[:, :, None] & free[:, None, :], M, eye.expand_as(M))
    scale = torch.diagonal(M, dim1=-2, dim2=-1).amax(-1).clamp_min(torch.finfo(M.dtype).tiny)
    M = M + (jitter * scale)[:, None, None] * eye
    L, info = torch.linalg.cholesky_ex(M)
    return torch.where((info > 0)[:, None, None], eye.expand_as(L), L)


def _joint_newton_pcg(T, W, H, max_steps=50, cg_max=60, rel_tol=0.0, damping=1e-12,
                     precond_jitter=1e-8, prep=None, verbose=False, nnal=None, deriv=None, tilt_H=None,
                     nonneg_W=True):
    # nonneg_W=False drops the W >= 0 constraint: every coefficient is free and
    # the line search does not clamp W. The pixel problem stays strictly convex in
    # w for any real w. Used to estimate H without the truncation bias that the
    # bound induces (see bias_corrected_spectra); the physical W is re-solved
    # with the constraint afterwards.
    # tilt_H, if given, adds the linear term <tilt_H, H> to the objective: its
    # gradient enters gH and the line search, its Hessian is zero. Used by
    # bias_corrected_spectra to solve the modified profile likelihood's
    # stationarity conditions with this solver unchanged otherwise.
    def tilt(Hx):
        return 0.0 if tilt_H is None else (tilt_H * Hx).sum(dtype=torch.float64)
    nnal = stable_nnal if nnal is None else nnal
    deriv = stable_nnal_derivatives if deriv is None else deriv
    prep = _nnal_prep(T) if prep is None else prep
    W = W.clone(); H = H.clone()
    rank = W.shape[1]
    rows, cols = torch.triu_indices(rank, rank, device=W.device)
    loss = nnal(W @ H, T, prep, dtype=torch.float64) + tilt(H)
    lam = damping
    total_cg = 0
    step = 0
    gnorm0 = None
    for step in range(1, max_steps + 1):
        X = W @ H
        G, Z = deriv(X, T, prep)
        gW, gH = G @ H.T, W.T @ G
        if tilt_H is not None:
            gH = gH + tilt_H
        fW = ~((W <= 0) & (gW > 0)) if nonneg_W else torch.ones_like(W, dtype=torch.bool)
        fH = ~((H <= 0) & (gH > 0))
        gW, gH = gW * fW, gH * fH
        gnorm2 = _joint_dot(gW, gW, gH, gH)
        if not torch.isfinite(gnorm2) or gnorm2 == 0:
            break
        # KKT fallback: for data a rank-`num_materials` model fits exactly, the
        # shifted loss goes to zero and its relative change stays O(1) forever, but
        # the projected gradient still vanishes. The primary test is on the loss,
        # below, because this one is relative to the gradient at the start and so
        # tightens with a better initialization. Since this test only ever fires in
        # the loss -> 0 regime, it is set tight there: in a quadratic basin
        # loss ~ g^2, so a gradient ratio of rel_tol is a loss ratio of only
        # rel_tol^2 -- 3e-14 from a warm-up loss of 286 -- and machine precision
        # needs a gradient ratio of a few tens of eps.
        gnorm = gnorm2.sqrt()
        if gnorm0 is None:
            gnorm0 = gnorm
        elif rel_tol > 0 and gnorm <= max(rel_tol ** 2, 100 * torch.finfo(T.dtype).eps) * gnorm0:
            break

        LW = _joint_blocks(Z @ (H[rows] * H[cols]).T, rows, cols, rank, fW, precond_jitter)
        LH = _joint_blocks(((W[:, rows] * W[:, cols]).T @ Z).T, rows, cols, rank, fH.T, precond_jitter)

        def precond(rW, rH):
            zW = torch.cholesky_solve(rW.unsqueeze(-1), LW).squeeze(-1) * fW
            zH = torch.cholesky_solve(rH.T.unsqueeze(-1), LH).squeeze(-1).T * fH
            # Same degeneracy as in _batched_spd_solve: rows with no curvature can
            # make the preconditioner overflow. Fall back to the unpreconditioned
            # residual there rather than poisoning CG with a non-finite direction.
            zW = torch.where(torch.isfinite(zW), zW, rW)
            zH = torch.where(torch.isfinite(zH), zH, rH)
            return zW, zH

        def hvp(dW, dH):
            dW, dH = dW * fW, dH * fH
            dX = dW @ H + W @ dH
            ZdX = Z * dX
            return ((ZdX @ H.T + G @ dH.T) * fW + lam * dW,
                    (W.T @ ZdX + dW.T @ G) * fH + lam * dH)

        xW = torch.zeros_like(gW); xH = torch.zeros_like(gH)
        rW, rH = -gW, -gH
        zW, zH = precond(rW, rH)
        pW, pH = zW.clone(), zH.clone()
        rz = _joint_dot(rW, zW, rH, zH)
        r0 = _joint_dot(rW, rW, rH, rH)
        tol2 = (torch.clamp(gnorm2.sqrt().sqrt(), max=0.5) ** 2) * r0
        ncg = 0
        for ncg in range(1, cg_max + 1):
            ApW, ApH = hvp(pW, pH)
            pAp = _joint_dot(pW, ApW, pH, ApH)
            if pAp <= 0:
                if ncg == 1: xW, xH = zW, zH
                break
            a = rz / pAp
            xW = xW + a * pW; xH = xH + a * pH
            rW = rW - a * ApW; rH = rH - a * ApH
            if _joint_dot(rW, rW, rH, rH) <= tol2:
                break
            zW, zH = precond(rW, rH)
            rz_new = _joint_dot(rW, zW, rH, zH)
            b = rz_new / rz
            pW = zW + b * pW; pH = zH + b * pH
            rz = rz_new
        total_cg += ncg

        slope = _joint_dot(gW, xW, gH, xH)
        if slope >= 0:
            xW, xH = -gW, -gH
            slope = -gnorm2
        a, accepted = 1.0, False
        for _ in range(30):
            Wn = (W + a * xW).clamp_(min=0) if nonneg_W else W + a * xW
            Hn = (H + a * xH).clamp_(min=0)
            new_loss = nnal(Wn @ Hn, T, prep, dtype=torch.float64) + tilt(Hn)
            if torch.isfinite(new_loss) and new_loss <= loss + 1e-4 * a * slope:
                accepted = True; break
            a *= 0.5
        if not accepted:
            lam = lam * 10.0 if lam > 0 else 1e-12
            if lam > 1e6: break
            continue
        lam = max(lam * 0.3, 1e-14)
        # No stagnation heuristic here either. At the precision floor the line
        # search stops finding an acceptable step, which escalates the damping and
        # terminates through the lam > 1e6 path above -- a real signal rather than
        # a threshold on a quantity too noisy to threshold.
        # rel_tol is the relative change in the loss per accepted step, summed in
        # float64, the same meaning it has for every other method.
        rel_change = torch.abs(loss - new_loss) / torch.abs(new_loss).clamp_min(torch.finfo(torch.float64).tiny)
        W, H, loss = Wn, Hn, new_loss
        if rel_tol > 0 and bool(rel_change <= rel_tol):
            break
        if verbose:
            print(f'  step {step:3d} cg {ncg:3d} a {a:.3g} loss {loss.item():.6e}', flush=True)
    return W, H, step, total_cg


def joint_newton_optimize(T, num_materials, max_steps, rel_tol, update_H=True,
                          convergence_check_interval=1, W_init=None, H_init=None,
                          warmup_steps=5, cg_max=10, compile_mode=None):
    """Block-Newton warm-up followed by a joint (W,H) preconditioned Newton solve.

    Alternating methods stall at a linear rate once the fit is good, because they
    discard the W<->H coupling block of the Hessian; on a problem where an exact
    factorization exists, block-Newton alone plateaus around 1e-7. Solving for both
    factors jointly restores fast convergence to machine precision. The alternating
    method is still the right way to get into the basin, so it runs first.

    update_H=False is not supported here (the joint step updates both factors);
    it falls back to block-Newton alone.

    warmup_steps=5 and cg_max=10 were chosen by interleaved A/B at equal converged
    quality (rel_tol=1e-8) against the previous 10/20: 0.91x at dosage 3 on
    64x64 (a real 9% loss), 2.43x at dosage 100, 1.31x at dosage 3 on 128x128,
    same loss to six figures in all three. The cost model behind it: a block
    warm-up step costs 1.4x a one-CG-iteration joint step, and past a handful of
    them the joint solver makes better use of the time.
    """
    nnal_fn, deriv_fn, _, step_fn = _kernels(compile_mode)
    prep = _nnal_prep(T)
    W, H = W_init, H_init
    X = W @ H
    for _ in range(min(warmup_steps, max_steps)):
        X = W @ H
        W, X, _ = step_fn(W, H, X, T, prep, 0)
        if update_H:
            H, X, _ = step_fn(H, W, X, T, prep, 1)
    if not update_H:
        return W, H, min(warmup_steps, max_steps)
    remaining = max(0, max_steps - warmup_steps)
    if remaining == 0:
        return W, H, min(warmup_steps, max_steps)
    W, H, steps, _ = _joint_newton_pcg(T, W, H, max_steps=remaining, cg_max=cg_max,
                                       rel_tol=rel_tol, prep=prep, nnal=nnal_fn, deriv=deriv_fn)
    return W, H, warmup_steps + steps


# -----------------------------------------------------------------------
# Streaming factorization for data that does not fit in memory
# -----------------------------------------------------------------------
def _h_stats_accumulate(W, H, T, prep, rows, cols, deriv, rowwise):
    """Per-chunk sufficient statistics for one Newton step on H.

    The H-step of block_newton needs, for every wavelength bin, the gradient
    W^T G[:, k] and the Hessian W^T diag(Z[:, k]) W. Both are sums over pixels, so
    they accumulate across chunks: this returns one chunk's share, and the caller
    adds. The Hessian's upper triangle for all K bins comes out of a single GEMM
    against the Khatri-Rao product (W[:, rows] * W[:, cols]), exactly as in
    block_newton_step. The per-bin loss is the line search's baseline.
    """
    X = W @ H
    G, Z = deriv(X, T, prep)
    return W.T @ G, (W[:, rows] * W[:, cols]).T @ Z, rowwise(X, T, prep, 0, dtype=torch.float64)


def _h_direction(H, grad, flat, rows, cols, jitter_rel=1e-9):
    """Projected-Newton direction on H from accumulated statistics.

    Mirrors the H axis of block_newton_step -- two-metric projection, KKT release
    of bound entries with an inward gradient, trust region, descent safeguard,
    feasibility bound -- operating on the (K, R) transpose. Returns (d, slope,
    alpha_max) with one row/entry per bin.
    """
    V = H.T
    g = grad.T
    rank = V.shape[1]
    M = flat.new_zeros(flat.shape[1], rank, rank)
    M[:, rows, cols] = flat.T
    M[:, cols, rows] = flat.T
    eps_active = _ACTIVE_TOL * V.abs().amax(-1, keepdim=True).mean()
    bound = (V <= eps_active) & (g > 0)
    free = ~bound
    eye = torch.eye(rank, dtype=V.dtype, device=V.device)
    M = torch.where(free[:, :, None] & free[:, None, :], M, eye.expand_as(M))
    rhs = torch.where(free, g, torch.zeros_like(g))
    d = _batched_spd_solve(M, rhs, jitter_rel)
    d = torch.where(free, d, torch.zeros_like(d))
    diag_M = torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(torch.finfo(V.dtype).tiny)
    d = torch.where((V <= eps_active) & (g < 0), g / diag_M, d)
    row_max = V.abs().amax(-1, keepdim=True)
    floor = torch.clamp(_TRUST_FLOOR * row_max.mean(), min=torch.finfo(V.dtype).eps)   # same floor as block_newton_step
    limit = 16.0 * torch.maximum(row_max, floor)
    d = torch.clamp(d, min=-limit, max=limit)
    slope = (g * d).sum(-1)
    d = torch.where((slope <= 0)[:, None], torch.clamp(rhs / diag_M, min=-limit, max=limit), d)
    slope = (g * d).sum(-1)
    ratio = torch.where(d > 0, V / d.clamp_min(torch.finfo(V.dtype).tiny), torch.full_like(d, float('inf')))
    return d, slope, torch.clamp(ratio.amin(-1), max=1.0)


def stream_factorization(chunks, num_materials, max_passes=5, rel_tol=1e-6, warmup_pixels=16384,
                         w_rel_tol=1e-8, w_max_steps=300, ls_trials=4, device='cuda',
                         compile_mode=None, random_state=0, verbose=False, polish_dtype=None,
                         kkt_tol=None, stats=None):
    """Factorize a dataset too large for device memory, one chunk at a time.

    Args:
        chunks: A sequence of CPU tensors, each (pixels, bins), together making
            up T. Any indexable sequence works, so chunks may be lazily loaded
            (e.g. views into a memory-mapped array or an HDF5 dataset).
        num_materials: Factorization rank R.
        max_passes: Full passes over the data for polishing H. 0 stops after
            the subsample fit, which is the existing batched path.
        rel_tol: Stop polishing when a pass changes the total loss by less than
            this, relatively (float64 sum).
        warmup_pixels: Pixels drawn from the leading chunks to fit H initially.
        polish_dtype: If set (e.g. torch.float64), the polish passes run in that
            dtype -- chunks, W and H are cast on the way in -- while the warm-up
            stays in the chunks' native dtype. On an H100, whose kernels here are
            memory-bound, float64 costs about 2x; it is insurance against a
            float32 elementwise ceiling on H at very large P. The accumulated
            statistics are float64 regardless.
        kkt_tol: If set, also stop once the relative KKT residual of H,
            ||P(grad_H L)||_F / ||W^T T||_F with P the projection onto the
            feasible directions, falls below it. The residual is scale-free and
            independent of the initialization. rel_tol alone can stop inside a
            precision plateau: at millions of pixels per bin the line search
            cannot accept an improvement below its noise floor, the loss stops
            changing, and H is still measurably non-stationary. The residual
            tells the two apart, and the run says so when it stops that way.
        stats: Optional dict; receives 'loss' and 'kkt' lists, one entry per pass.

    Returns:
        (W_chunks, H, passes): W as a list of CPU tensors aligned with `chunks`.

    The two factors are asymmetric in a way that makes this work. W is per-pixel
    and separable: with H fixed each pixel's problem is independent, so W is
    solved chunk by chunk and never held whole on the device. H is shared by
    every pixel but holds only R * K values, and its Newton step needs only sums
    over pixels -- gradient, R x R Hessian per bin, per-bin loss -- which
    accumulate across chunks. One full pass therefore delivers one exact
    block-Newton step on H from all the data; a second pass evaluates the line
    search at a few trial step lengths at once. H starts from a joint_newton
    fit on a subsample, which is already within a few hundredths of a percent
    of the full-data optimum, so a handful of passes polish it.

    joint_newton is deliberately not streamed: each of its CG iterations would
    be a full pass, and at 10 or more per step that is hundreds of passes.
    """
    nnal_fn, deriv, rowwise, _ = _kernels(compile_mode)
    R = num_materials
    rows, cols = None, None
    W_chunks = [None] * len(chunks)

    # ---- H from a subsample of the leading chunks
    parts, n = [], 0
    for c in chunks:
        parts.append(c[: warmup_pixels - n]); n += parts[-1].shape[0]
        if n >= warmup_pixels:
            break
    T_sub = torch.cat(parts, 0).to(device)
    _, H, _ = nnal_factorization(T_sub, method='joint_newton', num_materials=R, max_steps=300,
                                 rel_tol=1e-6, compile_mode=compile_mode, random_state=random_state)
    del T_sub
    if polish_dtype is not None:
        H = H.to(polish_dtype)
    rows, cols = torch.triu_indices(R, R, device=H.device)
    prev_loss = None
    passes = 0

    def to_device(c):
        c = c.pin_memory().to(device, non_blocking=True) if c.device.type == 'cpu' else c
        return c if polish_dtype is None else c.to(polish_dtype)

    for p in range(max_passes + 1):
        # ---- pass A: solve W per chunk with H fixed, accumulate H statistics
        # Accumulate across chunks in float64 whatever the working dtype: these are
        # sums over every pixel, and the per-bin loss in particular must resolve
        # H improvements far smaller than the float32 ulp of a sum near 1e8.
        grad = torch.zeros(H.shape, dtype=torch.float64, device=H.device)
        flat = torch.zeros(rows.numel(), H.shape[1], dtype=torch.float64, device=H.device)
        base = torch.zeros(H.shape[1], dtype=torch.float64, device=H.device)
        scale = torch.zeros(H.shape, dtype=torch.float64, device=H.device)     # W^T T, the gradient's natural scale
        nxt = to_device(chunks[0])
        for i in range(len(chunks)):
            Tc = nxt
            if i + 1 < len(chunks):
                nxt = to_device(chunks[i + 1])            # prefetch overlaps the solve below
            prep = _nnal_prep(Tc)
            kw = dict(W_init=W_chunks[i].to(device=device, dtype=H.dtype)) if W_chunks[i] is not None else {}
            W, _, _ = nnal_factorization(Tc, method='block_newton', num_materials=R, max_steps=w_max_steps,
                                         rel_tol=w_rel_tol, update_H=False, H_init=H.clone(),
                                         compile_mode=compile_mode, **kw)
            W_chunks[i] = W.cpu()
            g_c, f_c, b_c = _h_stats_accumulate(W, H, Tc, prep, rows, cols, deriv, rowwise)
            grad += g_c; flat += f_c; base += b_c; scale += (W.T @ Tc).to(torch.float64)
            del Tc, W
        loss = base.sum(dtype=torch.float64)
        # Projected gradient: where H is zero only a negative gradient (a wish to grow) counts.
        pg = torch.where(H > 0, grad, grad.clamp(max=0))
        kkt = (pg.norm() / scale.norm()).item()
        if stats is not None:
            stats.setdefault('loss', []).append(loss.item()); stats.setdefault('kkt', []).append(kkt)
        if verbose:
            print(f'  pass {p}: full-data loss {loss.item():.6e}  KKT residual {kkt:.2e}', flush=True)
        if kkt_tol is not None and kkt <= kkt_tol:
            if verbose:
                print(f'  H is stationary to {kkt_tol:g}', flush=True)
            break
        if prev_loss is not None and rel_tol > 0 and bool(torch.abs(loss - prev_loss) <= rel_tol * torch.abs(loss)):
            if verbose and kkt_tol is not None:
                print(f'  loss stalled with KKT residual {kkt:.2e} > {kkt_tol:g}: the line search is at its '
                      f'precision floor in {H.dtype}; polish_dtype=torch.float64 lowers it', flush=True)
            break
        prev_loss = loss
        if p == max_passes:
            break

        # ---- one exact Newton step on H from the accumulated statistics
        d, slope, alpha_max = _h_direction(H, grad.to(H.dtype), flat.to(H.dtype), rows, cols)
        alphas = alpha_max[None, :] * (0.5 ** torch.arange(ls_trials, dtype=H.dtype, device=H.device))[:, None]

        # ---- pass B: per-bin loss at every trial step, accumulated over chunks
        trial = torch.zeros(ls_trials, H.shape[1], dtype=torch.float64, device=H.device)
        nxt = to_device(chunks[0])
        for i in range(len(chunks)):
            Tc = nxt
            if i + 1 < len(chunks):
                nxt = to_device(chunks[i + 1])
            prep = _nnal_prep(Tc)
            W = W_chunks[i].to(device)
            X = W @ H
            B = W @ d.T
            for t in range(ls_trials):
                trial[t] += rowwise(X - alphas[t][None, :] * B, Tc, prep, 0, dtype=torch.float64)
            del Tc, W, X, B
        # Same floor as block_newton_step (see _ARMIJO_FLOOR): the float32 sums are
        # gone, but elements whose step falls below ulp(X) still do not move.
        noise = _ARMIJO_FLOOR * torch.finfo(H.dtype).eps * base.abs()
        ok = trial <= base[None, :] - 1e-4 * alphas.double() * slope.double()[None, :] + noise[None, :]   # Armijo, per bin and trial
        # largest accepted trial per bin, else zero
        accepted = torch.where(ok.any(0), alphas.gather(0, ok.float().argmax(0, keepdim=True)).squeeze(0), torch.zeros_like(alpha_max))
        Ht = (H.T - accepted[:, None] * d).clamp_(min=0)
        # Same epsilon-active snap as block_newton_step: a bin component at the
        # bound with an outward gradient becomes exactly zero, not a residue.
        eps_active = _ACTIVE_TOL * Ht.abs().amax(-1, keepdim=True).mean()
        Ht = torch.where((Ht <= eps_active) & (grad.T.to(Ht.dtype) > 0), torch.zeros_like(Ht), Ht)
        H = Ht.T.contiguous()
        passes = p + 1
    return W_chunks, H, passes


def optimize(T: torch.Tensor, update, num_materials, max_steps, rel_tol, update_H=True,
             convergence_check_interval=1, W_init: torch.Tensor = torch.tensor([]),
             H_init: torch.Tensor = torch.tensor([]), compile_mode=None,
             extrapolate=False, extrapolate_check_every=5) -> torch.Tensor:
    """Factorize T into W and H by minimizing nonnegative attenuation loss."""

    if W_init.numel() == 0 and H_init.numel() == 0:
        # Initialize in the ATTENUATION domain. The model is X = W @ H with
        # X = -log(T), so seeding from a nonnegative factorization of T itself
        # approximates the wrong quantity: T is not low rank when its logarithm
        # is.
        #
        # Zero counts need a floor, and it matters what it is. A zero count only
        # says the attenuation exceeds that of the faintest pixel that did
        # register -- so it is floored at half a count relative to the smallest
        # genuine transmission, i.e. -log(T_min_real) + log 2. Upstream code
        # marks zero counts with a tiny positive value (1e-30), and flooring THERE
        # gives attenuation 69: at dosage_rate=1 half the entries are then 69 and
        # the mean is 34.5, which the initialization's zero-fill smears into every
        # component, putting X above 2000 where no attenuation exceeds 1. Any
        # transmission below 1e-12 is treated as a zero count; no real measurement
        # gets anywhere near that.
        real = T > 1e-12
        if bool(real.any()) and not bool(real.all()):
            floor = 0.5 * T[real].min()
            T_for_init = torch.where(real, T, floor)
        else:
            T_for_init = T.clamp_min(torch.finfo(T.dtype).tiny)
        W_init, H_init = nndsvda(-torch.log(T_for_init), n_components=num_materials)
    elif W_init.numel() == 0:
        # lstsq is unconstrained, so clamp before handing it to a nonnegative solver
        W_init = torch.linalg.lstsq(H_init.T, T.T)[0].T.clamp(min=0)
    elif H_init.numel() == 0:
        H_init = torch.linalg.lstsq(W_init, T)[0].clamp(min=0)

    if update in (block_newton_optimize, joint_newton_optimize):
        return update(T, num_materials, max_steps, rel_tol,
                      update_H=update_H, W_init=W_init, H_init=H_init,
                      convergence_check_interval=convergence_check_interval,
                      compile_mode=compile_mode)

    prep = _nnal_prep(T)
    W, H, aux = (W_init, H_init, 0.001)
    num_steps = 0

    if extrapolate:
        # Nesterov extrapolation around the update. The multiplicative update is a
        # fixed-point map g with linear convergence, and the classic momentum
        # sequence beta_k = (k-1)/(k+2) applied as y = clamp(x_k + beta (x_k - x_{k-1}), 0)
        # cuts its sweeps to a target 17x on the demo problem (6580 -> 380) at 1.3x
        # the cost per sweep -- 13x in wall clock, which lifts it from 40-70x
        # slower than joint_newton to parity. The extrapolated point is only an
        # input; the answer is always the last PLAIN iterate, so the fixed-point
        # set is untouched. A function-value safeguard restarts the momentum
        # whenever an extrapolated sweep raises the loss; it uses the raw
        # sum(exp(-X) + T X) minus the constant that separates it from the shifted
        # loss (0.5 ms against 4.3 ms for stable_nnal -- checking with the full
        # loss every sweep is what limited an earlier version to 4x), and is
        # evaluated every extrapolate_check_every sweeps: the sweeps in between are
        # unguarded, which is the price of the speed. rel_tol keeps its per-step
        # meaning by comparing across the check interval.
        log_T, positive, _, _ = prep
        const = torch.sum(torch.where(positive, T * (1.0 - log_T), torch.zeros_like(T)),
                          dtype=torch.float64)
        cheap = lambda X: torch.sum(torch.exp(-X) + T * X, dtype=torch.float64) - const
        noise = 1e-9                                  # fp32 elementwise noise on the raw sum is ~1e-10 relative
        Wp, Hp = W, H                                 # last plain iterate: the answer
        Wy, Hy = W, H                                 # extrapolated input to the sweep
        prev, j = None, 0
        for i in range(max_steps):
            Wn, Hn, aux = update(Wy, Hy, T, aux, update_H=update_H, prep=prep)
            num_steps = i + 1
            j += 1
            if num_steps % extrapolate_check_every == 0:
                L = cheap(Wn @ Hn)
                if prev is not None and (not torch.isfinite(L) or bool(L > prev * (1.0 + noise))):
                    Wy, Hy, j = Wp, Hp, 0                 # momentum hurt: restart from the last plain point
                    continue
                if prev is not None and rel_tol > 0 and bool(
                        torch.abs(L - prev) <= rel_tol * extrapolate_check_every * torch.abs(L)):
                    Wp, Hp = Wn, Hn
                    break
                prev = L
            beta = (j - 1) / (j + 2) if j > 1 else 0.0
            Wy = (Wn + beta * (Wn - Wp)).clamp_(min=0)
            Hy = (Hn + beta * (Hn - Hp)).clamp_(min=0)
            Wp, Hp = Wn, Hn
        return Wp, Hp, num_steps

    # Converge on a float64 sum. In float32 the loss is quantized at ~1e-7
    # relative, so at low dosage the multiplicative methods saw two identical
    # consecutive losses and stopped after two iterations.
    prev_loss = stable_nnal(W @ H, T, prep, dtype=torch.float64)
    for i in range(max_steps):
        # prep is threaded in rather than rebuilt inside `update`: it holds a
        # Python bool, and recomputing it inside a torch.compile region forces a
        # graph break on the .all() that produces it.
        W, H, aux = update(W, H, T, aux, update_H=update_H, prep=prep)
        num_steps = i + 1
        # Only evaluate the loss when it is actually needed: it costs a
        # (pixels x rank x bins) matmul plus a full elementwise pass.
        if rel_tol > 0 and num_steps % convergence_check_interval == 0:
            loss_new = stable_nnal(W @ H, T, prep, dtype=torch.float64)
            converged = torch.abs(loss_new - prev_loss) / (prev_loss + 1e-30) < rel_tol
            prev_loss = loss_new
            if converged:
                break

    return W, H, num_steps

def nnal_factorization(T: torch.Tensor, method='quasi_newton', num_materials=3, max_steps=1000,
                       rel_tol=1e-10, batch_size=None, compile_mode=None, random_state=0,
                       **kwargs) -> torch.Tensor:
    """Factorize T ~= exp(-W @ H), W, H >= 0, by minimizing the non-negative attenuation loss.

    rel_tol is the relative change in the loss per step (summed in float64) at
    which a method stops, and means the same thing for every method. It is not
    worth the same amount of convergence, though: joint_newton is within 0.002%
    of its optimum at 1e-6, while block_newton and quadratic converge linearly
    and at 1e-6 can stop on a plateau with a percent still to gain --
    block_newton at dosage 100 stopped 1.4% short at 1e-6 and 0.001% short at
    1e-8. Use 1e-8 or tighter for those two. mann_multiplicative is extrapolated
    by default (see optimize) and reaches joint_newton's answer in comparable
    wall clock. On data the model fits exactly the loss goes to zero and this
    test never fires; a projected-gradient test then takes over and runs to
    machine precision.

    compile_mode: any non-None value compiles the elementwise hot kernels of
    block_newton and joint_newton (2-3x per solve, bit-identical) at a one-off
    cost of 4.5-18 s; worth it for repeated solves, not for one. For the other
    methods the update function itself is compiled, as before.

    random_state seeds the pixel permutation the batched path uses to choose the
    subsample H is fitted on. It defaults to 0 so two batched runs on the same
    data give the same answer; pass None for fresh entropy each call. Left
    unseeded, block_newton took 318 steps in one run and 565 in the next on
    identical data, which made the batched path impossible to benchmark.
    """
    if method == 'quasi_newton':
        update = newton_update
    elif method == 'mann_multiplicative':
        update = multiplicative_update
    elif method == 'quadratic':
        update = quadratic_update
    elif method == 'block_newton':
        update = block_newton_optimize
    elif method == 'joint_newton':
        update = joint_newton_optimize
    else:
        raise ValueError("Invalid method. Choose 'joint_newton', 'block_newton', "
                         "'quasi_newton', 'mann_multiplicative' or 'quadratic'.")

    if update in (block_newton_optimize, joint_newton_optimize):
        # These take compile_mode themselves and compile their hot kernels, not the
        # driver: see _kernels. Wrapping the driver in torch.compile is what the
        # old branch did for the other methods, and it does nothing useful here.
        kwargs['compile_mode'] = compile_mode
    elif compile_mode is not None:
        compile_options = {
            "triton.cudagraphs": False,
            "max_autotune": compile_mode == "max-autotune",
        }
        update = torch.compile(
            update,
            options=compile_options,
        )

    if update is multiplicative_update:
        # Nesterov extrapolation is on by default for the multiplicative update:
        # 13x in wall clock, same fixed points. Pass extrapolate=False to disable.
        kwargs.setdefault('extrapolate', True)
    kwargs.update({
        'update': update,
        'num_materials': num_materials,
        'max_steps': max_steps,
        'rel_tol': rel_tol
    })

    if batch_size is None:
        return optimize(T, **kwargs)

    num_pixels = T.shape[0]
    num_batches = int(np.ceil(num_pixels / batch_size))

    # Randomly permute the pixel indices for batching, reproducibly by default
    batch_idxs = np.random.default_rng(random_state).permutation(num_pixels)

    # H is shared by every pixel and holds only num_materials * N_k values, so one
    # batch determines it about as well as all of them do. Fit it once on a random
    # subsample, then solve the per-pixel coefficients batch by batch with H held
    # fixed -- that part is separable across pixels, so batching costs nothing but
    # the loop. The previous version instead factored every batch, then factored
    # the stacked spectra again with sklearn to reconcile them, which cost one full
    # solve per batch and a host round trip.
    _, H, i_total = optimize(T[batch_idxs[:batch_size]], **kwargs)

    # Compute material coefficients for each batch using the unified spectra
    W = torch.zeros((num_pixels, num_materials), dtype=T.dtype, device=T.device)
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_pixels)
        T_batch = T[batch_idxs[start_idx:end_idx]]
        W_batch, _, i = optimize(T_batch, **kwargs, update_H=False, H_init=H)
        W[batch_idxs[start_idx:end_idx]] = W_batch
        i_total += i

    return W, H, i_total

def hyper_denoise(data, dataset_type='attenuation', num_materials=None, safety_factor=2, beta_loss='frobenius',
                  max_iter=300, tolerance=1e-10, batch_size=2 ** 27, subspace_basis=None, random_state=None,
                  verbose=1):
    """
    Denoise a hyperspectral dataset using dehydration and rehydration as described in:

    M. S. N. Chowdhury, D. Yang, S. Tang, S. V. Venkatakrishnan, H. Z. Bilheux, G. T. Buzzard, and C. A. Bouman, "Fast Hyperspectral Neutron Tomography," IEEE Transactions on Computational Imaging, vol. 11, pp. 663–677, 2025. doi:10.1109/TCI.2025.3567854

    The function works for any rank array. However, the spectral axis must be the last axis.

    Args:
        data: Hyperspectral data array with arbitrary axes and a spectral axis in the last position.
        dataset_type: 'attenuation' or 'transmission' where attenuation = -log(transmission). Defaults to 'attenuation'.
        num_materials: Number of materials in the sample. If None, the number is estimated automatically from
            the data. Defaults to None.
        safety_factor: A multiplier (≥ 1) applied to the number of materials to set the subspace dimension.
            Defaults to 2.
        beta_loss: Beta divergence minimized in NMF. Can be 'frobenius' or 'kullback-leibler'. Defaults to 'frobenius'.
        max_iter: Maximum iterations for the NMF solver. Defaults to 300.
        tolerance: Convergence tolerance for the NMF solver. Defaults to 1e-10.
        batch_size: Size of data processed per batch. Useful for large datasets to limit memory usage. Defaults to 2^27.
        subspace_basis: Pre-computed subspace basis spectra of shape :math:`(N_s, N_k)`. If None, the basis spectra are
            estimated directly from the data. Defaults to None.
        random_state: Random seed for reproducibility of the NMF initialization and batch row sampling. If None,
            the factors vary from run to run. Defaults to None.
        verbose: Verbosity level. If 0, prints nothing; if 1, prints details; if >1, also generates plots. Defaults to 1.

    Returns:
        Denoised hyperspectral data with the same shape as the input data.

    Example:
        >>> denoised_data = hyper_denoise(data, num_materials=5, safety_factor=2)
        >>> data.shape, denoised_data.shape
        ((N_x, N_y, N_z, ..., N_k), (N_x, N_y, N_z, ..., N_k))

    """
    # --------------------- Dehydrate ----------------------
    dehydrated_data = dehydrate(data,
                                dataset_type=dataset_type,
                                num_materials=num_materials,
                                safety_factor=safety_factor,
                                beta_loss=beta_loss,
                                max_iter=max_iter,
                                tolerance=tolerance,
                                batch_size=batch_size,
                                subspace_basis=subspace_basis,
                                random_state=random_state,
                                verbose=verbose)

    # --------------------- Rehydrate ----------------------
    denoised_data = rehydrate(dehydrated_data)

    return denoised_data


def dehydrate(data, dataset_type='attenuation', num_materials=None, safety_factor=2, beta_loss='frobenius',
              max_iter=300, tolerance=1e-10, batch_size=2 ** 27, subspace_basis=None, random_state=None,
              verbose=1):
    """
    Dehydrate/compress a hyperspectral dataset onto a low-dimensional subspace as described in:

    M. S. N. Chowdhury, D. Yang, S. Tang, S. V. Venkatakrishnan, H. Z. Bilheux, G. T. Buzzard, and C. A. Bouman, "Fast Hyperspectral Neutron Tomography," IEEE Transactions on Computational Imaging, vol. 11, pp. 663–677, 2025. doi:10.1109/TCI.2025.3567854

    The function works for any rank array. However, the spectral axis must be the last axis.

    Args:
        data: Hyperspectral data array with arbitrary axes and a spectral axis of length :math:`N_k` in the last position.
        dataset_type: 'attenuation' or 'transmission' where attenuation = -log(transmission). Defaults to 'attenuation'.
        num_materials: Number of materials in the sample :math:`N_m`. If None, the number is estimated automatically from
            the data. Defaults to None.
        safety_factor: A multiplier (≥ 1) applied to the number of materials to set the subspace dimension :math:`N_s`.
            Defaults to 2.
        beta_loss: Beta divergence minimized in NMF. Can be 'frobenius' or 'kullback-leibler'. Defaults to 'frobenius'.
        max_iter: Maximum iterations for the NMF solver. Defaults to 300.
        tolerance: Convergence tolerance for the NMF solver. Defaults to 1e-10.
        batch_size: Size of data processed per batch. Useful for large datasets to limit memory usage. Defaults to 2^27.
        subspace_basis: Pre-computed subspace basis spectra of shape :math:`(N_s, N_k)`. If None, the basis spectra are
            estimated directly from the data. Defaults to None.
        random_state: Random seed for reproducibility of the NMF initialization and batch row sampling. The NMF
            factorization is not unique, so with the default of None the returned factors vary from run to run even
            though their product is stable; pass an int to make a run reproducible. Defaults to None.
        verbose: Verbosity level. If 0, prints nothing; if 1, prints details; if >1, also generates plots. Defaults to 1.

    Returns:
        A list containing the dehydrated hyperspectral dataset in the form [subspace_data, subspace_basis, dataset_type].
            - subspace_data: ndarray with same shape as input data except the last axis length is :math:`N_s`.
            - subspace_basis: ndarray of shape :math:`(N_s, N_k)`, where rows are subspace basis spectra.
            - dataset_type: Can be 'attenuation' or 'transmission' where attenuation = -log(transmission).

    Example:
        >>> [subspace_data, subspace_basis, dataset_type] = dehydrate(data, num_materials=5, safety_factor=2)
        >>> data.shape, subspace_data.shape, subspace_basis.shape
        ((N_x, N_y, N_z, ..., N_k), (N_x, N_y, N_z, ..., 10), (10, N_k))
    """
    epsilon = 1e-3  # Define epsilon

    # --------------- Dataset type validation --------------
    if dataset_type not in ('attenuation', 'transmission'):
        raise ValueError("'dataset_type' must be either 'attenuation' or 'transmission'.")

    # ------------------ Data preparation ------------------
    data_shape = data.shape
    num_bands = data_shape[-1]
    num_points = data.size // num_bands
    data = data.reshape(num_points, num_bands).astype(np.float64)  # Reshape to 2D and cast to float64 for stability

    if dataset_type == 'transmission':
        # Initial cleanup in the transmission domain to get rid of defective measurements
        data = hyper_denoise(data,
                             dataset_type='attenuation',
                             num_materials=num_materials,
                             safety_factor=safety_factor * 3,
                             beta_loss=beta_loss,
                             max_iter=max_iter,
                             tolerance=tolerance,
                             batch_size=batch_size,
                             random_state=random_state,
                             verbose=0)
        data[data < epsilon] = epsilon
        data = - np.log(data)  # Convert to attenuation

    data[data < 0] = 0  # Enforce non-negativity

    if subspace_basis is not None:
        subspace_basis = np.asarray(subspace_basis, dtype=np.float64)  # Cast to float64 for stability

    # --------------------- Batch setup ---------------------
    num_points_batch = max(1, batch_size // num_bands)  # Number of hyperspectral points per batch
    num_batches = int(np.ceil(num_points / num_points_batch))  # Number of batches

    # ------------------- NMF solver setup ------------------
    if beta_loss == 'frobenius':
        solver = 'cd'  # Coordinate Descent
    elif beta_loss == 'kullback-leibler':
        solver = 'mu'  # Multiplicative Update
    else:
        warnings.warn(f"Invalid beta_loss '{beta_loss}' specified: falling back to 'frobenius'.")
        beta_loss = 'frobenius'
        solver = 'cd'

    # ------------- Subspace dimension setup -----------------
    if subspace_basis is not None:
        subspace_dimension = subspace_basis.shape[0]
    elif num_materials is not None:
        subspace_dimension = int(np.ceil(safety_factor * num_materials))
    else:
        subspace_dimension = _estimate_subspace_dimension(data, safety_factor=safety_factor,
                                                          random_state=random_state, verbose=verbose)

    # ------- Subspace basis estimation for multi-batch ------
    if subspace_basis is None and num_batches > 1:
        row_idx = np.random.default_rng(random_state).permutation(num_points)
        subspace_basis_batch = [None] * num_batches

        # Estimate subspace basis for each batch using NMF
        for batch in range(num_batches):
            b_start = batch * num_points_batch
            b_stop = min((batch + 1) * num_points_batch, num_points)
            batch_data = data[row_idx[b_start: b_stop]]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, subspace_basis_batch[batch], _ = nmf(batch_data,
                                                        n_components=subspace_dimension,
                                                        init='nndsvd',
                                                        beta_loss=beta_loss,
                                                        solver=solver,
                                                        tol=tolerance,
                                                        max_iter=max(50, max_iter // num_batches),
                                                        random_state=random_state,
                                                        update_H=True)

        # Estimate final subspace basis from batch estimations using NMF
        subspace_basis_batch = np.reshape(np.array(subspace_basis_batch), (-1, num_bands))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, subspace_basis, _ = nmf(subspace_basis_batch,
                                       n_components=subspace_dimension,
                                       init='nndsvd',
                                       beta_loss=beta_loss,
                                       solver=solver,
                                       tol=tolerance,
                                       max_iter=max_iter,
                                       random_state=random_state)

    # --------------- Subspace data estimation ---------------
    if num_batches == 1:
        nmf_init, update_basis = 'nndsvd', True
    else:
        nmf_init, update_basis = 'custom', False

    # Estimate subspace data in batches using NMF
    subspace_data = np.zeros((num_points, subspace_dimension))
    for batch in range(num_batches):
        b_start = batch * num_points_batch
        b_stop = min((batch + 1) * num_points_batch, num_points)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            subspace_data[b_start: b_stop], subspace_basis, _ = nmf(data[b_start: b_stop],
                                                                    n_components=subspace_dimension,
                                                                    init=nmf_init,
                                                                    H=subspace_basis,
                                                                    beta_loss=beta_loss,
                                                                    solver=solver,
                                                                    tol=tolerance,
                                                                    max_iter=max_iter,
                                                                    random_state=random_state,
                                                                    update_H=update_basis)

    # ------------------ Final formatting -------------------
    subspace_data = subspace_data.reshape(*data_shape[:-1], -1)  # Reshape to original dimensions (except last axis)
    subspace_data = np.asarray(subspace_data, dtype=np.float32)  # Cast to float32 to reduce memory footprint
    subspace_basis = np.asarray(subspace_basis, dtype=np.float32)  # Cast to float32 to reduce memory footprint
    dehydrated_data = [subspace_data, subspace_basis, dataset_type]  # Package outputs for return

    # --------------- Print details if required -------------
    if verbose >= 1:
        print("dehydrate(): ")
        print("   -Number of data batches: ", num_batches)
        print("   -Original spectral dimension: ", data.shape[-1])
        print("   -Subspace dimension: ", subspace_data.shape[-1])

    return dehydrated_data


def rehydrate(dehydrated_data, hyperspectral_idx=None):
    """
    Rehydrate/decompress selected spectral bins from dehydrated hyperspectral data as described in:

    M. S. N. Chowdhury, D. Yang, S. Tang, S. V. Venkatakrishnan, H. Z. Bilheux, G. T. Buzzard, and C. A. Bouman, "Fast Hyperspectral Neutron Tomography," IEEE Transactions on Computational Imaging, vol. 11, pp. 663–677, 2025. doi:10.1109/TCI.2025.3567854

    Args:
        dehydrated_data: Dehydrated hyperspectral data in the form [subspace_data, subspace_basis, dataset_type]:

            - subspace_data: ndarray with arbitrary axes and a subspace axis of length :math:`N_s` in the last position.
            - subspace_basis: ndarray of shape :math:`(N_s, N_k)`, where rows are subspace basis spectra.
            - dataset_type: 'attenuation' or 'transmission' where attenuation = -log(transmission).
        hyperspectral_idx: A list of :math:`N_h` indices along the original spectral axis to rehydrate. If None, all :math:`N_k`
            spectral bins are rehydrated. Defaults to None.

    Returns:
        Rehydrated/decompressed hyperspectral data with the same shape as the input subspace_data except the last axis
        length is :math:`N_h (N_h <= N_k)`.

    Example:
        >>> hyper_data = rehydrate([subspace_data, subspace_basis, dataset_type], hyperspectral_idx=[5, 10, 15])
        >>> subspace_data.shape, subspace_basis.shape, hyper_data.shape
        ((N_x, N_y, N_z, ..., N_s), (N_s, N_k), (N_x, N_y, N_z, ..., 3))
    """
    [subspace_data, subspace_basis, dataset_type] = dehydrated_data  # Unpack data

    # Retrieve original data dimensions
    if hyperspectral_idx is None:
        rehydrated_data = subspace_data @ subspace_basis
    else:
        rehydrated_data = subspace_data @ subspace_basis[:, hyperspectral_idx]

    if dataset_type == 'transmission':
        rehydrated_data = np.exp(-rehydrated_data)  # Convert to transmission

    return rehydrated_data


def _estimate_subspace_dimension(data, safety_factor=2, noise_fit_window=[25.0, 75.0], threshold=1.5, random_state=None,
                                 verbose=1):
    """
    Estimate the signal subspace dimension using a log-linear fit to singular values.

    Args:
        data: 2D array of shape (num_samples, :math:`N_k`). Values should be real.
        safety_factor: Multiplicative factor ≥ 1 used to scale the initial estimate of subspace dimension and ensure
            safer final choice. Defaults to 2.
        noise_fit_window: Two-element list or tuple [start_percent, stop_percent] indicating the percentile window (0–100)
            over which the singular value fitting is performed. Defaults to [25.0, 75.0].
        threshold: Multiplicative factor to define the cutoff relative to the predicted singular values. Defaults to 1.5.
        random_state: Random seed for reproducibility of row sampling and SVD. Defaults to None.
        verbose: Verbosity level. If >1, plots singular values, fit, and threshold curves. Defaults to 1.

    Returns:
        Estimated dimension of the signal subspace (positive integer).
    """
    if data.ndim != 2:
        raise ValueError("`data` must be a 2D array shaped (samples, N_k).")

    n_points, n_bands = data.shape

    # Decide how many rows to sample for speed/robustness
    sample_size = min(n_points, n_bands)

    # Sample rows without replacement
    rng = np.random.default_rng(random_state)
    row_idx = rng.choice(n_points, size=sample_size, replace=False)

    # Cast to float64 for numerical stability in svd
    Y = np.asarray(data[row_idx, :], dtype=np.float64)

    # Compute singular values via randomized SVD
    _, s, _ = randomized_svd(Y, n_components=sample_size, random_state=random_state)

    # Guard against degenerate cases
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return 0

    # Extract start and stop percent from noise_fit_window
    start_percent, stop_percent = noise_fit_window
    # Fit window around percentile: [percentile-10, percentile+10], in s-index space
    start_idx = int(np.floor((start_percent / 100.0) * s.size))
    stop_idx = int(np.ceil((stop_percent / 100.0) * s.size))

    # Clip and ensure at least 2 points
    start_idx = max(0, min(start_idx, s.size - 2))
    stop_idx = max(start_idx + 2, min(stop_idx, s.size))

    # Fit log(s) ≈ a*n + b on [start_idx:stop_idx]
    n = np.arange(s.size)
    a, b = np.polyfit(n[start_idx:stop_idx], np.log(s[start_idx:stop_idx] + 1e-12), 1)

    # Predicted singular values for all indices
    s_pred = np.exp(a * n + b)

    # Compute tau by scaling the predicted singular values with the threshold
    tau = threshold * s_pred

    # Consider singular values > the corresponding tau values to be associated with signals
    signal_flag = s > tau
    num_materials = int(np.sum(signal_flag[:start_idx]))

    if verbose > 1:
        plt.figure()
        plt.semilogy(s, label='s: actual singular values from data (signal + noise)')
        plt.semilogy(s_pred, label='s_pred: predicted singular values from noise model')
        plt.semilogy(tau, label='tau: noise and signal discriminator (threshold x s_pred)')
        plt.title("Modeling noise singular values for number of material estimation")
        plt.xlabel("singular value index")
        plt.ylabel("singular value")
        plt.legend()

    # Multiply by safety factor
    subspace_dimension = int(np.ceil(safety_factor * num_materials))

    return max(1, subspace_dimension)


# -----------------------------------------------------------------------
# HDF5 Import/Export Utilities for Hyperspectral Neutron Data/Metadata
# -----------------------------------------------------------------------


# Description of the allowed keys
KEY_DESCRIPTIONS = {
    "dataset_name": "Character string with the name of the dataset.",
    "dataset_type": "'attenuation' or 'transmission'.",
    "dataset_modality": "'hyperspectral neutron'.",
    "wavelengths": "Array of wavelength values in Angstroms.",
    "alu_unit": "Character string defining geometry unit (e.g., 'mm' or 'cm').",
    "alu_value": "Float that represents the value of 1 ALU in the defined unit.",
    "delta_det_channel": "Detector channel spacing in ALU.",
    "delta_det_row": "Detector row spacing in ALU.",
    "dataset_geometry": "'parallel' or 'cone'.",
    "angles": "Array of view angles in degrees.",
    "det_channel_offset": "Assumed offset between center of rotation and center of detector in ALU.",
    "source_detector_dist": "Distance from source to detector in ALU.",
    "source_iso_dist": "Distance from source to iso in ALU."
}

# Acceptable input options for certain keys
VALIDATION_RULES = {
    "dataset_type": (None, "attenuation", "transmission"),
    "dataset_modality": (None, "hyperspectral neutron"),
    "dataset_geometry": (None, "parallel", "cone"),
}

# Allowed keys derived from the KEY_DESCRIPTIONS
ALLOWED_KEYS = list(KEY_DESCRIPTIONS.keys())


def _validate_key(key, value):
    """Validate categorical keys according to VALIDATION_RULES."""
    if key in VALIDATION_RULES and value not in VALIDATION_RULES[key]:
        valid_options = [v for v in VALIDATION_RULES[key] if v is not None]
        warnings.warn(f"Invalid '{key}': should be one of {valid_options}.")


def _with_key_docstring(style):
    """Function to insert key descriptions into docstrings."""
    indent = "\t- " if style == "dict" else "\t"
    text = "\n".join(f"{indent}{k}: {v}" for k, v in KEY_DESCRIPTIONS.items())

    def decorator(func):
        if func.__doc__:
            func.__doc__ = func.__doc__.replace("{_KEY_DOCS}", text)
        return func

    return decorator


@_with_key_docstring("dict")
def import_hsnt_data_hdf5(filename):
    """
    Import a hyperspectral dataset and metadata from an HDF5 file.

    Args:
        filename: Path to the HDF5 file.

    Returns:
        A list containing hyperspectral data and parameters in the form [data, metadata].
            - data: ndarray with spectral last axis (hyperspectral form), a list (dehydrated form), or None.
            - metadata: A dictionary with the keys shown below.

    Keys:
    {_KEY_DOCS}
    """
    data = None
    metadata = {key: None for key in ALLOWED_KEYS}

    try:
        with h5py.File(filename, "r") as f:
            group = f

            # Check if data is dehydrated/compressed
            dehydrated = all(k in group for k in ["subspace_data", "subspace_basis", "dataset_type"])

            # Importing data
            if dehydrated:
                dataset_type = group["dataset_type"][()]
                if isinstance(dataset_type, (bytes, np.bytes_)):
                    dataset_type = dataset_type.decode()
                data = [group["subspace_data"][()],
                        group["subspace_basis"][()],
                        dataset_type]
            elif "data" in group:
                data = group["data"][()]
            else:
                warnings.warn(f"No HSNT data found in HDF5 file '{filename}'. Returning data=None.")

            # Importing metadata
            for key in ALLOWED_KEYS:
                if key in group:
                    value = group[key][()]
                    if isinstance(value, (bytes, np.bytes_)):
                        value = value.decode()
                    elif isinstance(value, np.ndarray) and value.shape == ():
                        value = value.item()
                    metadata[key] = value
    except Exception as error:
        warnings.warn(f"Could not import HSNT data from HDF5 file '{filename}': {error}. Returning data=None.")
        data = None

    # Validate categorical keys
    for key, value in metadata.items():
        _validate_key(key, value)

    return [data, metadata]


@_with_key_docstring("arg")
def create_hsnt_metadata(**kwargs):
    """
    Create a dictionary of parameters (metadata) associated with a hyperspectral neutron dataset.

    Args:
    {_KEY_DOCS}

    Returns:
        dict: Dictionary containing hyperspectral neutron dataset parameters (metadata).

    Example:
        >>> metadata = create_hsnt_metadata(
        ...     dataset_name="sample1",
        ...     dataset_type="attenuation",
        ...     dataset_modality="hyperspectral neutron",
        ...     wavelengths=np.linspace(1.0, 5.0, 50),
        ...     alu_unit="mm",
        ...     alu_value=1.0,
        ...     dataset_geometry="parallel",
        ...     angles=np.linspace(0, 180, 10)
        ... )
        >>> print(metadata["dataset_name"])
        sample1
    """
    # Warn for unexpected keyword arguments
    for key in kwargs.keys():
        if key not in ALLOWED_KEYS:
            warnings.warn(f"Ignoring invalid key '{key}' in arguments.")

    metadata = {k: kwargs.get(k, None) for k in ALLOWED_KEYS}

    # Validation
    for key, value in metadata.items():
        _validate_key(key, value)

    return metadata


@_with_key_docstring("dict")
def export_hsnt_data_hdf5(filename, data, metadata=None):
    """
    Export a hyperspectral dataset and metadata to an HDF5 file.

    Args:
        filename: Path to the HDF5 file.
        data: ndarray with spectral last axis (hyperspectral form) or a list (dehydrated form).
        metadata: A dictionary with the keys shown below. Use create_hsnt_metadata to create a metadata dictionary.

    Keys:
    {_KEY_DOCS}

    Returns:
        None. Creates an HDF5 file with the corresponding structure.
    """
    if metadata is None:
        metadata = {}

    # Check if data is dehydrated/compressed
    dehydrated = (isinstance(data, list)
                  and len(data) == 3
                  and isinstance(data[2], str)
                  and data[2] in VALIDATION_RULES["dataset_type"][1:])

    # Validate categorical keys before writing
    for key, value in metadata.items():
        _validate_key(key, value)

    with h5py.File(filename, "w") as f:
        group = f

        # Exporting data
        if dehydrated:
            group.create_dataset("subspace_data", data=data[0])
            group.create_dataset("subspace_basis", data=data[1])
            group.create_dataset("dataset_type", data=np.bytes_(data[2]))
        else:
            group.create_dataset("data", data=data)

        # Exporting metadata
        for key, value in metadata.items():
            if key not in ALLOWED_KEYS:
                warnings.warn(f"Ignoring invalid key '{key}' in metadata.")
                continue
            if value is None or (key == "dataset_type" and dehydrated):
                continue
            if isinstance(value, str):
                group.create_dataset(key, data=np.bytes_(value))
            else:
                group.create_dataset(key, data=value)


# -----------------------------------------------------------------------
# Noisy Hyperspectral Neutron Data Simulation Function (Ni, Cu, and Al)
# -----------------------------------------------------------------------


def generate_hyper_data(material_basis, num_angles=1, detector_rows=64, detector_columns=64, dosage_rate=300,
                        material_density=None, noisy=True, verbose=1):
    """
    Simulate noisy hyperspectral neutron attenuation data for :math:`N_m=3` materials (Ni, Cu, Al) and :math:`N_k` wavelength bins.

    Args:
        material_basis: ndarray of shape :math:`(N_m, N_k)`, where rows are material linear attenuation coefficient spectra.
        num_angles: Number of view angles :math:`(N_v)`. Defaults to 1.
        detector_rows: Number of rows in the detector :math:`(N_r)`. Defaults to 64.
        detector_columns: Number of columns in the detector :math:`(N_c)`. Defaults to 64.
        dosage_rate: Neutron dosage rate during hyperspectral data collection. Defaults to 300.
        material_density: Material density (vol. fraction) for Ni, Cu, and Al. Defaults to {"Ni": 0.2, "Cu": 0.2, "Al": 1.0}.
        noisy: Whether to generate noisy data. Defaults to True.
        verbose: Verbosity level. If 0, prints nothing; if 1, prints details; if >1, also generates plots. Defaults to 1.

    Returns:
        A list in the form [noisy_hyper_projection, angles, gt_hyper_projection].
            - noisy_hyper_projection: Simulated noisy hyperspectral data of shape :math:`(N_v, N_r, N_c, N_k)`.
            - angles: ndarray of view angles in radians.
            - gt_hyper_projection: Ground truth noiseless hyperspectral data of same shape.

    """
    # Ensure material_basis has exactly 3 rows
    if material_basis.shape[0] != 3:
        raise ValueError("material_basis must have exactly 3 rows (Ni, Cu, Al).")

    # Validate geometry and inputs
    if detector_rows < 3 or detector_columns < 2:
        raise ValueError("detector_rows must be ≥3 and detector_columns ≥2.")
    if dosage_rate <= 0:
        raise ValueError("dosage_rate must be positive.")

    # Handle default material_density and verify required keys
    if material_density is None:
        material_density = {"Ni": 0.2, "Cu": 0.2, "Al": 1.0}
    required = {"Ni", "Cu", "Al"}
    missing = required - set(material_density)
    if missing:
        raise KeyError(f"material_density missing keys: {sorted(missing)}")

    # Basic sanity on basis values
    if np.any(material_basis < 0):
        raise ValueError("material_basis should be non-negative attenuation coefficients.")

    # Set variable values
    epsilon = 1e-30
    number_of_materials = material_basis.shape[0]
    number_of_wavelengths = material_basis.shape[1]

    # Generate view angles
    angles = np.linspace(0, np.pi, num_angles)

    # Generate simulated projection data for 3 materials (Ni, Cu, and Al)
    height = detector_rows // 3
    width = detector_columns // 2
    # -(width // 2), not -width // 2: Python floors, so for odd width the latter
    # is one further from zero than width // 2, the range is asymmetric, and the
    # square root goes negative at one end (NaN -> zeroed thickness). Bit at 91x91.
    thickness = 20 * np.sqrt((width//2)**2 - np.linspace(-(width // 2), width // 2, width)**2)/ width
    material_projection = np.zeros((num_angles, detector_rows, detector_columns, number_of_materials), dtype=material_basis.dtype)
    material_projection[:, :height, width // 2:width + width // 2, 0] = material_density["Ni"] * thickness
    material_projection[:, 2 * height:, width // 2:width + width // 2, 1] = material_density["Cu"] * thickness
    material_projection[:, height:2 * height, width // 2:width + width // 2, 2] = material_density["Al"] * thickness

    # Generate noiseless hyperspectral projection data using rehydrate function
    gt_hyper_projection = rehydrate([material_projection, material_basis, 'attenuation'])

    # Generate noiseless hyperspectral open beam data using the given dosage rate
    noiseless_open_beam = dosage_rate * np.ones((detector_rows, detector_columns, number_of_wavelengths), dtype=material_basis.dtype)

    # Generate noiseless raw hyperspectral neutron counts
    noiseless_object_scan = np.exp(-gt_hyper_projection) * noiseless_open_beam
    noiseless_object_scan = np.nan_to_num(noiseless_object_scan, nan=0, posinf=0, neginf=0)

    if noisy:
        # Generate noisy neutron counts from Poisson distribution
        noisy_object_scan = np.random.poisson(noiseless_object_scan)
    else:
        # Do not generate noisy data
        noisy_object_scan = noiseless_object_scan

    # Generate noisy hyperspectral projection data
    ratio = noisy_object_scan / noiseless_open_beam
    ratio[ratio < epsilon] = epsilon
    noisy_hyper_projection = -np.log(ratio)

    if verbose >= 1:
        print("generate_hyper_data(): ")
        print("   -Shape of material_basis (linear attenuation coefficients for Ni, Cu, and Al):", material_basis.shape)
        print("   -Shape of material_projection (density of Ni, Cu, and Al):", material_projection.shape)
        print("   -Shape of hyperspectral data: ", noisy_hyper_projection.shape)

    if verbose > 1:
        plt.figure()
        plt.plot(material_basis.T)  # each column is a basis function
        plt.xlabel("wavelength index")
        plt.ylabel("linear attenuation ($cm^{-1}$)")
        plt.title("Material basis functions (Ni, Cu, Al)")
        plt.legend(["Ni", "Cu", "Al"])

    return [noisy_hyper_projection, angles, gt_hyper_projection]


def compare_spectra(spectra_groups, ground_truth=None, labels=None, subtitles=None, title=None, x_label=None, y_label=None, x_lim=None, y_lim=None, wavelengths=None, filename=None,
                    font_size=20, legend_font_size=12, line_width=1.5):
    """
    Function to display and save multiple 2D arrays as images.

    Args:
        spectra_groups(list): list of groups of spectra to display
        ground_truth(list,optional): list of ground truth spectra for comparison
        labels(list,optional): labels for different spectra
        subtitles(list,optional): subtitles for different spectrum groups
        title(str,optional): title for the image
        x_label(str,optional): X axis label
        y_label(str,optional): Y axis label
        x_lim(tuple,optional): (x_min, x_max) to set x-axis display range
        y_lim(tuple,optional): (y_min, y_max) to set y-axis display range
        wavelengths(list,optional): list of wavelength values for the spectra
        filename(str,optional): path to save the image
        font_size(int,optional): base font size; raise for slides. Defaults to 20.
        legend_font_size(int,optional): legend font size. Defaults to 12.
        line_width(float,optional): width of the plotted spectra. Defaults to 1.5.
        """
    num_groups = len(spectra_groups)
    if num_groups == 0:
        raise ValueError("No spectra groups provided for comparison.")

    num_spectra = len(spectra_groups[0])  # Assume all groups have the same number of spectra

    if labels is None:
        labels = ['Spectrum: ' + str(i+1) for i in range(num_spectra)]

    if wavelengths is None:
        wavelengths = np.arange(len(spectra_groups[0][0]))

    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rc('font', size=font_size)
    plt.figure(figsize=(12, 4 * num_groups))
    plt.suptitle(title)

    for group_idx, spectra in enumerate(spectra_groups):
        ax = plt.subplot(num_groups, 1, group_idx + 1)

        group_labels = labels.copy()
        if ground_truth is not None:
            for i, gt_spectrum in enumerate(ground_truth):
                gt_label = "Ground Truth" if i == 0 else None
                ax.plot(wavelengths, gt_spectrum, 'k--', label=gt_label, lw=line_width)

                # Add signal-to-noise ratio annotation
                err = np.linalg.norm(gt_spectrum - spectra[i])
                snr = 20 * np.log10(np.linalg.norm(gt_spectrum) / err)
                group_labels[i] += f" (SNR: {snr:.1f} dB)"

        for i, spectrum in enumerate(spectra):
            ax.plot(wavelengths, spectrum, label=group_labels[i], lw=line_width)

        if subtitles is not None:
            ax.set_title(subtitles[group_idx])

        # Only add x label on final group
        if group_idx == num_groups - 1:
            ax.set_xlabel(x_label)
        else:
            ax.set_xticklabels([])  # Remove x label
        ax.set_ylabel(y_label)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        ax.legend(loc='lower left', fontsize=legend_font_size)

    if filename is not None:
        try:
            plt.savefig(filename)
        except:
            warnings.warn("Can't write to file.")

# -----------------------------------------------------------------------
# Bias-corrected spectra: the modified profile likelihood
# -----------------------------------------------------------------------
_CORRECTIONS = ('cox_reid', 'barndorff_nielsen', 'orthant', 'bootstrap')


def _kr_matrices(Z, A, B):
    """Per-pixel R x R matrices  M_p = sum_k Z_pk a_k b_k^T  for A, B of shape (R, K).

    One GEMM of Z (P x K) against the K x R^2 Khatri-Rao product of A and B.
    """
    R = A.shape[0]
    KR = (A[:, None, :] * B[None, :, :]).reshape(R * R, -1)      # row r*R+s holds a_r * b_s
    return (Z @ KR.T).reshape(Z.shape[0], R, R)


def _adjustment_value_and_grad(T, W, H, dose, correction, H_ref, chunk=8192):
    """Value of the modified profile objective and the total H-gradient of its adjustment.

    At W = w_hat(H), the ML solution for the pixels given H,

        F(H) = L(W, H) + a(H),

    with the Cox-Reid adjustment

        a = 1/(2 dose) sum_p [ log det M_p - log det H H^T ],     M_p = sum_k Z_pk h_k h_k^T,

    M_p being the observed information of pixel p's coefficients per unit dose
    (the same R x R matrix block_newton_step forms) and the Gram term fixing the
    gauge (see the comment in the code), or the Barndorff-Nielsen
    adjustment in Severini's (1998) form,

        a = 1/dose sum_p [ log|det N_p| - 1/2 log det M_p ],   N_p = sum_k Z_pk h_k hhat_k^T,

    hhat being the reference ML estimate of H. Each is restricted to the pixel's
    free coordinates: a coefficient at the bound is not estimated, so there is no
    estimation noise to correct for. A pixel with no material contributes nothing.

    The adjustment depends on H directly and through w_hat(H), and its gradient
    needs both. One Newton step of the W problem from its converged point,
    W1 = W - M^-1 g(W, H), is a differentiable function of H whose derivative at
    the fixed point equals that of w_hat(H) (implicit function theorem), so
    autograd through a(H, W1(H)) returns the exact total gradient.

    Returns (F, da/dH) with F in the units of stable_nnal and the gradient in
    H's dtype. Pixels are processed in chunks to bound autograd memory.
    """
    R = H.shape[0]
    Hg = H.detach().clone().requires_grad_(True)
    Href = (H if H_ref is None else H_ref).detach()
    eye = torch.eye(R, dtype=W.dtype, device=W.device)
    scale = W.abs().amax(-1, keepdim=True).mean()
    loss_total = 0.0
    adj_total = 0.0
    for s in range(0, T.shape[0], chunk):
        Tc = T[s:s + chunk]
        W0 = W[s:s + chunk].detach()
        prep = _nnal_prep(Tc)
        X0 = W0 @ Hg
        G0, Z0 = stable_nnal_derivatives(X0, Tc, prep)
        g = G0 @ Hg.T                                                   # gradient of the pixel loss in w
        free = ~((W0 <= _ACTIVE_TOL * scale) & (g.detach() > 0))        # block_newton_step's active set
        fm = free[:, :, None] & free[:, None, :]
        M0 = torch.where(fm, _kr_matrices(Z0, Hg, Hg), eye.expand_as(fm).to(W.dtype))
        d = torch.linalg.solve(M0, torch.where(free, g, torch.zeros_like(g))[..., None])[..., 0]
        W1 = W0 - torch.where(free, d, torch.zeros_like(d))
        X1 = W1 @ Hg
        with torch.no_grad():
            loss_total += stable_nnal(X1, Tc, prep, dtype=torch.float64).item()
        Z1 = torch.exp(-X1)
        M1 = torch.where(fm, _kr_matrices(Z1, Hg, Hg), eye.expand_as(fm).to(W.dtype))
        logdet_M = torch.logdet(M1)
        if correction == 'cox_reid':
            # Gauge. The likelihood is invariant under (W, H) -> (W A, A^-1 H), but
            # log det M_p shifts by -2 log|det A|, so along the exactly flat scaling
            # direction the adjusted objective would be linear and unbounded -- the
            # outer loop drifted 97% along it. Measuring the information relative to
            # the subspace's Gram matrix H H^T (a flat prior on the pixel's
            # attenuation profile in R^K rather than on its coordinates) makes the
            # adjustment invariant; it equals Cox-Reid in the gauge H H^T = I.
            gram = torch.where(fm, (Hg @ Hg.T)[None].expand_as(fm), eye.expand_as(fm).to(W.dtype))
            adj = (logdet_M - torch.logdet(gram)).sum() / (2.0 * dose)
        elif correction == 'barndorff_nielsen':
            N1 = torch.where(fm, _kr_matrices(Z1, Hg, Href), eye.expand_as(fm).to(W.dtype))
            adj = (torch.linalg.slogdet(N1)[1].sum() - 0.5 * logdet_M.sum()) / dose
        elif correction == 'orthant':
            # EXPERIMENTAL. Integrated likelihood over w >= 0: the invariant Laplace
            # term on the free coordinates plus, for each coordinate at the bound
            # with outward gradient g and curvature m, the one-dimensional integral
            #   int_0^inf exp(-dose (g w + m w^2 / 2)) dw
            #     = (dose m)^-1/2 sqrt(2 pi) exp(gamma^2/2) Phi(-gamma),  gamma = g sqrt(dose/m),
            # taken along the profile direction (|h_r| dw) so that it is gauge
            # invariant. It is the only curvature-type adjustment that sees the
            # truncation of coefficients at zero. In a Monte Carlo of the profile
            # score at the true H (K=300, dose 3) it removed 28% of the score bias
            # with the right sign, where Cox-Reid removed none. It is incomplete: the
            # bound coordinates of a pixel are treated as independent 1-D integrals
            # with marginal curvature, whereas the material spectra are strongly
            # correlated and the orthant integral is governed by that correlation
            # (using the conditional Schur-complement curvature instead made the
            # correction 2.2x too large with the wrong sign). The consistent version
            # needs a differentiable bivariate/trivariate orthant probability.
            gram = torch.where(fm, (Hg @ Hg.T)[None].expand_as(fm), eye.expand_as(fm).to(W.dtype))
            adj_free = (logdet_M - torch.logdet(gram)).sum() / (2.0 * dose)
            G1 = Tc - Z1
            bound = ~free
            g1 = torch.where(bound, (G1 @ Hg.T).clamp(min=0), torch.zeros_like(W0))     # outward gradient at the bound
            m_rr = Z1 @ (Hg * Hg).T                                                      # marginal curvature
            hn2 = (Hg * Hg).sum(1)[None, :]
            gamma = g1 * torch.sqrt(dose / m_rr)
            log_int = 0.5 * math.log(2 * math.pi) + 0.5 * gamma * gamma + torch.special.log_ndtr(-gamma)
            bound_term = 0.5 * torch.log(m_rr / hn2) - log_int
            adj = adj_free + torch.where(bound, bound_term, torch.zeros_like(bound_term)).sum() / dose
        else:
            raise ValueError(f"correction must be one of {_CORRECTIONS}")
        adj.backward()
        adj_total += adj.item()
    return loss_total + adj_total, Hg.grad.to(H.dtype)


def _bootstrap_score_bias(T, W, H, dose, n_sim, seed, chunk, w_steps, compile_mode, deriv):
    """Parametric-bootstrap estimate of the profile score's bias at (W, H).

    Counts are simulated from the fitted model, T_b ~ Poisson(dose exp(-W H)) / dose,
    the pixel coefficients are re-solved under the same nonnegativity constraints
    with H fixed, and the profile score W_b^T (T_b - Z_b) is averaged. Were (W, H)
    the truth, that average would be the score's expectation -- the quantity a
    bias-corrected estimating equation subtracts (Kuk 1995; Hahn & Newey 2004).
    It captures whatever biases the per-pixel solve, including the truncation of
    coefficients at zero, which no curvature-based adjustment sees. The noise of
    the estimate falls as 1/sqrt(P n_sim), so it is most precise where the bias
    matters. The seed is fixed across outer iterations (common random numbers)
    so the fixed-point map is smooth.
    """
    R = H.shape[0]
    gen = torch.Generator(device=T.device).manual_seed(seed)
    b = torch.zeros(H.shape, dtype=torch.float64, device=H.device)
    for s in range(0, T.shape[0], chunk):
        Wc = W[s:s + chunk]
        lam = dose * torch.exp(-(Wc @ H))
        for _ in range(n_sim):
            Tb = torch.poisson(lam, generator=gen) / dose
            Wb, _, _ = block_newton_optimize(Tb, R, w_steps, 1e-12, update_H=False, W_init=Wc, H_init=H,
                                             compile_mode=compile_mode)
            Gb, _ = deriv(Wb @ H, Tb, _nnal_prep(Tb))
            b += (Wb.T @ Gb).to(torch.float64)
    return b / n_sim


def bias_corrected_spectra(T, W, H, dose, correction='bootstrap', max_outer=20, rel_tol=1e-8, h_tol=1e-3,
                           joint_steps=None, cg_max=10, chunk=8192, n_sim=8, seed=0, w_steps=30,
                           relax=1.0, callback=None, compile_mode=None, verbose=False):
    """Move a converged ML factorization to the maximizer of the modified profile likelihood.

    H is shared by every pixel and estimated jointly with R nuisance coefficients
    per pixel, each pixel carrying a fixed amount of information (K bins at the
    given dose). That is the Neyman-Scott setting: the ML estimate of H is not
    consistent -- its bias is set by the information per pixel, not the pixel
    count -- and on this problem converging the ML solve further lowered the
    spectral SNR while the loss kept falling. The modified profile likelihood
    (Barndorff-Nielsen 1983; Cox & Reid 1987) subtracts 1/2 log det j_ww(H, w_hat(H))
    per pixel, the Laplace approximation to integrating the nuisance parameters
    out; it removes the leading term of the bias, and is exact (REML) in the
    Neyman-Scott normal example.

    Solved as a fixed point. At the current (W, H) the adjustment's total
    gradient c = da/dH is computed, including its dependence through w_hat(H);
    the ML problem with the linear tilt <c, H> is then re-solved jointly from the
    current point, whose stationarity conditions are grad_W L = 0 and
    grad_H L + c = 0 -- those of the modified profile likelihood once c has
    stopped changing. The tilt is a perturbation of relative size ~1/(dose K), so
    a handful of outer iterations suffice.

    Args:
        T, W, H: data and a converged ML factorization (e.g. joint_newton at 1e-8).
        dose: open-beam counts per pixel and bin, so that T = counts / dose. It sets
            the weight of the adjustment relative to the loss; a scalar.
        correction: 'bootstrap' (default) subtracts a parametric-bootstrap estimate of
            the profile score's bias (n_sim simulated data sets per outer iteration,
            common random numbers from `seed`); it is the only one of the three that
            sees the truncation of coefficients at zero, which dominates on data with
            pure-material pixels. 'cox_reid' and 'barndorff_nielsen' (Severini's
            approximation, with the input H as reference) are the curvature-based
            adjustments; they assume interior per-pixel maxima and an
            information-orthogonal parametrization, neither of which holds here.
        rel_tol: stop when an outer iteration changes F by less than this, relatively
            (also the tolerance of each inner joint solve).
        h_tol: or when it changes H by less than this, relative to |H|. The default
            1e-3 is below the correction's own noise (about 1/sqrt(n_sim) of the ML
            estimate's sampling error for the bootstrap).
        joint_steps: Newton steps of the tilted joint solve per outer iteration. None
            means 1 for the bootstrap and 100 (a full solve) for the gradient-based
            adjustments. A full solve with the bootstrap tilt is unsafe: a linear tilt
            on H with W >= 0 is unbounded below -- shrink a component's W column while
            growing its H row and the loss stands still while <c, H> falls -- and one
            run moved H by 176% at the step cap. One damped, CG-truncated Newton step
            per outer iteration is bounded by construction; the outer loop supplies
            the iteration.
        relax: under-relaxation of the outer update, H <- H_prev + relax (H_new - H_prev)
            (W re-solved), for fixed-point maps that are not contractions at relax=1.
        callback: optional f(iteration, W, H, F, tilt) called once per outer iteration.

    Returns:
        (W, H, outer_iterations).
    """
    if correction not in _CORRECTIONS:
        raise ValueError(f"correction must be one of {_CORRECTIONS}")
    if joint_steps is None:
        joint_steps = 1 if correction == 'bootstrap' else 100
    nnal_fn, deriv, _, _ = _kernels(compile_mode)
    prep = _nnal_prep(T)
    R = H.shape[0]
    H_ref = H.detach().clone()
    W, _, _ = block_newton_optimize(T, R, 50, 1e-12, update_H=False, W_init=W, H_init=H,
                                    compile_mode=compile_mode)
    F_prev, H_prev, rises = None, None, 0
    for it in range(max_outer):
        if correction == 'bootstrap':
            tilt = -_bootstrap_score_bias(T, W, H, dose, n_sim, seed, chunk, w_steps, compile_mode, deriv).to(H.dtype)
            F = nnal_fn(W @ H, T, prep, dtype=torch.float64).item()      # no adjusted objective: stop on dH
        else:
            F, tilt = _adjustment_value_and_grad(T, W, H, dose, correction, H_ref, chunk)
        # Along the gauge orbit {A H} the loss is exactly flat, so any component
        # of the tilt there makes the tilted objective unbounded: with the
        # bootstrap tilt |H| grew 200x in five iterations while the loss stood
        # still. Complementary slackness would make the score orthogonal to that
        # orbit exactly, but the per-pixel solves are converged only to a
        # tolerance, so the orthogonality is enforced: each row of the tilt is
        # projected onto the complement of H's row space. The removed part would
        # only have moved H within its own row space, which changes no fit.
        Hd = H.to(torch.float64)
        gram = Hd @ Hd.T
        tilt = (tilt.to(torch.float64) - torch.linalg.solve(gram, Hd @ tilt.to(torch.float64).T).T @ Hd).to(H.dtype)
        dH = float('nan') if H_prev is None else ((H - H_prev).norm() / H.norm()).item()
        if verbose:
            print(f'  outer {it}: F {F:.8e}  |dH|/|H| {dH:.2e}  |c| {tilt.norm().item():.3e}', flush=True)
        if callback is not None:
            callback(it, W, H, F, tilt)
        if F_prev is not None:
            if dH <= h_tol or (correction != 'bootstrap' and abs(F - F_prev) <= rel_tol * abs(F)):
                break
            rises = rises + 1 if (F > F_prev and correction != 'bootstrap') else 0
            if rises >= 3:
                if verbose:
                    print('  F rose three times in a row; stopping at the last iterate', flush=True)
                break
        F_prev, H_prev = F, H.clone()
        W, H, _, _ = _joint_newton_pcg(T, W, H, max_steps=joint_steps, cg_max=cg_max, rel_tol=rel_tol,
                                       prep=prep, nnal=nnal_fn, deriv=deriv, tilt_H=tilt)
        if relax != 1.0:
            H = (H_prev + relax * (H - H_prev)).clamp_(min=0)
            W, _, _ = block_newton_optimize(T, R, 50, 1e-12, update_H=False, W_init=W, H_init=H,
                                            compile_mode=compile_mode)
    return W, H, it + 1

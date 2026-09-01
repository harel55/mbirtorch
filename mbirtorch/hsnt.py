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


def stable_nnal(X, T, prep=None):
    """
    Compute a shifted form of the non-negative attentuation loss
    that is much more numerically stable

    Args:
        X: Attenuation estimate, broadcastable against T.
        T: Measured transmission ratio (counts / open beam).
        prep: Optional tuple from _nnal_prep(T). Pass it inside an iteration to
            avoid recomputing log(T) on every call.
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

    return torch.sum(loss, dim=(-2, -1))


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


def nndsvda(X, n_components):
    """NNDSVDA initialization for X ~= W @ H.

    Args:
        X: Nonnegative array of shape (n_samples, n_features).
        n_components: Factorization rank.

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

    # NNDSVDA replaces zero entries with the mean of X.
    fill_value = torch.mean(X)
    W = torch.where(W == 0, fill_value, W)
    H = torch.where(H == 0, fill_value, H)

    return W, H


def quadratic_update(W, H, T, prep=None, update_H=True):
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
        prep: Optional tuple from _nnal_prep(T).
        update_H: If False, keep H fixed and only update W.

    Returns:
        Updated (W, H) pair as PyTorch tensors
    """
    prep = _nnal_prep(T) if not isinstance(prep, tuple) else prep
    tiny = torch.finfo(T.dtype).tiny

    X = W @ H
    G, Z = stable_nnal_derivatives(X, T, prep)
    ZV = Z * X - G
    W = W * ((ZV.clamp(min=0) @ H.T)
             / ((Z * X) @ H.T + (-ZV).clamp(min=0) @ H.T).clamp(min=tiny))

    if update_H:
        # Relinearize before the H step: the model is only second order accurate
        # at the iterate it was expanded about, and W has just moved.
        X = W @ H
        G, Z = stable_nnal_derivatives(X, T, prep)
        ZV = Z * X - G
        H = H * ((W.T @ ZV.clamp(min=0))
                 / (W.T @ (Z * X) + W.T @ (-ZV).clamp(min=0)).clamp(min=tiny))

    return W, H, prep


def newton_update(W, H, T, lr_init, update_H=True):
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
    G, Z = stable_nnal_derivatives(X, T)
    init_loss = stable_nnal(X, T)

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
    candidate_losses = stable_nnal(X_candidates, T)
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

def multiplicative_update(W: torch.Tensor, H: torch.Tensor, T: torch.Tensor, unused: torch.Tensor, update_H: bool = True):
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

    W *= ((Z @ H.T) / torch.clip(T @ H.T, min=1e-30)) ** damping_factor
    if update_H:
        H *= ((W.T @ Z) / torch.clip(W.T @ T, min=1e-30)) ** damping_factor

    return W, H, 0.0

def _nnal_rowwise(X, T, prep, dim):
    """NNAL summed over `dim` only: per-pixel (dim=1) or per-wavelength (dim=0)."""
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
    return loss.sum(dim=dim)


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


def block_newton_step(V, other, X, T, prep, axis, ls_max=8, jitter_rel=1e-9):
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
        (V_new, X_new, num_backtracks)
    """
    log_T, positive, all_positive, taylor_cutoff = prep
    G, Z = stable_nnal_derivatives(X, T, prep)

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
    free = ~((V <= 0) & (grad > 0))
    eye = torch.eye(rank, dtype=V.dtype, device=V.device)
    M = torch.where(free[:, :, None] & free[:, None, :], M, eye.expand_as(M))
    rhs = torch.where(free, grad, torch.zeros_like(grad))

    d = _batched_spd_solve(M, rhs, jitter_rel)
    d = torch.where(free, d, torch.zeros_like(d))

    # Trust region, per row. The feasibility clamp below only bounds directions
    # that drive a factor toward zero; nothing bounds one that grows it. A row
    # whose Z has underflowed to zero (float32 loses exp(-X) past X ~ 88) carries
    # no curvature, so the damped solve returns a direction of order 1/tiny --
    # large but finite, so it survives every non-finite check and only overflows
    # on the next matmul. Keeping the step within the scale of the row itself
    # costs nothing when the Hessian is healthy.
    limit = 16.0 * V.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(V.dtype).eps)
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
        base = _nnal_rowwise(X, T, prep, 1)
        expand = lambda a: a[:, None]
    else:
        B = other @ d.T
        base = _nnal_rowwise(X, T, prep, 0)
        expand = lambda a: a[None, :]

    accepted = torch.zeros_like(alpha)
    done = torch.zeros_like(alpha, dtype=torch.bool)
    num_backtracks = 0
    for _ in range(ls_max):
        trial = torch.where(done, torch.zeros_like(alpha), alpha)
        dim = 1 if axis == 0 else 0
        # The Armijo decrease can fall below the roundoff of `base` itself,
        # which in float32 makes the test fail spuriously and backtrack to the
        # cap. Accept anything that is not measurably worse than the target.
        noise = 8.0 * torch.finfo(V.dtype).eps * base.abs()
        ok = (_nnal_rowwise(X - expand(trial) * B, T, prep, dim)
              <= base - 1e-4 * trial * slope + noise) | (trial == 0)
        accepted = torch.where(ok & ~done, trial, accepted)
        done = done | ok
        if bool(done.all()):
            break
        alpha = alpha * 0.5
        num_backtracks += 1

    V_new = (V - accepted[:, None] * d).clamp_(min=0.0)
    X_new = X - expand(accepted) * B
    if axis == 1:
        V_new = V_new.T.contiguous()
    return V_new, X_new, num_backtracks


def block_newton_optimize(T, num_materials, max_steps, rel_tol, update_H=True,
                          convergence_check_interval=1, W_init=None, H_init=None,
                          jitter_rel=1e-9):
    """Alternating exact projected-Newton minimization of the NNAL."""
    prep = _nnal_prep(T)
    W, H = W_init, H_init
    X = W @ H
    prev_loss = _nnal_rowwise(X, T, prep, 1).sum()
    num_steps = 0
    for step in range(max_steps):
        X = W @ H                      # resynchronize against incremental drift
        W, X, _ = block_newton_step(W, H, X, T, prep, 0, jitter_rel=jitter_rel)
        if update_H:
            H, X, _ = block_newton_step(H, W, X, T, prep, 1, jitter_rel=jitter_rel)
        num_steps = step + 1
        if rel_tol > 0 and num_steps % convergence_check_interval == 0:
            loss = _nnal_rowwise(X, T, prep, 1).sum()   # free: X is already in hand
            # Guard the denominator against prev_loss -> 0: on data that a rank
            # `num_materials` model fits exactly, the shifted loss goes to zero and
            # dividing by it makes the relative change diverge, so the test would
            # never fire. Fall back to an absolute roundoff floor there.
            floor = 8 * torch.finfo(T.dtype).eps * torch.abs(loss)
            change = torch.abs(loss - prev_loss)
            converged = bool(change <= floor) or bool(change <= rel_tol * torch.abs(prev_loss))
            prev_loss = loss
            if converged:
                break
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
                     precond_jitter=1e-8, prep=None, verbose=False):
    prep = _nnal_prep(T) if prep is None else prep
    W = W.clone(); H = H.clone()
    rank = W.shape[1]
    rows, cols = torch.triu_indices(rank, rank, device=W.device)
    loss = stable_nnal(W @ H, T, prep)
    lam = damping
    total_cg = 0
    step = 0
    gnorm0 = None
    stalled = 0
    eps = torch.finfo(T.dtype).eps
    for step in range(1, max_steps + 1):
        X = W @ H
        G, Z = stable_nnal_derivatives(X, T, prep)
        gW, gH = G @ H.T, W.T @ G
        fW = ~((W <= 0) & (gW > 0))
        fH = ~((H <= 0) & (gH > 0))
        gW, gH = gW * fW, gH * fH
        gnorm2 = _joint_dot(gW, gW, gH, gH)
        if not torch.isfinite(gnorm2) or gnorm2 == 0:
            break
        # Stop on the projected-gradient (KKT) residual, not on the relative change
        # in the loss: this objective is shifted so its minimum is 0, so once the
        # fit is near-exact the relative loss change stays O(1) forever and a
        # loss-ratio test never fires.
        gnorm = gnorm2.sqrt()
        if gnorm0 is None:
            gnorm0 = gnorm
        elif rel_tol > 0 and gnorm <= rel_tol * gnorm0:
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
            Wn = (W + a * xW).clamp_(min=0)
            Hn = (H + a * xH).clamp_(min=0)
            new_loss = stable_nnal(Wn @ Hn, T, prep)
            if torch.isfinite(new_loss) and new_loss <= loss + 1e-4 * a * slope:
                accepted = True; break
            a *= 0.5
        if not accepted:
            lam = lam * 10.0 if lam > 0 else 1e-12
            if lam > 1e6: break
            continue
        lam = max(lam * 0.3, 1e-14)
        # Stagnation: the step bought less than the roundoff of the loss itself.
        # Three in a row means we are at the precision floor, whatever the KKT
        # residual says (the scale ambiguity W -> WM, H -> M^-1 H leaves an
        # R^2-dimensional null space, so the gradient need not vanish outright).
        stalled = stalled + 1 if (loss - new_loss) <= 8 * eps * torch.abs(new_loss) else 0
        W, H, loss = Wn, Hn, new_loss
        if verbose:
            print(f'  step {step:3d} cg {ncg:3d} a {a:.3g} loss {loss.item():.6e}', flush=True)
        if stalled >= 3:
            break
    return W, H, step, total_cg


def joint_newton_optimize(T, num_materials, max_steps, rel_tol, update_H=True,
                          convergence_check_interval=1, W_init=None, H_init=None,
                          warmup_steps=10, cg_max=20):
    """Block-Newton warm-up followed by a joint (W,H) preconditioned Newton solve.

    Alternating methods stall at a linear rate once the fit is good, because they
    discard the W<->H coupling block of the Hessian; on a problem where an exact
    factorization exists, block-Newton alone plateaus around 1e-7. Solving for both
    factors jointly restores fast convergence to machine precision. The alternating
    method is still the right way to get into the basin, so it runs first.

    update_H=False is not supported here (the joint step updates both factors);
    it falls back to block-Newton alone.
    """
    prep = _nnal_prep(T)
    W, H = W_init, H_init
    X = W @ H
    for _ in range(min(warmup_steps, max_steps)):
        X = W @ H
        W, X, _ = block_newton_step(W, H, X, T, prep, 0)
        if update_H:
            H, X, _ = block_newton_step(H, W, X, T, prep, 1)
    if not update_H:
        return W, H, min(warmup_steps, max_steps)
    remaining = max(0, max_steps - warmup_steps)
    if remaining == 0:
        return W, H, min(warmup_steps, max_steps)
    W, H, steps, _ = _joint_newton_pcg(T, W, H, max_steps=remaining, cg_max=cg_max,
                                       rel_tol=rel_tol, prep=prep)
    return W, H, warmup_steps + steps


def optimize(T: torch.Tensor, update, num_materials, max_steps, rel_tol, update_H=True,
             convergence_check_interval=1, W_init: torch.Tensor = torch.tensor([]),
             H_init: torch.Tensor = torch.tensor([])) -> torch.Tensor:
    """Factorize T into W and H by minimizing nonnegative attenuation loss."""

    if W_init.numel() == 0 and H_init.numel() == 0:
        W_init, H_init = nndsvda(T, n_components=num_materials)
    elif W_init.numel() == 0:
        # lstsq is unconstrained, so clamp before handing it to a nonnegative solver
        W_init = torch.linalg.lstsq(H_init.T, T.T)[0].T.clamp(min=0)
    elif H_init.numel() == 0:
        H_init = torch.linalg.lstsq(W_init, T)[0].clamp(min=0)

    if update in (block_newton_optimize, joint_newton_optimize):
        return update(T, num_materials, max_steps, rel_tol,
                      update_H=update_H, W_init=W_init, H_init=H_init,
                      convergence_check_interval=convergence_check_interval)

    prep = _nnal_prep(T)
    W, H, aux = (W_init, H_init, 0.001)
    prev_loss = stable_nnal(W @ H, T, prep)
    num_steps = 0
    for i in range(max_steps):
        W, H, aux = update(W, H, T, aux, update_H=update_H)
        num_steps = i + 1
        # Only evaluate the loss when it is actually needed: it costs a
        # (pixels x rank x bins) matmul plus a full elementwise pass.
        if rel_tol > 0 and num_steps % convergence_check_interval == 0:
            loss_new = stable_nnal(W @ H, T, prep)
            converged = torch.abs(loss_new - prev_loss) / (prev_loss + 1e-30) < rel_tol
            prev_loss = loss_new
            if converged:
                break

    return W, H, num_steps

def nnal_factorization(T: torch.Tensor, method='quasi_newton', num_materials=3, max_steps=1000,
                       rel_tol=1e-10, batch_size=None, compile_mode=None, **kwargs) -> torch.Tensor:
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

    if compile_mode is not None and update not in (block_newton_optimize,
                                                   joint_newton_optimize):
        compile_options = {
            "triton.cudagraphs": False,
            "max_autotune": compile_mode == "max-autotune",
        }
        update = torch.compile(
            update,
            options=compile_options,
        )

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

    # Randomly permute the pixel indices for batching
    batch_idxs = np.random.permutation(num_pixels)

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
    thickness = 20 * np.sqrt((width//2)**2 - np.linspace(-width // 2, width // 2, width)**2)/ width
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


def compare_spectra(spectra_groups, ground_truth=None, labels=None, subtitles=None, title=None, x_label=None, y_label=None, x_lim=None, y_lim=None, wavelengths=None, filename=None):
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
    plt.rc('font', size=20)
    plt.figure(figsize=(12, 4 * num_groups))
    plt.suptitle(title)

    for group_idx, spectra in enumerate(spectra_groups):
        ax = plt.subplot(num_groups, 1, group_idx + 1)

        group_labels = labels.copy()
        if ground_truth is not None:
            for i, gt_spectrum in enumerate(ground_truth):
                gt_label = "Ground Truth" if i == 0 else None
                ax.plot(wavelengths, gt_spectrum, 'k--', label=gt_label)

                # Add signal-to-noise ratio annotation
                mse = np.linalg.norm(gt_spectrum - spectra[i])
                snr = 20 * np.log10(np.linalg.norm(gt_spectrum) / mse)
                group_labels[i] += f" (RMSE: {mse**0.5:2g}, SNR: {snr:.1f}dB)"

        for i, spectrum in enumerate(spectra):
            ax.plot(wavelengths, spectrum, label=group_labels[i])

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

        ax.legend(loc='upper left', fontsize=12)

    if filename is not None:
        try:
            plt.savefig(filename)
        except:
            warnings.warn("Can't write to file.")
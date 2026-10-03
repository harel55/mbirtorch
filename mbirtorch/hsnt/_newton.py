import torch

from ._linalg import (_attenuation_for_start, _batched_spd_solve, _joint_blocks, _joint_dot,
                      _nonneg_least_squares_start, _reseed_dead)
from ._loss import _nnal_prep, _nnal_rowwise, stable_nnal, stable_nnal_derivatives


# Armijo noise floor, in units of eps * |row loss|: a smaller decrease is float32 truncation, which grows with the
# row length, and is not trusted. 4 is the smallest value that does not backtrack spuriously.
_ARMIJO_FLOOR = 4.0
# Trust-region floor as a fraction of the mean row scale; without it a row pinned near zero crawls.
_TRUST_FLOOR = 1e-3
# epsilon-active set (Bertsekas): an entry within this fraction of its component's mean positive entry of zero, with
# an outward gradient, is frozen and snapped to zero, so a tiny residue at the feasibility limit cannot freeze its row
# at a non-stationary point.
_ACTIVE_TOL = 1e-6
# A block step that drives an entry to within this many ulps of zero sets it to exactly zero, so that fused (compiled)
# and separate rounding take the same active-set decision.
_SNAP_ULPS = 8.0
# The largest attenuation one component may contribute to an entry, max_p W_pk * max_b H_kb <= _X_MAX: a transmission
# of 1e-12, which any realistic dose records as zero counts. Well inside it the bound changes nothing; it keeps a
# component spent on zero-count entries from growing without limit. Each step bounds W and H by _X_MAX over the
# partner's current peak, a box fixed for the step.
_X_MAX = 27.631021115928547                              # -log(1e-12)
# The joint solve stops after this many consecutive accepted steps below rel_tol: its tail is slow and erratic, and a
# single quiet step is followed by larger decreases often enough to make a one-step stop irreproducible.
_PATIENCE = 5
# 'auto' compiles from this many data entries on CUDA (about 417k pixels at 1200 bins). At 1M pixels a compiled step
# takes about a third of the eager time; below the threshold solves run eager, since the break-even against a first
# compile has not been measured. The rank search always runs eager.
_COMPILE_MIN_ELEMENTS = 5e8
# Re-seeds of a dead component after the joint solve; a component that dies again each time has no support in the
# data, and the solve returns it dead, as it does when the solve after a re-seed ends no lower.
_MAX_RESEEDS = 2


def _resolve_compile(compile_mode, T, device=None):
    """'auto', 'on' or 'off' (None is 'off') as 'on' or 'off' for the data T solved on `device` (default: T's). 'auto'
    compiles on CUDA with a working Triton when T has at least _COMPILE_MIN_ELEMENTS entries."""
    if compile_mode is None or compile_mode == 'off':
        return 'off'
    if compile_mode not in ('auto', 'on'):
        raise ValueError(f"compile_mode must be 'auto', 'on' or 'off', got {compile_mode!r}")
    if compile_mode == 'on':
        return 'on'
    if torch.device(device or T.device).type != 'cuda' or T.numel() < _COMPILE_MIN_ELEMENTS:
        return 'off'
    from ..kernel_availability import triton_available
    return 'on' if triton_available()[0] else 'off'


def _kernels(compile_mode):
    """The four hot kernels (nnal, derivatives, rowwise, block step), compiled when compile_mode is 'on'.

    compile_mode is a resolved mode (see _resolve_compile). Compiling fuses the elementwise passes over the P x K data
    that dominate the Newton solvers; the GEMM-bound CG iteration stays eager. A kernel recompiles when P, K, R, the
    dtype or the presence of zero counts changes, and falls back to eager if the compile backend fails.
    """
    if compile_mode != 'on':
        return stable_nnal, stable_nnal_derivatives, _nnal_rowwise, block_newton_step
    from ..projectors import maybe_compile
    return tuple(maybe_compile(f, True) for f in (stable_nnal, stable_nnal_derivatives, _nnal_rowwise,
                                                  block_newton_step))


def _two_metric_direction(V, grad, flat, rows, cols, jitter_rel=1e-9, nonneg=True, ub=None):
    """Projected-Newton direction for a batch of rows of V (B, rank) under V >= 0.

    `flat` (B, Q) holds the upper triangle of each row's rank x rank Hessian
    (Q = rank (rank + 1) / 2, indexed by `rows`, `cols`). Two-metric projection
    (Bertsekas): variables within an epsilon of the bound with an outward
    gradient are frozen and snapped to zero, the rest take the Newton step; a
    bound-adjacent entry with an inward gradient gets the scaled gradient, and is
    kept out of the Newton system so that its partners' moves do not assume a
    step it does not take; a per-row trust region bounds directions from rows
    without curvature; a non-descent direction falls back to the scaled gradient;
    alpha is the largest step keeping V >= 0. The epsilon and the trust region are
    per component, in units of each component's mean positive entry, so rescaling a component
    (a gauge change of X = W H) rescales its direction and nothing else. With
    nonneg=False there is no active set, no feasibility limit and alpha = 1. The
    constants are documented where they are defined (_ARMIJO_FLOOR, _TRUST_FLOOR
    and _ACTIVE_TOL). ub (1, rank), when given, is an upper bound per component:
    an entry within _ACTIVE_TOL of it with an outward gradient is frozen, and alpha
    also keeps V <= ub.

    Returns (d, slope, alpha, bound, projected_gnorm2): d is the descent
    direction (V decreases along +d), slope = <grad, d> per row, alpha the
    per-row feasible step, bound the frozen mask, and the squared norm of the
    projected gradient per component (rank,), the KKT residual at the incoming
    iterate.
    """
    rank = V.shape[1]
    M = flat.new_zeros(flat.shape[0], rank, rank)
    M[:, rows, cols] = flat
    M[:, cols, rows] = flat
    # Each component's scale is its mean positive entry: equivariant under rescaling the component, and, unlike the
    # mean over all rows, not shrunk by sparsity (a material in a few percent of the pixels), which would let
    # residues just above the epsilon stall their rows.
    pos = V > 0
    unit = torch.where(pos, V, torch.zeros_like(V)).sum(0, keepdim=True) / pos.sum(0, keepdim=True).clamp_min(1)
    eps_active = _ACTIVE_TOL * unit
    unit = torch.where(unit > 0, unit, torch.ones_like(unit))
    bound = ((V <= eps_active) & (grad > 0)) if nonneg else torch.zeros_like(grad, dtype=torch.bool)
    if ub is not None:
        bound = bound | ((V >= ub * (1 - _ACTIVE_TOL)) & (grad < 0))
    free = ~bound
    projected_gnorm2 = ((grad * free) ** 2).sum(0)
    eye = torch.eye(rank, dtype=V.dtype, device=V.device)
    inward = ((V <= eps_active) & (grad < 0)) if nonneg else torch.zeros_like(grad, dtype=torch.bool)
    newton = free & ~inward
    diag_M = torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(torch.finfo(V.dtype).tiny)
    M = torch.where(newton[:, :, None] & newton[:, None, :], M, eye.expand_as(M))
    d = _batched_spd_solve(M, torch.where(newton, grad, torch.zeros_like(grad)), jitter_rel)
    d = torch.where(newton, d, torch.zeros_like(d))
    d = torch.where(inward, grad / diag_M, d)
    rhs = torch.where(free, grad, torch.zeros_like(grad))
    row_max = (V.abs() / unit).amax(-1, keepdim=True)
    floor = torch.clamp(_TRUST_FLOOR * row_max.mean(), min=torch.finfo(V.dtype).eps)
    limit = 16.0 * torch.maximum(row_max, floor) * unit
    d = torch.clamp(d, min=-limit, max=limit)
    slope = (grad * d).sum(-1)
    d = torch.where((slope <= 0)[:, None], torch.clamp(rhs / diag_M, min=-limit, max=limit), d)
    slope = (grad * d).sum(-1)
    ratio = torch.where(d > 0, V / d.clamp_min(torch.finfo(V.dtype).tiny), torch.full_like(d, float('inf')))
    if ub is not None:
        up = (ub - V).clamp_min(0) / (-d).clamp_min(torch.finfo(V.dtype).tiny)
        ratio = torch.minimum(ratio, torch.where(d < 0, up, torch.full_like(d, float('inf'))))
    alpha = torch.clamp(ratio.amin(-1), max=1.0) if nonneg else torch.ones_like(ratio.amin(-1))
    return d, slope, alpha, bound, projected_gnorm2


def _partner_bound(other, axis):
    """The upper bound (1, rank) of a factor's entries: _X_MAX over each component's peak in the other factor."""
    peak = other.amax(0) if axis == 0 else other.amax(1)
    return torch.where(peak > 0, _X_MAX / peak.clamp_min(torch.finfo(other.dtype).tiny),
                       torch.full_like(peak, float('inf')))[None, :]


def block_newton_step(V, other, X, T, prep, axis, jitter_rel=1e-9, nonneg=True):
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
        squared gradient norm is per component and measured at the incoming
        iterate, before the step.
    """
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

    ub = _partner_bound(other, 1 - axis) if nonneg else None
    d, slope, alpha, bound, projected_gnorm2 = _two_metric_direction(V, grad, flat, rows, cols, jitter_rel, nonneg,
                                                                     ub)

    # X(alpha) along the step is exactly X - alpha * B, so the line search needs no extra matmul.
    if axis == 0:
        B = d @ other
        base = _nnal_rowwise(X, T, prep, 1, dtype=torch.float64)
        expand = lambda a: a[:, None]
    else:
        B = other @ d.T
        base = _nnal_rowwise(X, T, prep, 0, dtype=torch.float64)
        expand = lambda a: a[None, :]

    accepted = torch.zeros_like(alpha)
    done = torch.zeros_like(alpha, dtype=torch.bool)
    num_backtracks = 0
    dim = 1 if axis == 0 else 0
    for _ in range(8):
        trial = torch.where(done, torch.zeros_like(alpha), alpha)
        # Accept a step that is not measurably worse than the Armijo target (see _ARMIJO_FLOOR).
        noise = _ARMIJO_FLOOR * torch.finfo(V.dtype).eps * base.abs()
        ok = (_nnal_rowwise(X - expand(trial) * B, T, prep, dim, dtype=torch.float64)
              <= base - 1e-4 * trial * slope + noise) | (trial == 0)
        accepted = torch.where(ok & ~done, trial, accepted)
        done = done | ok
        if bool(done.all()):
            break
        alpha = alpha * 0.5
        num_backtracks += 1

    step = accepted[:, None] * d
    V_new = V - step
    X_new = X - expand(accepted) * B
    if nonneg:
        reach = (d > 0) & (step >= V * (1.0 - _SNAP_ULPS * torch.finfo(V.dtype).eps))    # see _SNAP_ULPS
        V_new = torch.where(reach, torch.zeros_like(V_new), V_new).clamp_(min=0.0)
        # Snap the frozen entries to zero, but only in rows whose loss the snap does not raise: the line search has
        # not seen it, and a small entry times a large partner row can be a large change of X (at low dose, next to a
        # huge zero-count component). X is updated with the snap where it applies.
        snap = bound & (V_new > 0)
        if bool(snap.any()):
            dV = torch.where(snap, V_new, torch.zeros_like(V_new))
            X_try = X_new - (dV @ other if axis == 0 else other @ dV.T)
            l_plain = _nnal_rowwise(X_new, T, prep, dim, dtype=torch.float64)
            l_try = _nnal_rowwise(X_try, T, prep, dim, dtype=torch.float64)
            keep = l_try <= l_plain + _ARMIJO_FLOOR * torch.finfo(V.dtype).eps * l_plain.abs()
            V_new = torch.where(keep[:, None], V_new - dV, V_new)
            X_new = torch.where(expand(keep), X_try, X_new)
    if ub is not None:                                   # rounding past the bound: clamp, and X with it
        over = V_new > ub
        if bool(over.any()):
            V_new = torch.minimum(V_new, ub)
            X_new = V_new @ other if axis == 0 else other @ V_new.T
    if axis == 1:
        V_new = V_new.T.contiguous()
    return V_new, X_new, (num_backtracks, projected_gnorm2)


def block_newton_optimize(T, num_materials, max_steps, rel_tol, update_H=True, W_init=None, H_init=None,
                          compile_mode=None, nonneg_W=True, weights=None):
    """Alternating exact projected-Newton minimization of the NNAL (per-entry weights: see _nnal_prep)."""
    _, _, rowwise, step_fn = _kernels(compile_mode)
    prep = _nnal_prep(T, weights)
    W, H = W_init, H_init
    X = W @ H
    prev_loss = rowwise(X, T, prep, 1, dtype=torch.float64).sum()
    gnorm0 = None
    num_steps = 0
    for step in range(max_steps):
        X = W @ H                      # resynchronize against incremental drift
        W_in, H_in = W, H
        W, X, info_W = step_fn(W, H, X, T, prep, 0, nonneg=nonneg_W)
        gnorm = _stationarity(info_W[1], W_in, 0)
        if update_H:
            H, X, info_H = step_fn(H, W, X, T, prep, 1)
            gnorm = gnorm + _stationarity(info_H[1], H_in, 1)
            W, H, _ = _reseed_dead(W, H, T, seed=step + 1)
        num_steps = step + 1
        if rel_tol > 0:
            # The relative loss change per step, summed in float64; the projected-gradient (KKT) test catches data
            # the model fits exactly, whose loss goes to zero and whose relative change never becomes small.
            loss = rowwise(X, T, prep, 1, dtype=torch.float64).sum()
            if gnorm0 is None:
                gnorm0 = gnorm
            if (bool(torch.abs(loss - prev_loss) <= rel_tol * torch.abs(loss))
                    or bool(gnorm <= max(rel_tol ** 2, 100 * torch.finfo(T.dtype).eps) * gnorm0)):
                break
            prev_loss = loss
    return W, H, num_steps


def _stationarity(projected_gnorm2, V, axis):
    """Sum over components of |projected gradient of the factor| x |factor|: the KKT measure of the stops, which
    rescaling a component leaves unchanged (its gradient scales inversely to it), unlike a plain gradient norm, which a
    component in a large gauge dominates. projected_gnorm2 is per component; V is W (axis 0) or H (axis 1)."""
    norms = V.norm(dim=0) if axis == 0 else V.norm(dim=1)
    return (projected_gnorm2.clamp_min(0).sqrt() * norms).sum()



def solve_W(T, H, W_init=None, max_steps=100, rel_tol=1e-12, nonneg=True, compile_mode=None, weights=None):
    """The pixel coefficients for a fixed H: independent convex problems per pixel, solved by block-Newton W steps
    from W_init (default: a nonnegative least-squares fit of the attenuation). W >= 0 unless nonneg=False."""
    if W_init is None:
        W_init = _nonneg_least_squares_start(_attenuation_for_start(T), H)
    W, _, _ = block_newton_optimize(T, H.shape[0], max_steps, rel_tol, update_H=False, W_init=W_init, H_init=H,
                                    compile_mode=compile_mode, nonneg_W=nonneg, weights=weights)
    return W

def _joint_newton_pcg(T, W, H, max_steps=50, cg_max=60, rel_tol=0.0, prep=None, nnal=None, deriv=None,
                      nonneg_W=True, w_mask=None, patience=1):
    """Joint truncated-Newton solve on (W, H) by preconditioned CG, from a warm start.

    Each step forms the projected gradient (entries at zero with an outward
    gradient are frozen) and solves the Newton system approximately by CG. The
    Hessian-vector product is matrix-free -- dX = dW H + W dH, then Z * dX and the
    coupling terms through G, six GEMMs and no Hessian formed -- with Levenberg
    damping lam added to the diagonal. The preconditioner is the block-diagonal
    Hessian: the per-pixel and per-bin R x R blocks, Cholesky-factored in a batch.
    CG stops at cg_max iterations or by an Eisenstat-Walker forcing term, the
    residual reduced by min(sqrt(|g|), 0.5). A backtracking Armijo search on the
    float64 loss, clamping to the feasible set, accepts the step; a failed search
    multiplies lam by 10 and retries, an accepted one shrinks it. The solve stops
    after `patience` consecutive accepted steps whose relative loss change is at
    most rel_tol, on the KKT fallback below (for data the model fits exactly), or
    when lam runs away, which is what the precision floor looks like from here.
    Returns (W, H, steps, total_cg).

    Hooks: nonneg_W=False drops W >= 0 -- every coefficient is free and the line
    search does not clamp W; the pixel problem stays strictly convex for any real
    w -- so H can be estimated without the truncation bias the bound induces
    (_unconstrained_spectra). w_mask (bool, W's shape) holds coefficients outside
    the mask at their current value, zero for a selected support
    (_support_selected_spectra). With W >= 0 each step also keeps the attenuation
    bound: W and H under _X_MAX over the other factor's peak per component, a box
    fixed for the step (entries at its top with an outward gradient are frozen,
    and the line search clamps to it).
    """
    nnal = stable_nnal if nnal is None else nnal
    deriv = stable_nnal_derivatives if deriv is None else deriv
    prep = _nnal_prep(T) if prep is None else prep
    W = W.clone()
    H = H.clone()
    rank = W.shape[1]
    rows, cols = torch.triu_indices(rank, rank, device=W.device)
    loss = nnal(W @ H, T, prep, dtype=torch.float64)
    lam = 1e-12
    total_cg = 0
    step = 0
    gnorm0 = None
    quiet = 0
    for step in range(1, max_steps + 1):
        X = W @ H
        G, Z = deriv(X, T, prep)
        gW, gH = G @ H.T, W.T @ G
        fW = ~((W <= 0) & (gW > 0)) if nonneg_W else torch.ones_like(W, dtype=torch.bool)
        if w_mask is not None:
            fW = fW & w_mask
        fH = ~((H <= 0) & (gH > 0))
        if nonneg_W:                                     # the attenuation bound, a box fixed for this step
            uW, uH = _partner_bound(H, 1), _partner_bound(W, 0).T
            fW = fW & ~((W >= uW * (1 - _ACTIVE_TOL)) & (gW < 0))
            fH = fH & ~((H >= uH * (1 - _ACTIVE_TOL)) & (gH < 0))
        gW, gH = gW * fW, gH * fH
        gnorm2 = _joint_dot(gW, gW, gH, gH)
        if not torch.isfinite(gnorm2) or gnorm2 == 0:
            break
        # KKT fallback for data the model fits exactly, where the shifted loss goes to zero and its relative change
        # stays O(1). There loss ~ g^2, so a gradient ratio of rel_tol is a loss ratio of rel_tol^2, and machine
        # precision needs a gradient ratio of a few tens of eps. The loss test below is primary. The measure is
        # invariant to rescaling a component (_stationarity), so a start in a skewed gauge does not set it.
        gnorm = (gW.norm(dim=0) * W.norm(dim=0)).sum() + (gH.norm(dim=1) * H.norm(dim=1)).sum()
        if gnorm0 is None:
            gnorm0 = gnorm
        elif rel_tol > 0 and gnorm <= max(rel_tol ** 2, 100 * torch.finfo(T.dtype).eps) * gnorm0:
            break

        LW = _joint_blocks(Z @ (H[rows] * H[cols]).T, rows, cols, rank, fW, 1e-8)
        LH = _joint_blocks(((W[:, rows] * W[:, cols]).T @ Z).T, rows, cols, rank, fH.T, 1e-8)

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

        xW = torch.zeros_like(gW)
        xH = torch.zeros_like(gH)
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
            xW = xW + a * pW
            xH = xH + a * pH
            rW = rW - a * ApW
            rH = rH - a * ApH
            if _joint_dot(rW, rW, rH, rH) <= tol2:
                break
            zW, zH = precond(rW, rH)
            rz_new = _joint_dot(rW, zW, rH, zH)
            b = rz_new / rz
            pW = zW + b * pW
            pH = zH + b * pH
            rz = rz_new
        total_cg += ncg

        slope = _joint_dot(gW, xW, gH, xH)
        if slope >= 0:
            xW, xH = -gW, -gH
            slope = -gnorm2
        a, accepted = 1.0, False
        for _ in range(30):
            Wn = torch.minimum((W + a * xW).clamp_(min=0), uW) if nonneg_W else W + a * xW
            Hn = torch.minimum((H + a * xH).clamp_(min=0), uH) if nonneg_W else (H + a * xH).clamp_(min=0)
            if nonneg_W:                                 # both factors moved: W under _X_MAX over the new peaks
                Wn = torch.minimum(Wn, _partner_bound(Hn, 1))
            new_loss = nnal(Wn @ Hn, T, prep, dtype=torch.float64)
            if torch.isfinite(new_loss) and new_loss <= loss + 1e-4 * a * slope:
                accepted = True
                break
            a *= 0.5
        if not accepted:
            lam = lam * 10.0 if lam > 0 else 1e-12
            if lam > 1e6: break
            continue
        lam = max(lam * 0.3, 1e-14)
        # At the precision floor no step is accepted, the damping escalates, and the lam > 1e6 test above ends
        # the solve; rel_tol is the relative loss change per accepted step.
        rel_change = torch.abs(loss - new_loss) / torch.abs(new_loss).clamp_min(torch.finfo(torch.float64).tiny)
        W, H, loss = Wn, Hn, new_loss
        quiet = quiet + 1 if bool(rel_change <= rel_tol) else 0
        if rel_tol > 0 and quiet >= patience:
            break
    return W, H, step, total_cg


def joint_newton_optimize(T, num_materials, max_steps, rel_tol, update_H=True, W_init=None, H_init=None,
                          warmup_steps=5, cg_max=10, compile_mode=None, weights=None):
    """Block-Newton warm-up followed by a joint (W,H) preconditioned Newton solve.

    Alternating methods stall at a linear rate once the fit is good, because they
    discard the W<->H coupling block of the Hessian; on a problem where an exact
    factorization exists, block-Newton alone plateaus well short of machine
    precision. Solving for both factors jointly restores fast convergence. The
    alternating method is still the right way to get into the basin, so it runs
    first.

    With update_H=False only W is solved, by block Newton (the joint step needs both factors free). A block
    warm-up step costs more than a joint step with one CG iteration, so a few warm-up steps are enough. A component
    whose map or spectrum dies is re-seeded after every warm-up step and, up to _MAX_RESEEDS times, at the end of the
    joint solve, which then continues (_reseed_dead); a continued solve that ends no lower than the state before its
    re-seed is undone.
    """
    if not update_H:
        return block_newton_optimize(T, num_materials, max_steps, rel_tol, update_H=False, W_init=W_init,
                                     H_init=H_init, compile_mode=compile_mode, weights=weights)
    nnal_fn, deriv_fn, _, step_fn = _kernels(compile_mode)
    prep = _nnal_prep(T, weights)
    W, H = W_init, H_init
    steps = min(warmup_steps, max_steps)
    for i in range(steps):
        X = W @ H
        W, X, _ = step_fn(W, H, X, T, prep, 0)
        H, X, _ = step_fn(H, W, X, T, prep, 1)
        if i + 1 < max_steps:                             # re-seed only when a step follows
            W, H, _ = _reseed_dead(W, H, T, seed=i + 1)
    before = None                                         # (W, H, loss) before the last end-of-solve re-seed
    for attempt in range(_MAX_RESEEDS + 1):
        if steps >= max_steps:
            break
        W, H, taken, _ = _joint_newton_pcg(T, W, H, max_steps=max_steps - steps, cg_max=cg_max, rel_tol=rel_tol,
                                           prep=prep, nnal=nnal_fn, deriv=deriv_fn, patience=_PATIENCE)
        steps += taken
        if before is not None and bool(nnal_fn(W @ H, T, prep, dtype=torch.float64) >= before[2]):
            W, H, steps = before[0], before[1], before[3]           # the kept state, and the steps that reached it
            break
        if attempt == _MAX_RESEEDS or steps >= max_steps:
            break
        W_new, H_new, n_dead = _reseed_dead(W, H, T, seed=100 + attempt)
        if n_dead == 0:
            break
        before = (W, H, nnal_fn(W @ H, T, prep, dtype=torch.float64), steps)
        W, H = W_new, H_new
    return W, H, steps

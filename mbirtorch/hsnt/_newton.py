import torch

from ._linalg import (_attenuation_for_start, _batched_spd_solve, _joint_blocks, _joint_dot,
                      _nonneg_least_squares_start, _reseed_dead)
from ._loss import (BACKGROUND_MAX, _background_logs, _background_terms, _nnal_prep, _nnal_rowwise, stable_nnal,
                    stable_nnal_derivatives)


# The Armijo test treats a loss increase below this many times eps * |row loss| as float32 rounding.
_ARMIJO_FLOOR = 4.0
# The smallest step limit, as a fraction of the mean row scale.
_TRUST_FLOOR = 1e-3
# An entry is held at zero when it is within this fraction of its component's mean positive entry
# of zero and its gradient points toward zero.  This is the epsilon-active set of Bertsekas.
_ACTIVE_TOL = 1e-6
# A step that brings an entry within this many ulps of zero sets the entry to exactly zero.  Then
# compiled and eager runs give the same result.
_SNAP_ULPS = 8.0
# This is the largest attenuation that one component may add to an entry.  The bound is
# max_p W_pk * max_b H_kb <= _X_MAX.  This bound stops a component from growing without limit
# on entries with zero counts.
_X_MAX = 27.631021115928547                              # -log(1e-12)
# The joint solve stops after this many accepted steps in a row with a relative loss change below rel_tol.
_PATIENCE = 5
# With compile_mode 'auto', mbirtorch compiles on CUDA when the data have at least this many entries.
_COMPILE_MIN_ELEMENTS = 5e8
# The largest number of times the joint solve restarts after it replaces zero components with random values.
_MAX_RESEEDS = 2


def _resolve_compile(compile_mode, T, device=None):
    """This function converts compile_mode ('auto', 'on', 'off', or None) to 'on' or 'off'.  It
    converts 'auto' to 'on' when the device is CUDA, Triton works, and ``T`` has at least
    ``_COMPILE_MIN_ELEMENTS`` entries."""
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
    """This function returns the four most used functions: the loss, its derivatives, the loss
    per row, and the block Newton step.  It compiles them with torch.compile when compile_mode
    is 'on'."""
    if compile_mode != 'on':
        return stable_nnal, stable_nnal_derivatives, _nnal_rowwise, block_newton_step
    from ..projectors import maybe_compile
    return tuple(maybe_compile(f, True) for f in (stable_nnal, stable_nnal_derivatives, _nnal_rowwise,
                                                  block_newton_step))


def _two_metric_direction(V, grad, flat, rows, cols, jitter_rel=1e-9, nonneg=True, ub=None):
    """This function computes the projected Newton direction for each row of ``V`` (B, rank).
    The constraint is V >= 0, and also V <= ub when ``ub`` is given.

    The array ``flat`` (B, Q) holds the upper triangle of the Hessian of each row.  The arrays
    ``rows`` and ``cols`` give the indices of the upper triangle.  An entry is held fixed when it
    is at a bound and its gradient points out of the feasible set.  This rule is the two-metric
    projection method.  The other entries take the Newton step.  With nonneg=False, no entry is
    held fixed and alpha is 1.

    The function returns (d, slope, alpha, bound, projected_gnorm2).  ``d`` is the descent
    direction, and V decreases along +d.  ``slope`` is <grad, d> for each row.  ``alpha`` is the
    largest feasible step for each row.  ``bound`` is the mask of entries held fixed.
    ``projected_gnorm2`` is the squared norm of the projected gradient for each component.
    """
    rank = V.shape[1]
    M = flat.new_zeros(flat.shape[0], rank, rank)
    M[:, rows, cols] = flat
    M[:, cols, rows] = flat
    # The scale of each component is its mean positive entry.  The tolerances are relative to this scale.
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
    """This function returns the upper bound of a factor as an array of shape (1, rank).  For each
    component, the bound is ``_X_MAX`` divided by the largest entry of that component in the
    other factor."""
    peak = other.amax(0) if axis == 0 else other.amax(1)
    return torch.where(peak > 0, _X_MAX / peak.clamp_min(torch.finfo(other.dtype).tiny),
                       torch.full_like(peak, float('inf')))[None, :]


def block_newton_step(V, other, X, T, prep, axis, jitter_rel=1e-9, nonneg=True):
    """This function takes one projected Newton step on one factor.  The other factor stays
    fixed.

    With the other factor fixed, the problem splits into one small convex problem for each pixel
    (when updating W) or for each wavelength bin (when updating H).  This function uses the full
    rank x rank Hessian of each small problem.

    Args:
        V: The factor to update.  It is W (pixels x rank) when axis=0 and H (rank x bins) when
            axis=1.
        other: The fixed factor.
        X: The current product W @ H.
        T: The transmission.
        prep: The tuple returned by ``_nnal_prep(T)``.
        axis: 0 to update W, 1 to update H.

    Returns:
        (V_new, X_new, (num_backtracks, projected_gradient_norm_squared)).  The squared norm of
            the projected gradient is given for each component.  It is computed at the input
            value of V.
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

    # Along the step, X(alpha) = X - alpha * B exactly, so the line search needs no extra matmul.
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
        # A step is accepted when its loss exceeds the Armijo target by no more than float32 rounding.
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
        # Set the entries held fixed to zero, but only in rows where this does not raise the loss.
        # X is updated to match.
        snap = bound & (V_new > 0)
        if bool(snap.any()):
            dV = torch.where(snap, V_new, torch.zeros_like(V_new))
            X_try = X_new - (dV @ other if axis == 0 else other @ dV.T)
            l_plain = _nnal_rowwise(X_new, T, prep, dim, dtype=torch.float64)
            l_try = _nnal_rowwise(X_try, T, prep, dim, dtype=torch.float64)
            keep = l_try <= l_plain + _ARMIJO_FLOOR * torch.finfo(V.dtype).eps * l_plain.abs()
            V_new = torch.where(keep[:, None], V_new - dV, V_new)
            X_new = torch.where(expand(keep), X_try, X_new)
    if ub is not None:                                   # clamp entries that rounding put above ub, and recompute X
        over = V_new > ub
        if bool(over.any()):
            V_new = torch.minimum(V_new, ub)
            X_new = V_new @ other if axis == 0 else other @ V_new.T
    if axis == 1:
        V_new = V_new.T.contiguous()
    return V_new, X_new, (num_backtracks, projected_gnorm2)


def block_newton_optimize(T, num_materials, max_steps, rel_tol, update_H=True, W_init=None, H_init=None,
                          compile_mode=None, nonneg_W=True, background=None):
    """This function minimizes the non-negative attenuation loss by alternating projected Newton
    steps on W and on H.  The argument ``background`` is the fixed additive background of each
    bin (see ``_nnal_prep``), or None."""
    _, _, rowwise, step_fn = _kernels(compile_mode)
    prep = _nnal_prep(T, background)
    W, H = W_init, H_init
    X = W @ H
    prev_loss = rowwise(X, T, prep, 1, dtype=torch.float64).sum()
    gnorm0 = None
    num_steps = 0
    for step in range(max_steps):
        X = W @ H                      # recompute X to remove accumulated rounding error
        W_in, H_in = W, H
        W, X, info_W = step_fn(W, H, X, T, prep, 0, nonneg=nonneg_W)
        gnorm = _stationarity(info_W[1], W_in, 0)
        if update_H:
            H, X, info_H = step_fn(H, W, X, T, prep, 1)
            gnorm = gnorm + _stationarity(info_H[1], H_in, 1)
            W, H, _ = _reseed_dead(W, H, T, seed=step + 1)
        num_steps = step + 1
        if rel_tol > 0:
            # The first test is on the relative loss change.  The second (KKT) test
            # stops the solve when the data are fit exactly.
            loss = rowwise(X, T, prep, 1, dtype=torch.float64).sum()
            if gnorm0 is None:
                gnorm0 = gnorm
            if (bool(torch.abs(loss - prev_loss) <= rel_tol * torch.abs(loss))
                    or bool(gnorm <= max(rel_tol ** 2, 100 * torch.finfo(T.dtype).eps) * gnorm0)):
                break
            prev_loss = loss
    return W, H, num_steps


def _stationarity(projected_gnorm2, V, axis):
    """This function returns the sum over components of |projected gradient| x |factor|.  The
    stopping test uses this value because rescaling a component does not change it.  ``V`` is W
    when axis is 0 and H when axis is 1."""
    norms = V.norm(dim=0) if axis == 0 else V.norm(dim=1)
    return (projected_gnorm2.clamp_min(0).sqrt() * norms).sum()



def solve_W(T, H, W_init=None, max_steps=100, rel_tol=1e-12, nonneg=True, compile_mode=None):
    """This function computes the pixel coefficients W for a fixed H by block Newton steps.  It
    starts from ``W_init``, or from a nonnegative least squares fit when ``W_init`` is None.  It
    constrains W >= 0 unless nonneg=False."""
    if W_init is None:
        W_init = _nonneg_least_squares_start(_attenuation_for_start(T), H)
    W, _, _ = block_newton_optimize(T, H.shape[0], max_steps, rel_tol, update_H=False, W_init=W_init, H_init=H,
                                    compile_mode=compile_mode, nonneg_W=nonneg)
    return W

def _joint_newton_pcg(T, W, H, max_steps=50, cg_max=60, rel_tol=0.0, prep=None, nnal=None, deriv=None,
                      nonneg_W=True, w_mask=None, patience=1, fit_background=False):
    """This function solves for W and H together by a truncated Newton method with preconditioned
    CG.  It starts from the given W and H, and returns (W, H, steps, total_cg).

    Each step solves the damped Newton system by matrix free CG.  The preconditioner uses the
    Hessian blocks for each pixel and for each bin.  A backtracking line search then accepts the
    step.  The solve stops after ``patience`` accepted steps in a row with a relative loss change
    of at most rel_tol.  With nonneg_W=False, W is not constrained to be nonnegative.  When
    ``w_mask`` is given, the coefficients outside the mask stay at their current values.

    With fit_background=True, ``prep`` holds a background (see ``_nnal_prep``), and the solve
    also updates the background of each bin, within [0, BACKGROUND_MAX], with the Fisher
    information in place of its Hessian.  The function then returns (W, H, steps, total_cg, b).
    """
    nnal = stable_nnal if nnal is None else nnal
    deriv = stable_nnal_derivatives if deriv is None else deriv
    prep = _nnal_prep(T) if prep is None else prep
    W = W.clone()
    H = H.clone()
    b = torch.exp(prep[3][1]) if fit_background else None
    dot = lambda aW, bW, aH, bH, ab=None, bb=None: (_joint_dot(aW, bW, aH, bH)
                                                    + ((ab * bb).sum() if fit_background else 0.0))
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
        if nonneg_W:                                     # upper bounds from _X_MAX, fixed for this step
            uW, uH = _partner_bound(H, 1), _partner_bound(W, 0).T
            fW = fW & ~((W >= uW * (1 - _ACTIVE_TOL)) & (gW < 0))
            fH = fH & ~((H >= uH * (1 - _ACTIVE_TOL)) & (gH < 0))
        gW, gH = gW * fW, gH * fH
        gb = fb = Cb = Db = None
        if fit_background:
            gb, Cb, Db = _background_terms(X, T, b)
            fb = ~(((b <= 0) & (gb > 0)) | ((b >= BACKGROUND_MAX) & (gb < 0)))
            gb = gb * fb
        gnorm2 = dot(gW, gW, gH, gH, gb, gb)
        if not torch.isfinite(gnorm2) or gnorm2 == 0:
            break
        # This KKT test stops the solve when the model fits the data exactly.  In that case the
        # loss goes to zero, and the relative change of the loss stays large.  The loss test below
        # is the main test.
        gnorm = (gW.norm(dim=0) * W.norm(dim=0)).sum() + (gH.norm(dim=1) * H.norm(dim=1)).sum()
        if fit_background:
            gnorm = gnorm + gb.norm() * b.norm()
        if gnorm0 is None:
            gnorm0 = gnorm
        elif rel_tol > 0 and gnorm <= max(rel_tol ** 2, 100 * torch.finfo(T.dtype).eps) * gnorm0:
            break

        LW = _joint_blocks(Z @ (H[rows] * H[cols]).T, rows, cols, rank, fW, 1e-8)
        LH = _joint_blocks(((W[:, rows] * W[:, cols]).T @ Z).T, rows, cols, rank, fH.T, 1e-8)

        def precond(rW, rH, rb=None):
            zW = torch.cholesky_solve(rW.unsqueeze(-1), LW).squeeze(-1) * fW
            zH = torch.cholesky_solve(rH.T.unsqueeze(-1), LH).squeeze(-1).T * fH
            # A row with a zero Hessian block can overflow the preconditioner.  For such rows, CG
            # uses the residual without preconditioning.
            zW = torch.where(torch.isfinite(zW), zW, rW)
            zH = torch.where(torch.isfinite(zH), zH, rH)
            zb = rb / Db.clamp_min(torch.finfo(Db.dtype).tiny) * fb if fit_background else None
            return zW, zH, zb

        def hvp(dW, dH, db=None):
            dW, dH = dW * fW, dH * fH
            dX = dW @ H + W @ dH
            ZdX = Z * dX
            hb = None
            if fit_background:
                db = db * fb
                hb = ((Cb * dX).sum(0) + Db * db) * fb + lam * db
                ZdX = ZdX + Cb * db
            return ((ZdX @ H.T + G @ dH.T) * fW + lam * dW,
                    (W.T @ ZdX + dW.T @ G) * fH + lam * dH, hb)

        xW = torch.zeros_like(gW)
        xH = torch.zeros_like(gH)
        xb = torch.zeros_like(gb) if fit_background else None
        rW, rH, rb = -gW, -gH, (-gb if fit_background else None)
        zW, zH, zb = precond(rW, rH, rb)
        pW, pH, pb = zW.clone(), zH.clone(), (zb.clone() if fit_background else None)
        rz = dot(rW, zW, rH, zH, rb, zb)
        r0 = dot(rW, rW, rH, rH, rb, rb)
        tol2 = (torch.clamp(gnorm2.sqrt().sqrt(), max=0.5) ** 2) * r0
        ncg = 0
        for ncg in range(1, cg_max + 1):
            ApW, ApH, Apb = hvp(pW, pH, pb)
            pAp = dot(pW, ApW, pH, ApH, pb, Apb)
            if pAp <= 0:
                if ncg == 1: xW, xH, xb = zW, zH, zb
                break
            a = rz / pAp
            xW = xW + a * pW
            xH = xH + a * pH
            rW = rW - a * ApW
            rH = rH - a * ApH
            if fit_background:
                xb = xb + a * pb
                rb = rb - a * Apb
            if dot(rW, rW, rH, rH, rb, rb) <= tol2:
                break
            zW, zH, zb = precond(rW, rH, rb)
            rz_new = dot(rW, zW, rH, zH, rb, zb)
            beta = rz_new / rz
            pW = zW + beta * pW
            pH = zH + beta * pH
            if fit_background:
                pb = zb + beta * pb
            rz = rz_new
        total_cg += ncg

        slope = dot(gW, xW, gH, xH, gb, xb)
        if slope >= 0:
            xW, xH, xb = -gW, -gH, (-gb if fit_background else None)
            slope = -gnorm2
        a, accepted = 1.0, False
        for _ in range(30):
            Wn = torch.minimum((W + a * xW).clamp_(min=0), uW) if nonneg_W else W + a * xW
            Hn = torch.minimum((H + a * xH).clamp_(min=0), uH) if nonneg_W else (H + a * xH).clamp_(min=0)
            if nonneg_W:                                 # recompute the bound on W from the new H
                Wn = torch.minimum(Wn, _partner_bound(Hn, 1))
            prep_n = prep
            if fit_background:
                bn = (b + a * xb).clamp(0.0, BACKGROUND_MAX)
                prep_n = prep[:3] + (_background_logs(bn),)
            new_loss = nnal(Wn @ Hn, T, prep_n, dtype=torch.float64)
            if torch.isfinite(new_loss) and new_loss <= loss + 1e-4 * a * slope:
                accepted = True
                break
            a *= 0.5
        if not accepted:
            lam = lam * 10.0 if lam > 0 else 1e-12
            if lam > 1e6: break
            continue
        lam = max(lam * 0.3, 1e-14)
        # At the limit of float precision, no step is accepted, and the test lam > 1e6 ends the solve.
        rel_change = torch.abs(loss - new_loss) / torch.abs(new_loss).clamp_min(torch.finfo(torch.float64).tiny)
        W, H, loss, prep = Wn, Hn, new_loss, prep_n
        if fit_background:
            b = bn
        quiet = quiet + 1 if bool(rel_change <= rel_tol) else 0
        if rel_tol > 0 and quiet >= patience:
            break
    if fit_background:
        return W, H, step, total_cg, b
    return W, H, step, total_cg


def joint_newton_optimize(T, num_materials, max_steps, rel_tol, update_H=True, W_init=None, H_init=None,
                          warmup_steps=5, cg_max=10, compile_mode=None, background=None, fit_background=False):
    """This function minimizes the loss over W and H.  It first takes a few block Newton steps,
    and then it solves for W and H together by a preconditioned Newton method.

    The first block Newton steps bring the factors near the solution.  The joint solve then
    converges quickly.  With update_H=False, this function solves only for W, by block Newton
    steps.  When the map or spectrum of a component becomes zero, ``_reseed_dead`` replaces the
    component with small random values.  The argument ``background`` is the additive background
    of each bin (see ``_nnal_prep``), or None.  The background stays fixed unless fit_background
    is True.  In that case the joint solve also fits it, and the function returns
    (W, H, steps, background).
    """
    if not update_H:
        return block_newton_optimize(T, num_materials, max_steps, rel_tol, update_H=False, W_init=W_init,
                                     H_init=H_init, compile_mode=compile_mode, background=background)
    nnal_fn, deriv_fn, _, step_fn = _kernels(compile_mode)
    prep = _nnal_prep(T, background)
    W, H = W_init, H_init
    steps = min(warmup_steps, max_steps)
    for i in range(steps):
        X = W @ H
        W, X, _ = step_fn(W, H, X, T, prep, 0)
        H, X, _ = step_fn(H, W, X, T, prep, 1)
        if i + 1 < max_steps:                             # replace zero components only when another step follows
            W, H, _ = _reseed_dead(W, H, T, seed=i + 1)
    before = None                                 # (W, H, loss, steps, background) before zero components are replaced
    for attempt in range(_MAX_RESEEDS + 1):
        if steps >= max_steps:
            break
        out = _joint_newton_pcg(T, W, H, max_steps=max_steps - steps, cg_max=cg_max, rel_tol=rel_tol, prep=prep,
                                nnal=nnal_fn, deriv=deriv_fn, patience=_PATIENCE, fit_background=fit_background)
        W, H, taken = out[:3]
        if fit_background:
            background = out[4]
            prep = prep[:3] + (_background_logs(background),)
        steps += taken
        if before is not None and bool(nnal_fn(W @ H, T, prep, dtype=torch.float64) >= before[2]):
            W, H, steps = before[0], before[1], before[3]           # restore the saved W, H, and step count
            if fit_background:
                background = before[4]
            break
        if attempt == _MAX_RESEEDS or steps >= max_steps:
            break
        W_new, H_new, n_dead = _reseed_dead(W, H, T, seed=100 + attempt)
        if n_dead == 0:
            break
        before = (W, H, nnal_fn(W @ H, T, prep, dtype=torch.float64), steps, background)
        W, H = W_new, H_new
    if fit_background:
        return W, H, steps, background
    return W, H, steps

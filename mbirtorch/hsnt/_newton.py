import torch

from ._linalg import _batched_spd_solve, _joint_blocks, _joint_dot
from ._loss import _nnal_prep, _nnal_rowwise, stable_nnal, stable_nnal_derivatives
from ._multiplicative import _reseed_dead


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


def _two_metric_direction(V, grad, flat, rows, cols, jitter_rel=1e-9, nonneg=True):
    """Projected-Newton direction for a batch of rows of V (B, rank) under V >= 0.

    `flat` (B, Q) holds the upper triangle of each row's rank x rank Hessian
    (Q = rank (rank + 1) / 2, indexed by `rows`, `cols`). Two-metric projection
    (Bertsekas): variables within an epsilon of the bound with an outward
    gradient are frozen and snapped to zero, the rest take the Newton step; a
    bound-adjacent entry with an inward gradient gets the scaled gradient so one
    pinned entry cannot zero the step for its whole row; a per-row trust region
    bounds directions from rows without curvature; a non-descent direction falls
    back to the scaled gradient; alpha is the largest step keeping V >= 0. With
    nonneg=False there is no active set, no feasibility limit and alpha = 1. The
    constants and their measurements are documented at _ARMIJO_FLOOR,
    _TRUST_FLOOR and _ACTIVE_TOL and in docs/hsnt_solver_notes.md, section 1.

    Returns (d, slope, alpha, bound, projected_gnorm2): d is the descent
    direction (V decreases along +d), slope = <grad, d> per row, alpha the
    per-row feasible step, bound the frozen mask, and the squared norm of the
    projected gradient (the KKT residual at the incoming iterate).
    """
    rank = V.shape[1]
    M = flat.new_zeros(flat.shape[0], rank, rank)
    M[:, rows, cols] = flat
    M[:, cols, rows] = flat
    eps_active = _ACTIVE_TOL * V.abs().amax(-1, keepdim=True).mean()
    bound = ((V <= eps_active) & (grad > 0)) if nonneg else torch.zeros_like(grad, dtype=torch.bool)
    free = ~bound
    projected_gnorm2 = ((grad * free) ** 2).sum()
    eye = torch.eye(rank, dtype=V.dtype, device=V.device)
    M = torch.where(free[:, :, None] & free[:, None, :], M, eye.expand_as(M))
    rhs = torch.where(free, grad, torch.zeros_like(grad))
    d = _batched_spd_solve(M, rhs, jitter_rel)
    d = torch.where(free, d, torch.zeros_like(d))
    diag_M = torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(torch.finfo(V.dtype).tiny)
    inward = ((V <= eps_active) & (grad < 0)) if nonneg else torch.zeros_like(grad, dtype=torch.bool)
    d = torch.where(inward, grad / diag_M, d)
    row_max = V.abs().amax(-1, keepdim=True)
    floor = torch.clamp(_TRUST_FLOOR * row_max.mean(), min=torch.finfo(V.dtype).eps)
    limit = 16.0 * torch.maximum(row_max, floor)
    d = torch.clamp(d, min=-limit, max=limit)
    slope = (grad * d).sum(-1)
    d = torch.where((slope <= 0)[:, None], torch.clamp(rhs / diag_M, min=-limit, max=limit), d)
    slope = (grad * d).sum(-1)
    ratio = torch.where(d > 0, V / d.clamp_min(torch.finfo(V.dtype).tiny), torch.full_like(d, float('inf')))
    alpha = torch.clamp(ratio.amin(-1), max=1.0) if nonneg else torch.ones_like(ratio.amin(-1))
    return d, slope, alpha, bound, projected_gnorm2


def block_newton_step(V, other, X, T, prep, axis, ls_max=8, jitter_rel=1e-9, nonneg=True,
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

    d, slope, alpha, bound, projected_gnorm2 = _two_metric_direction(V, grad, flat, rows, cols, jitter_rel, nonneg)

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

    V_new = V - accepted[:, None] * d
    if nonneg:
        V_new = V_new.clamp_(min=0.0)
        V_new = torch.where(bound, torch.zeros_like(V_new), V_new)
    X_new = X - expand(accepted) * B
    if axis == 1:
        V_new = V_new.T.contiguous()
    return V_new, X_new, (num_backtracks, projected_gnorm2)


def block_newton_optimize(T, num_materials, max_steps, rel_tol, update_H=True,
                          convergence_check_interval=1, W_init=None, H_init=None,
                          jitter_rel=1e-9, compile_mode=None, nonneg_W=True):
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
        W, X, info_W = step_fn(W, H, X, T, prep, 0, jitter_rel=jitter_rel, nonneg=nonneg_W)
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



def solve_W(T, H, W_init=None, max_steps=100, rel_tol=1e-12, nonneg=True, compile_mode=None):
    """The pixel coefficients for a fixed H: independent convex problems per pixel,
    solved by block-Newton W steps from W_init (or a clamped least-squares warm
    start, the initialization optimize() uses). The single home of the "re-solve
    W given H" call; callers pass the tolerances they need."""
    if W_init is None:
        W_init = torch.linalg.lstsq(H.T, T.T)[0].T.clamp(min=0)
    W, _, _ = block_newton_optimize(T, H.shape[0], max_steps, rel_tol, update_H=False, W_init=W_init, H_init=H,
                                    compile_mode=compile_mode, nonneg_W=nonneg)
    return W

def _joint_newton_pcg(T, W, H, max_steps=50, cg_max=60, rel_tol=0.0, damping=1e-12,
                     precond_jitter=1e-8, prep=None, verbose=False, nnal=None, deriv=None, tilt_H=None,
                     nonneg_W=True, w_mask=None):
    # w_mask (bool, W's shape): coefficients outside the mask are held at their
    # current value (zero, for a selected support) and take no part in the step.
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
        if w_mask is not None:
            fW = fW & w_mask
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

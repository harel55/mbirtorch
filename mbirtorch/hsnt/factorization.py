import numpy as np
import torch

from ._linalg import nndsvda
from ._loss import _nnal_prep, stable_nnal
from ._multiplicative import multiplicative_update
from ._newton import block_newton_optimize, joint_newton_optimize


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
        # marks zero counts with a tiny positive value (1e-30); flooring THERE
        # gives an attenuation near 69 that the initialization's zero-fill smears
        # into every component. Any transmission below 1e-12 is treated as a zero
        # count; no real measurement gets anywhere near that. See
        # docs/hsnt_solver_notes.md, section 2.
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
        # cuts its sweeps by an order of magnitude. The extrapolated point is only
        # an input; the answer is always the last PLAIN iterate, so the fixed-point
        # set is untouched. A function-value safeguard restarts the momentum
        # whenever an extrapolated sweep raises the loss, since momentum on a
        # fixed-point map can overshoot; it uses the raw sum(exp(-X) + T X) minus
        # the constant that separates it from the shifted loss, far cheaper than
        # stable_nnal, and is evaluated every extrapolate_check_every sweeps: the
        # sweeps in between are unguarded, which is the price of the speed. rel_tol
        # keeps its per-step meaning by comparing across the check interval.
        # See docs/hsnt_solver_notes.md, section 3.
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

    # Converge on a float64 sum: in float32 the loss is quantized coarser than the
    # per-step progress at low dosage, and the test would fire on noise. See
    # docs/hsnt_solver_notes.md, section 1.
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

def nnal_factorization(T: torch.Tensor, method='joint_newton', num_materials=3, max_steps=1000,
                       rel_tol=1e-10, batch_size=None, compile_mode=None, random_state=0,
                       **kwargs) -> torch.Tensor:
    """Factorize T ~= exp(-W @ H), W, H >= 0, by minimizing the non-negative attenuation loss.

    method: 'joint_newton' (default) -- block warm-up then a matrix-free truncated
    Newton solve on (W, H); the fastest to a given loss and the only one that
    reaches machine precision on exactly factorizable data. 'block_newton' is the
    alternating exact projected-Newton method (linear convergence; also the warm-up
    and the fixed-H solver). 'mann_multiplicative' is the damped, extrapolated
    multiplicative update, at parity with joint_newton in wall clock. The former
    'quadratic' (IRLS) and 'quasi_newton' (diagonal Hessian) methods were removed
    in the 2026-09 cleanup as dominated. Measurements: docs/hsnt_solver_notes.md,
    section 3.

    rel_tol is the relative change in the loss per step (summed in float64) at
    which a method stops, and means the same thing for every method. It is not
    worth the same amount of convergence, though: joint_newton is close to its
    optimum at 1e-6, while block_newton and quadratic converge linearly and at
    1e-6 can stop on a plateau with a percent still to gain; use 1e-8 or tighter
    for those two. mann_multiplicative is extrapolated by default (see optimize)
    and reaches joint_newton's answer in comparable wall clock. On data the model
    fits exactly the loss goes to zero and this test never fires; a
    projected-gradient (KKT) test then takes over and runs to machine precision.
    See docs/hsnt_solver_notes.md, section 3.

    compile_mode: any non-None value compiles the elementwise hot kernels of
    block_newton and joint_newton (see _kernels), a one-off cost that pays for
    repeated solves, not for one. For the other methods the update function
    itself is compiled.

    random_state seeds the pixel permutation the batched path uses to choose the
    subsample H is fitted on. It defaults to 0 so two batched runs on the same
    data give the same answer; pass None for fresh entropy each call.
    """
    if method == 'mann_multiplicative':
        update = multiplicative_update
    elif method == 'block_newton':
        update = block_newton_optimize
    elif method == 'joint_newton':
        update = joint_newton_optimize
    else:
        raise ValueError("Invalid method. Choose 'joint_newton', 'block_newton' or 'mann_multiplicative'.")

    if update in (block_newton_optimize, joint_newton_optimize):
        # These take compile_mode themselves and compile their hot kernels, not the
        # driver: see _kernels. Wrapping the driver in torch.compile, as is done for
        # the other methods below, does nothing useful here.
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
        # same fixed points, far fewer sweeps (see optimize). Pass extrapolate=False to disable.
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
    # the loop.
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

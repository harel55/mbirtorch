import torch

from . import _newton
from ._loss import _nnal_prep
from ._newton import _kernels, solve_W
from .factorization import nnal_factorization


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
    """Projected-Newton direction on H from accumulated statistics: the H axis of
    block_newton_step on the (K, R) transpose. Returns (d, slope, alpha_max) with
    one row/entry per bin; see _newton._two_metric_direction."""
    d, slope, alpha, _, _ = _newton._two_metric_direction(H.T, grad.T, flat.T, rows, cols, jitter_rel)
    return d, slope, alpha


def stream_factorization(chunks, num_materials, max_passes=5, rel_tol=1e-6, warmup_pixels=16384,
                         w_rel_tol=1e-8, w_max_steps=300, ls_trials=4, device='cuda',
                         compile_mode=None, random_state=0, verbose=False, polish_dtype=None,
                         kkt_tol=None, stats=None, nonneg_W=True):
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
        nonneg_W: False estimates H with the bound on the pixel coefficients dropped
            during the polish passes (see unconstrained_spectra: it removes the
            truncation bias that capped H at 43.8 dB on 10M pixels in float64 as
            in float32), then re-solves W >= 0 for every chunk in one final pass.
            The warm-up fit stays constrained, since it finds the basin.

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
            W0 = W_chunks[i].to(device=device, dtype=H.dtype) if W_chunks[i] is not None else None
            W = solve_W(Tc, H.clone(), W0, w_max_steps, w_rel_tol, nonneg=nonneg_W, compile_mode=compile_mode)
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
        noise = _newton._ARMIJO_FLOOR * torch.finfo(H.dtype).eps * base.abs()
        ok = trial <= base[None, :] - 1e-4 * alphas.double() * slope.double()[None, :] + noise[None, :]   # Armijo, per bin and trial
        # largest accepted trial per bin, else zero
        accepted = torch.where(ok.any(0), alphas.gather(0, ok.float().argmax(0, keepdim=True)).squeeze(0), torch.zeros_like(alpha_max))
        Ht = (H.T - accepted[:, None] * d).clamp_(min=0)
        # Same epsilon-active snap as block_newton_step: a bin component at the
        # bound with an outward gradient becomes exactly zero, not a residue.
        eps_active = _newton._ACTIVE_TOL * Ht.abs().amax(-1, keepdim=True).mean()
        Ht = torch.where((Ht <= eps_active) & (grad.T.to(Ht.dtype) > 0), torch.zeros_like(Ht), Ht)
        H = Ht.T.contiguous()
        passes = p + 1
    if not nonneg_W:
        # The physical coefficients: one more pass, W >= 0 given the final H.
        for i in range(len(chunks)):
            Tc = to_device(chunks[i])
            W = solve_W(Tc, H.clone(), W_chunks[i].to(device=device, dtype=H.dtype).clamp(min=0),
                        w_max_steps, w_rel_tol, compile_mode=compile_mode)
            W_chunks[i] = W.cpu()
            del Tc, W
    return W_chunks, H, passes

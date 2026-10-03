import logging

import numpy as np
import torch

from . import _newton
from ._loss import _nnal_prep
from ._device import _default_device
from ._newton import _kernels, _resolve_compile, solve_W
from .factorization import _nnal_factorization

log = logging.getLogger("mbirtorch.hsnt")


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


def _h_direction(H, grad, flat, rows, cols, jitter_rel=1e-9, ub=None):
    """Projected-Newton direction on H from accumulated statistics: the H axis of
    block_newton_step on the (K, R) transpose, under the upper bound ub (1, R) when
    given. Returns (d, slope, alpha_max, bound) with one row/entry per bin, bound the
    entries frozen at a bound; see _newton._two_metric_direction."""
    d, slope, alpha, bound, _ = _newton._two_metric_direction(H.T, grad.T, flat.T, rows, cols, jitter_rel, ub=ub)
    return d, slope, alpha, bound


# Each chunk's W solve (block Newton, stopping tolerance and cap), and the step lengths tried per H line search.
_W_REL_TOL, _W_MAX_STEPS, _LS_TRIALS = 1e-8, 300, 4


def _stream_factorization(chunks, num_materials, max_passes=5, rel_tol=1e-6, warmup_pixels=16384, device=None,
                          compile_mode='off', verbose=0, stats=None, nonneg_W=True, support_selection=None,
                          chunk_sizes=None, weight_chunks=None):
    """Factorize a dataset too large for device memory, one chunk of pixels at a time.

    W is separable over pixels, so it is solved chunk by chunk and never held whole on the device. H holds only R * K
    values, and its Newton step needs only sums over pixels (gradient, per-bin R x R Hessian, per-bin loss), which
    accumulate across chunks: one pass over the data gives one exact Newton step on H, and a second pass evaluates
    its line search at a few step lengths at once. H starts from a joint-Newton fit on a random subsample of the
    pixels, so a handful of passes polish it. The joint Newton solver is not streamed: each of its CG iterations would
    be a full pass.

    Args:
        chunks (sequence of torch.Tensor): Chunks of the transmission ratio, each (pixels, bins), together making
            up T; any indexable sequence works, so chunks may be loaded lazily.
        num_materials (int): Rank R.
        max_passes (int, optional): Polish passes over the data; 0 keeps the subsample fit. A warning gives the last
            pass's relative loss change when max_passes ends the passes before rel_tol does. Defaults to 5.
        rel_tol (float, optional): Stop on the first pass that changes the total loss by at most this, relatively:
            one pass, not the five steps in a row of the solver held whole. 0 runs every pass. Defaults to 1e-6.
        warmup_pixels (int, optional): Pixels for the initial fit, a seeded random subset of all the chunks (each
            chunk holding one is read for it). Defaults to 16384.
        device (str, optional): Torch device. Defaults to None, meaning CUDA if available, else CPU.
        compile_mode (str, optional): 'auto', 'on' or 'off', as for the in-memory solver, judged on one chunk.
            Defaults to 'off'.
        verbose (int, optional): 1 prints the loss and KKT residual of every pass. Defaults to 0.
        stats (dict, optional): Receives 'loss' and 'kkt' lists, one entry per pass; the KKT residual of H is
            ||P(grad_H L)|| / ||W^T T||. Defaults to None.
        nonneg_W (bool, optional): False estimates H with the bound on W dropped during the polish passes (the
            unconstrained spectra), then re-solves W >= 0 for every chunk. Defaults to True.
        support_selection (dict, optional): Support selection after the passes: 'dose' (required) and optionally
            'penalty' ('auto' or a multiple of log K; 'auto' is judged once from every chunk), 'wald_screen'
            and 'max_passes' (the refit's pass budget). One pass selects each chunk's supports; the
            polish loop then runs again with W confined to them. The supports are returned in
            stats['support_chunks'], the refit's losses in stats['loss_refit'] and stats['kkt_refit'].
            Defaults to None.
        chunk_sizes (sequence of int, optional): The chunks' pixel counts, so that lazily loaded chunks need not be
            read for them. Defaults to None, from the chunks.
        weight_chunks (sequence of torch.Tensor, optional): Per-entry weights aligned with the chunks (see
            _nnal_prep); maximum-likelihood spectra only. Defaults to None, all ones.

    Returns:
        (W_chunks, H, passes): W as a list of CPU tensors aligned with the chunks, H, and the polish passes made.
    """
    device = _default_device(device)
    if weight_chunks is not None and (support_selection is not None or not nonneg_W):
        raise ValueError("weight_chunks is supported with the maximum-likelihood spectra only")
    compile_mode = _resolve_compile(compile_mode, chunks[0], device)
    _, deriv, rowwise, _ = _kernels(compile_mode)
    R = num_materials
    W_chunks = [None] * len(chunks)

    # H from a seeded random subsample spread over all the chunks: leading pixels alone can miss the sample.
    sizes = np.array(chunk_sizes if chunk_sizes is not None else [c.shape[0] for c in chunks])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    picks = np.sort(np.random.default_rng(0).choice(offsets[-1], min(warmup_pixels, offsets[-1]), replace=False))
    parts, wparts = [], []
    for i, c in enumerate(chunks):
        local = picks[(picks >= offsets[i]) & (picks < offsets[i + 1])] - offsets[i]
        if local.size:
            parts.append(c[torch.from_numpy(local).to(c.device)])
            if weight_chunks is not None:
                wparts.append(weight_chunks[i][torch.from_numpy(local).to(weight_chunks[i].device)])
    T_sub = torch.cat(parts, 0).to(device)
    A_sub = torch.cat(wparts, 0).to(device) if weight_chunks is not None else None
    _, H, _ = _nnal_factorization(T_sub, R, max_steps=300, rel_tol=1e-6, compile_mode=compile_mode, weights=A_sub)
    del T_sub, A_sub
    rows, cols = torch.triu_indices(R, R, device=H.device)
    pin = torch.device(device).type == 'cuda'

    def to_device(c):
        if c.device.type == 'cpu' and pin:
            c = c.pin_memory().to(device, non_blocking=True)
        else:
            c = c.to(device)
        return c

    def polish(solve_chunk, passes_max, tag):
        """Passes of: W per chunk (solve_chunk), H statistics accumulated over chunks, one exact Newton step on H
        with a per-bin line search evaluated in a second pass. Updates H and W_chunks; returns the passes made."""
        nonlocal H
        prev_loss = None
        passes = 0
        last = None                                   # the last pass's (W chunks, H), consistent with each other
        for p in range(passes_max + 1):
            # Pass A: W per chunk with H fixed; H's statistics summed over chunks in float64, since the per-bin
            # loss must resolve improvements far below the float32 ulp of a sum near 1e8.
            grad = torch.zeros(H.shape, dtype=torch.float64, device=H.device)
            flat = torch.zeros(rows.numel(), H.shape[1], dtype=torch.float64, device=H.device)
            base = torch.zeros(H.shape[1], dtype=torch.float64, device=H.device)
            scale = torch.zeros(H.shape, dtype=torch.float64, device=H.device)     # W^T T, the gradient's natural scale
            w_peak = torch.zeros(H.shape[0], dtype=H.dtype, device=H.device)      # for the attenuation bound on H
            nxt = to_device(chunks[0])
            for i in range(len(chunks)):
                Tc = nxt
                if i + 1 < len(chunks):
                    nxt = to_device(chunks[i + 1])            # prefetch overlaps the solve below
                Ac = to_device(weight_chunks[i]) if weight_chunks is not None else None
                prep = _nnal_prep(Tc, Ac)
                W0 = W_chunks[i].to(device=device, dtype=H.dtype) if W_chunks[i] is not None else None
                W = solve_chunk(Tc, W0, i, Ac)
                W_chunks[i] = W.cpu()
                g_c, f_c, b_c = _h_stats_accumulate(W, H, Tc, prep, rows, cols, deriv, rowwise)
                grad += g_c
                flat += f_c
                base += b_c
                scale += (W.T @ Tc).to(torch.float64)
                w_peak = torch.maximum(w_peak, W.amax(0))
                del Tc, W
            loss = base.sum(dtype=torch.float64)
            finite = all(bool(torch.isfinite(x).all()) for x in (loss, grad, flat))
            if not finite:
                # A component growing on the zero counts overflowed the float32 statistics: keep the last pass.
                log.warning("streamed fit: the %s statistics are not finite at pass %d (a component growing on the "
                            "zero counts overflows); the fit keeps the last finite pass",
                            "support refit" if tag else "polish", p)
                if last is not None:
                    W_chunks[:], H = last
                break
            last = (list(W_chunks), H)
            # Projected gradient: where H is zero only a negative gradient (a wish to grow) counts.
            pg = torch.where(H > 0, grad, grad.clamp(max=0))
            kkt = (pg.norm() / scale.norm()).item()
            if stats is not None:
                stats.setdefault('loss' + tag, []).append(loss.item())
                stats.setdefault('kkt' + tag, []).append(kkt)
            if verbose:
                print(f'  pass {p}{tag}: full-data loss {loss.item():.6e}  KKT residual {kkt:.2e}', flush=True)
            if prev_loss is not None and rel_tol > 0 and bool(torch.abs(loss - prev_loss) <= rel_tol * torch.abs(loss)):
                break
            if p == passes_max:
                if prev_loss is not None and rel_tol > 0:
                    change = abs(loss.item() - prev_loss.item()) / max(abs(loss.item()), np.finfo(np.float64).tiny)
                    log.warning("streamed fit: max_passes (%d) ended the %s before the rel_tol stop: the last pass "
                                "changed the loss by %.2e, relatively, against rel_tol %.2e", passes_max,
                                "support refit" if tag else "polish passes", change, rel_tol)
                break
            prev_loss = loss

            # One exact Newton step on H from the accumulated statistics.
            # The attenuation bound (see _newton._X_MAX): H under _X_MAX over each component's largest W, while the
            # chunks' W solves keep W under _X_MAX over H's peak. The free-signed W of the unconstrained spectra is
            # not bounded.
            ub = (torch.where(w_peak > 0, _newton._X_MAX / w_peak.clamp_min(torch.finfo(H.dtype).tiny),
                              torch.full_like(w_peak, float('inf')))[None, :] if nonneg_W else None)
            d, slope, alpha_max, bound = _h_direction(H, grad.to(H.dtype), flat.to(H.dtype), rows, cols, ub=ub)
            alphas = alpha_max[None, :] * (0.5 ** torch.arange(_LS_TRIALS, dtype=H.dtype, device=H.device))[:, None]
            # The epsilon-active snap, as in block_newton_step: the frozen entries (their step is zero) are set to zero,
            # but only in bins whose loss the snap does not raise, so pass B also scores every trial with the snap.
            snap = torch.where(bound & (H.T > 0), H.T, torch.zeros_like(H.T))
            with_snap = bool((snap > 0).any())

            # Pass B: the per-bin loss at every trial step, with and without the snap, summed over chunks.
            trial = torch.zeros(_LS_TRIALS, H.shape[1], dtype=torch.float64, device=H.device)
            trial_snap = torch.zeros_like(trial) if with_snap else None
            nxt = to_device(chunks[0])
            for i in range(len(chunks)):
                Tc = nxt
                if i + 1 < len(chunks):
                    nxt = to_device(chunks[i + 1])
                prep = _nnal_prep(Tc, to_device(weight_chunks[i]) if weight_chunks is not None else None)
                W = W_chunks[i].to(device)
                X = W @ H
                B = W @ d.T
                Bs = W @ snap.T if with_snap else None
                for t in range(_LS_TRIALS):
                    Xt = X - alphas[t][None, :] * B
                    trial[t] += rowwise(Xt, Tc, prep, 0, dtype=torch.float64)
                    if with_snap:
                        trial_snap[t] += rowwise(Xt - Bs, Tc, prep, 0, dtype=torch.float64)
                del Tc, W, X, B, Bs
            # Same floor as block_newton_step (see _ARMIJO_FLOOR): the float32 sums are
            # gone, but elements whose step falls below ulp(X) still do not move.
            noise = _newton._ARMIJO_FLOOR * torch.finfo(H.dtype).eps * base.abs()
            # Armijo, per bin and trial
            ok = trial <= base[None, :] - 1e-4 * alphas.double() * slope.double()[None, :] + noise[None, :]
            # largest accepted trial per bin, else zero
            first = ok.float().argmax(0, keepdim=True)
            accepted = torch.where(ok.any(0), alphas.gather(0, first).squeeze(0), torch.zeros_like(alpha_max))
            step = accepted[:, None] * d
            Ht = H.T - step
            reach = (d > 0) & (step >= H.T * (1.0 - _newton._SNAP_ULPS * torch.finfo(H.dtype).eps))  # see _SNAP_ULPS
            Ht = torch.where(reach, torch.zeros_like(Ht), Ht).clamp_(min=0)
            if ub is not None:
                Ht = torch.minimum(Ht, ub)
            if with_snap:
                chosen = trial.gather(0, first).squeeze(0)
                keep = ok.any(0) & (trial_snap.gather(0, first).squeeze(0) <= chosen + noise)
                Ht = torch.where(keep[:, None] & (snap > 0), torch.zeros_like(Ht), Ht)
            H = Ht.T.contiguous()
            passes = p + 1
        return passes

    def solve_mle(Tc, W0, i, Ac=None):
        return solve_W(Tc, H, W0, _W_MAX_STEPS, _W_REL_TOL, nonneg=nonneg_W, compile_mode=compile_mode, weights=Ac)

    passes = polish(solve_mle, max_passes, '')
    if not nonneg_W:
        # The physical coefficients: one more pass, W >= 0 given the final H.
        for i in range(len(chunks)):
            Tc = to_device(chunks[i])
            W = solve_W(Tc, H, W_chunks[i].to(device=device, dtype=H.dtype).clamp(min=0), _W_MAX_STEPS, _W_REL_TOL,
                        compile_mode=compile_mode)
            W_chunks[i] = W.cpu()
            del Tc, W

    if support_selection is not None:
        from .spectra import (_auto_penalty, _select_supports, _solve_W_on_support, _warn_collinear_rows,
                              _weak_components)
        opt = dict(support_selection)
        dose = opt.pop('dose')
        refit_passes = opt.pop('max_passes', max_passes)
        penalty = opt.pop('penalty', 'auto')
        wald_screen = opt.pop('wald_screen', 0.0)
        if opt:
            raise TypeError(f"unknown support_selection keys: {sorted(opt)}")
        if penalty == 'auto':                    # judged once from the whole data, not per chunk
            penalty = _auto_penalty(torch.cat([c.float().mean(1) for c in chunks]), dose)
        # One pass: each chunk's supports given the polished H (the MLE W is kept for the component guard).
        S_chunks = [None] * len(chunks)
        W_mle = list(W_chunks)
        counts = 0
        P_total = 0
        for i in range(len(chunks)):
            Tc = to_device(chunks[i])
            Wc = W_chunks[i].to(device=device, dtype=H.dtype)
            support, W0, _ = _select_supports(Tc, Wc, H, dose, penalty=penalty, wald_screen=wald_screen)
            S_chunks[i] = support.cpu()
            W_chunks[i] = W0.cpu()
            counts = counts + support.sum(0)
            P_total += Tc.shape[0]
            del Tc, Wc, support, W0
        weak = _weak_components(counts, P_total)
        if bool(weak.any()):
            weak_cpu = weak.cpu()
            for i in range(len(chunks)):
                S_chunks[i][:, weak_cpu] = True
                W_chunks[i][:, weak_cpu] = W_mle[i][:, weak_cpu]
        del W_mle

        # The polish loop again, W confined to the supports.
        def solve_masked(Tc, W0, i, Ac=None):
            return _solve_W_on_support(Tc, H, W0, S_chunks[i].to(device))

        refit_passes = polish(solve_masked, refit_passes, '_refit')
        _warn_collinear_rows(H)
        if stats is not None:
            stats['support_chunks'] = S_chunks
            stats['refit_passes'] = refit_passes
    return W_chunks, H, passes

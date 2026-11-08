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
    """This function computes the terms of one chunk for one Newton step on H.

    The terms are the gradient, the Hessian of each bin, and the loss of each bin.  The caller
    sums the terms over all chunks.
    """
    X = W @ H
    G, Z = deriv(X, T, prep)
    return W.T @ G, (W[:, rows] * W[:, cols]).T @ Z, rowwise(X, T, prep, 0, dtype=torch.float64)


def _h_direction(H, grad, flat, rows, cols, jitter_rel=1e-9, ub=None):
    """This function computes the projected Newton direction on H from the gradient and Hessian
    summed over all chunks.

    When ``ub`` is given, the direction respects the upper bound ``ub``, which has shape (1, R).
    The function returns (d, slope, alpha_max, bound), each with one row per bin.
    """
    d, slope, alpha, bound, _ = _newton._two_metric_direction(H.T, grad.T, flat.T, rows, cols, jitter_rel, ub=ub)
    return d, slope, alpha, bound


# _W_REL_TOL and _W_MAX_STEPS are the tolerance and the step limit of the block Newton solve for
# W in each chunk.  _LS_TRIALS is the number of step lengths that each line search on H tries.
_W_REL_TOL, _W_MAX_STEPS, _LS_TRIALS = 1e-8, 300, 4


def _stream_factorization(chunks, num_materials, max_passes=5, rel_tol=1e-6, warmup_pixels=16384, device=None,
                          compile_mode='off', verbose=0, stats=None, nonneg_W=True, support_selection=None,
                          chunk_sizes=None, W_init=None, H_init=None, quick=False, weights=None):
    """This function factorizes a dataset that is too large for device memory.

    The function processes the data one chunk of pixels at a time.  It first fits H by a joint
    Newton solve on a random subset of the pixels.  Then it makes passes over the data.  Each
    pass solves for W one chunk at a time and then takes one exact Newton step on H.

    Args:
        chunks (sequence of torch.Tensor): The chunks of the transmission ratio T.  Each chunk has
            shape (pixels, bins).  Any indexable sequence works, so the caller can load the chunks
            only when they are accessed.
        num_materials (int): The rank R of the factorization.
        max_passes (int, optional): The number of passes over the data after the fit on the
            subset.  The value 0 keeps the fit on the subset.  Defaults to 5.
        rel_tol (float, optional): The function stops after the first pass whose relative change
            in total loss is at most ``rel_tol``.  The value 0 runs every pass.  Defaults to 1e-6.
        warmup_pixels (int, optional): The number of pixels in the random subset for the first
            fit.  The function draws the subset from all chunks with a fixed seed.
            Defaults to 16384.
        device (str, optional): The torch device.  None means CUDA if it is available and the
            CPU otherwise.  Defaults to None.
        compile_mode (str, optional): 'auto', 'on', or 'off', with the same meaning as for the
            in-memory solver.  Defaults to 'off'.
        verbose (int, optional): The value 1 prints the loss and the KKT residual of every pass.
            Defaults to 0.
        stats (dict, optional): A dict that receives the lists 'loss' and 'kkt', with one entry
            for each pass.  Defaults to None.
        nonneg_W (bool, optional): If False, the function estimates H without the constraint
            W >= 0.  The resulting H holds the unconstrained spectra.  The function then solves
            for W >= 0 with that H.  Defaults to True.
        support_selection (dict, optional): The settings for support selection after the passes.
            The dict must hold 'dose' and can hold 'penalty', 'wald_screen', and 'max_passes'.
            The function returns the supports in stats['support_chunks'].  Defaults to None.
        chunk_sizes (sequence of int, optional): The number of pixels in each chunk.  With this
            argument, the function does not need to load a chunk to find its size.
            Defaults to None.
        W_init (torch.Tensor, optional): Starting maps for all pixels, with shape (pixels, R), on
            the CPU.  Defaults to None.
        H_init (torch.Tensor, optional): Starting spectra, with shape (R, bins).  When it is
            given, the function skips the fit on the subset and starts the passes from it.
            When only ``W_init`` is given, the fit on the subset starts from the subset's rows of
            ``W_init``.  Defaults to None.
        quick (bool, optional): If True, the fit on the subset uses the quick stop of the
            in-memory solver, and the passes also stop after _newton._QUICK_PATIENCE passes in
            a row whose change is below _newton._QUICK_TOL in units of the noise.  The change
            is measured over all parameters and over the spectra alone.  The function then
            sets stats['quick_stop'] to True, and it records stats['effective_dose'].
            Defaults to False.
        weights (torch.Tensor, optional): One weight per bin, as a (1, bins) tensor on the
            device.  Each entry's loss is multiplied by the weight of its bin (see
            ``_nnal_prep``).  With support selection, the dose times the weights must be the open
            beam count of each bin.  Defaults to None, which gives every entry the weight 1.

    Returns:
        (W_chunks, H, passes): W_chunks is a list of CPU tensors, one for each chunk.  H is the
            spectral factor.  passes is the number of passes made.
    """
    device = _default_device(device)
    compile_mode = _resolve_compile(compile_mode, chunks[0], device)
    _, deriv, rowwise, _ = _kernels(compile_mode)
    R = num_materials
    W_chunks = [None] * len(chunks)

    # The first fit of H uses a random subset of pixels, drawn from all chunks with seed 0.
    sizes = np.array(chunk_sizes if chunk_sizes is not None else [c.shape[0] for c in chunks])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    picks = np.sort(np.random.default_rng(0).choice(offsets[-1], min(warmup_pixels, offsets[-1]), replace=False))
    parts = []
    for i, c in enumerate(chunks):
        local = picks[(picks >= offsets[i]) & (picks < offsets[i + 1])] - offsets[i]
        if local.size:
            parts.append(c[torch.from_numpy(local).to(c.device)])
    if W_init is not None:
        W_chunks = [W_init[offsets[i]:offsets[i + 1]].clone() for i in range(len(chunks))]
    if H_init is not None:
        H = H_init.to(device)
    else:
        T_sub = torch.cat(parts, 0).to(device)
        W_sub = None if W_init is None else W_init[torch.from_numpy(picks)].to(device=device, dtype=T_sub.dtype)
        _, H, _ = _nnal_factorization(T_sub, R, max_steps=300, rel_tol=1e-6, compile_mode=compile_mode, W_init=W_sub,
                                      quick=quick, weights=weights)
        del T_sub, W_sub
    rows, cols = torch.triu_indices(R, R, device=H.device)
    pin = torch.device(device).type == 'cuda'

    def to_device(c):
        if c.device.type == 'cpu' and pin:
            c = c.pin_memory().to(device, non_blocking=True)
        else:
            c = c.to(device)
        return c

    def polish(solve_chunk, passes_max, tag):
        """This function runs the passes over the data and returns the number of passes made.

        Each pass solves for W in each chunk and then takes one Newton step on H.  The function
        updates H and W_chunks in place.
        """
        nonlocal H
        prev_loss = None
        passes = 0
        last = None                                   # (W_chunks, H) from the last finite pass.
        quick_quiet = 0
        q_H_step = None                               # The last step on H, in counts (sum of F dX^2 / dose).
        for p in range(passes_max + 1):
            pearson = q_W = torch.zeros((), dtype=torch.float64, device=H.device)
            n_entries = 0
            changed = True                            # False when every chunk has a W from the last pass.
            # The first loop over the chunks solves for W with H fixed.  It also sums the
            # gradient and Hessian of H over the chunks in float64.
            grad = torch.zeros(H.shape, dtype=torch.float64, device=H.device)
            flat = torch.zeros(rows.numel(), H.shape[1], dtype=torch.float64, device=H.device)
            base = torch.zeros(H.shape[1], dtype=torch.float64, device=H.device)
            scale = torch.zeros(H.shape, dtype=torch.float64, device=H.device)     # W^T T scales the KKT residual.
            w_peak = torch.zeros(H.shape[0], dtype=H.dtype, device=H.device)      # Used for the upper bound on H.
            nxt = to_device(chunks[0])
            for i in range(len(chunks)):
                Tc = nxt
                if i + 1 < len(chunks):
                    nxt = to_device(chunks[i + 1])            # This copy overlaps the solve below.
                prep = _nnal_prep(Tc, weights)
                W0 = W_chunks[i].to(device=device, dtype=H.dtype) if W_chunks[i] is not None else None
                W = solve_chunk(Tc, W0, i)
                W_chunks[i] = W.cpu()
                if quick:
                    # The Pearson residual gives the effective dose, and the change of W since the
                    # last pass is measured in the same units as the joint solve's quick stop.
                    # With weights, both sums weight each entry as the loss does.
                    E = torch.exp(-(W @ H))
                    r2 = (Tc - E) ** 2 / E.clamp_min(torch.finfo(E.dtype).tiny)
                    Ew = E
                    if weights is not None:
                        r2, Ew = r2 * weights, E * weights
                    pearson = pearson + r2.sum(dtype=torch.float64)
                    n_entries += Tc.numel()
                    if W0 is None:
                        changed = False
                    else:
                        dX = (W - W0) @ H
                        q_W = q_W + (Ew * dX).mul_(dX).sum(dtype=torch.float64)
                        del dX
                    del E, Ew, r2
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
                # A component that grows on the zero counts can overflow float32.  In that case
                # the function keeps the result of the last finite pass.
                log.warning("streamed fit: the %s statistics are not finite at pass %d (a component growing on the "
                            "zero counts overflows); the fit keeps the last finite pass",
                            "support refit" if tag else "polish", p)
                if last is not None:
                    W_chunks[:], H = last
                break
            last = (list(W_chunks), H)
            # In the projected gradient, an entry where H is zero counts only if its gradient is
            # negative.
            pg = torch.where(H > 0, grad, grad.clamp(max=0))
            kkt = (pg.norm() / scale.norm()).item()
            if stats is not None:
                stats.setdefault('loss' + tag, []).append(loss.item())
                stats.setdefault('kkt' + tag, []).append(kkt)
            if verbose:
                print(f'  pass {p}{tag}: full-data loss {loss.item():.6e}  KKT residual {kkt:.2e}', flush=True)
            if prev_loss is not None and rel_tol > 0 and bool(torch.abs(loss - prev_loss) <= rel_tol * torch.abs(loss)):
                break
            if quick:
                dose = n_entries / max(pearson.item(), np.finfo(np.float64).tiny)
                P_all, K = int(offsets[-1]), H.shape[1]
                if stats is not None:
                    stats['effective_dose'] = dose
                if changed and q_H_step is not None:
                    u_H = np.sqrt(dose * q_H_step / (R * K))
                    u_all = np.sqrt(dose * (q_W.item() + q_H_step) / (R * (P_all + K - R)))
                    quick_quiet = quick_quiet + 1 if max(u_H, u_all) < _newton._QUICK_TOL else 0
                if quick_quiet >= _newton._QUICK_PATIENCE:
                    if stats is not None:
                        stats['quick_stop' + tag] = True
                    break
            if p == passes_max:
                if prev_loss is not None and rel_tol > 0:
                    change = abs(loss.item() - prev_loss.item()) / max(abs(loss.item()), np.finfo(np.float64).tiny)
                    log.warning("streamed fit: max_passes (%d) ended the %s before the rel_tol stop: the last pass "
                                "changed the loss by %.2e, relatively, against rel_tol %.2e", passes_max,
                                "support refit" if tag else "polish passes", change, rel_tol)
                break
            prev_loss = loss

            # This step is one exact Newton step on H.  Each row of H stays below _X_MAX divided
            # by the largest W of that component (see _newton._X_MAX).  When nonneg_W is False,
            # W can be negative, and H has no upper bound.
            ub = (torch.where(w_peak > 0, _newton._X_MAX / w_peak.clamp_min(torch.finfo(H.dtype).tiny),
                              torch.full_like(w_peak, float('inf')))[None, :] if nonneg_W else None)
            d, slope, alpha_max, bound = _h_direction(H, grad.to(H.dtype), flat.to(H.dtype), rows, cols, ub=ub)
            alphas = alpha_max[None, :] * (0.5 ** torch.arange(_LS_TRIALS, dtype=H.dtype, device=H.device))[:, None]
            # As in block_newton_step, the entries in the active set are set to zero only in
            # bins where setting them to zero does not raise the loss.  The second loop over
            # the chunks therefore computes the loss of every trial step with and without
            # this change.
            snap = torch.where(bound & (H.T > 0), H.T, torch.zeros_like(H.T))
            with_snap = bool((snap > 0).any())

            # The second loop over the chunks sums the loss of each bin at every trial step.
            trial = torch.zeros(_LS_TRIALS, H.shape[1], dtype=torch.float64, device=H.device)
            trial_snap = torch.zeros_like(trial) if with_snap else None
            q_bin = torch.zeros(H.shape[1], dtype=torch.float64, device=H.device) if quick else None
            nxt = to_device(chunks[0])
            for i in range(len(chunks)):
                Tc = nxt
                if i + 1 < len(chunks):
                    nxt = to_device(chunks[i + 1])
                prep = _nnal_prep(Tc, weights)
                W = W_chunks[i].to(device)
                X = W @ H
                B = W @ d.T
                Bs = W @ snap.T if with_snap else None
                if quick:
                    q = (torch.exp(-X) * B).mul_(B).sum(0, dtype=torch.float64)
                    q_bin += q if weights is None else q * weights[0]
                for t in range(_LS_TRIALS):
                    Xt = X - alphas[t][None, :] * B
                    trial[t] += rowwise(Xt, Tc, prep, 0, dtype=torch.float64)
                    if with_snap:
                        trial_snap[t] += rowwise(Xt - Bs, Tc, prep, 0, dtype=torch.float64)
                del Tc, W, X, B, Bs
            # The Armijo test uses the same rounding tolerance as block_newton_step (see
            # _ARMIJO_FLOOR).  The test is applied to each bin and each trial step.
            noise = _newton._ARMIJO_FLOOR * torch.finfo(H.dtype).eps * base.abs()
            ok = trial <= base[None, :] - 1e-4 * alphas.double() * slope.double()[None, :] + noise[None, :]
            # Each bin takes its largest accepted step, or zero if no step is accepted.
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
            if quick:
                q_H_step = float((accepted.double() ** 2 * q_bin).sum())
            passes = p + 1
        return passes

    def solve_mle(Tc, W0, i):
        return solve_W(Tc, H, W0, _W_MAX_STEPS, _W_REL_TOL, nonneg=nonneg_W, compile_mode=compile_mode,
                       weights=weights)

    passes = polish(solve_mle, max_passes, '')
    if not nonneg_W:
        # One more pass solves for W >= 0 with the final H.
        for i in range(len(chunks)):
            Tc = to_device(chunks[i])
            W = solve_W(Tc, H, W_chunks[i].to(device=device, dtype=H.dtype).clamp(min=0), _W_MAX_STEPS, _W_REL_TOL,
                        compile_mode=compile_mode, weights=weights)
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
        if penalty == 'auto':                    # Chosen once from all the data, not per chunk.
            penalty = _auto_penalty(torch.cat([c.float().mean(1) for c in chunks]), dose)
        # This pass finds the supports of each chunk for the current H.  The code keeps the
        # maximum-likelihood W for the weak components.
        S_chunks = [None] * len(chunks)
        W_mle = list(W_chunks)
        counts = 0
        P_total = 0
        for i in range(len(chunks)):
            Tc = to_device(chunks[i])
            Wc = W_chunks[i].to(device=device, dtype=H.dtype)
            support, W0, _ = _select_supports(Tc, Wc, H, dose, penalty=penalty, wald_screen=wald_screen,
                                              weights=weights)
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

        # The passes run again, with each W restricted to its supports.
        def solve_masked(Tc, W0, i):
            return _solve_W_on_support(Tc, H, W0, S_chunks[i].to(device), weights=weights)

        refit_passes = polish(solve_masked, refit_passes, '_refit')
        _warn_collinear_rows(H)
        if stats is not None:
            stats['support_chunks'] = S_chunks
            stats['refit_passes'] = refit_passes
    return W_chunks, H, passes

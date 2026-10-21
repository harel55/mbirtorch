"""This module computes the factorization T = exp(-W @ H) for ``dehydrate`` and for the command
line.  It first estimates the device memory that the fit needs.  If all pixels fit on the device,
it solves for all pixels at once.  Otherwise, it streams chunks of pixels to the device.  After the
factorization, it computes the spectra by the method that the caller selects."""
import logging
import time

import numpy as np
import torch

log = logging.getLogger("mbirtorch.hsnt")

# This is the device memory of a full solve, in bytes per data entry, including T itself.
_BYTES_PER_ENTRY = dict(compiled=dict(mle=41, unconstrained=42, support=42, basis=54),
                        eager=dict(mle=58, unconstrained=60, support=60, basis=54))
# This is the extra memory of support selection, in bytes per entry of one block of
# ``spectra._FREE_SET_ELEMS`` entries.
_BYTES_PER_BLOCK_ENTRY = 32
# This is the device memory of a streamed solve, in bytes per entry of one chunk.
_BYTES_PER_CHUNK_ENTRY = 72
# This is the memory in bytes that the float64 blocks of the loss and the fit quality use at
# any data size.
_FIXED_BYTES = 2**29
# The automatic plan uses a full solve when it needs at most 0.8 of the available memory.
# Otherwise, it sizes the streamed chunks and the warm-up fit to use 0.6 of the available memory.
_FULL_SHARE, _STREAM_SHARE = 0.8, 0.6
_MIN_WARMUP_PIXELS = 4096
SPECTRA = ("mle", "unconstrained", "support")


def _device_memory(device):
    """This function returns the tuple (budget, total, name, remark) for a device.  The value
    ``budget`` is the number of bytes that a solve can allocate, ``total`` is the total memory of
    the device, and ``name`` is the name of the device.  The string ``remark`` notes how the budget
    was measured, and it is empty when there is nothing to note."""
    if device.startswith("cuda"):
        from .._memory_estimate import device_budget_bytes
        torch.cuda.empty_cache()
        budget, total = device_budget_bytes(device), torch.cuda.get_device_properties(device).total_memory
        remark = ""
        if budget > total:
            budget = torch.cuda.mem_get_info(device)[0]
            remark = (" (the driver's free memory alone: with the allocator's reserve it exceeded the device's "
                      "total)")
        budget = min(budget, total - torch.cuda.memory_allocated(device))
        return budget, total, torch.cuda.get_device_name(device), remark
    import psutil
    free = psutil.virtual_memory().available
    return free, free, "cpu", ""


def _plan(P, K, device, spectra="mle", mode="auto", chunk_pixels=None, compile_mode="auto", warmup_pixels=16384):
    """This function chooses a full solve or a streamed solve from the available device memory.
    It returns the tuple (mode, chunk_pixels, warmup_pixels, note).  For a full solve, the two
    pixel counts are None."""
    from ._newton import _resolve_compile
    budget, total, name, remark = _device_memory(device)
    compiled = _resolve_compile(compile_mode, torch.empty(P, K, device="meta"), device) == "on"
    if compiled and torch.device(device).type == "cuda":     # Without a working Triton, 'on' runs eager.
        from ..kernel_availability import triton_available
        compiled = triton_available()[0]
    need_full = P * K * _BYTES_PER_ENTRY["compiled" if compiled else "eager"][spectra] + _FIXED_BYTES
    if spectra == "support":
        from .spectra import _FREE_SET_ELEMS
        need_full += min(P * K, _FREE_SET_ELEMS) * _BYTES_PER_BLOCK_ENTRY
    note = (f"{name}: {budget / 2**30:.1f} GiB available of {total / 2**30:.1f}{remark}; a full "
            f"{'compiled' if compiled else 'eager'} solve needs about {need_full / 2**30:.1f} GiB")
    if mode == "auto":
        mode = "full" if need_full <= _FULL_SHARE * budget else "stream"
    if mode not in ("full", "stream"):
        raise ValueError(f"mode must be 'auto', 'full' or 'stream', got {mode!r}")
    warmup = None
    if mode == "stream":
        share = _STREAM_SHARE * budget
        if chunk_pixels is None:
            chunk_pixels = max(1024, int(share / (K * _BYTES_PER_CHUNK_ENTRY)) // 1024 * 1024)
        chunk_pixels = max(1, min(int(chunk_pixels), P))
        note += f"; streaming in chunks of {chunk_pixels:,} pixels ({-(-P // chunk_pixels)} chunks)"
        if spectra != "basis":
            asked = min(int(warmup_pixels), P)
            cap = max(_MIN_WARMUP_PIXELS, int(share / (K * _BYTES_PER_ENTRY["eager"]["mle"])) // 1024 * 1024)
            warmup = min(asked, cap)
            if warmup < asked:
                note += f"; the warm-up fits {warmup:,} pixels, not {asked:,}, to stay within the memory"
    log.info("plan: %s solve. %s", mode, note)
    return mode, chunk_pixels, warmup, note


def _rows(T, start, stop):
    """This function returns the rows start to stop of T as a tensor.  T is a numpy array or a
    tensor, and the rows of a tensor stay on its device."""
    return T[start:stop] if torch.is_tensor(T) else torch.from_numpy(T[start:stop])


def _loss(W, H, T_host, device):
    """This function computes the float64 NNAL loss of W @ H against the data.  The data are a
    numpy array or a tensor.  It moves the data to the device in chunks of pixels."""
    from ._loss import stable_nnal
    from .outputs import _CHUNK_ELEMENTS
    Hd = H.to(device).double()
    chunk = max(1, _CHUNK_ELEMENTS // T_host.shape[1])
    total = 0.0
    for i in range(0, T_host.shape[0], chunk):
        Tc = _rows(T_host, i, i + chunk).to(device).double()
        total += stable_nnal(W[i:i + chunk].to(device).double() @ Hd, Tc).item()
    return total


def _report_zero_counts(rep, W, H, T, device, chunk):
    """This function finds two problems in a fit and records them in ``rep``.  The first problem
    is zero-count entries whose fitted attenuation reaches the upper bound.  The second problem is
    components that model the zero counts instead of a material.  The function logs a warning for
    each problem it finds."""
    from ._newton import _X_MAX
    from .factorization import _CAPTURED_SHARE, _components_at_bound, _zero_count_divergence, _zero_count_mass
    x_max, n_above = 0.0, 0
    Hd = H.to(device)
    on_zero = torch.zeros(Hd.shape[0], dtype=torch.float64, device=device)
    at_bound = torch.zeros(Hd.shape[0], dtype=torch.int64, device=device)
    for i in range(0, T.shape[0], chunk):
        Tc = _rows(T, i, i + chunk)
        Wc, Tc = W[i:i + chunk].to(device), Tc.to(device)
        m, n = _zero_count_divergence(Wc, Hd, Tc)
        x_max, n_above = max(x_max, m), n_above + n
        on_zero += _zero_count_mass(Wc, Hd, Tc)
        at_bound += _components_at_bound(Wc, Hd, Tc)
    total = W.to(torch.float64).sum(0).to(device) * Hd.to(torch.float64).sum(1)
    share = torch.where(total > 0, on_zero / total.clamp_min(torch.finfo(torch.float64).tiny), torch.zeros_like(total))
    captured = [k for k in range(share.numel()) if float(share[k]) >= _CAPTURED_SHARE or int(at_bound[k]) > 0]
    rep.update(zero_count_max_attenuation=x_max, zero_count_entries_at_bound=n_above,
               zero_count_share=[round(float(v), 6) for v in share], zero_count_components=captured)
    if n_above:
        log.warning("the fit holds %d zero-count entries at the attenuation bound (%.3g, a transmission of 1e-12). "
                    "At a few counts per bin the likelihood keeps rising along a component that grows on the zero "
                    "counts; the bound stops it there", n_above, _X_MAX)
    if captured:
        log.warning("components on zero counts: %s (the share of the component's attenuation on zero-count entries, "
                    "and the entries where it alone reaches the attenuation bound). Such a component models the zero "
                    "counts rather than a material; the materials have %d of the %d components",
                    ", ".join(f"{k} ({100 * float(share[k]):.2f}%, {int(at_bound[k])} at the bound)" for k in captured),
                    share.numel() - len(captured), share.numel())


def _fit_fixed_basis(T, H, device="cpu", mode="auto", chunk_pixels=None, max_steps=1000, rel_tol=1e-8,
                     compile_mode="auto", report=None):
    """This function computes the maps W >= 0 for a fixed basis H.  It uses block Newton steps on
    chunks of pixels.  It returns (W, H, report), and it fills ``report`` in the same way as
    :func:`_fit`."""
    from ._newton import _resolve_compile, solve_W
    P, K = T.shape
    Hd = torch.as_tensor(np.ascontiguousarray(H, dtype=np.float32), device=device)
    mode, chunk, _, note = _plan(P, K, device, "basis", mode, chunk_pixels, compile_mode)
    chunk = P if mode == "full" else chunk
    rep = {} if report is None else report
    rep.update(mode=mode, memory_plan=note, spectra="given basis", chunks=-(-P // chunk))
    if mode == "stream":
        rep["chunk_pixels"] = chunk
    if mode != "full":                                  # Streamed chunks compile only when compile_mode is 'on'.
        compile_mode = "on" if compile_mode == "on" else "off"
    t0 = time.perf_counter()
    parts = []
    for i in range(0, P, chunk):
        Tc = _rows(T, i, i + chunk).to(device)
        parts.append(solve_W(Tc, Hd, max_steps=max_steps, rel_tol=rel_tol,
                             compile_mode=_resolve_compile(compile_mode, Tc)).cpu())
        del Tc
    W = torch.cat(parts)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    rep["solve_seconds"] = round(time.perf_counter() - t0, 2)
    rep["loss_mle"] = rep["loss_final"] = _loss(W, Hd, T, device)
    log.info("maps for the given basis: %s, %d chunk(s) in %.1f s, loss %.6g", mode, rep["chunks"],
             rep["solve_seconds"], rep["loss_final"])
    _report_zero_counts(rep, W, Hd, T, device, chunk)
    rep["W_zero_frac"], rep["H_zero_frac"] = (W == 0).double().mean().item(), (Hd == 0).double().mean().item()
    if device.startswith("cuda"):
        rep["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    return W.numpy().astype(np.float32), np.asarray(H, dtype=np.float32), rep


def _fit(T, rank, spectra="mle", dose=None, penalty="auto", wald_screen=0.0, device="cpu",
         mode="auto", chunk_pixels=None, max_steps=1000, rel_tol=1e-8, max_passes=5, warmup_pixels=16384,
         compile_mode="auto", report=None):
    """This function fits the transmission T at the given rank.  T is a float32 numpy array or
    tensor with shape (pixels, bins).  A tensor on the solve's device is used without a copy.  The function returns (W, H, report).  W and H are the numpy factors,
    and ``report`` is a dict that describes what the fit did.

    The argument ``spectra`` is 'mle', 'unconstrained', or 'support'.  The value 'support' needs
    the dose.  The argument ``penalty`` is the penalty per component of the support selection.
    It is either 'auto' or a multiple of log K.  The function fills ``report`` in place.  As a
    result, the caller has the memory plan even when the solve fails.
    """
    from ._streaming import _stream_factorization
    from .factorization import _nnal_factorization
    from .spectra import _support_selected_spectra, _unconstrained_spectra
    if spectra not in SPECTRA:
        raise ValueError(f"spectra must be one of {SPECTRA}, got {spectra!r}")
    if spectra == "support" and dose is None:
        raise ValueError("support selection needs the dose (open-beam counts per pixel and bin)")
    P, K = T.shape
    mode, chunk, warmup, note = _plan(P, K, device, spectra, mode, chunk_pixels, compile_mode, warmup_pixels)
    rep = {} if report is None else report
    rep.update(mode=mode, memory_plan=note, spectra=spectra)
    if mode == "stream":
        rep["chunk_pixels"] = chunk
        if str(device).startswith("cuda"):           # Each chunk and the next one are held as pinned host copies.
            from .loading import _check_host_memory
            _check_host_memory(2 * min(chunk, P) * K * T.dtype.itemsize, "the stream's pinned chunk copies")
    t0 = time.perf_counter()
    stats = {}
    if mode == "full":
        Td = _rows(T, 0, P).to(device)
        W, H, steps = _nnal_factorization(Td, rank, max_steps=max_steps, rel_tol=rel_tol, compile_mode=compile_mode)
        rep["steps"] = int(steps)
    else:
        chunks = [_rows(T, i, i + chunk) for i in range(0, P, chunk)]
        support = (dict(dose=dose, penalty=penalty, wald_screen=wald_screen)
                   if spectra == "support" else None)
        W_chunks, H, passes = _stream_factorization(chunks, rank, max_passes=max_passes, rel_tol=rel_tol,
                                                    warmup_pixels=warmup, device=device,
                                                    verbose=int(log.isEnabledFor(logging.DEBUG)), stats=stats,
                                                    nonneg_W=(spectra != "unconstrained"), support_selection=support,
                                                    compile_mode="on" if compile_mode == "on" else "off",
                                                    chunk_sizes=[c.shape[0] for c in chunks])
        W = torch.cat(W_chunks)
        rep.update(passes=int(passes), loss_per_pass=stats.get("loss"), kkt_per_pass=stats.get("kkt"))
        Td = None
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    rep["solve_seconds"] = round(time.perf_counter() - t0, 2)
    if mode == "full":
        rep["loss_mle"] = _loss(W, H, T, device)
    # Otherwise, the loss of the last pass is at the final W and H.  With unconstrained W, this
    # loss is not the MLE loss.
    else:
        rep["loss_mle"] = stats["loss"][-1] if spectra != "unconstrained" else None
    steps_text = f"{rep['steps']} steps" if "steps" in rep else f"{rep['passes']} polish passes"
    log.info("factorization: %s, %s in %.1f s, loss %s", mode, steps_text, rep["solve_seconds"],
             "n/a" if rep["loss_mle"] is None else f"{rep['loss_mle']:.6g}")
    if mode == "full":
        rep["mle_hit_max_steps"] = rep["steps"] >= max_steps
        if rep["mle_hit_max_steps"] and rel_tol > 0:
            log.warning("the maximum-likelihood fit stopped at max_steps (%d) before its rel_tol stop", max_steps)
    if spectra == "unconstrained" and mode == "full":
        t1 = time.perf_counter()
        W, H, st = _unconstrained_spectra(Td, W, H, compile_mode=compile_mode)
        rep.update(unconstrained_steps=int(st), unconstrained_seconds=round(time.perf_counter() - t1, 2))
    elif spectra == "support" and mode == "full":
        t1 = time.perf_counter()
        W, H, S, st = _support_selected_spectra(Td, W, H, dose, penalty=penalty, wald_screen=wald_screen,
                                                compile_mode=compile_mode)
        rep.update(support_steps=int(st), support_seconds=round(time.perf_counter() - t1, 2),
                   mean_support_size=S.sum(1).double().mean().item())
    elif spectra == "support":
        S = torch.cat(stats["support_chunks"])
        rep.update(mean_support_size=S.sum(1).double().mean().item(), support_refit_passes=int(stats["refit_passes"]),
                   loss_per_pass_refit=stats.get("loss_refit"))
    rep["loss_final"] = rep["loss_mle"] if spectra == "mle" else _loss(W, H, T, device)
    if spectra != "mle":
        support_text = (f", mean {rep['mean_support_size']:.2f} materials per pixel" if "mean_support_size" in rep
                        else "")
        log.info("%s spectra%s: loss %.6g", spectra, support_text, rep["loss_final"])
    # These checks apply to the returned factors in every mode.
    _report_zero_counts(rep, W, H, Td if mode == "full" else T, device, P if mode == "full" else chunk)
    if not (bool(torch.isfinite(W).all()) and bool(torch.isfinite(H).all())):
        log.warning("the fit's factors are not finite")
    rep["W_zero_frac"], rep["H_zero_frac"] = (W == 0).double().mean().item(), (H == 0).double().mean().item()
    if device.startswith("cuda"):
        rep["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    return W.cpu().numpy().astype(np.float32), H.cpu().numpy().astype(np.float32), rep

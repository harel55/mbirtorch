"""The fit behind dehydrate and the command line: a memory plan, then the maximum-likelihood factorization held whole on
the device or streamed by chunks of pixels, then the requested spectra estimator."""
import logging
import time

import numpy as np
import torch

log = logging.getLogger("mbirtorch.hsnt")

# Device working set of a full solve, in bytes per data entry with T, about 10% above the measured peaks: compiled
# (H100, 1M pixels x 1200 bins: the joint-Newton MLE 37.2, the unconstrained re-estimate 38.2, support selection no
# higher) and eager (H100, 262k and 1M pixels: the MLE 53.3, unconstrained 54.3; support selection's refit as
# unconstrained). A W solve against a fixed basis peaks at 49.4 on the CPU.
_BYTES_PER_ENTRY = dict(compiled=dict(mle=41, unconstrained=42, support=42, basis=54),
                        eager=dict(mle=58, unconstrained=60, support=60, basis=54))
# Support selection adds its free-set blocks: 29.3 bytes per entry of one block on the CPU (a block holds at most
# spectra._FREE_SET_ELEMS entries), while the selection's own 37 bytes per data entry stay below the solve's.
_BYTES_PER_BLOCK_ENTRY = 32
# A streamed chunk's working set per chunk entry, about 10% above the measured (52.5 to 58.1 bytes beyond the chunk on
# the CPU, 63 for a chunk's support selection, and on CUDA the chunk and its prefetched successor, 8 more); the
# streamed warm-up is a full eager MLE fit of its subsample.
_BYTES_PER_CHUNK_ENTRY = 72
# Held at any size: the float64 blocks of the loss and of the fit quality (outputs._CHUNK_ELEMENTS entries each).
_FIXED_BYTES = 2**29
# A full solve is planned when its estimate is within the first share of the budget; stream chunks and the streamed
# warm-up are sized to the second.
_FULL_SHARE, _STREAM_SHARE = 0.8, 0.6
_MIN_WARMUP_PIXELS = 4096
SPECTRA = ("mle", "unconstrained", "support")


def _device_memory(device):
    """(budget, total, name, remark): the bytes a solve can allocate on the device, the device's total, its name, and
    a remark on the reading for the plan's note ('' when there is none).

    On CUDA the caching allocator's unused blocks are released first. The budget is then device_budget_bytes (the
    driver's free memory plus what the allocator reserves but does not use), at most the total less what is allocated.
    Where that sum exceeds the total, as the WSL2 driver can report, the driver's free memory is taken alone.
    """
    if device.startswith("cuda"):
        from .._memory_ledger import device_budget_bytes
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
    """A full or streamed solve from the memory the device has available. Returns (mode, chunk_pixels,
    warmup_pixels, note): the chunk and the streamed warm-up's pixels are None in a full solve.

    The full solve's estimate uses the compiled or the eager constants as compile_mode resolves for the whole data,
    and is chosen when within _FULL_SHARE of the budget; a streamed solve's chunks and its warm-up subsample are sized
    to _STREAM_SHARE of it. spectra is 'mle', 'unconstrained', 'support' or 'basis' (a given basis: no warm-up).
    """
    from ._newton import _resolve_compile
    budget, total, name, remark = _device_memory(device)
    compiled = _resolve_compile(compile_mode, torch.empty(P, K, device="meta"), device) == "on"
    if compiled and torch.device(device).type == "cuda":     # 'on' without a working Triton runs eager
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


def _loss(W, H, T_host, device, weights=None):
    """The float64 NNAL loss of W @ H against the host data, in pixel chunks so no P x K float64 array is formed;
    weights (host numpy, as T) multiply each entry's term."""
    from ._loss import stable_nnal, _nnal_prep
    from .outputs import _CHUNK_ELEMENTS
    Hd = H.to(device).double()
    chunk = max(1, _CHUNK_ELEMENTS // T_host.shape[1])
    total = 0.0
    for i in range(0, T_host.shape[0], chunk):
        Tc = torch.from_numpy(T_host[i:i + chunk]).to(device).double()
        prep = None if weights is None else _nnal_prep(Tc, torch.from_numpy(weights[i:i + chunk]).to(device).double())
        total += stable_nnal(W[i:i + chunk].to(device).double() @ Hd, Tc, prep).item()
    return total


def _report_zero_counts(rep, W, H, T, device, chunk):
    """Record, and warn about, zero-count entries whose fitted attenuation sits at the solver's bound
    (_zero_count_divergence), and the components that model the zero counts rather than a material: those holding at
    least _CAPTURED_SHARE of their attenuation there (_zero_count_mass), or reaching the attenuation bound on them
    alone (_components_at_bound). Goes through T (host numpy or a device tensor)
    by blocks of `chunk` rows."""
    from ._newton import _X_MAX
    from .factorization import _CAPTURED_SHARE, _components_at_bound, _zero_count_divergence, _zero_count_mass
    x_max, n_above = 0.0, 0
    Hd = H.to(device)
    on_zero = torch.zeros(Hd.shape[0], dtype=torch.float64, device=device)
    at_bound = torch.zeros(Hd.shape[0], dtype=torch.int64, device=device)
    for i in range(0, T.shape[0], chunk):
        Tc = T[i:i + chunk] if torch.is_tensor(T) else torch.from_numpy(T[i:i + chunk])
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
    """The maps for a given basis: every pixel's maximum-likelihood W >= 0 with H fixed, independent problems solved
    by block Newton in chunks of pixels from the memory plan. Returns (W, H, report) as _fit does, and fills `report`
    as _fit does."""
    from ._newton import _resolve_compile, solve_W
    P, K = T.shape
    Hd = torch.as_tensor(np.ascontiguousarray(H, dtype=np.float32), device=device)
    mode, chunk, _, note = _plan(P, K, device, "basis", mode, chunk_pixels, compile_mode)
    chunk = P if mode == "full" else chunk
    rep = {} if report is None else report
    rep.update(mode=mode, memory_plan=note, spectra="given basis", chunks=-(-P // chunk))
    if mode == "stream":
        rep["chunk_pixels"] = chunk
    if mode != "full":                                  # as a streamed fit: chunks compile only when asked
        compile_mode = "on" if compile_mode == "on" else "off"
    t0 = time.perf_counter()
    parts = []
    for i in range(0, P, chunk):
        Tc = torch.from_numpy(T[i:i + chunk]).to(device)
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
         compile_mode="auto", report=None, weights=None, H_init=None):
    """Fit T (host numpy, pixels x bins, float32) at the given rank. Returns (W, H, report): numpy factors and a dict
    of what was done (mode, steps or passes, seconds, losses, support size).

    spectra: 'mle', 'unconstrained' or 'support' (needs the dose). penalty: support selection's charge per component,
    'auto' or a multiple of log K. compile_mode applies to the full solve; a streamed solve compiles only with 'on'.
    rel_tol stops a full solve after five steps in a row below it and a streamed one on the first pass below it, within
    max_passes. warmup_pixels is the streamed warm-up's subsample, capped by the memory plan. report, a dict, is
    filled in place of a new one, so that a caller has the memory plan even when the solve fails. weights (host numpy,
    as T, >= 0; maximum-likelihood spectra only) multiply each entry's loss: e.g. 1 - P for overlap-corrected
    MCP/Timepix counts (see _loss._nnal_prep). H_init (numpy, rank x bins): starting spectra, e.g. tabulated ones,
    for the full solve or the streamed warm-up.
    """
    from ._streaming import _stream_factorization
    from .factorization import _nnal_factorization
    from .spectra import _support_selected_spectra, _unconstrained_spectra
    if weights is not None and spectra != "mle":
        raise ValueError("weights are supported with the maximum-likelihood spectra only")
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
        if str(device).startswith("cuda"):           # each chunk and its successor are pinned host copies
            from .loading import _check_host_memory
            _check_host_memory(2 * min(chunk, P) * K * T.dtype.itemsize, "the stream's pinned chunk copies")
    t0 = time.perf_counter()
    stats = {}
    if mode == "full":
        Td = torch.from_numpy(T).to(device)
        Ad = torch.from_numpy(weights).to(device) if weights is not None else None
        W, H, steps = _nnal_factorization(Td, rank, max_steps=max_steps, rel_tol=rel_tol, compile_mode=compile_mode,
                                          weights=Ad, H_init=None if H_init is None else
                                          torch.as_tensor(np.asarray(H_init), dtype=Td.dtype, device=device))
        del Ad
        rep["steps"] = int(steps)
    else:
        chunks = [torch.from_numpy(T[i:i + chunk]) for i in range(0, P, chunk)]
        support = (dict(dose=dose, penalty=penalty, wald_screen=wald_screen)
                   if spectra == "support" else None)
        W_chunks, H, passes = _stream_factorization(chunks, rank, max_passes=max_passes, rel_tol=rel_tol,
                                                    warmup_pixels=warmup, device=device,
                                                    verbose=int(log.isEnabledFor(logging.DEBUG)), stats=stats,
                                                    nonneg_W=(spectra != "unconstrained"), support_selection=support,
                                                    compile_mode="on" if compile_mode == "on" else "off",
                                                    chunk_sizes=[c.shape[0] for c in chunks],
                                                    weight_chunks=None if weights is None else
                                                    [torch.from_numpy(weights[i:i + chunk]) for i in range(0, P, chunk)],
                                                    H_init=None if H_init is None else
                                                    torch.as_tensor(np.asarray(H_init), dtype=torch.float32))
        W = torch.cat(W_chunks)
        rep.update(passes=int(passes), loss_per_pass=stats.get("loss"), kkt_per_pass=stats.get("kkt"))
        Td = None
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    rep["solve_seconds"] = round(time.perf_counter() - t0, 2)
    if mode == "full":
        rep["loss_mle"] = _loss(W, H, T, device, weights)
    else:       # the last polish pass's loss is at the final W and H; with free-signed W it is not the MLE's
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
    # On the factors returned, in every mode: the maximum-likelihood fit, or the spectra estimator's with W >= 0.
    _report_zero_counts(rep, W, H, Td if mode == "full" else T, device, P if mode == "full" else chunk)
    if not (bool(torch.isfinite(W).all()) and bool(torch.isfinite(H).all())):
        log.warning("the fit's factors are not finite")
    rep["W_zero_frac"], rep["H_zero_frac"] = (W == 0).double().mean().item(), (H == 0).double().mean().item()
    if device.startswith("cuda"):
        rep["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    return W.cpu().numpy().astype(np.float32), H.cpu().numpy().astype(np.float32), rep

"""This module estimates the rank of hyperspectral data, which is the number of spectral
components.  It fits the data with ranks 1, 2, 3, and so on, and it uses likelihood-ratio tests
to decide when an added component fits only noise.  Each fit minimizes the non-negative
attenuation loss (NNAL) defined in ``_loss.py``.
"""
import math
import warnings

import numpy as np
import torch

from ._device import _default_device
from ._loss import _weights_tensor


def _loss_and_residual(T, W, H, weights=None):
    """This function returns two float64 values: the loss of X = W @ H against T, and the mean
    of (T - e^-X)^2 / e^-X.  It computes both values in blocks of pixels to limit memory.  With
    ``weights``, a (1, bins) tensor of the weight of each bin, both values weight each entry by
    the weight of its bin."""
    from ._loss import _nnal_prep, stable_nnal
    from .outputs import _CHUNK_ELEMENTS
    Hd = H.double()
    wd = None if weights is None else weights.double()
    chunk = max(1, _CHUNK_ELEMENTS // T.shape[1])
    loss = resid = 0.0
    for i in range(0, T.shape[0], chunk):
        Xd = W[i:i + chunk].double() @ Hd
        Th = torch.exp(-Xd)
        Td = T[i:i + chunk].double()
        loss += stable_nnal(Xd, Td, _nnal_prep(Td, wd)).item()
        r2 = ((Td - Th) ** 2) / Th.clamp_min(1e-12)
        resid += (r2 if wd is None else r2 * wd).sum().item()
    return loss, resid / T.numel()


def _lrt_rank(T, max_rank, label, verbose=0, weights=None):
    """This function runs the likelihood-ratio rank test on the transmission T.  It returns the
    tuple (rank, detail), where detail is a dict of the losses, gains, and threshold.  The
    argument ``weights`` is a (1, bins) tensor of the weight of each bin, or None."""
    from .factorization import _nnal_factorization
    P, K = T.shape
    losses, resid = [], []
    for r in range(1, max_rank + 1):
        # The fit is not compiled, because each rank has a new shape that would force a recompile.
        W, H, _ = _nnal_factorization(T, r, max_steps=200, rel_tol=1e-6, compile_mode="off", weights=weights)
        loss, res = _loss_and_residual(T, W, H, weights)
        losses.append(loss)
        resid.append(res)
    dose_eff = 1.0 / max(resid[-1], np.finfo(np.float64).tiny)
    gains = [dose_eff * (losses[i - 1] - losses[i]) for i in range(1, len(losses))]     # gains[i - 1] is for component i + 1.
    edge = 0.5 * (math.sqrt(P) + math.sqrt(K)) ** 2          # This is the gain of a component fitted to noise alone.
    quiet = [g for g in gains[-3:] if g < 2.0 * edge]
    floor = max(0.5 * (P + K), float(np.median(quiet))) if quiet else edge
    threshold = 2.0 * floor
    rank = 1
    for r, g in zip(range(2, max_rank + 1), gains):
        if g > threshold:
            rank = r
        else:
            break
    if verbose >= 1:
        table = ", ".join(f"{r}: {g:,.0f}" for r, g in zip(range(2, max_rank + 1), gains))
        print(f"rank search ({label}, {P:,} pixels x {K} bins): effective dose {dose_eff:.3g}; log-likelihood gain of "
              f"component {table}; noise floor {floor:.0f}, threshold {threshold:.0f} -> rank {rank}")
    return rank, dict(pixels=P, losses=losses, gains=gains, effective_dose=dose_eff, noise_floor=floor,
                      threshold=threshold, noise_tail=bool(quiet) or max_rank == 1)


def _pool_pixels(T, spatial_shape, block):
    """This function averages the transmission over square blocks of detector pixels within
    each view.  Each block is block x block pixels.  The function first crops the rows and the
    columns to multiples of the block size.

    Args:
        T (numpy.ndarray): The transmission ratio, with shape (pixels, bins).  The pixels are in
            (views, rows, cols) order.
        spatial_shape (tuple): The shape (views, rows, cols).
        block (int): The block size in pixels along the rows and the columns.

    Returns:
        numpy.ndarray: The pooled transmission, with shape
            (views * (rows // block) * (cols // block), bins).
    """
    V, rows, cols = spatial_shape
    r, c = rows // block * block, cols // block * block
    X = np.asarray(T).reshape(V, rows, cols, -1)
    out = np.empty((V, r // block, c // block, X.shape[-1]), dtype=X.dtype)
    for v in range(V):                                  # This loop never copies the whole array.
        for i in range(r // block):
            band = X[v, i * block:(i + 1) * block, :c]
            out[v, i] = band.reshape(block, c // block, block, -1).mean(axis=(0, 2))
    return out.reshape(-1, out.shape[-1])


def _subsample(T, n):
    """This function returns at most n rows of T as a float32 array.  It chooses the rows at
    random with a fixed seed, and it keeps them in their original order."""
    rows = np.sort(np.random.default_rng(0).choice(T.shape[0], n, replace=False)) if T.shape[0] > n else slice(None)
    return np.ascontiguousarray(T[rows], dtype=np.float32)


def estimate_rank(data, dataset_type="attenuation", max_rank=6, device=None, pool="auto", verbose=0):
    """This function estimates the number of spectral components in a hyperspectral dataset.

    The function fits the data with ranks 1 to ``max_rank`` in turn.  It keeps adding components
    while the gain in log-likelihood of each new component is well above the gain expected from
    noise.  The data may have the image axes (views, rows, cols) or (rows, cols) before the
    spectral axis.  In that case, the function also runs the test on blocks of averaged pixels,
    and it returns the larger of the two ranks.

    Args:
        data (numpy.ndarray or torch.Tensor): The data, with the spectral axis last.
        dataset_type (str, optional): Either 'attenuation' or 'transmission'.  Defaults to
            'attenuation'.
        max_rank (int, optional): The largest rank tested.  Defaults to 6.
        device (str, optional): The torch device.  Defaults to None, which selects CUDA if it is
            available and the CPU otherwise.
        pool (str or int, optional): The width in pixels of the blocks that are averaged.  The
            value 'auto' chooses the width from the estimated counts per bin, and 0 turns
            averaging off.  Defaults to 'auto'.
        verbose (int, optional): If 1, the function prints the gains and the chosen rank of each
            test.  Defaults to 0.

    Returns:
        (rank, note, detail): The estimated rank, a one-line note that says how the rank was
            chosen, and a dict with the details of each test.
    """
    from .denoise import _spatial_shape, _to_transmission
    T, shape = _to_transmission(data, dataset_type)
    return _estimate_rank(T, _spatial_shape(shape[:-1]), device, max_rank, pool=pool, verbose=verbose)


def _estimate_rank(T, spatial_shape=None, device=None, max_rank=6, subsample=16384, pool="auto", verbose=0,
                   weights=None):
    """This function performs :func:`estimate_rank` on a transmission ratio with shape
    (pixels, bins).  The argument ``subsample`` is the most pixels that each test uses.  The
    argument ``weights`` is None or a numpy array with the weight of each bin, as in
    ``_fit._fit``."""
    device = _default_device(device)
    wd = _weights_tensor(weights, device)
    T_np = T.detach().cpu().numpy() if torch.is_tensor(T) else np.asarray(T)
    pixels, K = T_np.shape
    rows_per = max(1, 2**24 // max(K, 1))
    if not any(bool((T_np[i:i + rows_per] > 0).any()) for i in range(0, pixels, rows_per)):
        raise ValueError("the data hold no counts: the transmission is zero everywhere")
    Tt = torch.from_numpy(_subsample(T_np, subsample)).to(device)
    rank_full, d_full = _lrt_rank(Tt, max_rank, "full resolution", verbose, wd)
    block = 0
    if spatial_shape is not None and pool:
        rows, cols = spatial_shape[1], spatial_shape[2]
        if pool == "auto":
            block = int(np.ceil(np.sqrt(64.0 / max(d_full["effective_dose"], 1e-9))))
            block = int(min(block, np.ceil(np.sqrt(2.0 * pixels / K)), max(1, min(rows, cols) // 4)))
        else:
            block = int(pool)
    detail = dict(full=d_full, pool_block=block, max_rank=max_rank, rank_full=rank_full, rank_pooled=None)
    rank = rank_full
    parts = [f"full resolution gave {rank_full}"]
    if spatial_shape is None and pool == "auto" and d_full["effective_dose"] < 64.0:
        # With pool='auto', averaging is used below 64 counts per bin.  At such low counts the
        # full-resolution test can miss faint components.
        parts.append("no pooling: the data have no image axes")
        dose = d_full["effective_dose"]
        warnings.warn(f"the rank was estimated at full resolution only ({dose:.3g} counts per bin): the data have no "
                      "image axes, so neighboring pixels could not be pooled, and at this count the full-resolution "
                      "test can miss components. Pass the data with its image axes, (views, rows, cols, bins) or "
                      "(rows, cols, bins), or give the number of components")
    if block > 1:
        Tp = torch.from_numpy(_subsample(_pool_pixels(T_np, spatial_shape, block), subsample)).to(device)
        rank_pool, d_pool = _lrt_rank(Tp, max_rank, f"pooled {block}x{block}", verbose, wd)
        detail.update(pooled=d_pool, rank_pooled=rank_pool)
        rank = max(rank_full, rank_pool)
        parts.append(f"pooled {block}x{block} ({d_pool['pixels']:,} pixels) gave {rank_pool}")
    decisive = detail.get("pooled") or d_full
    if rank == max_rank or not decisive["noise_tail"]:
        warnings.warn(f"the largest ranks tried still gain more than noise does; the rank may exceed max_rank "
                      f"{max_rank}, raise max_rank")
    note = f"rank {rank} estimated by likelihood-ratio tests ({'; '.join(parts)})"
    detail.update(gains=decisive["gains"], effective_dose=d_full["effective_dose"], threshold=decisive["threshold"])
    return rank, note, detail

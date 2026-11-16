"""This module compresses and restores hyperspectral neutron data with a factorization X = W H.
The function ``dehydrate`` computes the factors, and the function ``rehydrate`` multiplies them."""
import warnings

import numpy as np

from ..utilities import _to_host

# These keywords belong to the MBIRJAX version of dehydrate, which used scikit-learn.  mbirtorch rejects them.
_L2_KEYWORDS = ("safety_factor", "beta_loss", "max_iter", "tolerance", "batch_size", "random_state")


def _reject_unknown_keywords(name, kwargs):
    if not kwargs:
        return
    legacy = sorted(set(kwargs) & set(_L2_KEYWORDS))
    if legacy:
        raise TypeError(f"{name}() fits the Poisson likelihood and does not take {', '.join(legacy)}, which belong to "
                        "MBIRJAX's scikit-learn NMF dehydrate; MBIRTorch does not include that method")
    raise TypeError(f"{name}() got unexpected keyword argument(s) {', '.join(sorted(kwargs))}")


def _spatial_shape(lead):
    """This function returns the leading axes as (views, rows, cols).  It sets views to 1 when
    there are two leading axes, and it returns None when there are not two or three."""
    return tuple(lead) if len(lead) == 3 else (1,) + tuple(lead) if len(lead) == 2 else None


def _clean_transmission(T):
    """This function sets each NaN, infinite, or negative entry of a float32 transmission array to
    zero, in place.  It returns a dict with the fraction of entries that it changed."""
    info = {}
    bad = ~np.isfinite(T)
    if bad.any():
        info["nonfinite_frac"] = float(bad.mean())
        T[bad] = 0.0
    negative = T < 0
    if negative.any():
        info["negative_frac"] = float(negative.mean())
        T[negative] = 0.0
    return info


def _clean_transmission_tensor(T):
    """This function does what ``_clean_transmission`` does for a torch tensor, on the tensor's
    device.  It leaves T unchanged, and it copies T only when an entry needs cleaning.  It returns
    (T, info)."""
    import torch
    info = {}
    bad = ~torch.isfinite(T)
    if bool(bad.any()):
        info["nonfinite_frac"] = float(bad.double().mean())
        T = torch.where(bad, torch.zeros_like(T), T)
    negative = T < 0
    if bool(negative.any()):
        info["negative_frac"] = float(negative.double().mean())
        T = T.clamp_min(0.0)
    return T, info


def _warn_cleaning(info):
    if info.get("nonfinite_frac"):
        warnings.warn(f"{100 * info['nonfinite_frac']:.3g}% of the entries are NaN or have an infinite transmission; "
                      "treated as zero counts")
    if info.get("negative_frac"):
        warnings.warn(f"{100 * info['negative_frac']:.3g}% of the entries have a negative transmission; clipped at "
                      "zero")


def _to_transmission(data, dataset_type):
    """This function converts the data to a float32 transmission array of shape (pixels, bins).
    It cleans the transmission array with ``_clean_transmission``.  It returns the transmission
    array and the shape of the input.  A torch tensor gives a tensor on the same device, and it is
    copied only when the conversion or the cleaning needs a copy.  Any other input gives a numpy
    array."""
    import torch
    if dataset_type not in ("attenuation", "transmission"):
        raise ValueError("'dataset_type' must be either 'attenuation' or 'transmission'.")
    on_device = torch.is_tensor(data)
    a = data.detach() if on_device else _to_host(data)
    if a.ndim < 2:
        raise ValueError(f"data must have at least one leading axis and a spectral last axis; got shape {tuple(a.shape)}")
    shape = tuple(a.shape)
    if on_device:
        a = a.reshape(-1, shape[-1]).to(torch.float32)
        T = torch.exp(-a) if dataset_type == "attenuation" else a
    else:
        a = a.reshape(-1, shape[-1]).astype(np.float32, copy=False)
        if dataset_type == "attenuation":
            with np.errstate(over="ignore", invalid="ignore"):
                T = np.exp(-a)
        else:
            T = a.copy()
    T, info = _clean_transmission_tensor(T) if on_device else (T, _clean_transmission(T))
    _warn_cleaning(info)
    if not bool((T > 0).any()):
        raise ValueError("the data hold no counts: the transmission is zero everywhere")
    if on_device:
        return T.contiguous(), shape
    return np.ascontiguousarray(T, dtype=np.float32), shape


def dehydrate(data, dataset_input_type="transmission", num_materials=None, *, spectra=None,
              max_steps=1000, rel_tol=1e-8, max_num_est_materials=6, device=None,
              mode="auto", max_passes=5, sample_mask=None, open_beam=None,
              verbose=1, **kwargs):
    """This function dehydrates hyperspectral data.  It computes a non-negative factorization
    of the data in the attenuation domain.  The spectral axis of the data is always the last axis.

    For transmission data :math:`T`, this function computes the non-negative factorization
    :math:`T = \\exp(-W H)`, where ``W`` has one row per pixel and :math:`N_m` columns, and ``H``
    is :math:`N_m \\times N_k`.  The rows of ``H`` are the spectral basis, and the columns of
    ``W`` are the maps.  The function ``rehydrate`` turns the result back into data.

    The algorithm fits ``W`` and ``H`` by maximum likelihood.  It assumes that :math:`T` comes
    from Poisson counts, so zero counts are handled correctly.

    The input may be either 'attenuation' or 'transmission'.  The output ``W`` and ``H`` always
    describe attenuation.

    Args:
        data (numpy.ndarray or torch.Tensor): The data, with the :math:`N_k` spectral bins on
            the last axis and the pixels on the other axes, for example (views, rows, cols).
            A NaN or infinite transmission counts as zero counts.  A negative transmission is
            clipped to zero.
        dataset_input_type (str, optional): 'attenuation' or 'transmission'.  Defaults to
            'transmission'.
        num_materials (int, optional): The number of spectra :math:`N_m`.  Defaults to None,
            which estimates it from the data.
        spectra (numpy.ndarray or torch.Tensor, optional): Fixed spectra ``H`` of shape
            :math:`(N_m, N_k)`.  When given, only the maps ``W`` are computed.  Defaults to None.
        max_steps (int, optional): The maximum number of solver iterations.  Defaults to 1000.
        rel_tol (float, optional): The solver stops when the relative change in the loss falls
            below this value.  Defaults to 1e-8.
        max_num_est_materials (int, optional): Only used when ``num_materials`` is not given.
            The estimate tries 1 to this many materials.  Defaults to 6.
        device (str, optional): The torch device.  Defaults to None, which uses CUDA when
            available and the CPU otherwise.
        mode (str, optional): 'full' holds all pixels in device memory.  'stream' processes
            the pixels in chunks.  'auto' chooses by the available memory.  When the data have
            image axes and few counts, a streamed fit starts from spectra fitted to averaged
            blocks of neighboring pixels.  Defaults to 'auto'.
        max_passes (int, optional): The passes over the chunks in 'stream' mode.  Defaults
            to 5.
        sample_mask (numpy.ndarray or torch.Tensor, optional): A boolean array, True where the
            sample may be.  Its shape is (rows, cols) for one region shared by all views, or
            (views, rows, cols) for a region per view.  ``W`` is set to zero outside the mask
            and fit only inside it.  Defaults to None.
        open_beam (float or numpy.ndarray, optional): The open beam count of each spectral bin,
            as a 1-D array of length :math:`N_k`.  Use it when the open beam differs between
            bins.  Each bin's loss is then weighted by its open beam count divided by the mean
            count, so that the loss is the Poisson negative log-likelihood of the counts.  A
            scalar, or an array with the same count in every bin, gives every bin the same
            weight.  Defaults to None, which gives every bin the same weight.
        verbose (int, optional): 0 prints nothing.  1 prints a summary.  2 also prints the
            search for ``num_materials``.  Defaults to 1.

    Returns:
        list: The list [maps, spectra].  maps is ``W``, with the pixel axes of the data followed
            by an axis of length :math:`N_m`.  spectra is ``H``, with shape :math:`(N_m, N_k)`.
            Both are float32, and ``W H`` is the attenuation at each pixel and bin.
    """
    import torch
    from ._device import _default_device
    from ._fit import _fit
    from .rank import _estimate_rank
    _reject_unknown_keywords("dehydrate", kwargs)
    if mode not in ("auto", "full", "stream"):
        raise ValueError(f"mode must be 'auto', 'full' or 'stream', got {mode!r}")
    if int(max_passes) < 0:
        raise ValueError(f"max_passes must be at least 0, got {max_passes}")
    T_all, shape = _to_transmission(data, dataset_input_type)
    weights = _bin_weights(open_beam, shape[-1])
    device = _default_device(device)
    lead = shape[:-1]
    mask = _sample_mask(sample_mask, lead)
    if mask is None:
        T = T_all
    elif torch.is_tensor(T_all):
        T = T_all[torch.from_numpy(mask).to(T_all.device)].contiguous()
    else:
        T = np.ascontiguousarray(T_all[mask])
    pixels = f"{T.shape[0]}" if mask is None else f"{T.shape[0]} of {T_all.shape[0]} (sample_mask)"
    if spectra is not None:
        from ._fit import _fit_fixed_basis
        H = _check_basis(spectra, shape[-1], num_materials)
        W, H, rep = _fit_fixed_basis(T, H, device=device, mode=mode, max_steps=max_steps, rel_tol=rel_tol,
                                     weights=weights)
        if verbose >= 1:
            print("dehydrate(): ")
            print("   -Spectral dimension: ", shape[-1], " -> rank: ", H.shape[0], "(the given spectra)")
            print("   -Pixels: ", pixels, f"; maps for the given basis ({rep['chunks']} chunk(s))")
    else:
        note = f"rank {num_materials} given"
        if num_materials is None:
            num_materials, note, _ = _estimate_rank(T_all, spatial_shape=_spatial_shape(lead), device=device,
                                                    max_rank=max_num_est_materials,
                                                    verbose=max(0, verbose - 1), weights=weights)
        W, H, rep = _fit(T, int(num_materials), device=device, mode=mode, max_steps=max_steps, rel_tol=rel_tol,
                         max_passes=int(max_passes), weights=weights, spatial_shape=_spatial_shape(lead),
                         pool_T=None if mask is None else T_all, pool_mask=mask)
        if verbose >= 1:
            solve = f"{rep['steps']} steps" if "steps" in rep else f"streamed, {rep['passes']} polish passes"
            print("dehydrate(): ")
            print("   -Spectral dimension: ", shape[-1], " -> rank: ", int(num_materials), f"({note})")
            print("   -Pixels: ", pixels, f" ({solve})")
    if mask is not None:
        W_all = np.zeros((T_all.shape[0], W.shape[1]), dtype=np.float32)
        W_all[mask] = W
        W = W_all
    return [W.reshape(*lead, H.shape[0]), H]


def _bin_weights(open_beam, bins):
    """This function converts the ``open_beam`` of ``dehydrate`` to the weight of each bin.  The
    weight is the bin's open beam count divided by the mean count.  It returns None when every bin
    has the same weight, so that the fit is then the unweighted fit."""
    if open_beam is None:
        return None
    d = np.asarray(_to_host(open_beam), dtype=np.float64)
    if not (np.isfinite(d).all() and (d > 0).all()):
        raise ValueError("open_beam must be finite and positive in every bin")
    if d.ndim == 0:
        return None
    if d.ndim != 1 or d.shape[0] != bins:
        raise ValueError(f"open_beam must hold one count per spectral bin, with shape ({bins},); got shape {d.shape}")
    if (d == d[0]).all():
        return None
    return d / d.mean()


def _sample_mask(sample_mask, lead):
    """This function converts ``sample_mask`` to a flat boolean array over the pixels, in the C
    order of the leading axes.  It returns None when ``sample_mask`` is None."""
    if sample_mask is None:
        return None
    m = np.asarray(_to_host(sample_mask))
    if m.dtype != bool:
        if not np.isin(m, (0, 1)).all():
            raise ValueError("sample_mask must be boolean (or hold only 0 and 1)")
        m = m.astype(bool)
    try:
        m = np.broadcast_to(m, tuple(lead))
    except ValueError:
        raise ValueError(f"sample_mask of shape {m.shape} does not broadcast to the leading axes "
                         f"{tuple(lead)}") from None
    if not m.any():
        raise ValueError("sample_mask is False everywhere: there are no pixels to fit")
    return np.ascontiguousarray(m).reshape(-1)


def _check_basis(spectra, bins, num_materials):
    """This function checks the given ``spectra`` against the data and returns it as a float32 array
    of shape (rank, bins)."""
    H = np.asarray(_to_host(spectra), dtype=np.float32)
    if H.ndim != 2 or H.shape[1] != bins:
        raise ValueError(f"spectra must have shape (num_materials, {bins}) for data with {bins} bins; got {H.shape}")
    if not np.isfinite(H).all() or (H < 0).any():
        raise ValueError("spectra must be finite and nonnegative")
    if num_materials is not None and int(num_materials) != H.shape[0]:
        raise ValueError(f"num_materials={num_materials} disagrees with the {H.shape[0]} rows of spectra")
    return H


def hyper_denoise(data, dataset_type="attenuation", num_materials=None, **kwargs):
    """This function denoises hyperspectral data.  It calls ``dehydrate`` and then ``rehydrate``.
    The spectral axis of the data is always the last axis.

    Args:
        data (numpy.ndarray or torch.Tensor): The data.  The last axis is the spectral axis.
        dataset_type (str, optional): 'attenuation' or 'transmission'. Defaults to 'attenuation'.
        num_materials (int, optional): The rank.  Defaults to None, in which case
            ``dehydrate`` estimates the rank.
        **kwargs: The keyword arguments of :func:`~mbirtorch.hsnt.dehydrate`.

    Returns:
        numpy.ndarray: The denoised float32 data, with the shape and dataset_type of the input.
    """
    _reject_unknown_keywords("hyper_denoise", {k: v for k, v in kwargs.items() if k in _L2_KEYWORDS})
    return rehydrate(dehydrate(data, dataset_input_type=dataset_type, num_materials=num_materials, **kwargs),
                     dataset_output_type=dataset_type)


def rehydrate(dehydrated, dataset_output_type="attenuation", hyperspectral_idx=None):
    """This function restores hyperspectral data from the output of ``dehydrate``.  It multiplies
    the maps by the spectra, for all bins or for selected bins.  The spectral axis of the
    returned data is always the last axis.  The method is described in
    M. S. N. Chowdhury et al., "Fast Hyperspectral Neutron Tomography," IEEE TCI, vol. 11,
    pp. 663-677, 2025, doi:10.1109/TCI.2025.3567854.

    Args:
        dehydrated: The list [maps, spectra] that ``dehydrate`` returns.  maps holds ``W``,
            with the materials on the last axis.  spectra holds ``H``,
            with shape :math:`(N_m, N_k)`.
        dataset_output_type (str, optional): The quantity to return.  'attenuation' returns ``W H``.
            'transmission' returns ``exp(-W H)``.  Defaults to 'attenuation'.
        hyperspectral_idx: The indices of the bins to restore.  Defaults to None, which
            restores all bins.

    Returns:
        numpy.ndarray: The restored data.  Its last axis holds the bins, and its other axes match
            maps.
    """
    if dataset_output_type not in ("attenuation", "transmission"):
        raise ValueError("'dataset_output_type' must be either 'attenuation' or 'transmission'.")
    maps, spectra = dehydrated[0], dehydrated[1]

    if hyperspectral_idx is None:
        rehydrated_data = maps @ spectra
    else:
        rehydrated_data = maps @ spectra[:, hyperspectral_idx]

    if dataset_output_type == 'transmission':
        rehydrated_data = np.exp(-rehydrated_data)

    return rehydrated_data

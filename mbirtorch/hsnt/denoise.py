"""This module compresses and restores hyperspectral neutron data with a factorization X = W H.
The function ``dehydrate`` computes the factors, and the function ``rehydrate`` multiplies them."""
import warnings

import numpy as np

from ..utilities import _to_host

# These keywords belong to the MBIRJAX version of dehydrate, which used scikit-learn.  mbirtorch rejects them.
_L2_KEYWORDS = ("safety_factor", "beta_loss", "max_iter", "tolerance", "batch_size", "random_state")
# check_mask flags an outside pixel when twice its gain in log likelihood exceeds this chi-square quantile.
_MASK_CHECK_LEVEL = 0.999
# check_mask warns when the share of flagged outside pixels exceeds this value.
_MASK_CHECK_WARN = 0.01


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


def dehydrate(data, dataset_type="attenuation", num_materials=None, *, subspace_basis=None, spectra="mle", dose=None,
              penalty="auto", max_steps=1000, rel_tol=1e-8, max_rank=6, device=None,
              compile_mode="auto", mode="auto", chunk_pixels=None, max_passes=5, sample_mask=None, check_mask=False,
              verbose=1, **kwargs):
    """This function factors the attenuation of a hyperspectral dataset as X = W H by maximum
    likelihood.  The factors satisfy W >= 0 and H >= 0.

    The spectral axis of the data is the last axis.  The maps ``W`` keep the leading axes of the
    data.  This function handles zero counts correctly.  The components form a nonnegative basis
    of the data, but they are not necessarily the spectra of pure materials.

    Args:
        data (numpy.ndarray or torch.Tensor): The data.  The last axis is the spectral axis,
            of length :math:`N_k`.  This function treats a NaN or infinite transmission as zero
            counts.  It clips a negative transmission to zero.
        dataset_type (str, optional): 'attenuation' or 'transmission'. Defaults to 'attenuation'.
        num_materials (int, optional): The rank :math:`N_m`.  Defaults to None, in which case
            this function estimates the rank.
        subspace_basis (numpy.ndarray or torch.Tensor, optional): Spectra of shape
            :math:`(N_m, N_k)`.  When it is given, this function holds the spectra fixed and
            computes only the maps.  Defaults to None.
        spectra (str, optional): The method for estimating the spectra.  'mle' gives the
            maximum likelihood estimate.  'unconstrained' removes the bias that the constraint
            W >= 0 causes at low dose.  'support' selects which components are present in each
            pixel, and it requires ``dose``.  Defaults to 'mle'.
        dose (float, optional): The open beam counts per pixel and bin.  Only 'support' uses
            it.  Defaults to None.
        penalty (str or float, optional): The penalty per component that 'support' uses.  It
            is either a multiple of log(N_k) or 'auto'.  Defaults to 'auto'.
        max_steps (int, optional): The maximum number of solver iterations.  Defaults to 1000.
        rel_tol (float, optional): The solver stops when the relative change in the loss falls
            below this value.  Defaults to 1e-8.
        max_rank (int, optional): The largest rank that the rank estimate considers.  Defaults
            to 6.
        device (str, optional): The torch device.  Defaults to None, which uses CUDA when it is
            available and the CPU otherwise.
        compile_mode (str, optional): 'auto', 'on', or 'off'.  The value sets whether this
            function uses torch.compile.  Defaults to 'auto'.
        mode (str, optional): 'full', 'stream', or 'auto'.  'stream' processes the pixels in
            chunks.  Defaults to 'auto'.
        chunk_pixels (int, optional): The number of pixels in each chunk when streaming.
            Defaults to None, in which case this function sets it from the available memory.
        max_passes (int, optional): The number of passes over the chunks when streaming.
            Defaults to 5.
        sample_mask (numpy.ndarray or torch.Tensor, optional): A boolean array that is True
            where the sample may be.  It must broadcast to the leading axes of the data.  An
            image of shape (rows, cols) applies to every view, and an array of shape (views,
            rows, cols) gives each view its own region.  This function sets W to zero outside
            the mask and fits only the pixels inside it.  All views share the spectra.  The rank
            estimate uses every pixel.  Defaults to None.
        check_mask (bool, optional): When it is True and ``sample_mask`` is given, this
            function also fits the pixels outside the mask, and it warns when more of them hold
            attenuation than noise explains.  Defaults to False.
        verbose (int, optional): 0 prints nothing, 1 prints a summary, and 2 also prints the
            rank search.  Defaults to 1.

    Returns:
        list: The list [subspace_data, subspace_basis, dataset_type].  subspace_data is ``W``.
            Its axes are the leading axes of the data, followed by an axis of length :math:`N_m`.
            subspace_basis is ``H``, with shape :math:`(N_m, N_k)`.  Both are float32.
            dataset_type is the dataset_type of the input.
    """
    import torch
    from ._device import _default_device
    from ._fit import _fit
    from .rank import _estimate_rank
    _reject_unknown_keywords("dehydrate", kwargs)
    if mode not in ("auto", "full", "stream"):
        raise ValueError(f"mode must be 'auto', 'full' or 'stream', got {mode!r}")
    if chunk_pixels is not None and int(chunk_pixels) < 1:
        raise ValueError(f"chunk_pixels must be a positive number of pixels, got {chunk_pixels}")
    if int(max_passes) < 0:
        raise ValueError(f"max_passes must be at least 0, got {max_passes}")
    T_all, shape = _to_transmission(data, dataset_type)
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
    if subspace_basis is not None:
        from ._fit import _fit_fixed_basis
        H = _check_basis(subspace_basis, shape[-1], num_materials, spectra)
        W, H, rep = _fit_fixed_basis(T, H, device=device, mode=mode, chunk_pixels=chunk_pixels, max_steps=max_steps,
                                     rel_tol=rel_tol, compile_mode=compile_mode)
        if verbose >= 1:
            print("dehydrate(): ")
            print("   -Spectral dimension: ", shape[-1], " -> rank: ", H.shape[0], "(the given subspace_basis)")
            print("   -Pixels: ", pixels, f"; maps for the given basis ({rep['chunks']} chunk(s))")
    else:
        note = f"rank {num_materials} given"
        if num_materials is None:
            num_materials, note, _ = _estimate_rank(T_all, spatial_shape=_spatial_shape(lead), device=device,
                                                    max_rank=max_rank, verbose=max(0, verbose - 1))
        W, H, rep = _fit(T, int(num_materials), spectra=spectra, dose=dose, penalty=penalty, device=device, mode=mode,
                         chunk_pixels=chunk_pixels, max_steps=max_steps, rel_tol=rel_tol, max_passes=int(max_passes),
                         compile_mode=compile_mode)
        if verbose >= 1:
            solve = f"{rep['steps']} steps" if "steps" in rep else f"streamed, {rep['passes']} polish passes"
            print("dehydrate(): ")
            print("   -Spectral dimension: ", shape[-1], " -> rank: ", int(num_materials), f"({note})")
            print("   -Pixels: ", pixels, f"; spectra: {spectra} ({solve})")
    if mask is not None:
        if check_mask and not mask.all():
            _check_outside(_to_host(T), W, np.ascontiguousarray(_to_host(T_all)[~mask]), H, dose, device, mode,
                           chunk_pixels, max_steps, rel_tol, compile_mode, verbose)
        W_all = np.zeros((T_all.shape[0], W.shape[1]), dtype=np.float32)
        W_all[mask] = W
        W = W_all
    return [W.reshape(*lead, H.shape[0]), H, dataset_type]


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


def _check_outside(T_in, W_in, T_out, H, dose, device, mode, chunk_pixels, max_steps, rel_tol, compile_mode, verbose):
    """This function checks whether the pixels outside ``sample_mask`` hold attenuation.

    It fits each outside pixel to the spectra H and computes the gain in log likelihood over
    W = 0, in counts.  It uses ``dose`` when it is given.  Otherwise it uses the effective dose of
    the fit inside the mask, as the rank estimate does.  When the mask is correct, twice the gain
    is at most a chi-square variable with rank degrees of freedom.  This function flags the pixels
    above the _MASK_CHECK_LEVEL quantile of that variable, and it warns when the flagged share
    exceeds _MASK_CHECK_WARN.  It returns the flagged share."""
    import torch
    from scipy.stats import chi2
    from ._fit import _fit_fixed_basis
    from ._loss import _nnal_prep, _nnal_rowwise
    from .outputs import _CHUNK_ELEMENTS
    from .rank import _loss_and_residual
    if dose is None:
        resid = _loss_and_residual(torch.from_numpy(T_in), torch.from_numpy(W_in), torch.from_numpy(H))[1]
        dose = 1.0 / max(resid, np.finfo(np.float64).tiny)
    W_out, _, _ = _fit_fixed_basis(T_out, H, device=device, mode=mode, chunk_pixels=chunk_pixels, max_steps=max_steps,
                                   rel_tol=rel_tol, compile_mode=compile_mode)
    R, K = H.shape
    Hd = torch.as_tensor(H, device=device)
    rows = max(1, _CHUNK_ELEMENTS // K)
    gains = []
    for i in range(0, T_out.shape[0], rows):
        Tc = torch.from_numpy(T_out[i:i + rows]).to(device)
        prep = _nnal_prep(Tc)
        X = torch.from_numpy(W_out[i:i + rows]).to(device) @ Hd
        gains.append((_nnal_rowwise(torch.zeros_like(X), Tc, prep, 1, dtype=torch.float64)
                      - _nnal_rowwise(X, Tc, prep, 1, dtype=torch.float64)).cpu())
    gain = float(dose) * torch.cat(gains).numpy()
    share = float(np.mean(2 * gain > chi2.ppf(_MASK_CHECK_LEVEL, R)))
    chance = 100 * (1 - _MASK_CHECK_LEVEL)
    if verbose >= 1:
        print(f"   -Outside the sample_mask: {T_out.shape[0]} pixels.  {100 * share:.3g}% hold more attenuation "
              f"than noise explains, and {chance:.3g}% would by chance.")
    if share > _MASK_CHECK_WARN:
        warnings.warn(f"{100 * share:.3g}% of the pixels outside the sample_mask hold more attenuation than noise "
                      f"explains, and {chance:.3g}% would by chance.  The sample may extend beyond the mask.")
    return share


def _check_basis(subspace_basis, bins, num_materials, spectra):
    """This function checks ``subspace_basis`` against the data and returns it as a float32 array
    of shape (rank, bins)."""
    H = np.asarray(_to_host(subspace_basis), dtype=np.float32)
    if H.ndim != 2 or H.shape[1] != bins:
        raise ValueError(f"subspace_basis must have shape (rank, {bins}) for data with {bins} bins; got {H.shape}")
    if not np.isfinite(H).all() or (H < 0).any():
        raise ValueError("subspace_basis must be finite and nonnegative")
    if num_materials is not None and int(num_materials) != H.shape[0]:
        raise ValueError(f"num_materials={num_materials} disagrees with the {H.shape[0]} rows of subspace_basis")
    if spectra != "mle":
        raise ValueError(f"spectra={spectra!r} re-estimates the spectra, which a given subspace_basis holds fixed")
    return H


def hyper_denoise(data, dataset_type="attenuation", num_materials=None, **kwargs):
    """This function denoises hyperspectral data.  It calls ``dehydrate`` and then ``rehydrate``.

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
    return rehydrate(dehydrate(data, dataset_type=dataset_type, num_materials=num_materials, **kwargs))


def rehydrate(dehydrated_data, hyperspectral_idx=None):
    """This function restores hyperspectral data from the output of ``dehydrate``.  It multiplies
    the maps by the spectra, for all bins or for selected bins.  The method is described in
    M. S. N. Chowdhury et al., "Fast Hyperspectral Neutron Tomography," IEEE TCI, vol. 11,
    pp. 663-677, 2025, doi:10.1109/TCI.2025.3567854.

    Args:
        dehydrated_data: The list [subspace_data, subspace_basis, dataset_type].  subspace_data
            holds the maps, with the component axis last.  subspace_basis holds the spectra,
            with shape :math:`(N_s, N_k)`.  dataset_type is 'attenuation' or 'transmission'.
        hyperspectral_idx: The indices of the bins to restore.  Defaults to None, which
            restores all bins.

    Returns:
        numpy.ndarray: The restored data.  Its last axis holds the bins, and its other axes match
            subspace_data.  The values are attenuation or transmission, as dataset_type says.
    """
    [subspace_data, subspace_basis, dataset_type] = dehydrated_data

    if hyperspectral_idx is None:
        rehydrated_data = subspace_data @ subspace_basis
    else:
        rehydrated_data = subspace_data @ subspace_basis[:, hyperspectral_idx]

    if dataset_type == 'transmission':
        rehydrated_data = np.exp(-rehydrated_data)

    return rehydrated_data

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
    array and the shape of the input."""
    if dataset_type not in ("attenuation", "transmission"):
        raise ValueError("'dataset_type' must be either 'attenuation' or 'transmission'.")
    a = _to_host(data)
    if a.ndim < 2:
        raise ValueError(f"data must have at least one leading axis and a spectral last axis; got shape {a.shape}")
    shape = a.shape
    a = a.reshape(-1, shape[-1]).astype(np.float32, copy=False)
    if dataset_type == "attenuation":
        with np.errstate(over="ignore", invalid="ignore"):
            T = np.exp(-a)
    else:
        T = a.copy()
    _warn_cleaning(_clean_transmission(T))
    if not bool((T > 0).any()):
        raise ValueError("the data hold no counts: the transmission is zero everywhere")
    return np.ascontiguousarray(T, dtype=np.float32), shape


def dehydrate(data, dataset_type="attenuation", num_materials=None, *, subspace_basis=None, spectra="mle", dose=None,
              penalty="auto", max_steps=1000, rel_tol=1e-8, quick=False, init=None, fix_spectra=False, max_rank=6,
              device=None,
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
        subspace_basis (numpy.ndarray or torch.Tensor, optional): This argument is deprecated.
            Use init=(None, H) and fix_spectra=True instead.  Spectra of shape
            :math:`(N_m, N_k)`.  When it is given, this function holds the spectra fixed and
            computes only the maps.  Defaults to None.
        spectra (str, optional): The method for estimating the spectra.  'mle' gives the
            maximum likelihood estimate.  'unconstrained' removes the bias that the constraint
            W >= 0 causes at low dose.  'support' selects which components are present in each
            pixel, and it requires ``dose``.  Defaults to 'mle'.
        dose (float or numpy.ndarray, optional): The open beam counts per pixel and bin.  A
            scalar is used only by 'support'.  A 1-D array gives the open beam count of each
            bin, with length :math:`N_k`.  Use an array when the open beam differs between bins.
            Every fit then weights each bin by its open beam count, so that the loss is the
            Poisson likelihood of the counts, and 'support' uses the mean of the array as its
            dose.  Defaults to None.
        penalty (str or float, optional): The penalty per component that 'support' uses.  It
            is either a multiple of log(N_k) or 'auto'.  Defaults to 'auto'.
        max_steps (int, optional): The maximum number of solver iterations.  Defaults to 1000.
        rel_tol (float, optional): The solver stops when the relative change in the loss falls
            below this value.  Defaults to 1e-8.
        quick (bool, optional): If True, the solver also stops when its next step is small
            compared with the noise in the data.  This is often several times faster at high
            counts, and the result is usually within a few tenths of a dB of the full solve.  At
            low counts the maps can stop short, and this function then warns.  To refine a quick
            result, call this function again with ``init`` and quick=False.  Defaults to False.
        init (tuple, optional): Starting factors (W, H), for example the first two outputs of
            an earlier call on the same data.  Either factor may be None.  W has the leading axes
            of the data, or one axis of pixels, followed by an axis of length :math:`N_m`.  H has
            shape :math:`(N_m, N_k)`.  The solve continues from these factors, and the rank comes
            from them.  When the data are streamed, a given H replaces the first fit on a subset
            of the pixels.  Defaults to None.
        fix_spectra (bool, optional): If True, this function holds the spectra H of ``init``
            fixed and computes only the maps.  For example, a large scan can then be dehydrated
            piece by piece against spectra fitted on part of it.  It needs ``init`` with H, and
            ``spectra`` must be 'mle'.  The quick stop does not apply, because the maps of each
            pixel are solved separately.  Defaults to False.
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
    dose, weights = _dose_and_weights(dose, shape[-1])
    device = _default_device(device)
    lead = shape[:-1]
    mask = _sample_mask(sample_mask, lead)
    T = T_all if mask is None else np.ascontiguousarray(T_all[mask])
    pixels = f"{T.shape[0]}" if mask is None else f"{T.shape[0]} of {T_all.shape[0]} (sample_mask)"
    if subspace_basis is not None:
        warnings.warn("subspace_basis is deprecated and will be removed in a future release; use init=(None, H) "
                      "and fix_spectra=True.", DeprecationWarning, stacklevel=2)
        if init is not None:
            raise ValueError("init and subspace_basis cannot both be given")
        init, fix_spectra = (None, _check_basis(subspace_basis, shape[-1], num_materials, spectra)), True
    if init is not None:
        init, num_materials = _check_init(init, T_all.shape[0], shape[-1], num_materials, mask)
    if fix_spectra:
        if init is None or init[1] is None:
            raise ValueError("fix_spectra=True needs the spectra: give init=(None, H) or init=(W, H)")
        if spectra != "mle":
            raise ValueError(f"spectra={spectra!r} re-estimates the spectra, which fix_spectra=True holds fixed")
        from ._fit import _fit_fixed_basis
        W, H, rep = _fit_fixed_basis(T, init[1], device=device, mode=mode, chunk_pixels=chunk_pixels,
                                     max_steps=max_steps, rel_tol=rel_tol, compile_mode=compile_mode, W_init=init[0],
                                     weights=weights)
        if verbose >= 1:
            print("dehydrate(): ")
            print("   -Spectral dimension: ", shape[-1], " -> rank: ", H.shape[0], "(the given spectra, held fixed)")
            print("   -Pixels: ", pixels, f"; maps for the given basis ({rep['chunks']} chunk(s))")
    else:
        note = f"rank {num_materials} given"
        if num_materials is None:
            num_materials, note, _ = _estimate_rank(T_all, spatial_shape=_spatial_shape(lead), device=device,
                                                    max_rank=max_rank, verbose=max(0, verbose - 1),
                                                    weights=weights)
        W, H, rep = _fit(T, int(num_materials), spectra=spectra, dose=dose, penalty=penalty, device=device, mode=mode,
                         chunk_pixels=chunk_pixels, max_steps=max_steps, rel_tol=rel_tol, max_passes=int(max_passes),
                         compile_mode=compile_mode, init=init, quick=bool(quick), weights=weights)
        if verbose >= 1:
            solve = f"{rep['steps']} steps" if "steps" in rep else f"streamed, {rep['passes']} polish passes"
            if rep.get("quick_stop"):
                solve += ", quick stop"
            print("dehydrate(): ")
            print("   -Spectral dimension: ", shape[-1], " -> rank: ", int(num_materials), f"({note})")
            print("   -Pixels: ", pixels, f"; spectra: {spectra} ({solve})")
    if mask is not None:
        if check_mask and not mask.all():
            _check_outside(T, W, np.ascontiguousarray(T_all[~mask]), H, dose, device, mode, chunk_pixels, max_steps,
                           rel_tol, compile_mode, verbose, weights)
        W_all = np.zeros((T_all.shape[0], W.shape[1]), dtype=np.float32)
        W_all[mask] = W
        W = W_all
    return [W.reshape(*lead, H.shape[0]), H, dataset_type]


def _dose_and_weights(dose, bins):
    """This function splits the ``dose`` of ``dehydrate`` into a scalar dose and the weight of
    each bin.  A scalar or None is returned unchanged with no weights.  An array of the open beam
    count of each bin gives its mean as the dose and the array divided by its mean as the weights."""
    if dose is None or np.ndim(_to_host(dose)) == 0:
        return dose, None
    d = np.asarray(_to_host(dose), dtype=np.float64)
    if d.ndim != 1 or d.shape[0] != bins:
        raise ValueError(f"an array dose must hold one open beam count per bin, with shape ({bins},); got shape "
                         f"{d.shape}")
    if not (np.isfinite(d).all() and (d > 0).all()):
        raise ValueError("an array dose must be finite and positive in every bin")
    mean = float(d.mean())
    return mean, d / mean


def _check_init(init, pixels, bins, num_materials, mask):
    """This function checks the starting factors (W, H) of ``dehydrate``.  Either factor may be
    None.  It returns the factors as float32 numpy arrays, with W restricted to the pixels in
    ``mask``, and the rank."""
    try:
        W, H = init
    except (TypeError, ValueError):
        raise ValueError("init must be a pair (W, H)") from None
    if W is None and H is None:
        raise ValueError("init must give W, H, or both")
    W = None if W is None else np.asarray(_to_host(W), dtype=np.float32)
    H = None if H is None else np.asarray(_to_host(H), dtype=np.float32)
    if H is not None and (H.ndim != 2 or H.shape[1] != bins):
        raise ValueError(f"init H must have shape (num_materials, {bins}), got {H.shape}")
    rank = H.shape[0] if H is not None else W.shape[-1]
    if num_materials is not None and int(num_materials) != rank:
        raise ValueError(f"num_materials is {num_materials}, but init has {rank} components")
    if W is not None:
        if W.shape[-1] != rank or W.size != pixels * rank:
            raise ValueError(f"init W must hold {pixels} pixels of {rank} components, got shape {W.shape}")
        W = W.reshape(pixels, rank)
        if mask is not None:
            W = W[mask]
        W = np.ascontiguousarray(W)
    for f in (W, H):
        if f is not None and ((f < 0).any() or not np.isfinite(f).all()):
            raise ValueError("init W and H must be finite and nonnegative")
    return (W, None if H is None else np.ascontiguousarray(H)), rank


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


def _check_outside(T_in, W_in, T_out, H, dose, device, mode, chunk_pixels, max_steps, rel_tol, compile_mode, verbose,
                   weights=None):
    """This function checks whether the pixels outside ``sample_mask`` hold attenuation.

    It fits each outside pixel to the spectra H and computes the gain in log likelihood over
    W = 0, in counts.  It uses ``dose`` when it is given.  Otherwise it uses the effective dose of
    the fit inside the mask, as the rank estimate does.  When the mask is correct, twice the gain
    is at most a chi-square variable with rank degrees of freedom.  This function flags the pixels
    above the _MASK_CHECK_LEVEL quantile of that variable, and it warns when the flagged share
    exceeds _MASK_CHECK_WARN.  The argument ``weights`` holds the weight of each bin, as in
    ``dehydrate``.  It returns the flagged share."""
    import torch
    from scipy.stats import chi2
    from ._fit import _fit_fixed_basis
    from ._loss import _nnal_prep, _nnal_rowwise, _weights_tensor
    from .outputs import _CHUNK_ELEMENTS
    from .rank import _loss_and_residual
    if dose is None:
        resid = _loss_and_residual(torch.from_numpy(T_in), torch.from_numpy(W_in), torch.from_numpy(H))[1]
        dose = 1.0 / max(resid, np.finfo(np.float64).tiny)
    W_out, _, _ = _fit_fixed_basis(T_out, H, device=device, mode=mode, chunk_pixels=chunk_pixels, max_steps=max_steps,
                                   rel_tol=rel_tol, compile_mode=compile_mode, weights=weights)
    R, K = H.shape
    Hd = torch.as_tensor(H, device=device)
    wd = _weights_tensor(weights, device)
    rows = max(1, _CHUNK_ELEMENTS // K)
    gains = []
    for i in range(0, T_out.shape[0], rows):
        Tc = torch.from_numpy(T_out[i:i + rows]).to(device)
        prep = _nnal_prep(Tc, wd)
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

import warnings

import numpy as np
from sklearn.decomposition import non_negative_factorization as nmf
from sklearn.utils.extmath import randomized_svd


# -----------------------------------------------------------------------
# Hyperspectral Neutron Radiographic/Tomographic Data Denoising Functions
# -----------------------------------------------------------------------


def hyper_denoise(data, dataset_type='attenuation', num_materials=None, safety_factor=2, beta_loss='frobenius',
                  max_iter=300, tolerance=1e-10, batch_size=2 ** 27, subspace_basis=None, random_state=None,
                  verbose=1):
    """
    Denoise a hyperspectral dataset using dehydration and rehydration as described in:

    M. S. N. Chowdhury, D. Yang, S. Tang, S. V. Venkatakrishnan, H. Z. Bilheux, G. T. Buzzard, and C. A. Bouman, "Fast Hyperspectral Neutron Tomography," IEEE Transactions on Computational Imaging, vol. 11, pp. 663–677, 2025. doi:10.1109/TCI.2025.3567854

    The function works for any rank array. However, the spectral axis must be the last axis.

    Args:
        data: Hyperspectral data array with arbitrary axes and a spectral axis in the last position.
        dataset_type: 'attenuation' or 'transmission' where attenuation = -log(transmission). Defaults to 'attenuation'.
        num_materials: Number of materials in the sample. If None, the number is estimated automatically from
            the data. Defaults to None.
        safety_factor: A multiplier (≥ 1) applied to the number of materials to set the subspace dimension.
            Defaults to 2.
        beta_loss: Beta divergence minimized in NMF. Can be 'frobenius' or 'kullback-leibler'. Defaults to 'frobenius'.
        max_iter: Maximum iterations for the NMF solver. Defaults to 300.
        tolerance: Convergence tolerance for the NMF solver. Defaults to 1e-10.
        batch_size: Size of data processed per batch. Useful for large datasets to limit memory usage. Defaults to 2^27.
        subspace_basis: Pre-computed subspace basis spectra of shape :math:`(N_s, N_k)`. If None, the basis spectra are
            estimated directly from the data. Defaults to None.
        random_state: Random seed for reproducibility of the NMF initialization and batch row sampling. If None,
            the factors vary from run to run. Defaults to None.
        verbose: Verbosity level. If 0, prints nothing; if 1, prints details; if >1, also generates plots. Defaults to 1.

    Returns:
        Denoised hyperspectral data with the same shape as the input data.

    Example:
        >>> denoised_data = hyper_denoise(data, num_materials=5, safety_factor=2)
        >>> data.shape, denoised_data.shape
        ((N_x, N_y, N_z, ..., N_k), (N_x, N_y, N_z, ..., N_k))

    """
    # --------------------- Dehydrate ----------------------
    dehydrated_data = dehydrate(data,
                                dataset_type=dataset_type,
                                num_materials=num_materials,
                                safety_factor=safety_factor,
                                beta_loss=beta_loss,
                                max_iter=max_iter,
                                tolerance=tolerance,
                                batch_size=batch_size,
                                subspace_basis=subspace_basis,
                                random_state=random_state,
                                verbose=verbose)

    # --------------------- Rehydrate ----------------------
    denoised_data = rehydrate(dehydrated_data)

    return denoised_data


def dehydrate(data, dataset_type='attenuation', num_materials=None, safety_factor=2, beta_loss='frobenius',
              max_iter=300, tolerance=1e-10, batch_size=2 ** 27, subspace_basis=None, random_state=None,
              verbose=1):
    """
    Dehydrate/compress a hyperspectral dataset onto a low-dimensional subspace as described in:

    M. S. N. Chowdhury, D. Yang, S. Tang, S. V. Venkatakrishnan, H. Z. Bilheux, G. T. Buzzard, and C. A. Bouman, "Fast Hyperspectral Neutron Tomography," IEEE Transactions on Computational Imaging, vol. 11, pp. 663–677, 2025. doi:10.1109/TCI.2025.3567854

    The function works for any rank array. However, the spectral axis must be the last axis.

    Args:
        data: Hyperspectral data array with arbitrary axes and a spectral axis of length :math:`N_k` in the last position.
        dataset_type: 'attenuation' or 'transmission' where attenuation = -log(transmission). Defaults to 'attenuation'.
        num_materials: Number of materials in the sample :math:`N_m`. If None, the number is estimated automatically from
            the data. Defaults to None.
        safety_factor: A multiplier (≥ 1) applied to the number of materials to set the subspace dimension :math:`N_s`.
            Defaults to 2.
        beta_loss: Beta divergence minimized in NMF. Can be 'frobenius' or 'kullback-leibler'. Defaults to 'frobenius'.
        max_iter: Maximum iterations for the NMF solver. Defaults to 300.
        tolerance: Convergence tolerance for the NMF solver. Defaults to 1e-10.
        batch_size: Size of data processed per batch. Useful for large datasets to limit memory usage. Defaults to 2^27.
        subspace_basis: Pre-computed subspace basis spectra of shape :math:`(N_s, N_k)`. If None, the basis spectra are
            estimated directly from the data. Defaults to None.
        random_state: Random seed for reproducibility of the NMF initialization and batch row sampling. The NMF
            factorization is not unique, so with the default of None the returned factors vary from run to run even
            though their product is stable; pass an int to make a run reproducible. Defaults to None.
        verbose: Verbosity level. If 0, prints nothing; if 1, prints details; if >1, also generates plots. Defaults to 1.

    Returns:
        A list containing the dehydrated hyperspectral dataset in the form [subspace_data, subspace_basis, dataset_type].
            - subspace_data: ndarray with same shape as input data except the last axis length is :math:`N_s`.
            - subspace_basis: ndarray of shape :math:`(N_s, N_k)`, where rows are subspace basis spectra.
            - dataset_type: Can be 'attenuation' or 'transmission' where attenuation = -log(transmission).

    Example:
        >>> [subspace_data, subspace_basis, dataset_type] = dehydrate(data, num_materials=5, safety_factor=2)
        >>> data.shape, subspace_data.shape, subspace_basis.shape
        ((N_x, N_y, N_z, ..., N_k), (N_x, N_y, N_z, ..., 10), (10, N_k))
    """
    epsilon = 1e-3  # Define epsilon

    # --------------- Dataset type validation --------------
    if dataset_type not in ('attenuation', 'transmission'):
        raise ValueError("'dataset_type' must be either 'attenuation' or 'transmission'.")

    # ------------------ Data preparation ------------------
    data_shape = data.shape
    num_bands = data_shape[-1]
    num_points = data.size // num_bands
    data = data.reshape(num_points, num_bands).astype(np.float64)  # Reshape to 2D and cast to float64 for stability

    if dataset_type == 'transmission':
        # Initial cleanup in the transmission domain to get rid of defective measurements
        data = hyper_denoise(data,
                             dataset_type='attenuation',
                             num_materials=num_materials,
                             safety_factor=safety_factor * 3,
                             beta_loss=beta_loss,
                             max_iter=max_iter,
                             tolerance=tolerance,
                             batch_size=batch_size,
                             random_state=random_state,
                             verbose=0)
        data[data < epsilon] = epsilon
        data = - np.log(data)  # Convert to attenuation

    data[data < 0] = 0  # Enforce non-negativity

    if subspace_basis is not None:
        subspace_basis = np.asarray(subspace_basis, dtype=np.float64)  # Cast to float64 for stability

    # --------------------- Batch setup ---------------------
    num_points_batch = max(1, batch_size // num_bands)  # Number of hyperspectral points per batch
    num_batches = int(np.ceil(num_points / num_points_batch))  # Number of batches

    # ------------------- NMF solver setup ------------------
    if beta_loss == 'frobenius':
        solver = 'cd'  # Coordinate Descent
    elif beta_loss == 'kullback-leibler':
        solver = 'mu'  # Multiplicative Update
    else:
        warnings.warn(f"Invalid beta_loss '{beta_loss}' specified: falling back to 'frobenius'.")
        beta_loss = 'frobenius'
        solver = 'cd'

    # ------------- Subspace dimension setup -----------------
    if subspace_basis is not None:
        subspace_dimension = subspace_basis.shape[0]
    elif num_materials is not None:
        subspace_dimension = int(np.ceil(safety_factor * num_materials))
    else:
        subspace_dimension = _estimate_subspace_dimension(data, safety_factor=safety_factor,
                                                          random_state=random_state, verbose=verbose)

    # ------- Subspace basis estimation for multi-batch ------
    if subspace_basis is None and num_batches > 1:
        row_idx = np.random.default_rng(random_state).permutation(num_points)
        subspace_basis_batch = [None] * num_batches

        # Estimate subspace basis for each batch using NMF
        for batch in range(num_batches):
            b_start = batch * num_points_batch
            b_stop = min((batch + 1) * num_points_batch, num_points)
            batch_data = data[row_idx[b_start: b_stop]]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, subspace_basis_batch[batch], _ = nmf(batch_data,
                                                        n_components=subspace_dimension,
                                                        init='nndsvd',
                                                        beta_loss=beta_loss,
                                                        solver=solver,
                                                        tol=tolerance,
                                                        max_iter=max(50, max_iter // num_batches),
                                                        random_state=random_state,
                                                        update_H=True)

        # Estimate final subspace basis from batch estimations using NMF
        subspace_basis_batch = np.reshape(np.array(subspace_basis_batch), (-1, num_bands))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, subspace_basis, _ = nmf(subspace_basis_batch,
                                       n_components=subspace_dimension,
                                       init='nndsvd',
                                       beta_loss=beta_loss,
                                       solver=solver,
                                       tol=tolerance,
                                       max_iter=max_iter,
                                       random_state=random_state)

    # --------------- Subspace data estimation ---------------
    if num_batches == 1:
        nmf_init, update_basis = 'nndsvd', True
    else:
        nmf_init, update_basis = 'custom', False

    # Estimate subspace data in batches using NMF
    subspace_data = np.zeros((num_points, subspace_dimension))
    for batch in range(num_batches):
        b_start = batch * num_points_batch
        b_stop = min((batch + 1) * num_points_batch, num_points)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            subspace_data[b_start: b_stop], subspace_basis, _ = nmf(data[b_start: b_stop],
                                                                    n_components=subspace_dimension,
                                                                    init=nmf_init,
                                                                    H=subspace_basis,
                                                                    beta_loss=beta_loss,
                                                                    solver=solver,
                                                                    tol=tolerance,
                                                                    max_iter=max_iter,
                                                                    random_state=random_state,
                                                                    update_H=update_basis)

    # ------------------ Final formatting -------------------
    subspace_data = subspace_data.reshape(*data_shape[:-1], -1)  # Reshape to original dimensions (except last axis)
    subspace_data = np.asarray(subspace_data, dtype=np.float32)  # Cast to float32 to reduce memory footprint
    subspace_basis = np.asarray(subspace_basis, dtype=np.float32)  # Cast to float32 to reduce memory footprint
    dehydrated_data = [subspace_data, subspace_basis, dataset_type]  # Package outputs for return

    # --------------- Print details if required -------------
    if verbose >= 1:
        print("dehydrate(): ")
        print("   -Number of data batches: ", num_batches)
        print("   -Original spectral dimension: ", data.shape[-1])
        print("   -Subspace dimension: ", subspace_data.shape[-1])

    return dehydrated_data


def rehydrate(dehydrated_data, hyperspectral_idx=None):
    """
    Rehydrate/decompress selected spectral bins from dehydrated hyperspectral data as described in:

    M. S. N. Chowdhury, D. Yang, S. Tang, S. V. Venkatakrishnan, H. Z. Bilheux, G. T. Buzzard, and C. A. Bouman, "Fast Hyperspectral Neutron Tomography," IEEE Transactions on Computational Imaging, vol. 11, pp. 663–677, 2025. doi:10.1109/TCI.2025.3567854

    Args:
        dehydrated_data: Dehydrated hyperspectral data in the form [subspace_data, subspace_basis, dataset_type]:

            - subspace_data: ndarray with arbitrary axes and a subspace axis of length :math:`N_s` in the last position.
            - subspace_basis: ndarray of shape :math:`(N_s, N_k)`, where rows are subspace basis spectra.
            - dataset_type: 'attenuation' or 'transmission' where attenuation = -log(transmission).
        hyperspectral_idx: A list of :math:`N_h` indices along the original spectral axis to rehydrate. If None, all :math:`N_k`
            spectral bins are rehydrated. Defaults to None.

    Returns:
        Rehydrated/decompressed hyperspectral data with the same shape as the input subspace_data except the last axis
        length is :math:`N_h (N_h <= N_k)`.

    Example:
        >>> hyper_data = rehydrate([subspace_data, subspace_basis, dataset_type], hyperspectral_idx=[5, 10, 15])
        >>> subspace_data.shape, subspace_basis.shape, hyper_data.shape
        ((N_x, N_y, N_z, ..., N_s), (N_s, N_k), (N_x, N_y, N_z, ..., 3))
    """
    [subspace_data, subspace_basis, dataset_type] = dehydrated_data  # Unpack data

    # Retrieve original data dimensions
    if hyperspectral_idx is None:
        rehydrated_data = subspace_data @ subspace_basis
    else:
        rehydrated_data = subspace_data @ subspace_basis[:, hyperspectral_idx]

    if dataset_type == 'transmission':
        rehydrated_data = np.exp(-rehydrated_data)  # Convert to transmission

    return rehydrated_data


def _estimate_subspace_dimension(data, safety_factor=2, noise_fit_window=[25.0, 75.0], threshold=1.5, random_state=None,
                                 verbose=1):
    """
    Estimate the signal subspace dimension using a log-linear fit to singular values.

    Args:
        data: 2D array of shape (num_samples, :math:`N_k`). Values should be real.
        safety_factor: Multiplicative factor ≥ 1 used to scale the initial estimate of subspace dimension and ensure
            safer final choice. Defaults to 2.
        noise_fit_window: Two-element list or tuple [start_percent, stop_percent] indicating the percentile window (0–100)
            over which the singular value fitting is performed. Defaults to [25.0, 75.0].
        threshold: Multiplicative factor to define the cutoff relative to the predicted singular values. Defaults to 1.5.
        random_state: Random seed for reproducibility of row sampling and SVD. Defaults to None.
        verbose: Verbosity level. If >1, plots singular values, fit, and threshold curves. Defaults to 1.

    Returns:
        Estimated dimension of the signal subspace (positive integer).
    """
    if data.ndim != 2:
        raise ValueError("`data` must be a 2D array shaped (samples, N_k).")

    n_points, n_bands = data.shape

    # Decide how many rows to sample for speed/robustness
    sample_size = min(n_points, n_bands)

    # Sample rows without replacement
    rng = np.random.default_rng(random_state)
    row_idx = rng.choice(n_points, size=sample_size, replace=False)

    # Cast to float64 for numerical stability in svd
    Y = np.asarray(data[row_idx, :], dtype=np.float64)

    # Compute singular values via randomized SVD
    _, s, _ = randomized_svd(Y, n_components=sample_size, random_state=random_state)

    # Guard against degenerate cases
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return 0

    # Extract start and stop percent from noise_fit_window
    start_percent, stop_percent = noise_fit_window
    # Fit window around percentile: [percentile-10, percentile+10], in s-index space
    start_idx = int(np.floor((start_percent / 100.0) * s.size))
    stop_idx = int(np.ceil((stop_percent / 100.0) * s.size))

    # Clip and ensure at least 2 points
    start_idx = max(0, min(start_idx, s.size - 2))
    stop_idx = max(start_idx + 2, min(stop_idx, s.size))

    # Fit log(s) ≈ a*n + b on [start_idx:stop_idx]
    n = np.arange(s.size)
    a, b = np.polyfit(n[start_idx:stop_idx], np.log(s[start_idx:stop_idx] + 1e-12), 1)

    # Predicted singular values for all indices
    s_pred = np.exp(a * n + b)

    # Compute tau by scaling the predicted singular values with the threshold
    tau = threshold * s_pred

    # Consider singular values > the corresponding tau values to be associated with signals
    signal_flag = s > tau
    num_materials = int(np.sum(signal_flag[:start_idx]))

    if verbose > 1:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.semilogy(s, label='s: actual singular values from data (signal + noise)')
        plt.semilogy(s_pred, label='s_pred: predicted singular values from noise model')
        plt.semilogy(tau, label='tau: noise and signal discriminator (threshold x s_pred)')
        plt.title("Modeling noise singular values for number of material estimation")
        plt.xlabel("singular value index")
        plt.ylabel("singular value")
        plt.legend()

    # Multiply by safety factor
    subspace_dimension = int(np.ceil(safety_factor * num_materials))

    return max(1, subspace_dimension)

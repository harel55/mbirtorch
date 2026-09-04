"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

This script demonstrates the use of dehydration and rehydration for hyperspectral data denoising.
A simulated hyperspectral neutron dataset containing three materials (Ni, Cu, and Al) is used for the purpose.
"""

import os
import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

import torch

from mbirtorch.hsnt import dehydrate, generate_hyper_data, nnal_factorization, compare_spectra, stable_nnal, stable_nnal_derivatives


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simulation parameters
    num_angles = 1  # Number of projection angles
    detector_rows = 64  # Number of rows in the detector
    detector_columns = 64  # Number of columns in the detector
    dosage_rate = 3  # Neutron dosage rate
    material_density = {"Ni": 0.25, "Cu": 0.25, "Al": 0.75}  # Define material density (vol. fraction)
    dataset_type = 'attenuation'  # Choose between 'attenuation' or 'transmission'

    # Denoiser parameters
    num_materials_fit = 3  # Number of materials in reconstructed subspace
    verbose = 0  # Verbosity level

    # Fix seed for random number generation
    np.random.seed(129)

    # Load theoretical linear attenuation coefficients for Ni, Cu, and Al
    material_basis_path = '/home/harel/mbirjax/experiments/hsnt/binaries/'
    filename = os.path.join(material_basis_path, 'material_basis.npy')
    material_basis = np.load(filename)
    num_materials_true = material_basis.shape[0]

    # Generate simulated noisy hyperspectral data and ground truth
    [noisy_hyper_projection, _, _] = generate_hyper_data(
        material_basis,
        num_angles=num_angles,
        detector_rows=detector_rows,
        detector_columns=detector_columns,
        dosage_rate=dosage_rate,
        material_density=material_density,
        noisy=True,
        verbose=verbose
    )
    # float32: the solvers need no more (float64 changed nothing in the precision study) and it is
    # several times faster on consumer GPUs, whose double-precision throughput is a small fraction of single.
    noisy_hyper_projection = torch.tensor(noisy_hyper_projection, dtype=torch.float32, device=device)
    noisy_hyper_projection = torch.nan_to_num(noisy_hyper_projection, nan=0.0, posinf=0.0, neginf=0.0)  # Replace any NaNs or infs with zeros
    T = torch.exp(-noisy_hyper_projection).reshape(-1, noisy_hyper_projection.shape[-1])
    print(f"Range of T: {torch.min(T).item():.2g} to {torch.max(T).item():.2g}")

    # Spoof simulated projection data which is not returned by generate_hyper_data
    height = detector_rows // 3
    width = detector_columns // 2
    thickness = 20 * np.sqrt((width//2)**2 - np.linspace(-(width // 2), width // 2, width)**2)/ width
    material_projection = np.zeros((num_angles, detector_rows, detector_columns, num_materials_true))
    material_projection[:, :height, width // 2:width + width // 2, 0] = material_density["Ni"] * thickness
    material_projection[:, 2 * height:, width // 2:width + width // 2, 1] = material_density["Cu"] * thickness
    material_projection[:, height:2 * height, width // 2:width + width // 2, 2] = material_density["Al"] * thickness
    material_projection = material_projection.reshape(-1, num_materials_true)

    # Perform hyperspectral denoising (dehydrate + rehydrate)
    print("Performing L2 factorization...")
    start_time = time.time()
    W, H, _ = dehydrate(noisy_hyper_projection.cpu().numpy(),
                        dataset_type=dataset_type,
                        num_materials=num_materials_fit,
                        safety_factor=1,
                        verbose=verbose)
    W = W.reshape(np.prod(noisy_hyper_projection.shape[:-1]), -1)
    H = H.reshape(-1, noisy_hyper_projection.shape[-1])
    print(f'L2 factorization completed in: {time.time() - start_time} seconds')

    ### Refine using nonnegative attenuation loss

    # Move everything to the same device
    material_projection = torch.tensor(material_projection, dtype=T.dtype, device=device)
    material_basis = torch.tensor(material_basis, dtype=T.dtype, device=device)
    W = torch.tensor(W, dtype=T.dtype, device=device)
    H = torch.tensor(H, dtype=T.dtype, device=device)

    kwargs = {
        'num_materials': num_materials_fit,
        'max_steps': 1000,
        'batch_size': None,
        'rel_tol': 1e-6,
        # compile_mode='default' compiles the hot kernels: a one-off cost of seconds that pays only for
        # repeated solves, so it is left off for this single pass per method.
    }

    # Perform hyperspectral denoising
    start_time = time.time()
    W_newt, H_newt, i_newt = nnal_factorization(
        T, method='joint_newton', **kwargs
    )
    print(f'Joint-Newton reconstruction completed in: {time.time() - start_time} seconds after {i_newt} iterations')
    start_time = time.time()
    W_mu, H_mu, i_mu = nnal_factorization(
        T, method='multiplicative', **kwargs
    )
    print(f'Multiplicative reconstruction completed in: {time.time() - start_time} seconds after {i_mu} iterations')
    start_time = time.time()
    W_blk, H_blk, i_blk = nnal_factorization(
        T, method='block_newton', **kwargs
    )
    print(f'Block-Newton reconstruction completed in: {time.time() - start_time} seconds after {i_blk} iterations')

    print(f"attenuation loss Scipy:\t\t{stable_nnal(W @ H, T).item()}")
    print(f"attenuation loss Joint Newton:\t{stable_nnal(W_newt @ H_newt, T).item()}")
    print(f"attenuation loss Multiplicative:\t{stable_nnal(W_mu @ H_mu, T).item()}")
    print(f"attenuation loss Block Newton:\t{stable_nnal(W_blk @ H_blk, T).item()}")
    print()
    print(f"L2 loss Scipy:\t\t{torch.linalg.norm(torch.log(T) + (W @ H)).item()}")
    print(f"L2 loss Joint Newton:\t{torch.linalg.norm(torch.log(T) + (W_newt @ H_newt)).item()}")
    print(f"L2 loss Multiplicative:\t{torch.linalg.norm(torch.log(T) + (W_mu @ H_mu)).item()}")
    print(f"L2 loss Block Newton:\t{torch.linalg.norm(torch.log(T) + (W_blk @ H_blk)).item()}")

    # Compute least squares estimate of material coefficients for current projections
    theta_frob = torch.linalg.lstsq(H.T, material_basis.T)[0].T
    theta_newt = torch.linalg.lstsq(H_newt.T, material_basis.T)[0].T
    theta_mu = torch.linalg.lstsq(H_mu.T, material_basis.T)[0].T
    theta_blk = torch.linalg.lstsq(H_blk.T, material_basis.T)[0].T

    # Move everything back to CPU for plotting
    material_projection = material_projection.cpu().numpy()
    material_basis = material_basis.cpu().numpy()
    W = W.cpu().numpy()
    H = H.cpu().numpy()
    T = T.cpu().numpy()
    W_newt = W_newt.cpu().numpy()
    H_newt = H_newt.cpu().numpy()
    W_mu = W_mu.cpu().numpy()
    H_mu = H_mu.cpu().numpy()
    W_blk = W_blk.cpu().numpy()
    H_blk = H_blk.cpu().numpy()
    theta_frob = theta_frob.cpu().numpy()
    theta_newt = theta_newt.cpu().numpy()
    theta_mu = theta_mu.cpu().numpy()
    theta_blk = theta_blk.cpu().numpy()

    # Plot reconstructed spectra
    compare_spectra(
        spectra_groups=[
            theta_frob @ H,
            theta_newt @ H_newt,
            theta_mu @ H_mu,
            theta_blk @ H_blk,
        ],
        ground_truth=material_basis,
        labels=['Ni', 'Cu', 'Al'],
        subtitles=[
            r'Scipy L$^2$ Loss',
            'Joint-Newton',
            'Multiplicative',
            'Block-Newton',
        ],
        title=f'Material attenuation spectra reconstructions',
        x_label='Wavelength index',
        y_label='Attenuation',
        y_lim=(0, 1.1),
        filename=f'example_1_spectra_reconstruction.png'
    )

    # Plot reconstructed material coefficient maps
    plt.figure(figsize=(18, 6))
    plt.suptitle(f'Material projection reconstructions')
    row_max = np.max(material_projection, axis=0).reshape(1, 1, num_materials_true)
    image_dims = (detector_rows, detector_columns, num_materials_true)
    for i, (image, title) in enumerate([
            (material_projection.reshape(image_dims) / row_max, 'Ground Truth'),
            ((W @ np.linalg.pinv(theta_frob)).reshape(image_dims) / row_max, 'Scipy L$^2$ Loss'),
            ((W_newt @ np.linalg.pinv(theta_newt)).reshape(image_dims) / row_max, 'Joint-Newton'),
            ((W_mu @ np.linalg.pinv(theta_mu)).reshape(image_dims) / row_max, 'Multiplicative'),
            ((W_blk @ np.linalg.pinv(theta_blk)).reshape(image_dims) / row_max, 'Block-Newton'),
        ]):
        ax = plt.subplot(1, 5, i + 1)
        ax.set_title(title)
        ax.imshow(image)
    plt.savefig(f'example_1_material_maps.png')

if __name__ == "__main__":
    main()

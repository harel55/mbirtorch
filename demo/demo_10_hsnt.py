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
        verbose=verbose
    )
    noisy_hyper_projection = torch.tensor(noisy_hyper_projection, dtype=torch.float32, device=device)
    noisy_hyper_projection = torch.nan_to_num(noisy_hyper_projection, nan=0.0, posinf=0.0, neginf=0.0)  # Replace any NaNs or infs with zeros
    T = torch.exp(-noisy_hyper_projection).reshape(-1, noisy_hyper_projection.shape[-1])
    print(f"Range of T: {torch.min(T).item():.2g} to {torch.max(T).item():.2g}")

    # Spoof simulated projection data which is not returned by generate_hyper_data
    height = detector_rows // 3
    width = detector_columns // 2
    thickness = 20 * np.sqrt((width//2)**2 - np.linspace(-width // 2, width // 2, width)**2)/ width
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
        'compile_mode': 'reduce-overhead',
    }

    # Perform hyperspectral denoising
    start_time = time.time()
    W_newt1, H_newt1, i_newt = nnal_factorization(
        T, method='quasi_newton', **kwargs, W_init=W.clone(), H_init=H.clone()
    )
    print(f'Newton reconstruction completed in: {time.time() - start_time} seconds after {i_newt} iterations')
    start_time = time.time()
    W_newt2, H_newt2, i_newt = nnal_factorization(
        T, method='quasi_newton', **kwargs, W_init=material_projection.clone(), H_init=material_basis.clone()
    )
    print(f'Newton reconstruction completed in: {time.time() - start_time} seconds after {i_newt} iterations')
    start_time = time.time()
    W_mu1, H_mu1, i_mu = nnal_factorization(
        T, method='mann_multiplicative', **kwargs, W_init=W.clone(), H_init=H.clone()
    )
    print(f'Multiplicative reconstruction completed in: {time.time() - start_time} seconds after {i_mu} iterations')
    start_time = time.time()
    W_mu2, H_mu2, i_mu = nnal_factorization(
        T, method='mann_multiplicative', **kwargs, W_init=material_projection.clone(), H_init=material_basis.clone()
    )
    print(f'Multiplicative reconstruction completed in: {time.time() - start_time} seconds after {i_mu} iterations')

    print(f"attenuation loss L2:\t\t\t{stable_nnal(W @ H, T).item()}")
    print(f"attenuation loss Newton (L2 loss init):\t{stable_nnal(W_newt1 @ H_newt1, T).item()}")
    print(f"attenuation loss Mann (L2 loss init):\t{stable_nnal(W_mu1 @ H_mu1, T).item()}")
    print(f"attenuation loss Newton (true init):\t{stable_nnal(W_newt2 @ H_newt2, T).item()}")
    print(f"attenuation loss Mann (true init):\t{stable_nnal(W_mu2 @ H_mu2, T).item()}")
    print(f"T-weighted L2 loss L2:\t\t\t\t{(T*((torch.log(T) + (W @ H))**2)).sum().item()}")
    print(f"T-weighted L2 loss Newton (L2 loss init):\t{(T*((torch.log(T) + (W_newt1 @ H_newt1))**2)).sum().item()}")
    print(f"T-weighted L2 loss Mann (L2 loss init):\t\t{(T*((torch.log(T) + (W_mu1 @ H_mu1))**2)).sum().item()}")
    print(f"T-weighted L2 loss Newton (true init):\t\t{(T*((torch.log(T) + (W_newt2 @ H_newt2))**2)).sum().item()}")
    print(f"T-weighted L2 loss Mann (true init):\t\t{(T*((torch.log(T) + (W_mu2 @ H_mu2))**2)).sum().item()}")
    print(f"L2 loss L2:\t\t\t{torch.linalg.norm(torch.log(T) + (W @ H)).item()}")
    print(f"L2 loss Newton (L2 loss init):\t{torch.linalg.norm(torch.log(T) + (W_newt1 @ H_newt1)).item()}")
    print(f"L2 loss Mann (L2 loss init):\t{torch.linalg.norm(torch.log(T) + (W_mu1 @ H_mu1)).item()}")
    print(f"L2 loss Newton (true init):\t{torch.linalg.norm(torch.log(T) + (W_newt2 @ H_newt2)).item()}")
    print(f"L2 loss Mann (true init):\t{torch.linalg.norm(torch.log(T) + (W_mu2 @ H_mu2)).item()}")

    # Compute least squares estimate of material coefficients for current projections
    theta_frob = torch.linalg.lstsq(H.T, material_basis.T)[0].T
    theta_newt1 = torch.linalg.lstsq(H_newt1.T, material_basis.T)[0].T
    theta_mu1 = torch.linalg.lstsq(H_mu1.T, material_basis.T)[0].T
    theta_newt2 = torch.linalg.lstsq(H_newt2.T, material_basis.T)[0].T
    theta_mu2 = torch.linalg.lstsq(H_mu2.T, material_basis.T)[0].T

    # Move everything back to CPU for plotting
    material_projection = material_projection.cpu().numpy()
    material_basis = material_basis.cpu().numpy()
    W = W.cpu().numpy()
    H = H.cpu().numpy()
    T = T.cpu().numpy()
    W_newt1 = W_newt1.cpu().numpy()
    H_newt1 = H_newt1.cpu().numpy()
    W_mu1 = W_mu1.cpu().numpy()
    H_mu1 = H_mu1.cpu().numpy()
    W_newt2 = W_newt2.cpu().numpy()
    H_newt2 = H_newt2.cpu().numpy()
    W_mu2 = W_mu2.cpu().numpy()
    H_mu2 = H_mu2.cpu().numpy()
    theta_frob = theta_frob.cpu().numpy()
    theta_newt1 = theta_newt1.cpu().numpy()
    theta_mu1 = theta_mu1.cpu().numpy()
    theta_newt2 = theta_newt2.cpu().numpy()
    theta_mu2 = theta_mu2.cpu().numpy()

    # Plot reconstructed spectra
    compare_spectra(
        spectra_groups=[
            theta_frob @ H,
            theta_newt1 @ H_newt1,
            theta_mu1 @ H_mu1,
            theta_newt2 @ H_newt2,
            theta_mu2 @ H_mu2,
        ],
        ground_truth=material_basis,
        labels=['Ni', 'Cu', 'Al'],
        subtitles=[
            r'L$^2$ Loss',
            'Quasi-Newton (random init)',
            'Mann-Multiplicative (random init)',
            'Quasi-Newton (true init)',
            'Mann-Multiplicative (true init)',
        ],
        title=f'Material attenuation spectra reconstructions',
        x_label='Wavelength index',
        y_label='Attenuation',
        y_lim=(0, 1.1),
        filename=f'example_1_spectra_reconstruction.png'
    )

    # Plot reconstructed material coefficient maps
    plt.figure(figsize=(12, 12))
    plt.suptitle(f'Material projection reconstructions')
    row_max = np.max(material_projection, axis=0).reshape(1, 1, num_materials_true)
    image_dims = (detector_rows, detector_columns, num_materials_true)
    for i, (image, title) in enumerate([
            (material_projection.reshape(image_dims) / row_max, 'Ground Truth'),
            ((W @ np.linalg.pinv(theta_frob)).reshape(image_dims) / row_max, 'L$^2$ Loss'),
            ((W_newt1 @ np.linalg.pinv(theta_newt1)).reshape(image_dims) / row_max, 'Quasi-Newton'),
            ((W_mu1 @ np.linalg.pinv(theta_mu1)).reshape(image_dims) / row_max, 'Mann-Multiplicative'),
        ]):
        ax = plt.subplot(2, 2, i + 1)
        ax.set_title(title)
        ax.imshow(image)
    plt.savefig(f'example_1_material_maps.png')

    # plt.show()

if __name__ == "__main__":
    # dosages = np.logspace(0, 4, 101)

    # os.remove('snr.txt')
    # for dosage_rate in dosages:
    main()
    #     plt.close('all')

    # data = np.array(open("snr.txt").readlines(), dtype=float).reshape(-1,3,3)
    # plt.rcParams['figure.constrained_layout.use'] = True
    # plt.rc('font', size=12)
    # plt.figure(figsize=(15, 5))
    # for i, material in enumerate(["Ni", "Cu", "Al"]):
    #     ax = plt.subplot(1, 3, i+1)
    #     ax.plot(dosages, data[:,:,i])
    #     ax.legend(['L$^2$ Norm','Quasi-Newton','Mann-Multiplicative'])
    #     ax.set_xscale('log')
    #     ax.set_xlabel('Dosage (arbitrary units)')
    #     ax.set_ylabel('SNR (dB)')
    #     ax.set_title(material)
    # plt.suptitle('Denoising performance over dosage\nPixel count: 4096')
    # plt.savefig('example_1_snr_4096.png')
    # plt.show()

"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

Experiments 2 and 3 demonstrate the use of dehydration and rehydration for fast hyperspectral reconstruction.
A simulated hyperspectral neutron dataset containing three materials (Ni, Cu, and Al) is used for the purpose.
This script - experiment 2 - performs dehydration followed by MBIR and then exports the dehydrated reconstructions.
"""

import os
import time

import numpy as np

import mbirtorch as mt
import mbirtorch.hsnt as hsnt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    start_time = time.time()

    # Set output path and dataset name
    output_path = os.path.join(SCRIPT_DIR, 'processed_data')  # Path to export data
    dataset_name = "hsnt_dehydrated_recons"  # Name of the output dataset
    os.makedirs(output_path, exist_ok=True)

    # Simulation parameters
    num_angles = 16  # Number of view angles
    detector_rows = 64  # Number of rows in the detector
    detector_columns = 64  # Number of columns in the detector
    dosage_rate = 300  # Neutron dosage rate
    material_density = {"Ni": 0.25, "Cu": 0.25, "Al": 0.75}  # Define material density (vol. fraction)

    # Fast hyperspectral reconstruction parameters
    num_materials = 3  # Number of materials
    recon_snr_db = 20  # Assumed SNR for the reconstruction
    verbose = 2  # Verbosity level

    # Fix seed for random number generation
    np.random.seed(129)

    # Load theoretical linear attenuation coefficients for Ni, Cu, and Al (run exp_0_generate_material_basis.py once)
    basis_path = os.path.join(SCRIPT_DIR, 'binaries', 'material_basis.npy')
    if not os.path.exists(basis_path):
        raise SystemExit(f'{basis_path} not found: run exp_0_generate_material_basis.py first')
    material_basis = np.load(basis_path)

    # Generate simulated Poisson counts and the open beam
    counts, open_beam, angles, _ = hsnt.generate_hyper_data(material_basis,
                                                            num_angles=num_angles,
                                                            detector_rows=detector_rows,
                                                            detector_columns=detector_columns,
                                                            dosage_rate=dosage_rate,
                                                            material_density=material_density,
                                                            verbose=verbose)

    # Normalize the counts by the open beam (the detector calibration). Dehydration takes the transmission.
    transmission = counts / open_beam

    # MBIR model setup
    ct_model = mt.ParallelBeamModel((num_angles, detector_rows, detector_columns), angles)
    ct_model.set_params(snr_db=recon_snr_db, verbose=0)

    # Perform dehydration
    maps, spectra = hsnt.dehydrate(transmission, dataset_input_type='transmission', num_materials=num_materials,
                                   verbose=verbose)

    # Reconstruct the map of each material
    map_recons = []
    for idx in range(maps.shape[-1]):
        print(f"Reconstructing the map of material {idx}")
        map_recon, _ = ct_model.recon(maps[:, :, :, idx])
        map_recons.append(map_recon)
    map_recons = np.stack(map_recons, axis=-1)

    # Export dehydrated reconstructions into HDF5 file.
    filename = os.path.join(output_path, dataset_name + ".h5")
    hsnt.export_hsnt_data_hdf5(filename, [map_recons, spectra], domain='recon', name=dataset_name, model=ct_model)

    print(f'Total time elapsed: {time.time() - start_time:.2f} seconds')


if __name__ == "__main__":
    main()

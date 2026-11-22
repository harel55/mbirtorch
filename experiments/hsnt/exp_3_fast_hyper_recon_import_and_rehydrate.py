"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

Experiments 2 and 3 demonstrate the use of dehydration and rehydration for fast hyperspectral reconstruction.
A simulated hyperspectral neutron dataset containing three materials (Ni, Cu, and Al) is used for the purpose.
This script - experiment 3 - imports the dehydrated reconstructions from experiment 2 and performs rehydration.
"""

import os
import time
import warnings

import matplotlib.pyplot as plt

import mbirtorch.hsnt as hsnt
from plot_utils import plot_images, plot_spectra

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    start_time = time.time()

    # Set input path and dataset name
    input_path = os.path.join(SCRIPT_DIR, 'processed_data')  # Path to import data
    dataset_name = "hsnt_dehydrated_recons"  # Name of the input dataset

    # Display parameters
    display_wave_idx = 200  # Wavelength index of displayed image
    display_slice_idx = 8  # Slice index of displayed image
    display_vox_idx = [32, 32, 8]  # Voxel index [column, column, row] of displayed spectra
    vmax = 0.1  # Maximum voxel value for displayed image
    vmin = 0  # Minimum voxel value for displayed image
    y_lim_attenuation = (0.04, 0.09)  # (y_min, y_max) to set y-axis range for attenuation spectra

    # Import dehydrated reconstructions from HDF5 file
    filename = os.path.join(input_path, dataset_name + ".h5")
    if not os.path.exists(filename):
        warnings.warn(f"{filename} not found. Run exp_2_fast_hyper_recon_dehydrate_and_export.py first.")
        return
    dehydrated, _ = hsnt.import_hsnt_data_hdf5(filename)

    # Perform rehydration
    hsnt_recons = hsnt.rehydrate(dehydrated)

    # Plot hyperspectral reconstruction and spectra
    plot_images(images=[hsnt_recons[:, :, display_slice_idx, display_wave_idx]],
                titles=['Hyperspectral reconstruction'
                        f'\n\nSlice index: {display_slice_idx}'
                        f'\nWavelength index: {display_wave_idx}'],
                vmax=vmax, vmin=vmin)

    plot_spectra(spectra=[hsnt_recons[tuple(display_vox_idx)]],
                 title='Single voxel spectra (attenuation) for reconstructed data',
                 x_label='wavelength index',
                 y_label='attenuation',
                 y_lim=y_lim_attenuation)

    print(f'Total time elapsed: {time.time() - start_time:.2f} seconds')

    plt.show()


if __name__ == "__main__":
    main()

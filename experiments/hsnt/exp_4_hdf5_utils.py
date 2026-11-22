"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

This script shows the workflow for exporting, importing, and handling hyperspectral neutron datasets in HDF5 format.
It writes and reads both a dehydrated file and a hyperspectral file, and it compares their sizes.
"""

import os

import numpy as np

import mbirtorch.hsnt as hsnt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # Set output path
    output_path = os.path.join(SCRIPT_DIR, 'processed_data')  # Path to export data
    os.makedirs(output_path, exist_ok=True)

    # 1. Generate random sample dehydrated data
    maps = np.random.rand(3, 64, 64, 5).astype(np.float32)
    spectra = np.random.rand(5, 1000).astype(np.float32)
    print("Maps shape:", maps.shape, " spectra shape:", spectra.shape)

    # 2. Rehydrate to get full hyperspectral data
    hyperspectral = hsnt.rehydrate([maps, spectra])
    print("Rehydrated data shape:", hyperspectral.shape)

    # 3. Export the dehydrated and the hyperspectral data to separate files
    filename_dehydrated = os.path.join(output_path, "hsnt_dehydrated.h5")
    hsnt.export_hsnt_data_hdf5(filename_dehydrated, [maps, spectra], name="hsnt_dehydrated")

    filename_hyperspectral = os.path.join(output_path, "hsnt_hyperspectral.h5")
    hsnt.export_hsnt_data_hdf5(filename_hyperspectral, hyperspectral, "attenuation", name="hsnt_hyperspectral")

    # 4. Import both files back
    dehydrated_imported, metadata_dehydrated = hsnt.import_hsnt_data_hdf5(filename_dehydrated)
    hyperspectral_imported, metadata_hyperspectral = hsnt.import_hsnt_data_hdf5(filename_hyperspectral)

    print("\nImported maps shape:", dehydrated_imported[0].shape)
    print("--Imported metadata:", metadata_dehydrated)
    print("\nImported hyperspectral data shape:", hyperspectral_imported.shape)
    print("--Imported metadata:", metadata_hyperspectral)

    # 5. Compare file sizes
    size_dehydrated = os.path.getsize(filename_dehydrated)
    size_hyperspectral = os.path.getsize(filename_hyperspectral)

    print("\nDehydrated file size:", size_dehydrated, "bytes")
    print("Hyperspectral file size:", size_hyperspectral, "bytes")
    print("Size ratio (hyperspectral / dehydrated):", np.round(size_hyperspectral / size_dehydrated, 3))


if __name__ == "__main__":
    main()

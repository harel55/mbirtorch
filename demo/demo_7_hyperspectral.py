"""Demo 7: hyperspectral reconstruction.

A hyperspectral scan measures the transmission of every detector pixel in
many wavelength bins.  A sample of a few materials has a low-rank
attenuation: a map per material times a spectrum per material.  The scan is
dehydrated to those few maps, each map is reconstructed on its own, and the
reconstructions are rehydrated to a volume at every wavelength.  This is far
cheaper than reconstructing every wavelength bin, and the noise of the bins
is averaged away.
"""

import numpy as np
import mbirtorch
import mbirtorch.hsnt as hsnt

# The scan: a parallel-beam scan of three cylinders of three materials, with
# 200 wavelength bins and a few hundred counts per pixel and bin.
num_views = 16
num_det_rows = 32
num_det_channels = 64
num_wavelengths = 200
dosage_rate = 300

angles = np.linspace(0, np.pi, num_views, endpoint=False)
ct_model = mbirtorch.ParallelBeamModel((num_views, num_det_rows, num_det_channels), angles)
material_maps = hsnt.gen_material_phantom(ct_model.get_params('recon_shape'))
material_basis = hsnt.synthetic_material_basis(num_wavelengths)
counts, open_beam, _ = hsnt.generate_hyper_sinogram(ct_model, material_maps, material_basis,
                                                    dosage_rate=dosage_rate, seed=0)
print(f'hyperspectral sinogram: {counts.shape} (views, rows, channels, wavelengths)')

# Dehydrate: fit three component maps and three component spectra to the
# transmission.  The components span the materials' spectra but need not be
# the materials themselves.
transmission = counts / open_beam
maps, spectra = hsnt.dehydrate(transmission, dataset_input_type='transmission', num_materials=3, verbose=0)
print(f'dehydrated to component sinograms {maps.shape} and spectra {spectra.shape}')

# Reconstruct each component's sinogram as an ordinary scan.
map_recons = np.stack([ct_model.recon(maps[..., k])[0] for k in range(maps.shape[-1])], axis=-1)

# Rehydrate: the reconstructed maps times the spectra give the attenuation
# volume at every wavelength, and the true volume the same way from the
# material maps and their spectra.
volume = hsnt.rehydrate([map_recons, spectra])
truth = hsnt.rehydrate([np.moveaxis(material_maps, 0, -1), material_basis])

# The baseline: each wavelength bin reconstructed on its own by FBP from the
# negative log of its transmission.  A zero count has no finite attenuation,
# so the transmission is floored at one count.
attenuation_sino = -np.log(np.maximum(transmission, 1.0 / dosage_rate))
fbp = np.stack([ct_model.recon_fbp(attenuation_sino[..., k]) for k in range(num_wavelengths)], axis=-1)
for name, volumes in (('FBP per wavelength', fbp), ('dehydrated reconstruction', volume)):
    nrmse = np.linalg.norm(volumes - truth) / np.linalg.norm(truth)
    print(f'{name}: normalized RMS error over all wavelengths {nrmse:.3f}')

# The spectrum at the center of each cylinder: the true attenuation, the FBP
# of each wavelength, and the rehydrated reconstruction, over every bin.
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for k, ax in enumerate(axes):
    row, col, sl = np.round(np.argwhere(material_maps[k] > 0).mean(axis=0)).astype(int)
    ax.plot(truth[row, col, sl], label='ground truth')
    ax.plot(fbp[row, col, sl], label='FBP per wavelength', alpha=0.6)
    ax.plot(volume[row, col, sl], label='dehydrated reconstruction')
    ax.set_title(f'Material {k + 1}: spectrum at voxel ({row}, {col}, {sl})')
    ax.set_xlabel('wavelength bin')
    ax.set_ylabel('attenuation per voxel')
    ax.legend()
fig.suptitle('Attenuation spectra at the center of each cylinder')

# The fitted spectra span a subspace, not the materials themselves.  To check
# the subspace, each true spectrum is projected onto it: the least-squares
# combination of the three fitted spectra that comes closest to it.
coefficients, *_ = np.linalg.lstsq(spectra.T, material_basis.T, rcond=None)
projected = (spectra.T @ coefficients).T
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for k, ax in enumerate(axes):
    ax.plot(material_basis[k], label='true spectrum')
    ax.plot(projected[k], '--', label='projection onto the fitted subspace')
    ax.set_title(f'Material {k + 1}')
    ax.set_xlabel('wavelength bin')
    ax.set_ylabel('attenuation per unit density')
    ax.legend()
fig.suptitle('True material spectra and their projections onto the three fitted spectra')

# View them in the 4D viewer with wavelength in place of time: the frame
# slider steps through the wavelength bins, and the mean inside an ROI
# plotted against frame is the spectrum of that region.
mbirtorch.slice_viewer4d(np.moveaxis(truth, -1, 0), np.moveaxis(fbp, -1, 0), np.moveaxis(volume, -1, 0),
                         vmin=0.0, vmax=float(truth.max()),
                         slice_label=['GROUND TRUTH', 'FBP PER WAVELENGTH', 'DEHYDRATED RECON'],
                         title='Ground truth (left), FBP per wavelength (middle), dehydrated reconstruction (right)')

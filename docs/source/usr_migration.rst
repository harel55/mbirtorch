.. _MigrationDocs:

======================
Migrating from MBIRJAX
======================

MBIRTorch replaces MBIRJAX, which is now legacy software.  The two packages have the same
interface, so most scripts run after the import line is changed.  This page lists the
differences that a script is likely to hit.  See :ref:`InstallationDocs` to install MBIRTorch.


Imports
-------

Change the package name in the import line.  ``import mbirjax`` becomes ``import mbirtorch``.
The submodule paths are unchanged, so ``mbirjax.preprocess`` becomes ``mbirtorch.preprocess``.


Arrays
------

Both packages accept a numpy array wherever a sinogram, a weight array, or a volume is expected,
and both return numpy arrays by default.  A script that reads its data with numpy needs no
change.  Where MBIRJAX accepts a JAX array, MBIRTorch accepts a torch tensor.


Renamed methods
---------------

The direct reconstruction methods are renamed so that every reconstruction method starts with
``recon``.  The arguments are unchanged.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - MBIRJAX
     - MBIRTorch
   * - ``model.direct_recon(sinogram)``
     - ``model.recon_direct(sinogram)``
   * - ``model.fbp_recon(sinogram)``
     - ``model.recon_fbp(sinogram)``
   * - ``model.fdk_recon(sinogram)``
     - ``model.recon_fdk(sinogram)``
   * - ``model.split_sino_recon(sinogram)``
     - ``model.recon_split_sino(sinogram)``
   * - ``preprocess.mar.recon_plastic_metal(model, sinogram, weights)``
     - ``model.recon_plastic_metal(sinogram, weights)``
   * - ``model.scale_recon_shape(row_scale, col_scale, slice_scale)``
     - ``model.resize_recon_fov(row_scale, col_scale, slice_scale)``

``scale_recon_shape`` still works but prints a warning.

``TranslationModel.recon_fdk`` still works but prints a warning.  Its non-iterative reconstruction is now ``recon_direct_tct``.


Devices
-------

MBIRTorch chooses its devices on its own.  It prefers CUDA, then Apple's Metal, then the CPU.
To choose the devices yourself, call ``model.configure_devices``, for example
``model.configure_devices(devices=['cpu'])``.  The MBIRJAX parameter ``use_gpu`` does not exist.
See :ref:`usr_multi_gpu` for reconstruction on several GPUs.


Hyperspectral neutron data (hsnt)
---------------------------------

``hsnt.dehydrate`` and ``hsnt.hyper_denoise`` keep their names, but they fit the Poisson
likelihood of the counts instead of the scikit-learn NMF that MBIRJAX called, and they return a
basis of rank ``num_materials`` rather than ``safety_factor * num_materials``.  The arguments
after ``num_materials`` are keyword only, and MBIRJAX's NMF keywords (``safety_factor``,
``beta_loss``, ``max_iter``, ``tolerance``, ``batch_size``, ``random_state``) raise a
``TypeError``.  The same holds for ``mbirtorch.dehydrate`` and ``mbirtorch.hyper_denoise``.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - MBIRJAX
     - MBIRTorch
   * - ``hsnt.dehydrate(data, num_materials=3, safety_factor=2)``
     - ``hsnt.dehydrate(data, num_materials=3)``
   * - ``hsnt.hyper_denoise(data, num_materials=3, safety_factor=2)``
     - ``hsnt.hyper_denoise(data, num_materials=3)``
   * - ``hsnt.generate_hyper_data(material_basis, detector_rows, detector_columns, dosage_rate, material_thickness)``
       (MBIRJAX 0.6.11 to 0.6.15)
     - ``hsnt.generate_hyper_data(material_basis, num_angles, detector_rows, detector_columns, dosage_rate,
       material_density)``

Without ``num_materials`` the rank is estimated.  ``mode`` and ``chunk_pixels`` bound the
memory, as ``batch_size`` did, and ``verbose=2`` prints the rank search rather than plotting it.
``spectra``, formerly ``subspace_basis``, keeps its meaning, with the given spectra always held fixed (MBIRJAX refitted
them for data of up to 2**27 entries); leave out ``num_materials`` then, since the rank is the
basis's number of rows.  ``generate_hyper_data`` returns the Poisson counts and the open beam
(noiseless, ``dosage_rate`` everywhere) rather than an attenuation: divide the counts by the open
beam and pass the result to ``dehydrate`` with ``dataset_input_type='transmission'``, as measured counts
are normalized by their open beam.  A seed therefore gives different data, and ``noisy=False``
(keyword only) returns the noiseless counts.  Its ``material_density`` is a volume fraction that
scales a rounded bar about 10 thick at its center, not a thickness, so the MBIRJAX 0.6.11 to
0.6.15 values do not carry over (the defaults 2, 2, 10 became 0.2, 0.2, 1), and it returns
``[counts, open_beam, angles, truth]``, each of shape (views, rows, columns, bins), with ``truth``
the noiseless attenuation.  See :ref:`HSNTDocs`.


Removed names
-------------

MBIRJAX helper functions specific to JAX, and its debugging and plotting utilities, have no
counterpart in MBIRTorch.  A name that fails to import is one of these.  The features that
MBIRTorch adds are described in :doc:`overview`.

.. _HSNTDocs:

================
Hyperspectral CT
================

The ``hsnt`` module works on hyperspectral neutron data.  Each detector pixel holds a spectrum: one
value per wavelength bin.  The spectral axis of the data is always the last axis.

A sample made of a few materials has an attenuation of low rank, ``X = W H``.  Each row of ``H`` is the
spectrum of one material, and each column of ``W`` is the map of that material over the pixels.  Both are
nonnegative.  The module fits ``W`` and ``H`` to the data, which we call dehydration, and multiplies them back
into denoised data, which we call rehydration.  It also estimates the number of materials, and it reads and
writes the hsnt HDF5 file.  The scripts in
`experiments/hsnt <https://github.com/cabouman/mbirtorch/blob/main/experiments/hsnt/>`__ are worked examples.

.. currentmodule:: mbirtorch.hsnt


Dehydration and rehydration
---------------------------

``dehydrate`` fits ``W`` and ``H`` by maximum likelihood.  The counts are Poisson, and the likelihood is
written in terms of the transmission, ``T = exp(-W H)``.  So the natural input is the transmission, which is the
counts divided by the open beam.  A zero count is simply a zero transmission:

.. code-block:: python

   transmission = counts / open_beam                       # normalize by the detector calibration
   maps, spectra = hsnt.dehydrate(transmission, dataset_input_type='transmission')
   denoised = hsnt.rehydrate([maps, spectra], dataset_output_type='transmission')

The output ``[maps, spectra]`` takes far less memory than the data.  The data hold one value per pixel and
bin.  The maps hold one value per pixel and material, and the spectra hold one value per material and bin.
With a few materials and hundreds of bins, that is a reduction of about a hundred times.  The maps and spectra
always describe attenuation.  ``rehydrate`` returns attenuation or transmission, as its ``dataset_output_type`` says.

A few more facts about ``dehydrate``:

* The fitted spectra span the materials' spectra, but they need not be the spectra of pure materials.
* Data too large for the GPU are fitted in chunks of pixels.
* Given ``spectra``, only the maps are fitted.  This is how a large scan is dehydrated piece by piece,
  against spectra fitted on part of it.
* Given a ``sample_mask``, only the pixels inside the mask are fitted.  The maps are zero outside it.
* Given an ``open_beam`` with one count per bin, each bin's loss is weighted by its count.  Use it when the
  open beam differs between bins.

.. autofunction:: dehydrate
.. autofunction:: rehydrate
.. autofunction:: hyper_denoise


Import/Export
-------------

``export_hsnt_data_hdf5`` writes hyperspectral data or dehydrated data to one HDF5 file, and
``import_hsnt_data_hdf5`` reads it back.  The root of the file carries a format version and the mbirtorch
version that wrote it, so a later mbirtorch can tell what it is reading.

.. autofunction:: export_hsnt_data_hdf5
.. autofunction:: import_hsnt_data_hdf5


Synthetic data
--------------

A simulated scan for trying the method: a phantom of three materials, three synthetic spectra,
and the hyperspectral sinogram of the phantom through a tomography model, with Poisson counts.
``demo_7_hyperspectral.py`` dehydrates, reconstructs, and rehydrates one.

.. autofunction:: gen_material_phantom
.. autofunction:: synthetic_material_basis
.. autofunction:: generate_hyper_sinogram
.. autofunction:: generate_hyper_data

.. _AddingGeometry:

=================
Adding a geometry
=================

A scanner geometry is a subclass of :ref:`TomographyModelDocs`.  The subclass provides its
coordinate math and two projection functions.  The base class provides everything else:
reconstruction, parameter handling, batching over views, compilation, memory budgeting, and
multi-device execution.  :class:`~mbirtorch.ParallelBeamModel` and
:class:`~mbirtorch.ConeBeamModel` are the tested examples to copy from.


What to implement
-----------------

- **A forward projection function** that projects a batch of voxel cylinders into a batch of
  sinogram views.
- **A back projection function** that back projects a batch of views onto the voxel cylinders,
  summed over the batch.  Its ``coeff_power`` argument is 1 for the gradient and 2 for the
  diagonal of the Hessian.
- ``_view_batch_bodies``, which returns the pair of functions above.
- ``_view_batch_args``, which returns the keyword arguments the functions need.  It reads the
  parameters on every call, so a change made with ``set_params`` takes effect.
- ``verify_valid_params``, ``get_magnification``, and ``auto_set_recon_geometry``: parameter
  checks, the scale from the rotation axis to the detector, and the default reconstruction shape
  and voxel spacing.
- ``_get_psf_radii``, which returns three integer radii in the order ``det_channel_radius``,
  ``det_row_radius``, ``vox_slice_radius``.  ``det_channel_radius`` is the number of detector
  channels on each side of a voxel's center channel that a loop visits.  ``det_row_radius`` is
  the same number for detector rows.  ``vox_slice_radius`` is the number of recon slices on each
  side of a detector row's center slice.  A radius that no loop of the geometry reads is 0.
  ``_view_batch_args`` passes the three radii to the projection functions under these names.
- ``rows_track_slices``, a class attribute.  Set it True only when detector row ``r`` maps to
  reconstruction slice ``r`` alone, as in parallel beam.
- ``_transient_cols``, which returns the width of the largest temporary array a projection
  function holds per view.  The base class returns the slice band length, which is right only
  when ``rows_track_slices`` is True.  Every other geometry overrides it, because the driver
  divides its memory budget by this width.


How the driver runs the functions
---------------------------------

The two projection functions must be module-level functions, not methods.  The driver compiles
each one with ``torch.compile`` once per device and caches the compiled instances.  If
compilation fails, the function runs uncompiled, and the failure is recorded.

The driver divides the views into batches and calls a projection function once per batch.  The
model parameter ``view_batch_size`` sets the batch, and the driver reduces it when the memory
budget requires.


Multi-device requirements
-------------------------

A reconstruction can be spread across several devices, with the sinogram divided by view and
the reconstruction by slice.  See :doc:`dev_sharding_overview`.  A geometry needs three things
for this to work.

- Both projection functions accept ``slice_start`` and ``band_slices``, through which the
  multi-device back projection asks for one band of slices at a time.  A geometry with
  ``rows_track_slices`` True can leave them at their defaults.  Any other geometry must apply
  them.
- The number of devices need not divide the number of views or slices, so every per-view or
  per-slice computation must work on the block length it is given, including zero.
- ``_transient_cols`` must be right, as described above, or the memory check underestimates
  the run.


Skeleton
--------

The following is a starting point for a new geometry.  Study the parallel-beam and cone-beam
classes alongside it.

.. include:: _static/new_model_template.py
   :code: python


Optional GPU kernels
--------------------

The projection functions above run on any device.  On CUDA, a geometry can also provide
hand-written kernels in `Triton <https://triton-lang.org>`__, which compiles GPU kernels from
Python at import time.  A kernel wrapper has the same signature as the projection function it
replaces, so the driver runs both kinds without change.  Parallel beam, cone beam, and multi-axis
parallel beam each have a forward and a back kernel, in ``triton_parallel.py``,
``triton_cone.py``, and ``triton_multiaxis.py``.  Translation has none.

A kernel is used only after two checks pass, both in ``kernel_availability.py``.  Once per
process, a small Triton kernel is compiled and run to confirm that Triton works on the machine.
Then, on first use, each kernel is run against its projection function on a small problem on
the device that will run it.  A kernel whose result differs by more than a relative tolerance
of ``1e-4`` is not used.  The projection function is kept in every case as the fallback, so
results do not depend on which path runs.  Each geometry selects its kernels in its own
``_view_batch_bodies``.

Two environment variables control the kernels.  ``MBIRTORCH_DISABLE_TRITON=1`` turns every
kernel off.  It is read once, when the first model is built, so set it before then.
``MBIRTORCH_SORTED_FORWARD=0`` makes the parallel-beam forward projection use its simpler
kernel instead of the default one, which sorts each view's pixels by detector channel first.

"""This module defines the class TomographyModel.  The class holds the VCD reconstruction loop,
FBP, and the methods that call the projectors.

The reconstruction loop always splits its arrays into one part per device.  A run on one device
uses the same code with a single part.  A run with a fixed random seed repeats the same
reconstruction at every iteration.  For this reason, the order of operations in each formula
must not be changed.
"""

import concurrent.futures
import contextlib
import datetime
import io
import math
import os
import threading
import warnings

import numpy as np
import torch

from . import _memory_estimate
from . import _sharding
from . import _min_problem_sizes
from . import qggmrf as _qggmrf
from . import tomography_utils, vcd_utils
from .memory_stats import get_memory_stats
from ._utils import _AUTO_REGULARIZATION_PARAM_NAMES, recon_param_names
from .parameter_handler import ParameterHandler
from .projectors import Projectors, maybe_compile

_F32_EPS = float(np.finfo(np.float32).eps)

# This is the number of pixels in each batch of voxel cylinders that the multi-device forward
# projection transfers between devices.
FORWARD_PIXEL_BATCH = 32768


def _warn_deprecated_name(old, new):
    """This function issues a DeprecationWarning that names the old and the new attribute."""
    warnings.warn(f'{old} is deprecated and will be removed in a future release; use {new}.',
                  DeprecationWarning, stacklevel=3)


# torch.compile compiles each of these module-level functions once per process.
def _diagonal_update_direction(forward_grad, prior_grad, forward_hess, prior_hess):
    return -((forward_grad + prior_grad) / (forward_hess + prior_hess))


def _prior_line_terms(prior_grad, prior_hess, delta):
    return (torch.sum(prior_grad * delta),
            torch.sum(prior_hess * delta * delta))


def _forward_lin_quad_const(weighted_error_sinogram, delta_sinogram, fm_constant):
    return (fm_constant * torch.sum(weighted_error_sinogram * delta_sinogram),
            fm_constant * torch.sum(delta_sinogram * delta_sinogram))


def _forward_lin_quad_weighted(error_sinogram, delta_sinogram, weights, fm_constant):
    return (fm_constant * torch.sum(weights * error_sinogram * delta_sinogram),
            fm_constant * torch.sum(delta_sinogram * delta_sinogram * weights))


def _apply_update(flat_recon, error_sinogram, pixel_indices, delta_scaled,
                  alpha, delta_sinogram):
    # This function updates its inputs in place.  The callers assign the returned tensors to
    # their state variables.
    flat_recon.index_add_(0, pixel_indices, delta_scaled)
    delta_sumsq = torch.sum(delta_scaled * delta_scaled, dim=0)
    # addcmul_ reads alpha from a tensor on the device.  Using sub_ would force a host
    # synchronization.
    error_sinogram.addcmul_(delta_sinogram, alpha, value=-1)
    l1 = torch.sum(torch.abs(delta_scaled))
    return flat_recon, error_sinogram, delta_sumsq, l1


def _resolve_device(device):
    """This function converts 'auto' to cuda if CUDA is available, to mps if MPS is available,
    and otherwise to cpu.  It returns any other value as a torch.device."""
    if device != 'auto':
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def gpu_devices():
    """This function returns the GPU devices that torch can use, as a tuple.

    If CUDA is available, the tuple holds every CUDA device.  If only MPS is available, the
    tuple holds the MPS device.  If there is no GPU, the tuple is empty.  The function does not
    read any environment variable.
    """
    if torch.cuda.is_available():
        return tuple(torch.device('cuda', i) for i in range(torch.cuda.device_count()))
    if torch.backends.mps.is_available():
        return (torch.device('mps'),)
    return ()


def cpu_devices():
    """This function returns the CPU device, as a one-element tuple."""
    return (torch.device('cpu'),)


def default_devices():
    """This function returns the devices that a run uses when the user names no devices.  It
    returns the GPU devices as a list if there are any, and otherwise the CPU device."""
    return list(gpu_devices()) or list(cpu_devices())


def _array_extremes(array):
    """This function returns the minimum and the maximum of ``array``.  It computes them on the
    device that already holds the array.  If the array holds a NaN, both values are NaN.
    """
    if torch.is_tensor(array):
        tensor = array
    else:
        with warnings.catch_warnings():
            # The code only reads the array, so the torch warning about a non-writable array
            # does not apply.
            warnings.filterwarnings('ignore', message='.*not writable.*')
            tensor = torch.as_tensor(array)
    return float(tensor.min()), float(tensor.max())


class TomographyModel(ParameterHandler):
    """
    This class is used as the base class of all tomography geometries.  It provides projection
    (:meth:`forward_project`, :meth:`back_project`) and reconstruction (:meth:`recon`,
    :meth:`prox_map`, :meth:`recon_direct`).  It also provides device configuration
    (:meth:`~mbirtorch.TomographyModel.configure_devices`) and parameter handling.  Users
    construct a geometry subclass, such as ``ConeBeamModel`` or ``ParallelBeamModel``, rather
    than this class.

    All geometries share these constructor arguments:

    * sinogram_shape (tuple of int): The shape (num_views, num_det_rows, num_det_channels).
    * view_batch_size (int, optional): The number of views in each projection call.  If None
      (the default), each projection method chooses the number.
    * compile_mode (str, optional): If 'auto' (the default), mbirtorch compiles the
      computational kernels with torch.compile.  If 'off', mbirtorch runs without compilation.
    """

    def __init__(self, sinogram_shape, view_batch_size=None,
                 compile_mode='auto', **kwargs):
        super().__init__()
        # The device is chosen on first use, so mbirtorch does not touch a device it will not use.
        self._torch_device = None
        self._sino_placement = None
        self._recon_placement = None
        self._projector_functions = None
        # This is the number of views per projector call.  None means the projector's default.
        self.view_batch_size = view_batch_size
        self.compile_mode = compile_mode
        # This holds the stored initialization of prox_map, which later prox_map calls reuse.
        self.prox_data = None
        self._dc_damping_cache = None
        self.dev2dev_safe = True     # configure_devices tests and sets this value.
        # This becomes False when the user calls configure_devices.
        self.device_layout_is_automatic = True
        # This is the (sinogram_shape, recon_shape) pair for which mbirtorch chose the devices.
        self._device_choice_shapes = None
        # This is the task, 'recon' or 'direct', for which mbirtorch checked the memory of the
        # chosen GPUs.
        self._device_choice_workload = None
        # These are the device counts that the automatic choice rejected.  The run log lists them.
        self.device_choice_rejections = []
        self._size_limit_fallback = None
        self._size_limit_held = None
        # The memory check runs when mbirtorch chooses the devices.  A later change to this value
        # has no effect until the shapes change.
        self.skip_memory_check = False
        self.memory_check_margin = 0.15
        # Only the tests read these two values.
        self.last_memory_estimate = None
        self.last_memory_calibration = None
        # This is the thread pool of _vcd_recon, with one thread per device.  It is None outside
        # a reconstruction.
        self._per_device_pool = None

        from ._utils import Param
        for key, val in kwargs.items():
            self.params[key] = Param(val, True)
        self.set_params(no_compile=True, no_warning=True,
                        sinogram_shape=tuple(int(s) for s in sinogram_shape))

        # The order matters.  The geometry defaults come first, then the projectors, and then the
        # validity check.
        self.auto_set_recon_geometry(no_compile=True, no_warning=True)
        self.verify_valid_params()

    @property
    def compile_enabled(self):
        return self.compile_mode != 'off'

    @property
    def torch_device(self):
        """This property returns the first device of the model.  If ``configure_devices`` has
        not set the device, the property chooses it on first use.  It chooses cuda if CUDA is
        available, mps if MPS is available, and otherwise cpu."""
        if self._torch_device is None:
            self._torch_device = _resolve_device('auto')
        return self._torch_device

    @torch_device.setter
    def torch_device(self, value):
        self._torch_device = torch.device(value)

    # Sinograms are split across devices by view (axis 0).  Reconstructions are split across
    # devices by slice (the last axis).
    @property
    def sino_placement(self):
        if self._sino_placement is None:
            self._sino_placement = _sharding.Placement([self.torch_device],
                                                       axis=0)
        return self._sino_placement

    @sino_placement.setter
    def sino_placement(self, value):
        self._sino_placement = value

    @property
    def recon_placement(self):
        if self._recon_placement is None:
            self._recon_placement = _sharding.Placement([self.torch_device],
                                                        axis=-1)
        return self._recon_placement

    @recon_placement.setter
    def recon_placement(self, value):
        self._recon_placement = value

    @property
    def projector_functions(self):
        """This property returns the projector object.  The property builds the object on first
        use, for the devices that the model uses at that time."""
        if self._projector_functions is None:
            self.create_projectors()
        return self._projector_functions

    @projector_functions.setter
    def projector_functions(self, value):
        self._projector_functions = value

    # ── hooks for geometry subclasses ─────────────────────────────────────────
    def create_projectors(self):
        self.projector_functions = Projectors(self)

    def get_magnification(self):
        """This method returns the magnification of this geometry.  The parallel-beam geometry
        returns 1.0."""
        raise NotImplementedError

    def _get_psf_radii(self):
        """This method returns the three tap radii of the projector loops as integers, in the
        order (det_channel_radius, det_row_radius, vox_slice_radius).

        A radius is the number of cells on each side of the center that a loop visits, so the
        loop visits 2 * radius + 1 cells.  ``det_channel_radius`` counts detector channels around
        a voxel.  ``det_row_radius`` counts detector rows around a voxel.  ``vox_slice_radius``
        counts recon slices around a detector row.  A radius that no loop of the geometry reads
        is 0.  Each geometry implements this method."""
        raise NotImplementedError

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """
        This method sets the automatic values of the recon shape and the voxel pitch.  It
        computes these values from the geometry parameters and the sinogram shape.

        The user must call this method after any change to a detector or geometry parameter,
        such as ``delta_det_channel``.  Otherwise, the reconstruction has the wrong scale.  The
        method sets ``recon_shape``, ``delta_voxel``, and ``recon_slice_offset``.  For this
        reason, any resize or shift of the region of reconstruction must come after this call.

        Args:
            no_compile (bool, optional): If True, the method does not rebuild the projectors.
                Defaults to False.
            no_warning (bool, optional): If True, the method does not issue warnings.
                Defaults to False.
        """
        raise NotImplementedError

    def _view_batch_bodies(self):
        """This method returns the two module-level functions of this geometry that project one
        batch of views.  It returns them as the pair (forward, back)."""
        raise NotImplementedError(
            f'{type(self).__name__} defines no per-view-batch projection '
            'bodies.')

    def _view_batch_args(self):
        """This method returns the dictionary of arguments for the projection functions of this
        geometry.  It reads the parameters on every call, outside the compiled code."""
        raise NotImplementedError

    def _transient_cols(self, band_cols):
        """This method returns the number of columns of the largest temporary array that this
        geometry uses for one view.  The projector uses this number to set its view batch
        size."""
        return band_cols

    def recon_direct(self, sinogram, filter_name=None, output_sharded=False):
        """
        This method performs a direct, non-iterative reconstruction, which is typically a form
        of filtered back projection.  Some geometries do not provide this method.

        Args:
            sinogram (numpy or tensor): The 3D sinogram, with shape
                (num_views, num_det_rows, num_det_channels).
            filter_name (string, optional): The name of the filter.  Every geometry uses 'ramp'
                as the default.
            output_sharded (bool, optional): If False (the default), the method returns a numpy
                array.  If True, the method returns the result on the devices.  On a model with
                one device, the result is a torch tensor.  On a model with several devices, the
                result is a Shards container with one tensor per device.

        Returns:
            recon (numpy or tensor): The reconstructed volume.

        Note:
            This note is for a developer who writes a new geometry.  The version of this method
            in the new geometry must call ``self._apply_device_policy(workload='direct')``
            before it allocates any large array.  If the user has not chosen the GPUs, that call
            chooses them.  The call also checks the memory that a direct reconstruction needs.
        """
        raise NotImplementedError

    def recon_split_sino(self, sino, weights=None, half_overlap=5, init_recon=None,
                         max_iterations=15, stop_threshold_change_pct=0.2,
                         first_iteration=0, compute_prior_loss=False,
                         logfile_path='~/.mbirtorch/logs/recon.log', print_logs=True,
                         align_split_grid=False):
        """
        This method reduces the memory that MBIR reconstruction needs.  It reconstructs the
        volume in sections and joins the results.  It is meant for a volume that does not fit on
        the GPUs as a whole.

        The method divides the detector rows into overlapping sections.  Cone beam uses two
        halves, and parallel beam uses one or more sections.  The method reconstructs each
        section with ordinary MBIR.  It then joins the sections with a linear blend across the
        overlap.  The method divides the GPUs into the largest number of equal groups that can
        each hold a section, and the groups run at the same time.

        The user calls this method in the same way as :meth:`recon`.  The result is
        approximately equal to the result of :meth:`recon`.  If the user chose devices with
        ``configure_devices``, the method uses only those GPUs.  A geometry that does not
        implement this method raises ``NotImplementedError``.

        Args:
            sino (numpy or tensor): The full sinogram, with shape (num_views, num_rows,
                num_cols).  The method does not accept a sharded array.
            weights (numpy or tensor, optional): The sinogram weights, with the same shape as
                ``sino``.  The method does not accept a sharded array.
            half_overlap (int, optional): The number of overlapping detector rows that each half
                keeps past the split.  Defaults to 5.
            init_recon (optional): The same as in :meth:`recon`.  The method does not accept a
                sharded array.
            max_iterations (int, optional): The same as in :meth:`recon`.
            stop_threshold_change_pct (float, optional): The same as in :meth:`recon`.
            first_iteration (int, optional): The same as in :meth:`recon`.
            compute_prior_loss (bool, optional): The method accepts this argument so that its
                interface matches :meth:`recon`.  The method ignores this argument.
            logfile_path (str, optional): The same as in :meth:`recon`.  The method merges the
                logs of the two halves into this one file.
            print_logs (bool, optional): The same as in :meth:`recon`.
            align_split_grid (bool, optional): If True, the method shifts the grid of recon
                slices by up to half a slice.  The shift aligns the split with the cut in the
                sinogram.  The alignment removes stripes at the seam.  Defaults to False.

        Returns:
            Tuple[np.ndarray, dict]: The reconstructed volume and a metadata dictionary.  The
            dictionary holds the recon and model parameters of each half.  It also holds
            ``'split_params'``, which lists the overlaps, any alignment shift, ``'num_groups'``,
            and ``'gpus_per_group'``.  If the split would leave either half too thin, the method
            issues a warning and performs a standard :meth:`recon` instead.  In that case it
            returns the dictionary of that reconstruction, which has no entries for the halves.
        """
        raise NotImplementedError(
            f'recon_split_sino is not implemented for {type(self).__name__}.')

    # ── the sections of a split reconstruction ────────────────────────────────

    def _split_devices(self):
        """This method returns the devices that a split reconstruction divides into groups.  If
        the user chose devices, it returns those devices.  Otherwise, it returns every GPU that
        the automatic device choice could use."""
        if not self.device_layout_is_automatic:
            return list(self.sino_placement.devices)
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        env_count = _memory_estimate.env_device_count()
        if env_count is not None:
            visible = min(env_count, visible)
        return self._candidate_devices(visible) if visible > 1 else [self.torch_device]

    def _section_fits(self, section_model, devices, weighted):
        """This method returns True if a reconstruction of ``section_model`` fits in the memory
        of ``devices``."""
        if self.skip_memory_check:
            return True
        call_arrays = {}
        if weighted:
            # This is a read-only array with the sinogram shape that uses almost no memory.
            shape = tuple(int(n) for n in section_model.get_params('sinogram_shape'))
            call_arrays['weights'] = np.broadcast_to(np.ones(1, dtype=np.float32), shape)
        estimate = section_model._estimate_memory(devices=devices, workload='recon',
                                                  **call_arrays)
        if estimate is None:
            return True
        fits, _rows = section_model._layout_capacity(devices, estimate, call_arrays)
        return fits

    @staticmethod
    def _choose_gpus_per_group(devices, fits, max_groups=None):
        """This method returns the number of devices in each group of a split reconstruction.
        It returns the smallest group size that divides the device count, gives at most
        ``max_groups`` groups, and passes the test ``fits``.  If no size meets these conditions,
        the method puts all the devices in one group.
        """
        num_devices = len(devices)
        for size in range(1, num_devices):
            if num_devices % size or (max_groups is not None and num_devices // size > max_groups):
                continue
            if fits(devices[:size]):
                return size
        return num_devices

    @staticmethod
    def _run_split_sections(jobs, devices, gpus_per_group):
        """This method runs the sections of a split reconstruction, and it returns their results
        in order.  If there are several groups, each group runs in its own thread.  In that case,
        each section gets its own random generator, seeded from the global np.random state, and
        each thread uses the caller's torch thread count, so that the compiled sums add in the
        same order as they do when a section runs alone.
        """
        if gpus_per_group >= len(devices):
            return [job(None, None) for job in jobs]
        groups = [devices[i:i + gpus_per_group] for i in range(0, len(devices), gpus_per_group)]
        seeds = np.random.randint(0, 2 ** 31 - 1, size=len(jobs))
        results = [None] * len(jobs)
        lock = threading.Lock()
        state = {'next': 0, 'failed': False}
        num_threads = torch.get_num_threads()

        def run_group(group):
            torch.set_num_threads(num_threads)
            while True:
                with lock:
                    index = state['next']
                    if index >= len(jobs) or state['failed']:
                        return
                    state['next'] += 1
                try:
                    results[index] = jobs[index](group, np.random.default_rng(int(seeds[index])))
                except BaseException:
                    # After a section fails, the other groups do not start new sections.
                    with lock:
                        state['failed'] = True
                    raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = [executor.submit(run_group, group) for group in groups]
            for future in futures:
                future.result()
        return results

    def recon_plastic_metal(self, sino, weights, num_BH_iterations=3, num_constraint_update_iter=10,
                            stop_threshold_change_pct=0.2, num_metal=1, order=3, alpha=1, beta=0.002,
                            gamma=0.1, verbose=0, max_iterations=15,
                            logfile_path='~/.mbirtorch/logs/recon.log',
                            radial_margin=None, top_margin=None, bottom_margin=None):
        """
        This method performs iterative metal artifact reduction with plastic-metal beam
        hardening correction.  If ``num_metal`` is 0, the method performs a standard MBIR recon.

        The method alternates between beam hardening correction and reconstruction.  The
        function ``correct_sino_plastic_metal`` performs the beam hardening correction.  The
        method works on any geometry that provides ``recon_direct`` and ``recon``.  A cone-beam
        model reconstructs with ``recon_split_sino``, and every other geometry reconstructs with
        ``recon``.

        Args:
            sino (numpy or tensor): The sinogram to correct.  The method converts a tensor to
                numpy on entry.  The method does not accept a sharded array.
            weights (numpy or tensor): The transmission weights for the reconstruction.  The
                method converts a tensor to numpy on entry.  The method does not accept a
                sharded array.
            num_BH_iterations (int, optional): The number of correction and reconstruction
                iterations.  Defaults to 3.
            num_constraint_update_iter (int, optional): The number of iterations that update the
                constraints.  At each iteration, the method activates the most violated
                constraints and solves the quadratic program again with OSQP.
            stop_threshold_change_pct (float, optional): The threshold, in percent, on the
                relative change for early stopping in MBIR.  Defaults to 0.2.
            num_metal (int, optional): The number of metal materials to segment and correct.
                Defaults to 1.
            order (int, optional): The maximum total degree of the beam hardening correction
                polynomial.  Defaults to 3.
            alpha (float, optional): The scaling factor of the regularization weights, which
                depends on the degree.  Higher values penalize the higher-order terms more
                strongly.  Defaults to 1.
            beta (float, optional): The regularization strength of the ridge regression.
                Defaults to 0.002.
            gamma (float, optional): The stabilization factor of the plastic correction.  The
                method multiplies the mean of ``s_p`` by ``gamma`` to set a positive lower bound
                on the denominator.  Defaults to 0.1.
            verbose (int, optional): The verbosity level for printing intermediate information.
                Defaults to 0.
            max_iterations (int, optional): The maximum number of MBIR iterations in each
                reconstruction pass.  Defaults to 15.
            logfile_path (str, optional): The same as in :meth:`recon`.  The method merges the
                logs of the beam hardening passes into this one file, each under a section
                header.
            radial_margin, top_margin, bottom_margin (int or None, optional): The margins of the
                segmentation mask that classifies plastic and metal.  If None (the default), the
                margins scale with the volume size.  See ``segment_plastic_metal``.

        Returns:
             (recon, recon_dict): The final corrected reconstruction as a NumPy array in host
             memory, and the reconstruction dictionary of the final reconstruction pass.

        Example:
            >>> recon, recon_dict = ct_model.recon_plastic_metal(
            ...     sino, weights, num_BH_iterations=3, num_metal=1, order=3, beta=0.005)
            >>> mbirtorch.slice_viewer(recon)
        """
        import functools
        from .preprocess.beam_hardening import correct_sino_plastic_metal
        from .preprocess.recon_utils import segment_plastic_metal
        from .utilities import merge_log_files
        from .view_utils import slice_viewer

        if num_metal < 0:
            raise ValueError("num_metal must be >= 0")

        # This check must come before np.asarray, because np.asarray would accept a sharded input.
        if (isinstance(sino, _sharding.Shards)
                or isinstance(weights, _sharding.Shards)):
            raise ValueError(
                'recon_plastic_metal does not accept a sinogram or weights '
                'in sharded form.  Pass the host (numpy or tensor) sinogram and the '
                'host weights.')
        if isinstance(sino, torch.Tensor):
            sino = sino.detach().cpu().numpy()
        sino = np.asarray(sino)
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
            weights = np.asarray(weights)

        # Cone beam uses recon_split_sino, which never puts the full sinogram on the devices.
        if ('cone' in self.get_params('geometry_type')
                and type(self).recon_split_sino is not TomographyModel.recon_split_sino):
            recon_function = self.recon_split_sino
        else:
            recon_function = functools.partial(self.recon, output_sharded=True)

        def to_output_form(r):
            return r if isinstance(r, np.ndarray) else self._gather_recon(r)

        if num_metal == 0:
            recon, recon_dict = recon_function(sino, weights=weights, max_iterations=max_iterations,
                                               stop_threshold_change_pct=stop_threshold_change_pct,
                                               logfile_path=logfile_path)
            return to_output_form(recon), recon_dict

        if verbose >= 1:
            print("\n************ Perform initial FDK reconstruction  **************")
        recon = self.recon_direct(sino, output_sharded=True)

        # Each pass writes its own log file.  The finally block merges the files.
        if logfile_path:
            log_path = os.path.expanduser(logfile_path)
            pass_log_paths = [log_path + '.pass{}'.format(i + 1) for i in range(num_BH_iterations)]
        else:
            log_path, pass_log_paths = None, [None] * num_BH_iterations
        try:
            for i in range(num_BH_iterations):
                if verbose >= 1:
                    print(f"\n************ Correct sino plastic metal {i + 1}  **************")
                corrected_sinogram = correct_sino_plastic_metal(self, sino, recon, num_metal=num_metal, order=order, alpha=alpha, beta=beta, gamma=gamma, num_constraint_update_iter=num_constraint_update_iter,
                                                                radial_margin=radial_margin, top_margin=top_margin, bottom_margin=bottom_margin)

                if verbose >= 1:
                    print(f"\n************ Perform MBIR reconstruction {i + 1} **************")
                # The recon methods need host arrays or tensors, so the sharded recon is gathered
                # into one array.
                init = (self._gather_recon(recon)
                        if isinstance(recon, _sharding.Shards) else recon)
                recon, recon_dict = recon_function(corrected_sinogram, weights=weights, init_recon=init,
                                          max_iterations=max_iterations,
                                          stop_threshold_change_pct=stop_threshold_change_pct,
                                          logfile_path=pass_log_paths[i])

                if verbose >= 2:
                    print(f"\n************ BH Iteration {i + 1}: Display plastic and metal mask **************")
                    plastic_mask, metal_masks, plastic_scale, metal_scales = segment_plastic_metal(
                        recon, num_metal, radial_margin=radial_margin, top_margin=top_margin,
                        bottom_margin=bottom_margin)
                    labels = ['Plastic Mask'] + [f'Metal {j + 1} Mask' for j in range(len(metal_masks))]
                    slice_viewer(plastic_mask, *metal_masks, vmin=0, vmax=1.0,
                                    slice_label=labels,
                                    title=f'Iteration {i + 1}: Comparison of Plastic and Metal Masks')
        finally:
            if log_path:
                # A failed pass may have left its log file open.  The file must be closed because
                # the merge deletes the files.
                self.close_log_file()
                labels = ['recon_plastic_metal: BH pass {}'.format(i + 1) for i in range(num_BH_iterations)]
                merge_log_files(log_path, zip(labels, pass_log_paths))

        return to_output_form(recon), recon_dict

    # ── projection wrappers ───────────────────────────────────────────────────
    def sparse_forward_project(self, voxel_values, pixel_indices):
        """This method forward projects the voxel values at ``pixel_indices`` to a full
        sinogram.  The voxel values at one pixel index form a cylinder of voxels through all
        slices.  If the input is a tensor, the output is a tensor.  If the input is ``Shards``,
        or if the model uses several devices, the output is a sinogram split by view across the
        devices."""
        voxel_values = self._shard_recon(voxel_values)
        if isinstance(voxel_values, _sharding.Shards):
            return self._sparse_forward_project_sharded(voxel_values, pixel_indices)
        return self.projector_functions._sparse_forward_project_single_device(
            voxel_values, pixel_indices)

    def sparse_back_project(self, sinogram, pixel_indices, coeff_power=1):
        """This method back projects a sinogram to the voxel cylinders at ``pixel_indices``.
        If the input is a tensor, the output is a tensor.  If the input is ``Shards``, or if the
        model uses several devices, the output is split by slice across the devices."""
        sinogram = self._shard_sinogram(sinogram)
        if isinstance(sinogram, _sharding.Shards):
            return self._sparse_back_project_sharded(sinogram, pixel_indices,
                                                     coeff_power=coeff_power)
        return self.projector_functions._sparse_back_project_single_device(
            sinogram, pixel_indices, coeff_power=coeff_power)

    def _band_pool(self, n):
        """This method returns the thread pool for a sharded projection."""
        if self._per_device_pool is not None:
            return contextlib.nullcontext(self._per_device_pool)
        return _sharding.device_pool(n)

    @staticmethod
    def _slice_band_length(slices_per_dev, n_dev, num_pixels, fixed_band=None):
        """This method returns the number of slices in each group of slices that the
        multi-device back projection processes at one time.  By default, each device processes
        all its slices at once.  The model attribute ``back_project_slice_band`` sets a smaller
        number."""
        b = fixed_band if fixed_band else slices_per_dev
        return min(int(b), slices_per_dev)

    def _forward_pixel_batch(self):
        """This method returns the number of pixels in each batch of voxel cylinders that the
        forward projection transfers between devices.  The default is
        :data:`FORWARD_PIXEL_BATCH`.  The model attribute ``forward_project_pixel_batch``
        overrides the default."""
        fixed = getattr(self, 'forward_project_pixel_batch', None)
        return max(1, int(fixed)) if fixed else FORWARD_PIXEL_BATCH

    @staticmethod
    def _balanced_slice_bounds(extent, band_len):
        """This method splits ``[0, extent)`` into the fewest intervals of length at most
        ``band_len``.  The lengths of the intervals differ by at most one."""
        if extent <= 0:
            # A device with no slices gets no intervals.
            return []
        num_bands = -(-extent // band_len)            # Ceiling division.
        base, rem = divmod(extent, num_bands)
        bounds, start = [], 0
        for k in range(num_bands):
            length = base + (1 if k < rem else 0)
            bounds.append((start, start + length))
            start += length
        return bounds

    def _banded_setup(self, pixel_indices):
        """This method builds the values that the multi-device projectors share.  These are
        the view range of each device, the slice range of each device, and a copy of the pixel
        indices on each device."""
        sp, rp = self.sino_placement, self.recon_placement
        if type(self)._view_batch_bodies is TomographyModel._view_batch_bodies:
            raise NotImplementedError(
                f'{type(self).__name__} has no per-view-batch projection '
                'bodies, so the multi-device drivers cannot run.')
        # The view ranges and slice ranges are half-open and in device order.  The last devices
        # may hold no views or slices.
        view_spans = [span for _d, span in sp.shard_ranges()]
        band_ranges = rp.shard_ranges()
        idx_per_dev = [torch.as_tensor(pixel_indices, dtype=torch.int64).to(d)
                       for d in sp.devices]
        return sp, rp, view_spans, band_ranges, idx_per_dev

    def _sparse_forward_project_sharded(self, voxel_shards, pixel_indices):
        """This method runs the forward projection on voxel values that are split across
        devices."""
        if voxel_shards.placement.is_trivial:
            return _sharding.Shards(
                [self.projector_functions._sparse_forward_project_single_device(
                    voxel_shards.tensors[0], pixel_indices)],
                self.sino_placement)
        return self._sparse_forward_project_cylinders(voxel_shards,
                                                      pixel_indices)

    def _sparse_forward_project_cylinders(self, voxel_shards, pixel_indices):
        """This method runs the multi-device forward projection.  The voxel cylinders are
        transferred between devices in batches of pixels.  Each device that holds views
        collects each batch of full-height cylinders from the devices that hold slices.  The
        device then projects the batch onto its own views."""
        sp, rp, view_spans, _band_ranges, idx_per = self._banded_setup(
            pixel_indices)
        pf = self.projector_functions
        num_channels = int(self.get_params('sinogram_shape')[2])
        num_rows = (int(rp.axis_len) if self.rows_track_slices
                    else int(self.get_params('sinogram_shape')[1]))
        num_pixels = int(idx_per[0].shape[0])
        shards = voxel_shards.tensors        # The device order is the order of the slices.
        pixel_batch = self._forward_pixel_batch()
        batch_bounds = [(p0, min(p0 + pixel_batch, num_pixels))
                        for p0 in range(0, num_pixels, pixel_batch)]

        def worker(i, dev):
            v0, v1 = view_spans[i]
            if v1 <= v0:
                return torch.zeros((0, num_rows, num_channels),
                                   dtype=voxel_shards.dtype, device=dev)
            local_idx = idx_per[i]
            owned = None

            def transfer(k):
                p0, p1 = batch_bounds[k]
                return _sharding.transfer_cylinder_batch_async(
                    shards, p0, p1, dev, self.dev2dev_safe)

            # This holds the batch that is transferred while the current batch is projected.
            ahead = transfer(0) if batch_bounds else None
            for k, (p0, p1) in enumerate(batch_bounds):
                full_cyl, ready = ahead
                # The copy of the next batch starts before this batch is projected.  The wait below
                # makes the projection wait for the copies of this batch.
                ahead = transfer(k + 1) if k + 1 < len(batch_bounds) else None
                _sharding.wait_for_cylinder_batch(dev, ready)
                # The first batch allocates the output block, and later batches add into it.
                if owned is None:
                    owned = pf.sparse_forward_project_view_range(
                        full_cyl, local_idx[p0:p1], (v0, v1), slice_start=0,
                        dev_index=i)
                else:
                    pf.sparse_forward_project_view_range(
                        full_cyl, local_idx[p0:p1], (v0, v1), slice_start=0,
                        dev_index=i, accumulate_into=owned)
                full_cyl = None
            if owned is None:
                # If there are no pixels, the output block for the views of this device is zero.
                owned = torch.zeros((v1 - v0, num_rows, num_channels),
                                    dtype=voxel_shards.dtype, device=dev)
            return owned

        # The thread that uses a transfer to a device also starts that transfer.
        transfer_devices = (list(voxel_shards.placement.devices)
                            + list(sp.devices))
        _sharding.open_copy_streams(transfer_devices)
        try:
            with self._band_pool(sp.n_devices) as pool:
                tensors = _sharding.run_per_device(sp.devices, worker,
                                                   executor=pool)
        finally:
            # The streams are closed even after an error, because copies in progress still read
            # the voxel tensors.
            _sharding.close_copy_streams(transfer_devices)
        return _sharding.Shards(tensors, sp)

    def _sparse_back_project_sharded(self, sino_shards, pixel_indices,
                                     coeff_power=1):
        """This method runs the multi-device back projection.  The method processes the slices
        in groups.  For each group of slices, every device that holds views back projects its
        views onto the group.  The method then sums the partial results onto the device that
        holds the group of slices."""
        if sino_shards.placement.is_trivial:
            return _sharding.Shards(
                [self.projector_functions._sparse_back_project_single_device(
                    sino_shards.tensors[0], pixel_indices,
                    coeff_power=coeff_power)],
                self.recon_placement)
        sp, rp, view_spans, band_ranges, idx_per = self._banded_setup(pixel_indices)
        pf = self.projector_functions
        aligned = self.rows_track_slices
        recon_tensors = []
        num_pixels = int(idx_per[0].shape[0])
        fixed_band = getattr(self, 'back_project_slice_band', None)
        with self._band_pool(sp.n_devices) as pool:
            for oi, (odev, (s0, s1)) in enumerate(band_ranges):
                band_len = self._slice_band_length(
                    s1 - s0, sp.n_devices, num_pixels, fixed_band)
                owner_parts = []
                for (l0, l1) in self._balanced_slice_bounds(s1 - s0, band_len):
                    # A device that holds no views is skipped.
                    if aligned:
                        partials = _sharding.run_per_device(
                            sp.devices,
                            lambda i, d: (
                                pf.sparse_back_project_view_range(
                                    sino_shards.tensors[i][
                                        :, s0 + l0:s0 + l1, :],
                                    idx_per[i], view_spans[i],
                                    coeff_power=coeff_power, dev_index=i)
                                if view_spans[i][1] > view_spans[i][0]
                                else None),
                            executor=pool)
                    else:
                        partials = _sharding.run_per_device(
                            sp.devices,
                            lambda i, d: (
                                pf.sparse_back_project_view_range(
                                    sino_shards.tensors[i],
                                    idx_per[i], view_spans[i],
                                    slice_start=s0 + l0, band_slices=l1 - l0,
                                    coeff_power=coeff_power, dev_index=i)
                                if view_spans[i][1] > view_spans[i][0]
                                else None),
                            executor=pool)
                    owner_parts.append(_sharding.sum_band_to_owner(
                        [p for p in partials if p is not None], odev,
                        self.dev2dev_safe))
                    # This release must come before the run_per_device call of the next group.
                    partials = None
                if not owner_parts:
                    # A device with no slices has no partial results, so this code sets the dtype
                    # and device of its empty output.
                    recon_tensors.append(torch.zeros(
                        (num_pixels, 0), dtype=sino_shards.dtype, device=odev))
                else:
                    recon_tensors.append(owner_parts[0] if len(owner_parts) == 1
                                         else torch.cat(owner_parts, dim=1))
        return _sharding.Shards(recon_tensors, rp)

    def _full_indices(self):
        recon_shape, use_ror_mask = self.get_params(['recon_shape', 'use_ror_mask'])
        return vcd_utils.gen_full_indices(recon_shape, use_ror_mask=use_ror_mask)

    def full_index_count(self):
        """This method returns the number of pixels that the ROR mask keeps.  The method
        caches the count for each pair (recon_shape, use_ror_mask)."""
        recon_shape, use_ror_mask = self.get_params(['recon_shape', 'use_ror_mask'])
        key = (tuple(recon_shape),
               use_ror_mask if isinstance(use_ror_mask, bool) else None)
        cache = getattr(self, '_full_index_count_cache', None)
        if key[1] is None or cache is None or cache[0] != key:
            count = int(np.shape(self._full_indices())[0])
            if key[1] is None:
                return count
            self._full_index_count_cache = (key, count)
        return self._full_index_count_cache[1]

    def full_indices_device(self):
        """This method returns the indices of all pixels inside the ROR mask, as an int64
        tensor on the model device.  The method caches the tensor for each triple
        (recon_shape, use_ror_mask, device).  The method does not cache the result for a custom
        mask array."""
        recon_shape, use_ror_mask = self.get_params(['recon_shape', 'use_ror_mask'])
        key = (tuple(recon_shape), use_ror_mask if isinstance(use_ror_mask, bool) else None,
               str(self.torch_device))
        if key[1] is None:
            return torch.as_tensor(self._full_indices(), dtype=torch.int64,
                                   device=self.torch_device)
        cache = getattr(self, '_full_indices_cache', None)
        if cache is None or cache[0] != key:
            idx = torch.as_tensor(self._full_indices(), dtype=torch.int64,
                                  device=self.torch_device)
            self._full_indices_cache = (key, idx)
        return self._full_indices_cache[1]

    def refresh_device_bindings(self):
        """This method rebuilds the assignment of views and slices to devices from the current
        parameters.  It keeps the configured devices, and it then rebuilds the projectors.
        mbirtorch calls this method after any ``set_params`` call that changes the geometry."""
        devices = self.sino_placement.devices
        sinogram_shape, recon_shape = self.get_params(
            ['sinogram_shape', 'recon_shape'])
        self.sino_placement = _sharding.Placement(
            devices, axis=0, axis_len=int(sinogram_shape[0]))
        self.recon_placement = _sharding.Placement(
            devices, axis=-1, axis_len=int(recon_shape[2]))
        self._check_no_empty_shard()
        self._invalidate_device_caches()
        if self._projector_functions is not None:
            self.create_projectors()

    def _check_no_empty_shard(self):
        """This method raises an error if the chosen devices would leave a device with no views
        and no slices."""
        sp, rp = self.sino_placement, self.recon_placement
        if sp.axis_len is None or rp.axis_len is None:
            return
        if sp.n_devices > sp.axis_len and sp.n_devices > rp.axis_len:
            raise ValueError(
                f'{sp.n_devices} devices would leave at least one device with '
                f'no views AND no slices ({sp.axis_len} views, '
                f'{rp.axis_len} slices); use at most '
                f'{max(sp.axis_len, rp.axis_len)} devices for this '
                f'geometry.')

    def _invalidate_device_caches(self):
        """This method clears every cache that depends on the devices or on the geometry."""
        self.prox_data = None
        self._dc_damping_cache = None

    # ── device configuration ──────────────────────────────────────────────────
    def configure_devices(self, num_devices=1, devices=None, like=None):
        """
        This method chooses the devices that the model computes on.

        The method takes a number of GPUs, a list of devices, or another model to match::

            ct_model.configure_devices(2)                             # the first two GPUs
            ct_model.configure_devices(devices=['cuda:0', 'cuda:2'])  # these two GPUs
            ct_model.configure_devices(devices=['cpu'])               # the CPU
            denoiser.configure_devices(like=ct_model)                 # the same devices as ct_model

        If the user does not call this method, the model uses a GPU when there is one.  On a
        machine with several GPUs, the model uses as many GPUs as speed up a problem of that
        size.  A call to this method turns off the automatic choice for this model.  For
        example, ``configure_devices(1)`` keeps a run on one GPU, which gives a reproducible
        result.  The results differ slightly with the number of GPUs.

        The argument ``like=`` lets a Plug-and-Play loop pass a volume between a reconstruction
        model and a denoiser while the volume stays on the GPUs.  The user must configure the
        other model first.

        The user must call this method again after any change to the sinogram shape or the
        reconstruction shape.

        Args:
            num_devices (int, optional): The number of GPUs.  Defaults to 1.
            devices (list, optional): The device names, such as ``['cuda:0', 'cuda:1']`` or
                ``['cpu']``.  This argument overrides ``num_devices``.
            like (TomographyModel, optional): A model whose devices this model copies.  The two
                models must have the same recon shape.
        """
        if like is not None and devices is not None:
            raise ValueError(
                'configure_devices takes like= or devices=, not both: like= '
                "copies another model's device list, and devices= names one "
                'directly.  Pass whichever one expresses the intent.')
        self.device_layout_is_automatic = False
        # The user chose the devices, so the records of the automatic choice are cleared.
        self.device_choice_rejections = []
        self._device_choice_shapes = None
        self._device_choice_workload = None
        if like is not None:
            devices = self._devices_like(like)
        if devices is None:
            if num_devices == 1:
                devices = [self.torch_device]
            else:
                if not torch.cuda.is_available() or \
                        torch.cuda.device_count() < num_devices:
                    raise ValueError(
                        f"configure_devices({num_devices}) needs {num_devices} "
                        f"CUDA devices; found "
                        f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}.")
                devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]
        self._install_device_layout(devices)

    def _devices_like(self, other):
        """This method returns the device list of ``other``.  It first checks that this model
        and ``other`` can exchange reconstruction arrays."""
        placement = getattr(other, 'recon_placement', None)
        other_get_params = getattr(other, 'get_params', None)
        if not isinstance(placement, _sharding.Placement) \
                or not getattr(placement, 'devices', None) \
                or other_get_params is None:
            raise ValueError(
                'configure_devices(like=...) copies the device list from '
                'another tomography model (a geometry model or a '
                f'QGGMRFDenoiser).  Got {type(other).__name__}, which has no '
                'device placement to copy.')
        own_shape = tuple(int(s) for s in self.get_params('recon_shape'))
        other_shape = tuple(int(s) for s in other_get_params('recon_shape'))
        if own_shape != other_shape:
            raise ValueError(
                'configure_devices(like=...) needs the two models to agree on '
                'the whole recon shape, because that is what makes a volume '
                'from one usable by the other.  The slice count alone decides '
                'how the shards are cut, but a mismatch in the rows or the '
                'columns would pass this check and then fail deep inside a '
                'reconstruction as an unreadable tensor error.  This '
                f'{type(self).__name__} has recon_shape {own_shape}, and the '
                f'{type(other).__name__} given as like= has recon_shape '
                f'{other_shape}, so recon-like arrays could not pass between '
                "them.  A QGGMRFDenoiser is built at the other model's RECON "
                "shape (ct_model.get_params('recon_shape')), not its sinogram "
                'shape.  To place two models on the same devices without '
                'exchanging arrays between them, pass devices=... instead.')
        return list(placement.devices)

    def _install_device_layout(self, devices):
        """This method assigns the views and the slices to ``devices``.  It does not change
        ``device_layout_is_automatic``."""
        devices = [torch.device(d) for d in devices]
        self.torch_device = devices[0]
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape', 'recon_shape'])
        self.sino_placement = _sharding.Placement(
            devices, axis=0, axis_len=int(sinogram_shape[0]))
        self.recon_placement = _sharding.Placement(
            devices, axis=-1, axis_len=int(recon_shape[2]))
        self._check_no_empty_shard()
        # Direct copies between devices are used only if they are correct on this hardware.
        self.dev2dev_safe = _sharding.is_dev2dev_safe(devices)
        self._invalidate_device_caches()
        if self._projector_functions is not None:
            self.create_projectors()

    # ── memory estimates and the automatic device count ───────────────────────
    def _estimate_memory(self, devices=None, workload='recon',
                         **call_arrays):
        """This method returns the estimated peak memory of each device for a list of devices.
        If ``devices`` is None, it uses the current devices.  The argument ``workload`` is
        'recon' or 'direct'."""

        devices = list(self.sino_placement.devices if devices is None
                       else devices)
        return _memory_estimate.estimate_peak_device_bytes(
            _memory_estimate.plan_from_model(self, devices, workload=workload,
                                             **call_arrays))

    def _candidate_devices(self, num_devices):
        return [torch.device(f'cuda:{i}') for i in range(num_devices)]

    def _shape_pair(self):
        """This method returns the (sinogram_shape, recon_shape) pair."""
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape',
                                                       'recon_shape'])
        return tuple(sinogram_shape), tuple(recon_shape)

    def _apply_device_policy(self, workload='recon', **call_arrays):
        """This method chooses the GPUs for the reconstruction that is about to run, and it
        returns the memory estimate for those GPUs.  It chooses the GPUs once per model, and
        again when the shapes change.  The argument ``workload`` is ``'recon'``, ``'direct'``,
        ``'forward'``, ``'back'``, or ``'denoise'``.
        """
        calibrating = _memory_estimate.calibration_enabled()
        # The devices are chosen for a full reconstruction.  A denoiser is the exception,
        # because it only denoises.
        sizing = 'denoise' if workload == 'denoise' else 'recon'
        if not self.device_layout_is_automatic:
            # When the user chose the devices, the memory estimate runs only for calibration.
            estimate = self._estimate_memory(workload=sizing,
                                             **call_arrays) \
                if calibrating else None
            return self._record_memory_estimate(estimate, sizing)

        if self._device_choice_shapes is not None:
            if self._device_choice_shapes == self._shape_pair():
                if _memory_estimate.workload_covers(self._device_choice_workload,
                                                    workload):
                    # The earlier memory check covers this call, so the same devices are reused.
                    estimate = self._estimate_memory(workload=sizing,
                                                     **call_arrays) \
                        if calibrating else None
                    return self._record_memory_estimate(estimate, sizing)
                # This call needs more memory than the earlier check assumed, so the check runs
                # again.
                estimate = self._recheck_memory_for_chosen_devices(workload, call_arrays)
                self._device_choice_workload = workload
                return self._record_memory_estimate(estimate, sizing)
            self._device_choice_shapes = None
            self._device_choice_workload = None

        env_count = _memory_estimate.env_device_count()
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible < 2:
            # There is no choice to make.  On one device, the allocator reports any lack of
            # memory.
            return self._record_memory_estimate(None, sizing)

        if env_count is not None:
            # If the environment sets the device count for the process, that count is used.
            candidates, held = [min(env_count, visible)], {}
        else:
            candidates, held = self._speed_ordered_candidates(visible)

        rejected, best = [], None
        self._size_limit_fallback = None
        self._size_limit_held = held
        for count in candidates:
            if count in held:
                # The sinogram is below the minimum size for this count.  The count is tried only
                # because larger counts did not fit in memory.
                held_note, taken_note = held[count]
                rejected.append((count, held_note))
                self._size_limit_fallback = (count, taken_note)
            devices = (self._candidate_devices(count) if count > 1
                       else [self.torch_device])
            if not self._layout_is_valid(devices):
                rejected.append((count, 'a device would own no real data'))
                continue
            # The memory estimate is for the largest task that the model may run later.
            estimate = self._estimate_memory(devices=devices,
                                             workload=sizing, **call_arrays)
            if estimate is None or self.skip_memory_check:
                # Either there is no memory limit to check, or the caller turned off the check.
                return self._use_chosen_devices(devices, estimate, rejected, sizing)
            fits, rows = self._layout_capacity(devices, estimate, call_arrays)
            if fits:
                return self._use_chosen_devices(devices, estimate, rejected, sizing)
            shortfall = max((d - b) for _dev, d, b in rows if b is not None)
            rejected.append((count, f'{shortfall / 2 ** 30:.2f} GB short'))
            if best is None or shortfall < best[0]:
                best = (shortfall, estimate, rows, count)

        if workload != sizing and best is not None:
            # No count fits the largest task.  For this reason, the counts are tried again with
            # the memory that this call needs.
            best = None
            for count in candidates:
                devices = (self._candidate_devices(count) if count > 1
                           else [self.torch_device])
                if not self._layout_is_valid(devices):
                    continue
                estimate = self._estimate_memory(
                    devices=devices, workload=workload, **call_arrays)
                fits, rows = self._layout_capacity(devices, estimate,
                                                   call_arrays)
                if fits:
                    # The first loop rejected this count for a task that this call does not run.
                    self._size_limit_fallback = None
                    rejected = [(c, why) for c, why in rejected
                                if c != count]
                    rejected.append(
                        (count, f'chosen for the {workload} reconstruction '
                                'in progress: no device count fits a full '
                                'recon at this size'))
                    return self._use_chosen_devices(devices, estimate, rejected, workload)
                shortfall = max((d - b) for _dev, d, b in rows if b is not None)
                if best is None or shortfall < best[0]:
                    best = (shortfall, estimate, rows, count)

        # No device count fits.  The call raises an error before the reconstruction starts.
        if best is None:
            raise _memory_estimate.MemoryCheckError(
                'no device layout is valid for this geometry: '
                + '; '.join(f'{c} devices ({why})' for c, why in rejected))
        _shortfall, estimate, rows, count = best
        raise _memory_estimate.MemoryCheckError(
            _memory_estimate.format_shortfall(
                estimate, rows, num_devices_tried=candidates,
                closest_count=count, remedies=self._memory_remedies()))

    def _speed_ordered_candidates(self, visible):
        """This method returns the order in which to try the device counts.  The counts for
        which the sinogram meets the minimum size in ``_min_problem_sizes`` come first, from
        largest to smallest.  The other counts come after them.

        Returns:
            (list, dict): The device counts in the order to try them, and a dict
            ``{count: (held_note, taken_note)}`` for the counts below their minimum size.
        """
        candidates = list(range(visible, 0, -1))
        if not _min_problem_sizes.size_limit_enabled():
            return candidates, {}
        elements = _min_problem_sizes.sinogram_elements(
            self.get_params('sinogram_shape'))
        family = self._size_table_key
        note = _min_problem_sizes.stale_note()
        if note is not None:
            self.logger.info('Note: ' + note + '.')
        if family is None and self.get_params('verbose') >= 2:
            self.logger.debug(
                f'  {type(self).__name__} sets no _size_table_key, so the '
                f'{_min_problem_sizes.DEFAULT_FAMILY} minimum problem sizes '
                f'apply to its automatic device count.')
        allowed_counts, held = [], {}
        for count in candidates:
            ok, why = _min_problem_sizes.allowed(family, count, elements)
            if ok:
                allowed_counts.append(count)
            else:
                held[count] = (why, _min_problem_sizes.fallback_reason(
                    family, count, elements))
        # One device is always allowed, so the list is never empty.
        return allowed_counts + list(held), held

    def _memory_remedies(self):
        """This method returns extra suggestion lines for the error message that this geometry
        gives when the reconstruction does not fit in memory."""
        if type(self).recon_split_sino is not TomographyModel.recon_split_sino:
            return ['  model.recon_split_sino(...)                '
                    '# reconstructs one row band at a time; raises',
                    '                                             '
                    '# the feasible size at a fixed device count']
        return []

    def _layout_is_valid(self, devices):
        """This method returns True if ``devices`` passes the test in
        :meth:`_check_no_empty_shard`."""
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape',
                                                       'recon_shape'])
        return len(devices) <= max(int(sinogram_shape[0]), int(recon_shape[2]))

    def _layout_capacity(self, devices, estimate, call_arrays):
        """This method returns True if the estimated peak memory in ``estimate`` fits on
        ``devices``.  It also returns the table rows that an error message uses to report the
        result."""
        budgets = [_memory_estimate.device_budget_bytes(d) for d in devices]
        bytes_already_on_devices = _memory_estimate.bytes_already_on_devices(
            devices, list(call_arrays.values()))
        return _memory_estimate.layout_fits(
            estimate, budgets, bytes_already_on_devices, margin=self.memory_check_margin)

    def _fits_available_devices(self, workload='recon', **call_arrays):
        """This method returns True if the estimated peak memory for ``workload`` fits on some
        set of devices that this model could use.  The method treats a device other than a CUDA
        device as having unlimited memory.
        """
        if self.skip_memory_check:
            return True
        if not self.device_layout_is_automatic:
            candidates = [list(self.sino_placement.devices)]
        else:
            env_count = _memory_estimate.env_device_count()
            visible = (torch.cuda.device_count() if torch.cuda.is_available()
                       else 0)
            if visible < 2:
                counts = [1]
            elif env_count is not None:
                counts = [min(env_count, visible)]
            else:
                counts = list(range(visible, 0, -1))
            candidates = [self._candidate_devices(count) if count > 1
                          else [self.torch_device] for count in counts]
        for devices in candidates:
            if not self._layout_is_valid(devices):
                continue
            estimate = self._estimate_memory(devices=devices,
                                             workload=workload,
                                             **call_arrays)
            if estimate is None:
                return True
            fits, _rows = self._layout_capacity(devices, estimate, call_arrays)
            if fits:
                return True
        return False

    def _recheck_memory_for_chosen_devices(self, workload, call_arrays):
        """This method runs the memory check for ``workload`` on the chosen GPUs, and it
        returns the memory estimate.  The method does not change the devices.
        """
        if self.skip_memory_check:
            # The caller has turned off the memory check.
            return None
        devices = list(self.sino_placement.devices)
        estimate = self._estimate_memory(devices=devices, workload=workload,
                                         **call_arrays)
        fits, rows = self._layout_capacity(devices, estimate, call_arrays)
        if fits:
            return estimate
        raise _memory_estimate.MemoryCheckError(
            _memory_estimate.format_shortfall(
                estimate, rows, num_devices_tried=[len(devices)],
                closest_count=len(devices),
                remedies=self._memory_remedies()))

    def _use_chosen_devices(self, devices, estimate, rejected, workload='recon'):
        """This method switches the model to the chosen devices if they differ from the current
        devices.  It also logs the choice and records the memory estimate."""
        chosen, current = len(devices), self.sino_placement.n_devices
        # If the chosen count is below its minimum sinogram size, the log says that memory decided
        # the choice.
        fallback = getattr(self, '_size_limit_fallback', None)
        if fallback is not None and fallback[0] == chosen:
            rejected = [fallback if count == chosen else (count, why)
                        for count, why in rejected]
        # The log names the larger counts that were not used, to explain idle GPUs.
        held = getattr(self, '_size_limit_held', None) or {}
        rejected = list(rejected)
        already = {count for count, _why in rejected}
        for count in sorted(held, reverse=True):
            if count > chosen and count not in already:
                rejected.append((count, held[count][0]))
        self._size_limit_fallback = None
        self._size_limit_held = None
        # The run log reads this to explain any idle GPUs.
        self.device_choice_rejections = list(rejected)
        if chosen != current:
            self.logger.info(
                f'Using {chosen} CUDA device(s) for this reconstruction '
                f'(was {current}).  configure_devices(num_devices=n) fixes it.')
            self._install_device_layout(devices)
        if rejected and self.get_params('verbose') >= 2:
            for count, why in rejected:
                self.logger.debug(f'  device count {count} rejected: {why}')
        # Later calls reuse these devices while the shapes stay the same.
        self._device_choice_shapes = self._shape_pair()
        self._device_choice_workload = workload
        return self._record_memory_estimate(estimate, workload)

    def _record_memory_estimate(self, estimate, workload='recon'):
        """This method stores the memory estimate for the tests to read.  In calibration mode,
        it computes an estimate if the caller has none."""
        if estimate is not None:
            self.last_memory_estimate = estimate
        if _memory_estimate.calibration_enabled() and estimate is None:
            estimate = self._estimate_memory(workload=workload)
            self.last_memory_estimate = estimate
        return estimate

    # ── deprecated attribute names ────────────────────────────────────────────
    @property
    def skip_memory_preflight(self):
        """This property is the deprecated name of ``skip_memory_check``."""
        _warn_deprecated_name('skip_memory_preflight', 'skip_memory_check')
        return self.skip_memory_check

    @skip_memory_preflight.setter
    def skip_memory_preflight(self, value):
        _warn_deprecated_name('skip_memory_preflight', 'skip_memory_check')
        self.skip_memory_check = value

    @property
    def memory_preflight_margin(self):
        """This property is the deprecated name of ``memory_check_margin``."""
        _warn_deprecated_name('memory_preflight_margin', 'memory_check_margin')
        return self.memory_check_margin

    @memory_preflight_margin.setter
    def memory_preflight_margin(self, value):
        _warn_deprecated_name('memory_preflight_margin', 'memory_check_margin')
        self.memory_check_margin = value

    @property
    def last_memory_ledger(self):
        """This property is the deprecated name of ``last_memory_estimate``."""
        _warn_deprecated_name('last_memory_ledger', 'last_memory_estimate')
        return self.last_memory_estimate

    def _shard_sinogram(self, sinogram):
        """This method copies a sinogram or a weights array to the devices as float32.  On
        several devices, the method splits the array by view."""
        num_views = self.get_params('sinogram_shape')[0]
        if isinstance(sinogram, _sharding.Shards):
            if sinogram.placement != self.sino_placement:
                raise ValueError(
                    'Sinogram shards belong to a different device '
                    'configuration: the shards are placed as '
                    f'{sinogram.placement}, and this model uses '
                    f'{self.sino_placement}; re-place the array.')
            return sinogram
        if self.sino_placement.is_trivial:
            sinogram = torch.as_tensor(sinogram, dtype=torch.float32,
                                       device=self.torch_device)
            if sinogram.shape[0] != num_views:
                raise ValueError(
                    'Cannot place the sinogram: its view axis has size '
                    f'{sinogram.shape[0]}, but the model expects {num_views} views.')
            return sinogram
        return self._split_to_shards(sinogram, self.sino_placement, num_views,
                                     what='sinogram (view axis)')

    # This is True when detector row r maps to reconstruction slice r, as in parallel beam.
    rows_track_slices = False

    # This is the smallest number of pixels that the compiled projection functions accept.
    # Calls with fewer pixels are padded before the call.
    min_compiled_pixel_width = 1

    # This names the model type in the table of minimum sinogram sizes per device count.  None
    # means parallel beam.
    _size_table_key = None

    def prepare_sino_for_devices(self, sinogram, weights=None):
        """
        This method copies a sinogram, and optionally its weights, to the devices of the model.

        Every reconstruction method copies the sinogram by itself.  This method is useful when
        the user reconstructs one large sinogram several times.  The user passes the result of
        this method to :meth:`recon` or :meth:`prox_map` in place of the sinogram.  In this way,
        the copy to the GPUs happens only once.  The user must call this method again after any
        change to the devices.

        Args:
            sinogram (numpy or tensor): The sinogram, with the shape ``sinogram_shape`` of the
                model.
            weights (numpy or tensor, optional): The weights, with the same shape.

        Returns:
            The prepared sinogram, or the tuple (sinogram, weights) if weights were given.
        """
        # The GPUs are chosen before the sinogram is placed on the devices.
        self._apply_device_policy(workload='direct')
        sino = self._shard_sinogram(sinogram)
        if weights is None:
            return sino
        return sino, self._shard_sinogram(weights)

    def _shard_recon(self, recon):
        """This method copies a reconstruction array to the devices as float32.  The array is
        either 3D or flat with shape (num_pixels, num_slices).  On several devices, the method
        splits the array by slice."""
        num_slices = self.get_params('recon_shape')[2]
        if isinstance(recon, _sharding.Shards):
            # Placements compare by value, so the method accepts shards from another model on the
            # same devices.
            if recon.placement != self.recon_placement:
                raise ValueError(
                    'Recon shards belong to a different device '
                    'configuration: the shards are placed as '
                    f'{recon.placement}, and this model uses '
                    f'{self.recon_placement}; re-place the array.')
            return recon
        if self.recon_placement.is_trivial:
            recon = torch.as_tensor(recon, dtype=torch.float32,
                                    device=self.torch_device)
            if recon.shape[-1] != num_slices:
                raise ValueError(
                    'Cannot place the reconstruction: its slice axis has size '
                    f'{recon.shape[-1]}, but the model expects {num_slices} slices.')
            return recon
        return self._split_to_shards(recon, self.recon_placement, num_slices,
                                     what='reconstruction (slice axis)')

    def _split_to_shards(self, x, placement, axis_len, what='array'):
        """This method splits an array into one contiguous block per device.  The lengths of
        the blocks differ by at most one."""
        x = torch.as_tensor(x, dtype=torch.float32)
        axis = placement.axis % x.ndim
        if x.shape[axis] != axis_len:
            raise ValueError(
                f'Cannot place the {what}: got shape {tuple(x.shape)}, '
                f'but the model expects size {axis_len} on axis {axis}.')
        tensors = []
        for dev, (start, end) in placement.shard_ranges(axis_len):
            idx = [slice(None)] * x.ndim
            idx[axis] = slice(start, end)
            tensors.append(x[tuple(idx)].to(dev))
        return _sharding.Shards(tensors, placement)

    def _gather_sinogram(self, sinogram):
        """This method returns a sinogram-shaped array as a numpy array in host memory.  It
        concatenates the shards along the view axis."""
        if isinstance(sinogram, _sharding.Shards):
            out = self._gather_shards(sinogram)
        else:
            out = sinogram.detach().cpu().numpy()
        return out

    def _gather_recon(self, recon):
        """This method returns a recon-shaped array as a numpy array in host memory.  It
        concatenates the shards along the slice axis."""
        if isinstance(recon, _sharding.Shards):
            out = self._gather_shards(recon)
        else:
            out = recon.detach().cpu().numpy()
        return out

    def _constant_recon(self, value):
        """This method returns a reconstruction with a constant value on the devices.  It builds
        the reconstruction one shard at a time."""
        recon_shape = self.get_params('recon_shape')
        if self.recon_placement.is_trivial:
            recon = torch.full(tuple(recon_shape), float(value),
                               dtype=torch.float32, device=self.torch_device)
        else:
            tensors = [
                torch.full(tuple(recon_shape[:2]) + (e - s,), float(value),
                           dtype=torch.float32, device=d)
                for d, (s, e) in self.recon_placement.shard_ranges()]
            recon = _sharding.Shards(tensors, self.recon_placement)
        return recon

    def _initial_error_state(self, sinogram, init_recon, weights,
                             constant_weights, scale_recon_to_sinogram):
        """This method returns the initial pair (error_sinogram, init_recon).  Both are scaled
        by the scale factor alpha that best fits the projection to the sinogram."""
        self.logger.info('Initializing error sinogram')
        fwd = self._forward_project(init_recon, output_sharded=True)
        if isinstance(fwd, _sharding.Shards):
            def dots_worker(i, d):
                # The sums run over one block of views at a time, so no full array of products
                # is formed.
                f = fwd.tensors[i]
                w = None if constant_weights else weights.tensors[i]
                return (float(_memory_estimate.weighted_square_sum(f, w)),
                        float(_memory_estimate.weighted_dot(
                            f, sinogram.tensors[i], w)))
            dots = _sharding.run_per_device(self.sino_placement.devices,
                                            dots_worker)
            wtd_err_sino_norm = sum(a for a, _ in dots)
            if wtd_err_sino_norm > 0 and scale_recon_to_sinogram:
                alpha = sum(b for _, b in dots) / wtd_err_sino_norm
            else:
                alpha = 1
            # The error sinogram, sinogram - alpha * fwd, is computed in place in the shards of
            # the projection.
            error_sinogram = _sharding.Shards(
                _sharding.run_per_device(
                    self.sino_placement.devices,
                    lambda i, d: fwd.tensors[i].mul_(-alpha).add_(
                        sinogram.tensors[i])),
                self.sino_placement)
            fwd = None
            init_recon = _sharding.Shards(
                [alpha * t for t in init_recon.tensors], self.recon_placement)
        else:
            w = None if constant_weights else weights
            wtd_err_sino_norm = _memory_estimate.weighted_square_sum(fwd, w)
            if wtd_err_sino_norm > 0 and scale_recon_to_sinogram:
                alpha = (_memory_estimate.weighted_dot(fwd, sinogram, w)
                         / wtd_err_sino_norm).item()
            else:
                alpha = 1
            error_sinogram = fwd.mul_(-alpha).add_(sinogram)
            fwd = None
            init_recon = alpha * init_recon
        return error_sinogram, init_recon

    def _flatten_recon(self, recon):
        """This method returns the reconstruction as a contiguous flat array with shape
        (num_pixels, slices), which is the form that the VCD loop uses."""
        if isinstance(recon, _sharding.Shards):
            # reshape cannot infer the number of rows of a shard that holds no slices.
            flat = _sharding.Shards(
                [t.reshape((math.prod(t.shape[:-1]), t.shape[-1])).contiguous()
                 for t in recon.tensors], recon.placement)
        else:
            flat = self._shard_recon(
                recon.reshape((-1, recon.shape[-1]))).contiguous()
        return flat

    def _flatten_prox_shards(self, prox_input, recon_shape):
        """This method converts a prox input that is already on the devices to the flat form
        (num_pixels, local_slices) that the VCD loop uses.  It does not copy the data through
        host memory.  It first checks that the shards cover the full pixel grid and all slices.
        """
        rows, cols, num_slices = (int(recon_shape[0]), int(recon_shape[1]),
                                  int(recon_shape[2]))
        num_pixels = rows * cols
        tensors = prox_input.tensors
        covers_grid = True
        total_slices = 0
        for t in tensors:
            if t.ndim == 3:
                pixels = int(t.shape[0]) * int(t.shape[1])
            elif t.ndim == 2:
                pixels = int(t.shape[0])
            else:
                covers_grid = False
                break
            total_slices += int(t.shape[-1])
            covers_grid = covers_grid and pixels == num_pixels
        if not covers_grid or total_slices != num_slices:
            raise ValueError(
                'prox_input does not have the correct size. \n'
                f'Expected shards covering {tuple(recon_shape)}: each shard '
                f'({rows}, {cols}, local_slices) or ({num_pixels}, '
                'local_slices), with the local slice counts summing to '
                f'{num_slices}.  Got shapes '
                f'{[tuple(t.shape) for t in tensors]} for prox_input.')
        # _shard_recon raises an error for shards from a model on different devices.
        return self._shard_recon(_sharding.Shards(
            [t.reshape(num_pixels, t.shape[-1]) for t in tensors],
            prox_input.placement))

    def _check_sinogram_shards(self, sinogram, sinogram_shape):
        """This method checks that a sinogram that is already on the devices covers the whole
        sinogram of the model.  The method checks only the shapes.
        """
        views, rows, channels = (int(sinogram_shape[0]), int(sinogram_shape[1]),
                                 int(sinogram_shape[2]))
        tensors = sinogram.tensors
        total_views = 0
        detector_matches = True
        for tensor in tensors:
            if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (rows, channels):
                detector_matches = False
                break
            total_views += int(tensor.shape[0])
        if not detector_matches or total_views != views:
            raise ValueError(
                'sinogram does not have the shape in sinogram_shape. \n'
                f'Expected shards covering {(views, rows, channels)}: each '
                f'shard (local_views, {rows}, {channels}), with the local view '
                f'counts summing to {views}.  Got shapes '
                f'{[tuple(t.shape) for t in tensors]}.')

    def _flatten_hessian(self, fm_hessian):
        """This method returns the Hessian diagonal in the flat form that the VCD loop uses."""
        if isinstance(fm_hessian, _sharding.Shards):
            flat = _sharding.Shards(
                [t.reshape((math.prod(t.shape[:-1]), t.shape[-1]))
                 for t in fm_hessian.tensors], fm_hessian.placement)
        else:
            flat = fm_hessian.reshape((-1, fm_hessian.shape[-1]))
        return flat

    def _recon_from_flat(self, flat_recon, recon_shape):
        """This method converts a reconstruction from the flat form that the VCD loop uses to a
        3D array."""
        if isinstance(flat_recon, _sharding.Shards):
            recon = _sharding.Shards(
                [t.reshape(tuple(recon_shape[:2]) + (t.shape[-1],))
                 for t in flat_recon.tensors], flat_recon.placement)
        else:
            recon = flat_recon.reshape(tuple(recon_shape[:2])
                                       + (flat_recon.shape[-1],))
        return recon

    def _gather_shards(self, shards):
        return shards.gather()

    def _as_shards(self, x, placement):
        """This method returns an array on the devices as a Shards container.  If the array is
        a plain tensor, the method wraps it in a container with one shard that shares its
        memory."""
        if isinstance(x, _sharding.Shards):
            shards = x
        else:
            shards = _sharding.Shards([x], placement)
        return shards

    def _as_device_form(self, x):
        """This method reverses :meth:`_as_shards`.  For a container with one shard, it returns
        the tensor in the container.  It returns a container with several shards unchanged."""
        if isinstance(x, _sharding.Shards) and x.placement.is_trivial:
            out = x.tensors[0]
        else:
            out = x
        return out

    def _sino_ones_device_form(self, sino_like=None):
        """This method returns a sinogram of all ones on the devices.  The sinogram has the
        dtype of ``sino_like``."""
        dtype = torch.float32 if sino_like is None else sino_like.dtype
        if self.sino_placement.is_trivial:
            return torch.ones(tuple(self.get_params('sinogram_shape')),
                              dtype=dtype, device=self.torch_device)
        shape = list(self.get_params('sinogram_shape'))
        tensors = []
        for dev, (start, end) in self.sino_placement.shard_ranges():
            local = list(shape)
            local[0] = end - start
            tensors.append(torch.ones(local, dtype=dtype, device=dev))
        return _sharding.Shards(tensors, self.sino_placement)

    def forward_project(self, recon, output_sharded=False):
        """
        This method performs a full forward projection of a volume to a sinogram.  If the
        parameter ``use_ror_mask`` is True (the default), the projection uses only the pixels
        inside the region-of-reconstruction mask.  If ``use_ror_mask`` is False, the projection uses
        every pixel.

        Args:
            recon (numpy or tensor): The 3D volume, with shape
                (num_recon_rows, num_recon_cols, num_recon_slices).
            output_sharded (bool, optional): If False (the default), the method returns a numpy
                array.  If True, the method returns the result on the devices.  On a model with
                one device, the result is a torch tensor.  On a model with several devices, the
                result is a Shards container with one tensor per device.

        Returns:
            The sinogram, with shape (num_views, num_det_rows, num_det_channels).

        Note:
            If the user has not chosen the devices of the model, this call chooses the GPUs
            and checks the memory that the projection needs, as ``recon_direct`` does.
        """
        # The GPUs are chosen before the volume is copied to the devices.
        self._apply_device_policy(workload='forward')
        return self._forward_project(recon, output_sharded=output_sharded)

    def _forward_project(self, recon, output_sharded=False):
        """This method performs the forward projection of :meth:`forward_project` on the
        current devices.  It does not choose GPUs or check memory.  Code inside mbirtorch calls
        this method."""
        recon_shape = self.get_params('recon_shape')
        recon = self._shard_recon(recon)
        indices = self.full_indices_device()
        if isinstance(recon, _sharding.Shards):
            # reshape cannot infer the number of rows of a shard that holds no slices.
            num_pixels = int(recon_shape[0]) * int(recon_shape[1])
            voxel_values = _sharding.Shards(
                [t.reshape(num_pixels, t.shape[-1])[indices.to(t.device)]
                 for t in recon.tensors], recon.placement)
        else:
            voxel_values = recon.reshape(-1, recon.shape[-1])[indices]
        # The projection reads only voxel_values, which is a copy.  When the input is a host
        # array, this line frees the volume on the devices.
        del recon
        sinogram = self.sparse_forward_project(voxel_values, indices)
        return sinogram if output_sharded else self._gather_sinogram(sinogram)

    def back_project(self, sinogram, output_sharded=False):
        """
        This method performs a full back projection of a sinogram to a volume.  If the
        parameter ``use_ror_mask`` is True (the default), the result is zero outside the
        region-of-reconstruction mask.  If ``use_ror_mask`` is False, the method computes every
        pixel.

        Args:
            sinogram (numpy or tensor): The 3D sinogram, with shape
                (num_views, num_det_rows, num_det_channels).
            output_sharded (bool, optional): If False (the default), the method returns a numpy
                array.  If True, the method returns the result on the devices.  On a model with
                one device, the result is a torch tensor.  On a model with several devices, the
                result is a Shards container with one tensor per device.

        Returns:
            The back projection, with shape (num_recon_rows, num_recon_cols,
            num_recon_slices).

        Note:
            If the user has not chosen the devices of the model, this call chooses the GPUs
            and checks the memory that the projection needs, as ``recon_direct`` does.
        """
        # The GPUs are chosen before the sinogram is copied to the devices.
        self._apply_device_policy(workload='back')
        return self._back_project(sinogram, output_sharded=output_sharded)

    def _back_project(self, sinogram, output_sharded=False):
        """This method performs the back projection of :meth:`back_project` on the current
        devices.  It does not choose GPUs or check memory.  Code inside mbirtorch calls this
        method."""
        recon_shape = self.get_params('recon_shape')
        sinogram = self._shard_sinogram(sinogram)
        indices = self.full_indices_device()
        cylinders = self.sparse_back_project(sinogram, indices)
        if isinstance(cylinders, _sharding.Shards):
            def scatter_worker(i, d):
                c = cylinders.tensors[i]
                r = torch.zeros((recon_shape[0] * recon_shape[1], c.shape[-1]),
                                dtype=torch.float32, device=d)
                r[indices.to(d)] = c
                return r.reshape(tuple(recon_shape[:2]) + (c.shape[-1],))
            recon = _sharding.Shards(
                _sharding.run_per_device(cylinders.placement.devices,
                                         scatter_worker), cylinders.placement)
        else:
            recon = torch.zeros((recon_shape[0] * recon_shape[1], cylinders.shape[-1]),
                                dtype=torch.float32, device=self.torch_device)
            recon[indices] = cylinders
            recon = recon.reshape(tuple(recon_shape[:2]) + (cylinders.shape[-1],))
        return recon if output_sharded else self._gather_recon(recon)

    def project_points(self, points_xyz, view_index):
        """This method maps points in the object to fractional detector indices, for one view
        or for several views.  It uses the same geometry as the projectors.

        The points are given in the object frame.  In this frame, voxel (i, j, k) of the
        reconstruction has its center at

            x = delta_voxel * (j - (num_cols - 1) / 2)
            y = voxel_row_aspect * delta_voxel * (i - (num_rows - 1) / 2)
            z = voxel_slice_aspect * delta_voxel * (k - (num_slices - 1) / 2) + recon_slice_offset

        The z axis is the rotation axis.  For each view, the method applies the motion of the
        object in that view.  This motion is a rotation about z, a helical shift in z, or a
        translation.

        The integer index m is the center of detector row m, and the index -0.5 is the outer
        edge of row 0.  A point outside the detector gets an index outside
        ``[-0.5, num - 0.5]``.  The method does not clip the indices.

        Args:
            points_xyz (array_like): The points in ALU, as an (N, 3) array of (x, y, z), or one
                point as a (3,) array.
            view_index (int or sequence of int): One view index, or several view indices.

        Returns:
            tuple of ndarray: The pair ``(row, channel)`` as float64 arrays.  For one view, each
            array has shape (N,).  For a sequence of views, each array has shape
            (num_selected_views, N).

        Raises:
            ValueError: If ``points_xyz`` does not have shape (N, 3) or (3,), or if a view index
                is not an integer.
            IndexError: If a view index is outside ``[0, num_views)``.

        Example:
            >>> row, channel = model.project_points([[0.0, 0.0, 0.0]], 0)
        """
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.ndim == 1 and points.shape == (3,):
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points_xyz must have shape (N, 3) or (3,); '
                             f'got {points.shape}.')
        single_view = np.ndim(view_index) == 0
        indices = np.atleast_1d(np.asarray(view_index))
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('view_index must be an integer or a sequence of '
                             f'integers; got {view_index!r}.')
        num_views = int(self.get_params('sinogram_shape')[0])
        if indices.size and (indices.min() < 0 or indices.max() >= num_views):
            raise IndexError(f'view_index {view_index!r} is outside '
                             f'[0, {num_views}).')
        view_params = np.asarray(self.get_params(self.get_params('view_params_name')))[indices]
        # The computation runs in float64 on the CPU.
        row, channel = self._project_points_batch(
            torch.as_tensor(points, dtype=torch.float64),
            torch.as_tensor(view_params, dtype=torch.float64))
        row = np.ascontiguousarray(row.numpy())
        channel = np.ascontiguousarray(channel.numpy())
        if single_view:
            return row[0], channel[0]
        return row, channel

    def _project_points_batch(self, points, view_params):
        """This method computes the part of :meth:`project_points` that depends on the
        geometry.

        Args:
            points: A float64 CPU tensor with shape (N, 3) that holds points in the object frame.
            view_params: A float64 CPU tensor that holds the selected rows of the per-view
                parameter array.  Its shape is (V,) for parallel beam, (V, 2) for cone beam and
                multiaxis, and (V, 3) for translation.

        Returns:
            (row, channel): Two float64 CPU tensors, each with shape (V, N).
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not implement project_points.')

    def compute_hessian_diagonal(self, weights=None, output_sharded=False,
                                 indices=None):
        """
        This method computes the diagonal of the Hessian matrix.  The diagonal is the back
        projection of the weights with squared projector coefficients.

        Args:
            weights (numpy or tensor, optional): The 3D positive weights, with the same shape as
                the sinogram.  Defaults to all ones.
            output_sharded (bool, optional): If False (the default), the method returns a numpy
                array.  If True, the method returns the result on the devices.
            indices (tensor, optional): The flat pixel indices at which to back project.  The
                entries outside this set are zero.  If None (the default), the method uses all
                pixels of the grid.

        Returns:
            The diagonal of the Hessian matrix, with the same shape as the recon.

        Note:
            If the user has not chosen the devices of the model, this call also chooses the
            GPUs.  Every later reconstruction on the model reuses these GPUs.
        """
        # The GPUs must be chosen before the full weights and volume are built.
        self._apply_device_policy(workload='direct')
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape', 'recon_shape'])
        if weights is None:
            weights = self._sino_ones_device_form()
        elif (not isinstance(weights, _sharding.Shards)
              and tuple(weights.shape) != tuple(sinogram_shape)):
            raise ValueError('Weights must be constant or an array compatible with sinogram'
                             f'\nGot weights.shape = {tuple(weights.shape)}, but '
                             f'sinogram.shape = {tuple(sinogram_shape)}')
        else:
            weights = self._shard_sinogram(weights)
        num_grid = int(recon_shape[0] * recon_shape[1])
        dense = indices is None
        if dense:
            indices = torch.arange(num_grid, dtype=torch.int64,
                                   device=self.torch_device)
        else:
            indices = torch.as_tensor(indices, dtype=torch.int64,
                                      device=self.torch_device)
        hessian = self.sparse_back_project(weights, indices, coeff_power=2)

        # A back projection at all pixels is reshaped.  A back projection at selected pixels is
        # copied into a volume of zeros.
        def to_volume(cylinders, device):
            if dense:
                return cylinders.reshape((recon_shape[0], recon_shape[1],
                                          cylinders.shape[-1]))
            volume = torch.zeros((num_grid, cylinders.shape[-1]),
                                 dtype=cylinders.dtype, device=device)
            volume.index_copy_(0, indices.to(device), cylinders)
            return volume.reshape((recon_shape[0], recon_shape[1],
                                   cylinders.shape[-1]))

        if isinstance(hessian, _sharding.Shards):
            hessian = _sharding.Shards(
                [to_volume(t, d) for t, d in zip(hessian.tensors,
                                                 hessian.placement.devices)],
                hessian.placement)
        else:
            hessian = to_volume(hessian, hessian.device)
        return hessian if output_sharded else self._gather_recon(hessian)

    def get_voxels_at_indices(self, recon, indices):
        """This method returns the voxel cylinders at ``indices`` as an array with shape
        (num_indices, num_slices).  If the recon is split across devices, the result is split
        in the same way.
        """
        if isinstance(recon, _sharding.Shards):
            recon_shape = self.get_params('recon_shape')
            # reshape cannot infer the number of rows of a shard that holds no slices.
            num_pixels = int(recon_shape[0]) * int(recon_shape[1])
            indices = torch.as_tensor(indices, dtype=torch.int64)
            return _sharding.Shards(
                [t.reshape(num_pixels, t.shape[-1])[indices.to(t.device)]
                 for t in recon.tensors], recon.placement)
        return recon.reshape((-1, recon.shape[-1]))[indices]

    # ── auto-regularization ───────────────────────────────────────────────────
    def auto_set_regularization_params(self, sinogram, weights=None):
        """
        This method sets the regularization parameters sigma_y, sigma_x, and sigma_prox from
        the sinogram and the optional weights.  It returns the parameters as a dict.  The method
        changes the parameters only if ``auto_regularize_flag`` is True.
        """
        if torch.is_tensor(sinogram):
            sinogram = sinogram.cpu().numpy()
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()
        if self.get_params('auto_regularize_flag'):
            small_sinogram = self.subsample_views(sinogram)
            small_weights = 1 if weights is None else self.subsample_views(weights)

            sino_indicator = self._get_sino_indicator(small_sinogram,
                                                      verbose=self.get_params('verbose'))
            self._check_lateral_truncation(sino_indicator)
            self.auto_set_sigma_y(small_sinogram, sino_indicator, small_weights)

            recon_std = self._get_estimate_of_recon_std(small_sinogram, sino_indicator)
            self.auto_set_sigma_x(recon_std)
            self.auto_set_sigma_prox(recon_std)

        values = [float(v) for v in self.get_params(list(_AUTO_REGULARIZATION_PARAM_NAMES))]
        return dict(zip(_AUTO_REGULARIZATION_PARAM_NAMES, values))

    def _check_lateral_truncation(self, sino_indicator):
        """This method warns if the sinogram support reaches the edge channels of the
        detector.  The support is the indicator from :meth:`_get_sino_indicator`."""
        if np.all(sino_indicator):
            # An indicator of all ones means either that the background was not found or that
            # the support is the whole sinogram.
            return
        edge_frac = float(np.mean(np.logical_or(sino_indicator[:, :, 0],
                                                sino_indicator[:, :, -1])))
        if edge_frac > 0.02 and self.get_params('verbose') > 0:
            warnings.warn(
                f"Lateral FOV truncation detected: the object support reaches the detector's "
                f"edge channels in {edge_frac:.0%} of the sampled view-rows.  Consider using "
                f"resize_recon_fov(s, s) where s >= 1.1 to improve image quality.")

    def auto_set_sigma_y(self, sinogram, sino_indicator, weights=1):
        """This method sets sigma_y from the sinogram, its support indicator, and the optional
        weights.  The sinogram is typically subsampled in views.  The method does not accept a
        sharded sinogram.  The method :meth:`subsample_views` converts a sharded sinogram to a
        host array.
        """
        _sharding.reject_shards('auto_set_sigma_y', sinogram=sinogram,
                                sino_indicator=sino_indicator, weights=weights)
        snr_db = self.get_params('snr_db')
        magnification = self.get_magnification()
        delta_voxel, delta_det_channel = self.get_params(['delta_voxel', 'delta_det_channel'])

        signal_rms = float(np.average(weights * np.asarray(sinogram) ** 2, None,
                                      sino_indicator) ** 0.5)

        rel_noise_std = 10 ** (-snr_db / 20)

        # The default pixel pitch is the detector pitch divided by the magnification.
        default_pixel_pitch = delta_det_channel / magnification
        pixel_pitch_relative_to_default = delta_voxel / default_pixel_pitch

        sigma_y = np.float32(rel_noise_std * signal_rms *
                             (pixel_pitch_relative_to_default ** 0.5))
        self.set_params(no_warning=True, sigma_y=float(sigma_y), auto_regularize_flag=True)

    def auto_set_sigma_x(self, recon_std):
        """This method sets sigma_x, the scale of the qGGMRF prior, from the estimated
        standard deviation of the recon."""
        sharpness = self.get_params('sharpness')
        # The constant 0.2 is empirical.
        sigma_x = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_x=float(sigma_x), auto_regularize_flag=True)

    def auto_set_sigma_prox(self, recon_std):
        """This method sets sigma_prox, the scale of the proximal map prior, from the estimated
        standard deviation of the recon."""
        sharpness = self.get_params('sharpness')
        # The constant 0.2 is empirical.
        sigma_prox = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_prox=float(sigma_prox),
                        auto_regularize_flag=True)

    @staticmethod
    def subsample_views(array, max_views_to_use=20):
        """This method returns about ``max_views_to_use`` evenly spaced views (axis 0) as a
        numpy array in host memory.  The stride depends only on the number of views.  As a
        result, a second call with the same arguments subsamples a matching array, such as the
        weights, in the same way.  For a sharded sinogram, the method subsamples each shard on
        its device.  The result is the same as for the full sinogram.
        """
        if isinstance(array, _sharding.Shards):
            placement = array.placement
            # The sharded axis may be negative, so the code converts it with the number of
            # dimensions.
            if placement.axis % array.tensors[0].ndim != 0:
                raise ValueError(
                    'A sinogram must be sharded on its first (view) axis; got '
                    f'a placement on axis {placement.axis}.')
            num_views = sum(int(t.shape[0]) for t in array.tensors)
            axis_len = placement.axis_len
            if axis_len is not None and int(axis_len) != num_views:
                raise ValueError(
                    f'The shards cover {num_views} views, but their placement '
                    f'says the view axis is {int(axis_len)} long.')
            max_views_to_use = min(max_views_to_use, num_views)
            step_size = max(num_views // max_views_to_use, 1)
            blocks = []
            for tensor, (_dev, (start, _end)) in zip(
                    array.tensors, placement.shard_ranges(num_views)):
                block = tensor[(-start) % step_size::step_size]
                blocks.append(block.detach().contiguous().cpu().numpy())
            return np.concatenate(blocks, axis=0)
        num_views = array.shape[0]
        max_views_to_use = min(max_views_to_use, num_views)
        step_size = max(num_views // max_views_to_use, 1)
        return np.array(array[::step_size])

    @staticmethod
    def _get_sino_indicator(sinogram, verbose=1):
        """This method returns an int8 mask of the sinogram support.  The mask has the same
        shape as the input.  The support is the set of entries that the object attenuates."""
        # The negative logarithm of a sinogram can give complex values, so the input is checked.
        sinogram = np.asarray(sinogram)
        if np.iscomplexobj(sinogram):
            raise TypeError("sinogram must be real-valued; got complex dtype.")
        if not np.isfinite(sinogram).all():
            raise ValueError("sinogram contains NaN and/or Inf values.")

        left, right = vcd_utils.estimate_background_cluster_boundaries(sinogram)
        threshold = right + (right - left)

        max_sino = np.max(sinogram)
        if max_sino <= 0:
            if verbose > 0:
                warnings.warn("Sinogram contains no positive values. This may lead to a "
                              "contrast reversed reconstruction.")
            return np.ones_like(sinogram, dtype=np.int8)
        if max_sino < threshold:
            if verbose > 0:
                warnings.warn('\nUnable to determine sinogram background. This may affect '
                              'regularization.\n')
            return np.ones_like(sinogram, dtype=np.int8)

        object_level = 0.25
        object_median = np.median(sinogram[sinogram >= threshold])
        object_threshold = object_level * object_median
        return np.int8(sinogram >= object_threshold)

    def _get_estimate_of_recon_std(self, sinogram, sino_indicator):
        """This method estimates the standard deviation of the reconstruction from the
        sinogram and its support indicator."""
        delta_det_channel = self.get_params('delta_det_channel')
        delta_voxel = self.get_params('delta_voxel')
        recon_shape = self.get_params('recon_shape')
        magnification = self.get_magnification()
        num_det_channels = sinogram.shape[-1]

        typical_sinogram_value = np.average(np.abs(sinogram), weights=sino_indicator)

        typical_path_length_space = (2 * recon_shape[0] * recon_shape[1]) / (
                recon_shape[0] + recon_shape[1]) * delta_voxel

        typical_path_length_sino = num_det_channels * delta_det_channel / magnification

        typical_path_length = np.minimum(typical_path_length_space, typical_path_length_sino)

        return typical_sinogram_value / typical_path_length

    # ── direct recon (FBP) machinery ──────────────────────────────────────────
    def _apply_direct_recon_filter(self, sinogram, filter_name, filter_scale,
                                   output_sharded=False, row_weight=None):
        """This method applies the row filter of filtered back projection for a direct
        reconstruction.  The filter is scaled by ``filter_scale * pi / num_views``.

        Args:
            sinogram: The sinogram, as a numpy array or tensor with shape
                (num_views, num_rows, num_channels).
            filter_name (str): The name of the filter for ``generate_direct_recon_filter``.
            filter_scale (float): The filter scale factor of the geometry.
            output_sharded (bool): If True, the method returns the result on the devices.
            row_weight (tensor or None): An optional (rows, channels) weight that multiplies
                each detector pixel before the filter.  For FDK, this is the cosine weight.  If
                None, the method applies plain FBP filtering.

        Returns:
            The filtered sinogram.
        """
        sinogram = self._shard_sinogram(sinogram)
        num_views, _, num_channels = self.get_params('sinogram_shape')
        recon_filter = tomography_utils.generate_direct_recon_filter(
            num_channels, filter_name=filter_name)
        recon_filter = recon_filter * np.float32(filter_scale * (np.pi / num_views))
        if isinstance(sinogram, _sharding.Shards):
            def filter_worker(i, d):
                ft = torch.as_tensor(recon_filter, device=d)
                rw = None if row_weight is None else row_weight.to(d)
                return tomography_utils.apply_row_filter(
                    sinogram.tensors[i], ft, row_weight=rw)
            filtered = _sharding.Shards(
                _sharding.run_per_device(self.sino_placement.devices,
                                         filter_worker), self.sino_placement)
        else:
            filter_t = torch.as_tensor(recon_filter, device=self.torch_device)
            filtered = tomography_utils.apply_row_filter(sinogram, filter_t,
                                                         row_weight=row_weight)
        return filtered if output_sharded else self._gather_sinogram(filtered)

    # ── loss and per-iteration statistics ─────────────────────────────────────
    @staticmethod
    def get_forward_model_loss(error_sinogram, sigma_y, weights=None, normalize=True):
        """
        This method computes the forward model loss from the error sinogram and the weights.
        The error sinogram is error_sinogram = measured_sinogram - forward_proj(recon).

        Args:
            error_sinogram (tensor): The 3D error sinogram.
            sigma_y (float): The standard deviation of the sinogram noise.
            weights (tensor, optional): The sinogram weights.  Defaults to all ones.
            normalize (bool, optional): If True (the default), the method returns the RMSE
                normalized by the weights.  Otherwise, it returns the weighted squared error
                without normalization.

        Returns:
            The loss as a scalar tensor on the device.
        """
        # The method does not accept a sharded array, because the sums run on one device.
        _sharding.reject_shards('get_forward_model_loss',
                                error_sinogram=error_sinogram, weights=weights)
        if weights is None:
            weights = 1
            avg_weight = 1
        elif np.ndim(weights) == 0:
            avg_weight = weights
        else:
            weights = torch.as_tensor(weights, dtype=torch.float32,
                                      device=error_sinogram.device)
            avg_weight = torch.mean(weights)
        weighted_sq_sum = _memory_estimate.weighted_square_sum(error_sinogram,
                                                             weights)
        if normalize:
            loss = torch.sqrt(weighted_sq_sum
                              / (avg_weight * float(error_sinogram.numel()))) / sigma_y
        else:
            loss = (1.0 / (2 * sigma_y ** 2)) * weighted_sq_sum
        return loss

    @staticmethod
    def _vcd_iteration_stats(error_sinogram, flat_recon, sigma_y, weights=None):
        """This method returns the VCD statistics of one iteration as the tuple
        (fm_loss, recon_l1, es_rmse)."""
        fm_loss = TomographyModel.get_forward_model_loss(
            error_sinogram, sigma_y, weights)
        # ``image_l1_norm`` sums in chunks, which can change the last digits of the NMAE stopping
        # test.
        recon_l1 = _memory_estimate.image_l1_norm(flat_recon)
        es_rmse = torch.sqrt(_memory_estimate.weighted_square_sum(error_sinogram)
                             / float(error_sinogram.numel()))
        return fm_loss, recon_l1, es_rmse

    def get_forward_lin_quad(self, weighted_error_sinogram, delta_sinogram, weights,
                             fm_constant, const_weights):
        """
        This method computes the two forward model terms of the line search.  The first term
        is ``fm_constant * sum(weighted_error_sinogram * delta_sinogram)``.  The second term is
        ``fm_constant * sum(delta_sinogram^2 * weights)``.  The method returns both terms as
        scalar tensors on the device.  The inputs must be plain tensors, and the method does not
        accept a sharded array.
        """
        _sharding.reject_shards('get_forward_lin_quad',
                                weighted_error_sinogram=weighted_error_sinogram,
                                delta_sinogram=delta_sinogram, weights=weights)
        forward_linear = fm_constant * torch.sum(weighted_error_sinogram * delta_sinogram)
        if const_weights:
            forward_quadratic = fm_constant * torch.sum(delta_sinogram * delta_sinogram)
        else:
            forward_quadratic = fm_constant * torch.sum(
                delta_sinogram * delta_sinogram * weights)
        return forward_linear, forward_quadratic

    # ── the VCD loop ────────────────────────────────────────────────────────
    def _get_update_direction(self, forward_grad, prior_grad, forward_hess,
                              prior_hess, pixel_indices, dev_index=0, slice_sums=None):
        """This method returns the update direction for one subset of pixels.  The direction
        is the preconditioned gradient -(forward_grad + prior_grad) / (forward_hess + prior_hess).

        The method returns the pair (direction, slice_pair_sum).  ``slice_pair_sum`` is the slice
        term of the exact line search, computed from ``slice_sums`` with
        :func:`qggmrf.qggmrf_slice_pair_sum`.  It is None when ``slice_sums`` is None."""
        fn = maybe_compile(_diagonal_update_direction, self.compile_enabled)
        direction = fn(forward_grad, prior_grad, forward_hess, prior_hess)
        if slice_sums is None:
            return direction, None
        return direction, _qggmrf.qggmrf_slice_pair_sum(slice_sums)

    def create_vcd_subset_updater(self, fm_hessian, weights, prox_input=None):
        """
        This method creates the function that updates one subset of pixels in the recon and
        in the error sinogram.

        Args:
            fm_hessian (tensor or Shards): The Hessian diagonal of the forward model loss, with
                shape (num_pixels, num_slices).
            weights (tensor, Shards, or 1): The sinogram weights, or the constant 1.
            prox_input (tensor or Shards, optional): The input of the proximal map, flattened to
                shape (num_pixels, num_slices).

        Returns:
            (callable) The function vcd_subset_updater(flat_recon, error_sinogram,
            pixel_indices, subset_index), which updates the recon and the error sinogram in
            place.  ``subset_index`` selects the pair lists of the subset.  Before the subsets
            of a partition are updated, the function attribute ``stage_halos`` must be called.
            With the qGGMRF prior, the function attribute ``stage_pair_lists`` must also be
            called with the pair lists of the partition from
            :func:`vcd_utils.gen_neighbor_pair_lists`.
        """
        sino_placement, recon_placement = self.sino_placement, self.recon_placement
        devices = sino_placement.devices
        num_devices = sino_placement.n_devices
        fm_hessian = self._as_shards(fm_hessian, recon_placement)
        const_weights = not (torch.is_tensor(weights)
                             or isinstance(weights, _sharding.Shards))
        if const_weights and abs(weights - 1) > 1e-5:
            raise ValueError('Constant weights must have value 1.')
        weights = None if const_weights else self._as_shards(weights, sino_placement)
        prox_input = (None if prox_input is None
                      else self._as_shards(prox_input, recon_placement))

        positivity_flag = self.get_params('positivity_flag')
        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(
            ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        b = _qggmrf.get_b_from_nbr_wts(qggmrf_nbr_wts)
        qggmrf_params = (b, sigma_x, p, q, T)
        sigma_prox = self.get_params('sigma_prox')
        recon_shape = self.get_params('recon_shape')
        max_alpha = self.get_params('max_alpha')

        # Each device thread gets its own compiled function, because a compiled function holds
        # state that the threads must not share.
        def per_dev(fn):
            return [maybe_compile(fn, self.compile_enabled, instance_key=i)
                    for i in range(num_devices)]
        prior_pass = per_dev(_qggmrf.qggmrf_prior_pass_at_indices)
        edge_term = per_dev(_qggmrf.qggmrf_edge_term)
        pair_sum = per_dev(_qggmrf.qggmrf_pair_sum)
        prior_line_terms = per_dev(_prior_line_terms)
        lin_quad_const = per_dev(_forward_lin_quad_const)
        lin_quad_weighted = per_dev(_forward_lin_quad_weighted)
        apply_update = per_dev(_apply_update)

        dev0 = devices[0]

        def combine_on_lead(parts):
            """This function sums the scalar results of each shard on the first device."""
            total = parts[0]
            for part in parts[1:]:
                total = total + _sharding.move_shard(part, dev0,
                                                     self.dev2dev_safe)
            return total

        # A halo holds the boundary slices of the neighboring device.  A halo of None means a
        # reflected boundary at a true edge of the volume.
        halos = {'left': [None] * num_devices, 'right': [None] * num_devices}

        def stage_halos(flat_shards):
            halos['left'], halos['right'] = _sharding.exchange_qggmrf_halos(
                flat_shards, self.dev2dev_safe)

        # The pair lists of the current partition hold, for each direction in the slice, the
        # pairs array on each device, the pair counts, and the neighbor weight of the pairs.
        pair_lists = []

        def stage_pair_lists(lists):
            pair_lists[:] = [([torch.as_tensor(pairs).to(dev) for dev in devices], counts,
                              b_pair)
                             for (pairs, counts), b_pair in zip(lists, (b[0] + b[1],
                                                                        b[2] + b[3]))]

        def vcd_subset_updater(flat_recon, error_sinogram, pixel_indices, subset_index=0):
            """This function runs one VCD update on one subset of the partition, in place.  It
            returns flat_recon, error_sinogram, ``l1_for_subset`` (the L1 norm of the update),
            alpha_for_subset, and delta_sumsq_subset."""
            pixel_indices_per_device = [torch.as_tensor(pixel_indices, dtype=torch.int64).to(dev)
                       for dev in devices]

            # The forward model loss is 1/(2 sigma_y^2) || error_sinogram - A delta ||_weights^2.
            if const_weights:
                weighted_error_sinogram = error_sinogram
            else:
                weighted_error_sinogram = _sharding.Shards(
                    _sharding.run_per_device(
                        devices, lambda i, dev: weights.tensors[i]
                        * error_sinogram.tensors[i],
                        executor=self._per_device_pool), sino_placement)

            # The back projection gives the gradient, and fm_constant is 1/sigma_y^2.
            back_projected_error = self.sparse_back_project(weighted_error_sinogram, pixel_indices)
            if not const_weights:
                weighted_error_sinogram = None

            # The halos supply the prior terms that cross a boundary between devices.  The
            # forward model gradient is computed in place in the output of the back projection.
            def prior_worker(i, dev):
                forward_grad = back_projected_error.tensors[i].mul_(-fm_constant)
                forward_hess = fm_constant * fm_hessian.tensors[i][pixel_indices_per_device[i]]
                if prox_input is None:
                    grad, hess, slice_sums = _qggmrf.qggmrf_prior_pass_in_chunks(
                        flat_recon.tensors[i], recon_shape, pixel_indices_per_device[i],
                        qggmrf_params, forward_grad, forward_hess,
                        left_halo=halos['left'][i], right_halo=halos['right'][i],
                        chunk_function=prior_pass[i])
                else:
                    grad = _qggmrf.prox_gradient_at_indices(
                        flat_recon.tensors[i], prox_input.tensors[i],
                        pixel_indices_per_device[i], sigma_prox)
                    hess = 1 / (sigma_prox ** 2)
                    slice_sums = None
                return forward_grad, forward_hess, grad, hess, slice_sums

            prior_terms = _sharding.run_per_device(devices, prior_worker,
                                             executor=self._per_device_pool)
            del back_projected_error

            # The exact quadratic term of the qGGMRF prior is computed from the approximate
            # term sum(prior_hess * delta^2).  The code subtracts the edge term and the pair
            # terms of the slice pairs and of the pairs in each slice.
            def direction_worker(i, dev):
                forward_grad, forward_hess, prior_grad, prior_hess, slice_sums = prior_terms[i]
                indices = pixel_indices_per_device[i]
                delta_recon, slice_pair_sum = self._get_update_direction(
                    forward_grad, prior_grad, forward_hess, prior_hess, indices,
                    dev_index=i, slice_sums=slice_sums)
                prior_hess_t = (prior_hess if torch.is_tensor(prior_hess)
                                else torch.as_tensor(prior_hess,
                                                     dtype=torch.float32,
                                                     device=delta_recon.device))
                prior_linear_part, prior_quadratic_part = prior_line_terms[i](
                    prior_grad, prior_hess_t, delta_recon)
                if prox_input is None:
                    prior_quadratic_part = (
                        prior_quadratic_part
                        - edge_term[i](delta_recon, indices, recon_shape, qggmrf_params,
                                       halos['left'][i] is None, halos['right'][i] is None)
                        - slice_pair_sum)
                    for pairs, counts, b_pair in pair_lists:
                        prior_quadratic_part = prior_quadratic_part - \
                            _qggmrf.qggmrf_pair_sum_in_chunks(
                                flat_recon.tensors[i], indices, delta_recon,
                                pairs[i][subset_index], int(counts[subset_index]), b_pair,
                                qggmrf_params, chunk_function=pair_sum[i])
                return delta_recon, prior_linear_part, prior_quadratic_part
            direction_results = _sharding.run_per_device(devices, direction_worker,
                                            executor=self._per_device_pool)
            delta_recon_per_device = [result[0] for result in direction_results]
            prior_linear = combine_on_lead([result[1] for result in direction_results])
            prior_quadratic = combine_on_lead([result[2] for result in direction_results])

            # The slice pairs that cross a boundary between devices belong to the device before
            # the boundary.  Each device receives the first slice of the direction of the next
            # device.
            if prox_input is None and num_devices > 1:
                _, delta_right = _sharding.exchange_qggmrf_halos(
                    _sharding.Shards(delta_recon_per_device, recon_placement),
                    self.dev2dev_safe)

                def boundary_worker(i, dev):
                    indices = pixel_indices_per_device[i]
                    return _qggmrf.qggmrf_boundary_pair_sum(
                        flat_recon.tensors[i][indices, -1], halos['right'][i][indices],
                        delta_recon_per_device[i][:, -1], delta_right[i], qggmrf_params)
                boundary_devices = [i for i in range(num_devices - 1)
                                    if delta_right[i] is not None]
                if boundary_devices:
                    prior_quadratic = prior_quadratic - combine_on_lead(
                        [boundary_worker(i, devices[i]) for i in boundary_devices])
                del delta_right
            prior_quadratic = prior_quadratic.to(torch.float32)

            # These buffers are freed before the projection of the update, because that projection
            # uses much more memory.
            del prior_terms, direction_results

            delta_sinogram = self.sparse_forward_project(
                _sharding.Shards(delta_recon_per_device, recon_placement), pixel_indices)

            def lin_quad_worker(i, dev):
                local_delta_sinogram = delta_sinogram.tensors[i]
                if const_weights:
                    return lin_quad_const[i](error_sinogram.tensors[i], local_delta_sinogram,
                                             fm_constant)
                return lin_quad_weighted[i](error_sinogram.tensors[i], local_delta_sinogram,
                                            weights.tensors[i], fm_constant)
            forward_line_terms = _sharding.run_per_device(devices, lin_quad_worker,
                                          executor=self._per_device_pool)
            forward_linear = combine_on_lead(
                [linear for linear, _ in forward_line_terms])
            forward_quadratic = combine_on_lead(
                [quadratic for _, quadratic in forward_line_terms])

            # alpha stays a tensor on the device, so no subset forces a host synchronization.
            alpha_numerator = forward_linear - prior_linear
            alpha_denominator = forward_quadratic + prior_quadratic + _F32_EPS
            alpha = alpha_numerator / alpha_denominator
            alpha = torch.clamp(alpha, _F32_EPS, max_alpha)
            alpha_per_device = ([alpha] if num_devices == 1 else
                         [_sharding.move_shard(alpha, dev, self.dev2dev_safe)
                          for dev in devices])

            # The positivity constraint clips the update so that recon + alpha * delta >= 0.  The
            # projection of the update is then computed again.
            if positivity_flag is True:
                def positivity_worker(i, dev):
                    recon_at_indices = flat_recon.tensors[i][pixel_indices_per_device[i]]
                    pos_constant = 1.0 / (alpha_per_device[i] + _F32_EPS)
                    return torch.maximum(-pos_constant * recon_at_indices,
                                         delta_recon_per_device[i])
                delta_recon_per_device[:] = _sharding.run_per_device(
                    devices, positivity_worker, executor=self._per_device_pool)
                delta_sinogram = self.sparse_forward_project(
                    _sharding.Shards(delta_recon_per_device, recon_placement), pixel_indices)

            def apply_worker(i, dev):
                delta_scaled = alpha_per_device[i] * delta_recon_per_device[i]
                _, _, delta_sumsq_local, l1_local = apply_update[i](
                    flat_recon.tensors[i], error_sinogram.tensors[i],
                    pixel_indices_per_device[i], delta_scaled, alpha_per_device[i],
                    delta_sinogram.tensors[i])
                return delta_sumsq_local, l1_local
            apply_results = _sharding.run_per_device(devices, apply_worker,
                                               executor=self._per_device_pool)
            delta_sumsq_subset = (apply_results[0][0] if num_devices == 1
                                  else torch.cat(
                [_sharding.move_shard(sumsq, dev0, self.dev2dev_safe)
                 for sumsq, _ in apply_results]))
            l1_for_subset = combine_on_lead(
                [l1 for _, l1 in apply_results])
            return (flat_recon, error_sinogram, l1_for_subset, alpha,
                    delta_sumsq_subset)

        vcd_subset_updater.stage_halos = stage_halos
        if prox_input is None:
            vcd_subset_updater.stage_pair_lists = stage_pair_lists
        return vcd_subset_updater

    def vcd_partition_iterator(self, vcd_subset_updater, flat_recon, error_sinogram,
                               partition, rng=None, pair_lists=None):
        """
        This method runs one full iteration of the VCD algorithm over the subsets of the
        partition.  It updates flat_recon and error_sinogram in place.

        Args:
            rng (numpy.random.Generator, optional): The generator that draws the order in which
                the subsets are visited.  Defaults to None, which uses the global np.random
                state.
            pair_lists (tuple, optional): The pair lists of the partition from
                :func:`vcd_utils.gen_neighbor_pair_lists`.  If None, the method makes them.

        Returns:
            (flat_recon, error_sinogram, l1_for_partition, alpha, delta_sumsq_partition): The
            updated recon and error sinogram, the summed L1 norm of the recon change, the
            average of alpha over the subsets, and the sum of squared updates of each slice over
            the partition.
        """
        if hasattr(vcd_subset_updater, 'stage_halos'):
            vcd_subset_updater.stage_halos(flat_recon)
        if hasattr(vcd_subset_updater, 'stage_pair_lists'):
            if pair_lists is None:
                pair_lists = vcd_utils.gen_neighbor_pair_lists(
                    partition, self.get_params('recon_shape'))
            vcd_subset_updater.stage_pair_lists(pair_lists)
        # This random draw must not change, because the tests compare the values at each
        # iteration.
        l1_for_partition = 0
        alpha_sum = 0
        delta_sumsq_partition = 0
        draw = np.random if rng is None else rng
        subset_indices = draw.permutation(partition.shape[0])

        for index in subset_indices:
            subset = partition[index]
            (flat_recon, error_sinogram, l1_for_subset, alpha_for_subset,
             delta_sumsq_subset) = vcd_subset_updater(flat_recon, error_sinogram, subset,
                                                      subset_index=index)
            l1_for_partition += l1_for_subset
            alpha_sum += alpha_for_subset
            delta_sumsq_partition = delta_sumsq_partition + delta_sumsq_subset

        return (flat_recon, error_sinogram, l1_for_partition,
                alpha_sum / partition.shape[0], delta_sumsq_partition)

    def _vcd_recon(self, sinogram, partitions, partition_sequence,
                   stop_threshold_change_pct, weights=None, init_recon=None,
                   prox_input=None, compute_prior_loss=False, first_iteration=0,
                   init_error_sinogram=None, fm_hessian=None,
                   return_checkpoint=False, rng=None):
        """
        This method performs MBIR reconstruction by Multi-Granular Vector Coordinate
        Descent with the given partitions and partition sequence.  The methods :meth:`recon`
        and :meth:`prox_map` call this method.  A user needs to call it directly only to resume
        a run with ``init_error_sinogram`` and ``return_checkpoint``.

        Args:
            sinogram (numpy or tensor or Shards): The 3D sinogram, with shape
                (num_views, num_det_rows, num_det_channels).  It may also be the sinogram on the
                devices that ``prepare_sino_for_devices`` returns.
            partitions (list): The K partitions.  Each partition is an integer tensor with shape
                (N_subsets, N_indices) that holds the indices of the voxels to update.
            partition_sequence (ndarray): The index of the partition to use at each iteration.
            stop_threshold_change_pct (float): The method stops when the NMAE change between
                iterations, in percent, falls below this value.
            weights (numpy or tensor or Shards, optional): The 3D positive weights, with the
                same shape as the sinogram.  The weights may be a plain array or on the devices.
                Defaults to all ones.
            init_recon (array or int or None): The initial reconstruction.  If None, the method
                uses ``recon_direct``.  If an int, the method uses a constant volume.
            prox_input (array or Shards, optional): The input of the proximal map, as a full
                volume or on the devices.
            compute_prior_loss (bool, optional): If True, the method also computes the prior
                loss.  This option is for debugging small recons.
            first_iteration (int, optional): The iteration number offset for restarts.
            init_error_sinogram (array or tensor, optional): A precomputed error sinogram,
                sinogram - A @ init_recon, to resume from.  This argument requires
                ``init_recon``.  The method updates both arrays in place.  For this reason, a
                caller who needs the earlier values must copy the arrays first.
            fm_hessian (array or tensor, optional): A precomputed Hessian diagonal of the forward
                model, for the same weights and geometry.  If None, the method computes it.
            return_checkpoint (bool, optional): If True, the method also returns the dict
                {'error_sinogram': ..., 'fm_hessian': ...}.  The dict holds the tensors that the
                loop used on the devices, not copies.
            rng (numpy.random.Generator, optional): The generator that draws the order of the
                subsets at each iteration.  Defaults to None, which uses the global np.random
                state.

        Returns:
            (recon, recon_stats): The 3D reconstruction tensor and a tuple of statistics for
            each iteration.  The tuple is (fm_rmse, pm_loss, nmae_update, alpha_values,
            delta_norm_per_slice), where nmae_update is
            ||recon(i+1) - recon(i)||_1 / ||recon(i+1)||_1.  If ``return_checkpoint`` is True,
            the method returns (recon, recon_stats, checkpoint).
        """
        self.verify_valid_params()
        dev = self.torch_device
        recon_shape = self.get_params('recon_shape')
        sinogram_shape = self.get_params('sinogram_shape')
        if isinstance(sinogram, _sharding.Shards):
            self._check_sinogram_shards(sinogram, sinogram_shape)
        elif tuple(sinogram.shape) != tuple(sinogram_shape):
            raise ValueError('sinogram does not have the shape in sinogram_shape. \n'
                             f'Expected {tuple(sinogram_shape)}, got '
                             f'{tuple(sinogram.shape)}.')

        # The GPUs must be chosen before the first large allocation.
        memory_estimate = self._apply_device_policy(
            partition_sequence=partition_sequence, weights=weights,
            init_recon=init_recon, fm_hessian=fm_hessian,
            prox_input=prox_input, init_error_sinogram=init_error_sinogram)
        if _memory_estimate.calibration_enabled():
            # The peak memory counters are reset here, once per reconstruction.
            _memory_estimate.calibration_start(self.sino_placement.devices)
        self._log_device_report()

        constant_weights = weights is None
        if constant_weights:
            weights = 1
        else:
            weights = self._shard_sinogram(weights)

        if init_error_sinogram is not None and init_recon is None:
            raise ValueError('init_error_sinogram requires init_recon (the pair must be a '
                             'consistent resume state; see the docstring).')

        # When the run resumes, the sinogram is not copied to the devices.
        if init_error_sinogram is None:
            sinogram = self._shard_sinogram(sinogram)

        scale_recon_to_sinogram = init_recon is None
        if init_recon is None:
            self.logger.info('Starting direct recon for initial reconstruction')
            init_recon = self.recon_direct(sinogram, output_sharded=True)
        elif isinstance(init_recon, int):
            init_recon = self._constant_recon(init_recon)
        else:
            if tuple(np.shape(init_recon)) != tuple(recon_shape):
                raise ValueError(f"init_recon does not have the correct shape. Expected "
                                 f"{tuple(recon_shape)}, got {tuple(np.shape(init_recon))}.")
            init_recon = self._shard_recon(init_recon)

        if init_error_sinogram is not None:
            # The loop uses the caller's arrays as its working arrays.
            self.logger.info('Resuming from init_error_sinogram')
            error_sinogram = self._shard_sinogram(init_error_sinogram)
        else:
            error_sinogram, init_recon = self._initial_error_state(
                sinogram, init_recon, weights, constant_weights,
                scale_recon_to_sinogram)

        sinogram = None
        error_sinogram = self._shard_sinogram(error_sinogram)

        if prox_input is not None:
            if isinstance(prox_input, _sharding.Shards):
                prox_input = self._flatten_prox_shards(prox_input, recon_shape)
            else:
                # The shape is checked before the reshape, so an input with the wrong shape
                # raises an error.
                if tuple(prox_input.shape) != tuple(recon_shape):
                    raise ValueError('prox_input does not have the correct size. \n'
                                     f'Expected {tuple(recon_shape)}, got shape '
                                     f'{tuple(prox_input.shape)} for prox_input shape.')
                prox_input = self._shard_recon(
                    prox_input.reshape((-1, prox_input.shape[-1])))

        verbose, sigma_y = self.get_params(['verbose', 'sigma_y'])

        # np.prod would overflow past 2^31 elements.
        total_sino_size = math.prod(sinogram_shape)

        if fm_hessian is None:
            if constant_weights:
                hess_weights = self._sino_ones_device_form(error_sinogram)
            else:
                hess_weights = weights
            self.logger.info('Computing Hessian diagonal')
            # The loop reads the Hessian only at pixels inside the mask, so only those pixels
            # are computed.
            hess_indices = (None if self.get_params('use_ror_mask') is False
                            else self.full_indices_device())
            fm_hessian = self.compute_hessian_diagonal(weights=hess_weights,
                                                       output_sharded=True,
                                                       indices=hess_indices)
        else:
            self.logger.info('Using precomputed Hessian diagonal')
            fm_hessian = self._shard_recon(fm_hessian)
        fm_hessian = self._flatten_hessian(fm_hessian)

        flat_recon = self._flatten_recon(init_recon)

        # On one device, the container with one shard shares memory with the caller's array.
        flat_recon = self._as_shards(flat_recon, self.recon_placement)
        error_sinogram = self._as_shards(error_sinogram,
                                             self.sino_placement)

        vcd_subset_updater = self.create_vcd_subset_updater(
            fm_hessian, weights=weights, prox_input=prox_input)

        self.logger.info('Starting VCD iterations')
        if verbose >= 2:
            if memory_estimate is not None:
                self.logger.debug('Modeled peak device memory by phase:')
                self.logger.debug(memory_estimate.format_table())
                self.logger.debug('--------')
            output = io.StringIO()
            get_memory_stats(file=output)
            self.logger.debug(output.getvalue())
            self.logger.debug('--------')

        max_iters = partition_sequence.size
        # The pair lists of each partition are made once, when the partition is first used.
        pair_lists = {}
        fm_rmse = np.zeros(max_iters)
        pm_loss = np.zeros(max_iters)
        nmae_update = np.zeros(max_iters)
        alpha_values = np.zeros(max_iters)
        delta_norm_per_slice = np.zeros((max_iters, recon_shape[2]))
        num_iters = 0
        if not self.sino_placement.is_trivial:
            self._per_device_pool = _sharding.device_pool(
                self.sino_placement.n_devices)
        try:
            for i in range(max_iters):
                partition = partitions[partition_sequence[i]]
                if prox_input is None and partition_sequence[i] not in pair_lists:
                    pair_lists[partition_sequence[i]] = vcd_utils.gen_neighbor_pair_lists(
                        partition, recon_shape)
                (flat_recon, error_sinogram, l1_for_partition, alpha,
                 delta_sumsq_partition) = self.vcd_partition_iterator(
                    vcd_subset_updater, flat_recon, error_sinogram, partition,
                    rng=rng, pair_lists=pair_lists.get(partition_sequence[i]))

                fm_loss_i, recon_l1, es_rmse = self._iteration_stats(
                    error_sinogram, flat_recon, sigma_y, weights,
                    constant_weights, float(total_sino_size))
                fm_rmse[i] = float(fm_loss_i)
                recon_l1_f = float(recon_l1)
                # A reconstruction of all zeros gives nan rather than ZeroDivisionError.
                nmae_update[i] = (float(l1_for_partition) / recon_l1_f
                                  if recon_l1_f else float('nan'))
                alpha_values[i] = float(alpha)
                delta_norm_per_slice[i] = np.sqrt(
                    delta_sumsq_partition.cpu().numpy())[:recon_shape[2]]

                if verbose >= 1:
                    iter_output = (
                        '\nAfter iteration {} of a max of {}: Pct change={:.4f}, '
                        'Forward loss={:.4f}'.format(i + first_iteration,
                                                     max_iters + first_iteration,
                                                     100 * nmae_update[i], fm_rmse[i]))
                    if compute_prior_loss:
                        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(
                            ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
                        b = _qggmrf.get_b_from_nbr_wts(qggmrf_nbr_wts)
                        qggmrf_params = (b, sigma_x, p, q, T)
                        # The prior loss is computed on the full volume gathered from the devices.
                        total_recon_size = math.prod(recon_shape)
                        loss_recon = self._gather_recon(flat_recon).reshape(
                            tuple(recon_shape))
                        pm_loss[i] = _qggmrf.qggmrf_loss(loss_recon, qggmrf_params)
                        pm_loss[i] /= total_recon_size
                        total_loss = ((fm_rmse[i] * total_sino_size
                                       + pm_loss[i] * total_recon_size)
                                      / (0.5 * (total_sino_size + total_recon_size)))
                        iter_output += ', Prior loss={:.4f}, Weighted total loss={:.4f}'.format(
                            pm_loss[i], total_loss)
                    self.logger.info(iter_output)
                    self.logger.info(f'Relative step size (alpha)={alpha_values[i]:.2f}, '
                                     f'Error sino RMSE={float(es_rmse):.4f}')
                    self.logger.info('Number subsets = {}'.format(partition.shape[0]))
                    if verbose >= 2:
                        output = io.StringIO()
                        get_memory_stats(file=output)
                        self.logger.debug(output.getvalue())
                        self.logger.debug('--------')
                num_iters += 1
                if nmae_update[i] < stop_threshold_change_pct / 100:
                    self.logger.warning('Change threshold stopping condition reached')
                    break
        finally:
            if self._per_device_pool is not None:
                self._per_device_pool.shutdown(wait=True)
                self._per_device_pool = None

        # This code must run before the loop releases its arrays, so that the measured peak is
        # correct.
        if memory_estimate is not None and _memory_estimate.calibration_enabled():
            rows = _memory_estimate.calibration_report(
                memory_estimate, self.sino_placement.devices)
            self.last_memory_calibration = rows
            if rows:
                self.logger.warning(_memory_estimate.format_calibration(rows))

        flat_recon = self._as_device_form(flat_recon)
        error_sinogram = self._as_device_form(error_sinogram)
        recon_3d = self._recon_from_flat(flat_recon, recon_shape)
        losses = (fm_rmse[:num_iters], pm_loss[:num_iters], nmae_update[:num_iters],
                  alpha_values[:num_iters], delta_norm_per_slice[:num_iters])
        if return_checkpoint:
            checkpoint = {'error_sinogram': error_sinogram, 'fm_hessian': fm_hessian}
            return recon_3d, losses, checkpoint
        return recon_3d, losses

    def initialize_recon(self, sinogram, weights=None, init_recon=None,
                         max_iterations=15, first_iteration=0,
                         logfile_path='~/.mbirtorch/logs/recon.log',
                         print_logs=True, rng=None):
        """
        This method prepares a run of :meth:`recon`.  It generates the voxel partitions and
        the partition sequence, checks the inputs, and sets the regularization automatically.

        Args:
            The same as :meth:`recon`.  ``rng`` is the generator that draws the partitions.
            If ``rng`` is None, the method uses the global np.random state.

        Returns:
            sinogram, weights, init_recon, partitions, partition_sequence,
            granularity, regularization_params
        """
        # A Plug-and-Play loop that passes do_initialization=False skips this method.  As a
        # result, the whole loop writes one log.
        self._log_run_header(first_iteration, logfile_path, print_logs)
        recon_shape, granularity, use_ror_mask = self.get_params(
            ['recon_shape', 'granularity', 'use_ror_mask'])
        partitions = vcd_utils.gen_set_of_pixel_partitions(
            recon_shape, granularity, device=self.torch_device,
            use_ror_mask=use_ror_mask, rng=rng)

        partition_sequence = self.get_params('partition_sequence')
        partition_sequence = vcd_utils.gen_partition_sequence(
            partition_sequence, max_iterations=max_iterations)
        partition_sequence = partition_sequence[first_iteration:]

        # A sharded sinogram is checked on the devices that hold it.
        if isinstance(sinogram, _sharding.Shards):
            for tensor in sinogram.tensors:
                if tensor.is_complex():
                    raise TypeError(
                        "sinogram must be real-valued; got complex dtype.")
                low, high = _array_extremes(tensor)
                if not (math.isfinite(low) and math.isfinite(high)):
                    raise ValueError("sinogram contains NaN and/or Inf values.")
            sinogram_for_stats = sinogram
        else:
            sinogram_np = np.asarray(sinogram) if not torch.is_tensor(sinogram) \
                else sinogram.cpu().numpy()
            if np.iscomplexobj(sinogram_np):
                raise TypeError("sinogram must be real-valued; got complex dtype.")
            low, high = _array_extremes(sinogram_np)
            if not (math.isfinite(low) and math.isfinite(high)):
                raise ValueError("sinogram contains NaN and/or Inf values.")
            sinogram_for_stats = sinogram_np
        if weights is not None:
            if isinstance(weights, _sharding.Shards):
                all_zero = True
                for tensor in weights.tensors:
                    low, high = _array_extremes(tensor)
                    if not (math.isfinite(low) and math.isfinite(high)):
                        raise ValueError("weights contains NaN and/or Inf values.")
                    if low < 0:
                        raise ValueError("weights contain negative values.")
                    all_zero = all_zero and low == 0 and high == 0
                if all_zero:
                    raise ValueError("all weights are zero.")
            else:
                weights_np = np.asarray(weights) if not torch.is_tensor(weights) \
                    else weights.cpu().numpy()
                low, high = _array_extremes(weights_np)
                if not (math.isfinite(low) and math.isfinite(high)):
                    raise ValueError("weights contains NaN and/or Inf values.")
                if low < 0:
                    raise ValueError("weights contain negative values.")
                if low == 0 and high == 0:
                    raise ValueError("all weights are zero.")

        regularization_params = self.auto_set_regularization_params(
            sinogram_for_stats, weights=weights)
        return (sinogram, weights, init_recon, partitions, partition_sequence,
                granularity, regularization_params)

    def recon(self, sinogram, weights=None, init_recon=None, max_iterations=15,
              stop_threshold_change_pct=0.2, first_iteration=0,
              logfile_path='~/.mbirtorch/logs/recon.log', print_logs=True,
              output_sharded=False, rng=None):
        """
        This method reconstructs a volume from a sinogram by MBIR.

        The method runs the Multi-Granular Vector Coordinate Descent algorithm for up to
        ``max_iterations`` iterations.  It stops early if the change between iterations falls
        below ``stop_threshold_change_pct``.  The pixel partitions are drawn at random, so two
        runs differ slightly.  The user can make the result repeatable with ``np.random.seed``
        or ``rng``.

        Args:
            sinogram (numpy or tensor): The 3D sinogram, with shape (num_views, num_det_rows,
                num_det_channels).
            weights (numpy or tensor, optional): The positive weights, with the same shape as
                the sinogram.  Defaults to None, which means all ones.
            init_recon (array, optional): The initial reconstruction.  Defaults to None, which
                means a direct reconstruction.
            max_iterations (int, optional): The maximum number of iterations.  Defaults to 15.
            stop_threshold_change_pct (float, optional): The method stops when the percent
                change in the volume between iterations drops below this value.  Defaults to
                0.2.  The value 0 runs exactly ``max_iterations`` iterations.
            first_iteration (int, optional): The number of iterations already done, when the
                user continues a reconstruction.  In this case, ``init_recon`` is the previous
                result.  Defaults to 0.
            logfile_path (str, optional): The path of the log file.  Defaults to
                '~/.mbirtorch/logs/recon.log'.  If None, the method writes no log file.
            print_logs (bool, optional): If True, the method prints the log to the console.
                Defaults to True.
            output_sharded (bool, optional): If True, the method returns the volume on the
                devices instead of as a numpy array.  Defaults to False.
            rng (numpy.random.Generator, optional): The generator for every random draw of the
                run.  Defaults to None, which uses the global numpy state.

        Returns:
            (recon, recon_dict): The reconstructed volume and a dict.  The dict has the entries
            'recon_params', 'recon_log', 'notes', and 'model_params'.  The entry 'recon_params'
            holds the settings and the values at each iteration.  The entry 'recon_log' holds
            the log text, and 'model_params' holds the model parameters.  The user can pass the
            dict to :func:`~mbirtorch.view_utils.slice_viewer` and
            :func:`~mbirtorch.export_recon_hdf5`.

        Example:
            >>> recon, recon_dict = ct_model.recon(sinogram, weights=weights)
        """
        # A sharded init_recon is refused, because the code cannot check its full shape.
        _sharding.reject_shards('recon', init_recon=init_recon)
        (sinogram, weights, init_recon, partitions, partition_sequence, granularity,
         regularization_params) = self.initialize_recon(
            sinogram, weights, init_recon, max_iterations, first_iteration,
            logfile_path=logfile_path, print_logs=print_logs, rng=rng)

        # torch.compile fails on in-place updates inside inference_mode, so the code uses
        # no_grad.
        with torch.no_grad():
            recon, loss_vectors = self._vcd_recon(
                sinogram, partitions, partition_sequence,
                stop_threshold_change_pct, weights=weights, init_recon=init_recon,
                first_iteration=first_iteration, rng=rng)

        partition_sequence = [int(val) for val in partition_sequence]
        fm_rmse = [float(val) for val in loss_vectors[0]]
        prior_loss = [0]
        stop_pct = [100 * float(val) for val in loss_vectors[2]]
        alpha_values = [float(val) for val in loss_vectors[3]]
        delta_norm_per_slice = [[float(v) for v in row] for row in loss_vectors[4]]
        num_iterations = len(fm_rmse)
        recon_params = dict(zip(recon_param_names,
                                [num_iterations, granularity, partition_sequence,
                                 fm_rmse, prior_loss, regularization_params,
                                 stop_pct, alpha_values, delta_norm_per_slice]))

        if logfile_path:
            self.logger.info('Logs written to {}'.format(
                os.path.abspath(os.path.expanduser(logfile_path))))
        for h in list(self.logger.handlers):
            h.flush()
        # A later call that continues this run reopens the log file.
        self.close_log_file()

        notes = 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now())
        recon_dict = self.get_recon_dict(recon_params, notes=notes)
        return (recon if output_sharded else self._gather_recon(recon)), recon_dict

    def _iteration_stats(self, error_sinogram, flat_recon, sigma_y, weights,
                         constant_weights, total_sino_size):
        """This method returns the statistics of one iteration.  These are the forward model
        loss, the L1 norm of the reconstruction, and the RMSE of the error sinogram."""
        if (isinstance(error_sinogram, _sharding.Shards)
                and not error_sinogram.placement.is_trivial):
            error_shards, flat_shards = error_sinogram, flat_recon
            weights_shards = None if constant_weights else weights

            def sino_worker(i, d):
                e = error_shards.tensors[i]
                sq = float(_memory_estimate.weighted_square_sum(e))
                if weights_shards is None:
                    return sq, 0.0, sq
                w = weights_shards.tensors[i]
                return (float(_memory_estimate.weighted_square_sum(e, w)),
                        float(torch.sum(w)), sq)
            parts = _sharding.run_per_device(error_shards.placement.devices,
                                             sino_worker)
            weighted_sq = sum(a for a, _, _ in parts)
            sq = sum(c for _, _, c in parts)
            if weights_shards is None:
                avg_weight = 1.0
            else:
                avg_weight = sum(b for _, b, _ in parts) / total_sino_size
            fm_loss = ((weighted_sq / (avg_weight * total_sino_size)) ** 0.5
                       / sigma_y)
            recon_l1 = sum(
                float(_memory_estimate.image_l1_norm(t))
                for t in flat_shards.tensors)
            es_rmse = (sq / total_sino_size) ** 0.5
        else:
            if isinstance(error_sinogram, _sharding.Shards):
                error_sinogram = error_sinogram.tensors[0]
                flat_recon = flat_recon.tensors[0]
            fm_loss, recon_l1, es_rmse = TomographyModel._vcd_iteration_stats(
                error_sinogram, flat_recon, sigma_y, weights)
        return fm_loss, recon_l1, es_rmse

    def initialize_prox(self, sinogram, weights=None, init_recon=None,
                        max_iterations=3, first_iteration=0,
                        logfile_path='~/.mbirtorch/logs/prox.log',
                        print_logs=True, rng=None):
        """
        This method prepares :meth:`prox_map` and stores the result in ``prox_data``.  Later
        calls of ``prox_map`` reuse this result.  The method ``prox_map`` calls this method when
        the caller asks for initialization or when there is no stored result.

        Args:
            The same as :meth:`prox_map`.  ``rng`` is the generator that draws the partitions.
            If ``rng`` is None, the method uses the global np.random state.

        Returns:
            sinogram, weights, init_recon, partitions, partition_sequence,
            granularity, regularization_params
        """
        initialized = self.initialize_recon(
            sinogram, weights, init_recon, max_iterations, first_iteration,
            logfile_path=logfile_path, print_logs=print_logs, rng=rng)
        # The stored entries are the partitions, the partition sequence, the granularity, and the
        # regularization parameters.
        self.prox_data = tuple(initialized[3:])
        return initialized

    def prox_map(self, prox_input, sinogram, sigma_prox=None, weights=None,
                 init_recon=None, do_initialization=True,
                 stop_threshold_change_pct=0.2, max_iterations=3,
                 first_iteration=0,
                 logfile_path='~/.mbirtorch/logs/prox.log', print_logs=True,
                 output_sharded=False, rng=None):
        """
        This method computes the proximal map of the data term for Plug-and-Play loops.

        The method is the same as :meth:`recon`, except that a Gaussian prior replaces the
        usual prior.  The Gaussian has mean ``prox_input`` and standard deviation
        ``sigma_prox``.  A Plug-and-Play loop alternates this method with a denoiser.

        Args:
            prox_input (numpy or tensor): The input of the proximal map, with the shape of the
                reconstruction.
            sinogram (numpy or tensor): The 3D sinogram, with shape (num_views, num_det_rows,
                num_det_channels).
            sigma_prox (float, optional): The standard deviation of the prior.  Defaults to
                None, which means the value is set from the sinogram.
            weights (numpy or tensor, optional): The positive weights, with the same shape as
                the sinogram.  Defaults to None, which means all ones.
            init_recon (numpy or tensor, optional): The initial reconstruction.  Defaults to
                None.
            do_initialization (bool, optional): The user sets this to False after the first call
                of a loop on the same sinogram.  In that case, the method does not compute the
                partitions and the regularization again.  Defaults to True.
            stop_threshold_change_pct (float, optional): The method stops when the percent
                change between iterations drops below this value.  Defaults to 0.2.
            max_iterations (int, optional): The maximum number of iterations, counted from
                iteration 0.  Defaults to 3.
            first_iteration (int, optional): The total number of proximal map iterations done so
                far in the loop.  The method uses it to continue the partition sequence.
                Defaults to 0.
            logfile_path (str, optional): The path of the log file.  Defaults to
                '~/.mbirtorch/logs/prox.log'.  If None, the method writes no log file.
            print_logs (bool, optional): If True, the method prints the log to the console.
                Defaults to True.
            output_sharded (bool, optional): If True, the method returns the volume on the
                devices instead of as a numpy array.  A loop can pass this volume to a denoiser
                on the same devices.  Defaults to False.
            rng (numpy.random.Generator, optional): The generator for every random draw.
                Defaults to None, which uses the global numpy state.

        Returns:
            (recon, recon_dict): The volume and a dict with the same entries as in
            :meth:`recon`.

        Example:
            >>> recon, _ = ct_model.prox_map(prox_input, sinogram, sigma_prox=0.5)
        """
        # A sharded init_recon is refused, because the code cannot check its full shape.
        _sharding.reject_shards('prox_map', init_recon=init_recon)
        prior_loss = [0]
        if do_initialization or self.prox_data is None:
            (sinogram, weights, init_recon, partitions, partition_sequence,
             granularity, regularization_params) = self.initialize_prox(
                sinogram, weights, init_recon, max_iterations, first_iteration,
                logfile_path=logfile_path, print_logs=print_logs, rng=rng)
        else:
            (partitions, partition_sequence, granularity,
             regularization_params) = self.prox_data
            # A change of the devices or of the geometry clears this stored result.  The
            # partition sequence is computed again.
            partition_sequence = vcd_utils.gen_partition_sequence(
                self.get_params('partition_sequence'),
                max_iterations=max_iterations)
            partition_sequence = partition_sequence[first_iteration:]
            # This call skipped the run header, which normally reopens the log file.  For this
            # reason, the code reopens the log file here.
            self._reopen_log_file()

        # A sigma_prox from the caller overrides the automatic value until the end of the call.
        self_sigma_prox = self.get_params('sigma_prox')
        if sigma_prox is not None:
            regularization_params = dict(regularization_params,
                                         sigma_prox=sigma_prox)
            self.set_params(no_warning=True, sigma_prox=sigma_prox,
                            auto_regularize_flag=self.get_params('auto_regularize_flag'))

        with torch.no_grad():
            recon, loss_vectors = self._vcd_recon(
                sinogram, partitions, partition_sequence,
                stop_threshold_change_pct, weights=weights,
                init_recon=init_recon, prox_input=prox_input,
                first_iteration=first_iteration, rng=rng)

        partition_sequence = [int(val) for val in partition_sequence]
        fm_rmse = [float(val) for val in loss_vectors[0]]
        stop_pct = [100 * float(val) for val in loss_vectors[2]]
        alpha_values = [float(val) for val in loss_vectors[3]]
        delta_norm_per_slice = [[float(v) for v in row] for row in loss_vectors[4]]
        num_iterations = len(fm_rmse)
        recon_params = dict(zip(recon_param_names,
                                [num_iterations, granularity, partition_sequence,
                                 fm_rmse, prior_loss, regularization_params,
                                 stop_pct, alpha_values, delta_norm_per_slice]))
        self.set_params(no_warning=True, sigma_prox=self_sigma_prox)

        if logfile_path:
            self.logger.info('Logs written to {}'.format(
                os.path.abspath(os.path.expanduser(logfile_path))))
        for h in list(self.logger.handlers):
            h.flush()
        self.close_log_file()

        notes = 'Proximal map completed: {}\n\n'.format(datetime.datetime.now())
        recon_dict = self.get_recon_dict(recon_params, notes=notes)
        return (recon if output_sharded else self._gather_recon(recon)), recon_dict

    @staticmethod
    def gen_weights(sinogram, weight_type):
        return vcd_utils.gen_weights(sinogram, weight_type)

    def resize_recon_fov(self, row_scale=1.0, col_scale=1.0, slice_scale=1.0):
        """
        This method resizes the reconstruction field of view (FOV) by the given scale
        factors.  The voxel size does not change, and the method updates ``recon_shape``.
        After the warning "Lateral FOV truncation detected", a typical call is
        ``resize_recon_fov(s, s)`` with ``s >= 1.1``.

        Args:
            row_scale (float): The scale factor for the number of recon rows.
            col_scale (float): The scale factor for the number of recon columns.
            slice_scale (float): The scale factor for the number of recon slices.

        Returns:
            tuple[int, int, int]: The number of voxels added to the rows, the columns, and the
            slices.
        """
        old_rows, old_cols, old_slices = self.get_params('recon_shape')
        new_rows = int(old_rows * row_scale)
        new_cols = int(old_cols * col_scale)
        new_slices = int(old_slices * slice_scale)
        self.set_params(recon_shape=(new_rows, new_cols, new_slices))
        return new_rows - old_rows, new_cols - old_cols, new_slices - old_slices

    def scale_recon_shape(self, row_scale=1.0, col_scale=1.0, slice_scale=1.0):
        warnings.warn('scale_recon_shape is deprecated; use resize_recon_fov instead.', FutureWarning, stacklevel=2)
        return self.resize_recon_fov(row_scale, col_scale, slice_scale)

    def recon_slice_z(self, slice_indices=None):
        """This method returns the axial coordinate, in ALU, of the center of each recon
        slice.  The slices are spaced by ``voxel_slice_aspect * delta_voxel``, and they are
        centered on ``recon_slice_offset``.

        Args:
            slice_indices (int, sequence of int, or None): The slices to map.  If None (the
                default), the method maps every slice.

        Returns:
            float or ndarray: The coordinate of each requested slice.
        """
        recon_shape = self.get_params('recon_shape')
        delta_voxel, voxel_slice_aspect = self.get_params(['delta_voxel', 'voxel_slice_aspect'])
        offset = self.get_params('recon_slice_offset') if 'recon_slice_offset' in self.params else 0.0
        num_slices = int(recon_shape[2])
        k = np.arange(num_slices) if slice_indices is None else np.asarray(slice_indices)
        return voxel_slice_aspect * delta_voxel * (k - (num_slices - 1) / 2.0) + offset

    def _fractional_slice_index(self, z):
        """This method returns the fractional slice index of the axial coordinate ``z``.  This
        method is the inverse of :meth:`recon_slice_z`.  The result has the same type as ``z``."""
        recon_shape = self.get_params('recon_shape')
        delta_voxel, voxel_slice_aspect = self.get_params(['delta_voxel', 'voxel_slice_aspect'])
        offset = self.get_params('recon_slice_offset') if 'recon_slice_offset' in self.params else 0.0
        num_slices = int(recon_shape[2])
        return (z - offset) / (voxel_slice_aspect * delta_voxel) + (num_slices - 1) / 2.0

    def nearest_recon_slice(self, z):
        """This method returns the index of the recon slice whose center is nearest to the
        axial coordinate ``z`` in ALU.  The method clips the index to the volume.  This method is
        the inverse of :meth:`recon_slice_z`."""
        num_slices = int(self.get_params('recon_shape')[2])
        index = int(round(float(self._fractional_slice_index(z))))
        return min(max(index, 0), num_slices - 1)

    def reshape_recon(self, recon):
        """This method reshapes a recon-shaped array to the ``recon_shape`` of the model.  The
        method does not accept a sharded array.
        """
        _sharding.reject_shards('reshape_recon', recon=recon)
        recon_shape = self.get_params('recon_shape')
        return recon.reshape(recon_shape)

    # ── model description and HDF5 persistence ────────────────────────────────

    def get_all_params(self):
        """
        This method returns the parameters of this model as three dicts,
        ``(required_params, optional_params, regularization)``.

        The dict ``required_params`` holds the geometry arguments of the constructor and
        ``geometry_type``.  The dict ``optional_params`` holds the other geometry and detector
        parameters, which ``set_params`` applies.  The dict ``regularization`` holds
        ``sigma_y``, ``sigma_x``, ``sigma_prox``, ``snr_db``, ``sharpness``, and
        ``auto_regularize_flag``.

        Returns:
            tuple: ``(required_params, optional_params, regularization)``.
        """
        import inspect

        regularization_names = _AUTO_REGULARIZATION_PARAM_NAMES + (
            'snr_db', 'sharpness', 'auto_regularize_flag')
        construction_derived_names = ('geometry_type', 'view_params_name', 'file_format',
                                      'version', 'use_gpu')
        environment_args = ('self', 'view_batch_size', 'compile_mode')

        ctor_names = [n for n in inspect.signature(type(self).__init__).parameters
                      if n not in environment_args]
        view_params_name = self.get_params('view_params_name')
        view_array = np.asarray(self.get_params(view_params_name))

        required_params = {}
        for name in ctor_names:
            if name in self.params:
                required_params[name] = self.get_params(name)
            elif name == 'angles':
                required_params[name] = view_array[:, 0] if view_array.ndim == 2 else view_array
            elif name == 'helical_z_shifts' and view_array.ndim == 2:
                required_params[name] = view_array[:, 1]

        optional_params = {}
        for key in self.params:
            if (key in ctor_names or key == view_params_name
                    or key in construction_derived_names or key in regularization_names):
                continue
            optional_params[key] = self.get_params(key)

        required_params['geometry_type'] = str(type(self))

        regularization = {name: self.get_params(name)
                          for name in regularization_names if name in self.params}

        return required_params, optional_params, regularization

    def get_recon_dict(self, recon_params=None, notes=None, save_log=True, save_model=True, str_format=False):
        """
        This method collects the recon parameters, the logs, the notes, and optionally the
        model parameters into a dict.  The user can pass the dict to
        :func:`mbirtorch.view_utils.slice_viewer` and :func:`mbirtorch.export_recon_hdf5`.

        Args:
            recon_params (dict, optional): The dict of reconstruction parameters.  Defaults to
                None.
            notes (str, optional): The user notes to attach to the dataset.  Defaults to None.
            save_log (bool, optional): If True, the method saves the internal log buffer if it
                exists.  Defaults to True.
            save_model (bool, optional): If True, the method saves the model parameters.
                Defaults to True.
            str_format (bool, optional): If True, the method converts each top-level entry to a
                string.

        Returns:
            dict: A dict with entries 'recon_params', 'notes', 'recon_log', and 'model_params'.

        Example:
            >>> recon, recon_dict = ct_model.recon(sinogram)
            >>> print(recon_dict['recon_log'])
        """
        recon_dict = dict()
        if recon_params is None:
            recon_dict['recon_params'] = "# Recon params not saved."
        else:
            recon_dict['recon_params'] = recon_params

        log_buffer = getattr(self, 'log_buffer', None)
        if log_buffer is None or not save_log:
            recon_dict['recon_log'] = "# Log info not saved."
        else:
            recon_dict['recon_log'] = log_buffer.getvalue()

        if notes is None:
            notes = '# No notes saved'
        recon_dict['notes'] = notes

        if save_model:
            recon_dict['model_params'] = {k: v.val for k, v in self.params.items()}
        else:
            recon_dict['model_params'] = '# Model not saved'

        if str_format:
            from .view_utils import convert_subdicts_to_strings
            recon_dict = convert_subdicts_to_strings(recon_dict)

        return recon_dict

    def save_recon_hdf5(self, filepath, recon, recon_dict=None):
        """This method is deprecated.  The function :func:`mbirtorch.export_recon_hdf5`
        replaces it."""
        warnings.warn('save_recon_hdf5 is deprecated; use mbirtorch.export_recon_hdf5.', FutureWarning, stacklevel=2)
        from .utilities import export_recon_hdf5
        export_recon_hdf5(filepath, recon, recon_dict)

    @staticmethod
    def load_recon_hdf5(filepath):
        """This method is deprecated.  The function :func:`mbirtorch.import_recon_hdf5`
        replaces it."""
        warnings.warn('load_recon_hdf5 is deprecated; use mbirtorch.import_recon_hdf5.', FutureWarning, stacklevel=2)
        from .utilities import import_recon_hdf5
        return import_recon_hdf5(filepath)

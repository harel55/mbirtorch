"""This module defines ParallelBeamModel, which is used for reconstruction in a parallel-beam
scanner geometry.  In this geometry, detector row r corresponds to reconstruction slice r.
"""

import os
import warnings

import numpy as np
import torch

from .geometry_rules import channel_index, pixel_xy, rotate_about_z
from .horizontal_fan import fan_back_batch, fan_forward_batch
from .tomography_model import TomographyModel

_F32 = torch.float32


def _parallel_hfan_math(pixel_indices, view_params_batch, num_rows, num_cols,
                        num_channels, delta_det_channel, det_channel_offset,
                        delta_voxel, delta_voxel_row):
    """This function computes, from the view angles, the inputs that the functions in
    horizontal_fan.py need for the parallel-beam geometry.

    It returns (n_p, centers, W_p_c, weight_scale).
    """
    x_tilde, y_tilde = pixel_xy(pixel_indices, num_rows, num_cols, delta_voxel,
                                delta_voxel_row)
    # After rotation, x is the channel coordinate of rays that travel along -y.
    cosine = torch.cos(view_params_batch)[:, None]
    sine = torch.sin(view_params_batch)[:, None]
    x, _ = rotate_about_z(x_tilde, y_tilde, cosine, sine)
    n_p = channel_index(x, delta_det_channel, det_channel_offset, num_channels)

    footprint_xy = torch.maximum(cosine.abs() * delta_voxel,
                                 sine.abs() * delta_voxel_row)
    W_p_c = footprint_xy / delta_det_channel
    weight_scale = (delta_voxel_row * delta_voxel) / footprint_xy
    centers = torch.round(n_p).to(torch.int32)
    return n_p, centers, W_p_c, weight_scale


def _parallel_forward_view_batch(values, pixel_indices, view_params_batch,
                                 num_rows, num_cols, num_channels,
                                 delta_det_channel, det_channel_offset,
                                 delta_voxel, delta_voxel_row, det_channel_radius,
                                 det_row_radius, vox_slice_radius,
                                 slice_start=0, plan=None):
    """This function computes the parallel-beam forward projection of one batch of views.

    Detector row r corresponds to reconstruction slice r.  The function ignores
    ``det_row_radius``, ``vox_slice_radius``, and ``plan``.
    """
    assert slice_start == 0
    hfan_data = _parallel_hfan_math(
        pixel_indices, view_params_batch, num_rows, num_cols, num_channels,
        delta_det_channel, det_channel_offset, delta_voxel, delta_voxel_row)
    block = fan_forward_batch(hfan_data, values, num_channels, det_channel_radius)
    return block.permute(0, 2, 1)


def _parallel_back_view_batch(sino_batch, pixel_indices, view_params_batch,
                              num_rows, num_cols, num_channels,
                              delta_det_channel, det_channel_offset,
                              delta_voxel, delta_voxel_row, det_channel_radius,
                              det_row_radius, vox_slice_radius,
                              coeff_power=1, slice_start=0, band_slices=None,
                              plan=None):
    """This function computes the parallel-beam back projection of one batch of views.

    The function sums the result over the views in the batch.  The function is the adjoint of
    :func:`_parallel_forward_view_batch`.  The function ignores ``det_row_radius``,
    ``vox_slice_radius``, and ``plan``.
    """
    assert slice_start == 0 and band_slices is None
    hfan_data = _parallel_hfan_math(
        pixel_indices, view_params_batch, num_rows, num_cols, num_channels,
        delta_det_channel, det_channel_offset, delta_voxel, delta_voxel_row)
    sino_T = sino_batch.permute(0, 2, 1).contiguous()
    return fan_back_batch(sino_T, hfan_data, num_channels, det_channel_radius,
                          coeff_power=coeff_power, reduce_views=True)


class ParallelBeamModel(TomographyModel):
    """
    This class is used to model the parallel-beam geometry.  In each view the rays are parallel.
    The object rotates about an axis parallel to the detector columns.

    Args:
        sinogram_shape (tuple): (num_views, num_det_rows, num_det_channels).  The channels run
            across the detector, perpendicular to the rotation axis.
        angles (ndarray): The rotation angle of each view, in radians.
        view_batch_size (int or None, optional): The number of views projected at one time.
            Smaller values use less memory and can be slower.  If None, mbirtorch chooses the
            value.  Defaults to None.
        compile_mode (str, optional): If 'auto', mbirtorch compiles the projection kernels with
            ``torch.compile``.  If 'off', mbirtorch does not compile them.  Defaults to 'auto'.

    Example:
        >>> angles = np.linspace(0, np.pi, 180, endpoint=False)
        >>> ct_model = mbirtorch.ParallelBeamModel((180, 256, 10), angles)
    """

    def __init__(self, sinogram_shape, angles,
                 view_batch_size=None, compile_mode='auto'):
        angles = np.asarray(angles, dtype=np.float32)
        super().__init__(sinogram_shape, view_batch_size=view_batch_size,
                         compile_mode=compile_mode,
                         geometry_type='parallel', view_params_name='angles',
                         angles=angles)

    def _project_points_batch(self, points, view_params):
        # points has shape (N, 3), float64, on the CPU.  view_params holds the V view angles.
        ddc, dco = self.get_params(['delta_det_channel', 'det_channel_offset'])
        num_channels = self.get_params('sinogram_shape')[2]
        cosine = torch.cos(view_params)[:, None]
        sine = torch.sin(view_params)[:, None]
        x, _ = rotate_about_z(points[:, 0], points[:, 1], cosine, sine)
        channel = channel_index(x, ddc, dco, num_channels)
        # The detector row of a point equals the fractional slice index of its z coordinate.
        row = self._fractional_slice_index(points[:, 2])[None, :].expand(
            channel.shape[0], -1)
        return row, channel

    def get_magnification(self):
        """This method returns the magnification, which is 1 for parallel beam."""
        return 1.0

    # Detector row r corresponds to slice r.  The multi-GPU code uses this fact.
    rows_track_slices = True

    # This selects the parallel-beam table in ``_min_problem_sizes``.  That table gives the
    # minimum sinogram size for each number of GPUs.
    _size_table_key = 'parallel'

    # On Linux, the torch 2.13.0 CPU compiler gives wrong code for a call with one pixel.
    # For this reason, mbirtorch pads such a call to two pixels before the compiled code.
    min_compiled_pixel_width = 2

    def _get_psf_radii(self):
        """This method returns the tap radii of the parallel-beam projector loops as
        (det_channel_radius, det_row_radius, vox_slice_radius).

        The channel radius is the largest number of detector channels on either side of the
        center channel that a voxel projects onto.  The row and slice radii are 0, because
        detector row r corresponds to slice r.
        """
        delta_det_channel, delta_voxel, voxel_row_aspect = self.get_params(
            ['delta_det_channel', 'delta_voxel', 'voxel_row_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        max_footprint = max(delta_voxel, delta_voxel_row)
        det_channel_radius = int(np.ceil(np.ceil(max_footprint / delta_det_channel) / 2))
        return det_channel_radius, 0, 0

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """This method sets ``recon_shape`` and ``delta_voxel`` from the detector pitches and the
        number of channels.

        mbirtorch calls this method after a geometry parameter such as ``delta_det_channel``
        changes.
        """
        delta_det_row, delta_det_channel = self.get_params(
            ['delta_det_row', 'delta_det_channel'])
        voxel_row_aspect = self.get_params('voxel_row_aspect')

        delta_voxel = self.get_params('delta_det_channel') / self.get_magnification()
        delta_voxel_row = voxel_row_aspect * delta_voxel

        sinogram_shape = self.get_params('sinogram_shape')
        num_det_rows, num_det_channels = sinogram_shape[1:3]
        magnification = self.get_magnification()
        num_recon_rows = int(np.ceil(num_det_channels * delta_det_channel
                                     / (delta_voxel_row * magnification)))
        num_recon_cols = int(np.ceil(num_det_channels * delta_det_channel
                                     / (delta_voxel * magnification)))
        num_recon_slices = int(np.round(num_det_rows * ((delta_det_row / delta_voxel)
                                                        / magnification)))
        recon_shape = (num_recon_rows, num_recon_cols, num_recon_slices)
        self.set_params(no_compile=no_compile, no_warning=no_warning,
                        recon_shape=recon_shape, delta_voxel=delta_voxel)

    def verify_valid_params(self):
        """This method raises a ValueError if the parameters are not valid for a reconstruction."""
        super().verify_valid_params()
        sinogram_shape, angles, voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['sinogram_shape', 'angles', 'voxel_row_aspect', 'voxel_slice_aspect'])

        if voxel_row_aspect <= 0:
            raise ValueError('Voxel row aspect ratio must be positive. \n'
                             f'Got {voxel_row_aspect} for voxel_row_aspect.')
        if voxel_slice_aspect != 1.0:
            raise ValueError('Setting voxel slice aspect ratio is not supported for '
                             f'parallel beam model. \nGot {voxel_slice_aspect}.')
        if np.asarray(angles).shape[0] != sinogram_shape[0]:
            raise ValueError('Number of view dependent parameter vectors must equal '
                             'the number of views.')
        recon_shape = self.get_params('recon_shape')
        if recon_shape[2] != sinogram_shape[1]:
            raise ValueError('Number of recon slices must match number of sinogram '
                             f'rows. \nGot {recon_shape} for recon_shape and '
                             f'{sinogram_shape} for sinogram_shape')

    def _view_batch_bodies(self):
        # The environment variable MBIRTORCH_DISABLE_TRITON=1 turns off the Triton kernels.
        from .kernel_availability import (parallel_back_kernel_usable,
                                          parallel_forward_kernel_usable)
        if parallel_back_kernel_usable(self)[0]:
            from .triton_parallel import _parallel_back_view_batch_triton
            back_body = _parallel_back_view_batch_triton
        else:
            back_body = _parallel_back_view_batch
        if parallel_forward_kernel_usable(self)[0]:
            from .triton_parallel import _parallel_forward_view_batch_triton
            fwd_body = _parallel_forward_view_batch_triton
        else:
            fwd_body = _parallel_forward_view_batch
        return fwd_body, back_body

    def _view_batch_args(self):
        gp_names = ['delta_det_channel', 'det_channel_offset', 'delta_voxel',
                    'voxel_row_aspect']
        delta_det_channel, det_channel_offset, delta_voxel, voxel_row_aspect = \
            self.get_params(gp_names)
        num_channels = self.get_params('sinogram_shape')[2]
        recon_shape = self.get_params('recon_shape')
        det_channel_radius, det_row_radius, vox_slice_radius = self._get_psf_radii()
        return dict(num_rows=recon_shape[0], num_cols=recon_shape[1],
                    num_channels=num_channels,
                    delta_det_channel=delta_det_channel,
                    det_channel_offset=det_channel_offset,
                    delta_voxel=delta_voxel,
                    delta_voxel_row=voxel_row_aspect * delta_voxel,
                    det_channel_radius=det_channel_radius,
                    det_row_radius=det_row_radius,
                    vox_slice_radius=vox_slice_radius)

    def fbp_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        This method applies the FBP filter to a sinogram.

        Args:
            sinogram (numpy or tensor): The sinogram, with shape
                (num_views, num_rows, num_channels).
            filter_name (string, optional): The name of the filter.  Defaults to "ramp".
            output_sharded (bool, optional): If False, the method returns a numpy array.  If
                True, it returns the tensor on the device.  Defaults to False.

        Returns:
            The filtered sinogram.
        """
        # The scale adjusts the filter for the voxel size.  See
        # https://mbirtorch.readthedocs.io/en/latest/theory.html
        delta_voxel, voxel_row_aspect = self.get_params(['delta_voxel',
                                                         'voxel_row_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        scaling_factor = 1.0 / (delta_voxel * delta_voxel_row)
        return self._apply_direct_recon_filter(sinogram, filter_name,
                                               filter_scale=scaling_factor,
                                               output_sharded=output_sharded)

    def recon_fbp(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        This method reconstructs a volume from a sinogram by filtered back projection (FBP).

        The back projection is the exact adjoint of the forward projector.

        Note:
            FBP assumes equally spaced view angles over the full angular range.  For other data,
            FBP gives an approximate result.  This result is useful as an initial value for
            ``recon()``.

        Args:
            sinogram (numpy or tensor): The sinogram, with shape
                (num_views, num_rows, num_channels).
            filter_name (string, optional): The name of the filter.  Defaults to "ramp".
            output_sharded (bool, optional): If False, the method returns a numpy array.  If
                True, the method returns the result on the devices.  This result is a torch
                tensor for a model on one device, or a Shards container with one tensor per
                device for a model on several devices.  Defaults to False.

        Returns:
            recon (numpy or tensor): The reconstructed volume.
        """
        # This call chooses the GPUs and checks memory for a direct reconstruction before any allocation.
        self._apply_device_policy(workload='direct')
        # The filter places the sinogram on the devices.  This method keeps no reference to the
        # device sinogram during the back projection.
        filtered_sinogram = self.fbp_filter(sinogram, filter_name=filter_name,
                                            output_sharded=True)
        recon = self._back_project(filtered_sinogram, output_sharded=True)
        return recon if output_sharded else self._gather_recon(recon)

    def recon_direct(self, sinogram, filter_name="ramp", output_sharded=False):
        """This method reconstructs a volume by filtered back projection (FBP).

        This method is the same as :meth:`recon_fbp`.  See :meth:`TomographyModel.recon_direct`
        for the arguments and return value.
        """
        return self.recon_fbp(sinogram, filter_name=filter_name,
                              output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        return self.fbp_filter(sinogram, filter_name=filter_name,
                               output_sharded=output_sharded)

    def recon_split_sino(self, sino, weights=None, half_overlap=5, init_recon=None, max_iterations=15,
                         stop_threshold_change_pct=0.2, first_iteration=0, compute_prior_loss=False,
                         logfile_path='~/.mbirtorch/logs/recon.log', print_logs=True,
                         align_split_grid=False, slices_per_section=None, min_slices_per_section=200,
                         slices_per_part=None):
        """
        This method reconstructs a parallel-beam volume in sections to reduce the GPU memory
        needed.  It is meant for a volume that does not fit on the GPUs as a whole.

        Detector row r corresponds to reconstruction slice r.  For this reason, each section of
        detector rows gives the matching section of slices.  The method reconstructs each section
        with ordinary MBIR.  Each section includes a few extra rows beyond each boundary with its
        neighbors.  The method joins the sections with a linear blend.  The method divides the
        GPUs into as many equal groups as possible such that each group can hold a section.  The
        groups reconstruct their sections at the same time.

        This method is used like :meth:`recon`, and its result is approximately equal to the
        result of :meth:`recon`.  This method does not accept ``output_sharded``.

        Args:
            sino (numpy or tensor): The full sinogram, with shape (num_views, num_rows, num_cols).
                The method does not accept a sharded array.
            weights (numpy or tensor, optional): The sinogram weights, with the same shape as
                ``sino``.  The method does not accept a sharded array.
            half_overlap (int): The number of extra detector rows, and slices, that each section
                includes on each side of a boundary.  Neighboring sections overlap by
                ``2 * half_overlap`` slices.
            init_recon (optional): The same as in the recon method.  The method does not accept a
                sharded array.
            max_iterations (int, optional): The same as in the recon method.
            stop_threshold_change_pct (float, optional): The same as in the recon method.
            first_iteration (int, optional): The same as in the TomographyModel.recon() method.
            compute_prior_loss (bool, optional): This argument is accepted so that the interface
                matches other methods.  The method does not use it.
            logfile_path (str, optional): The same as in the TomographyModel.recon() method.  The
                method merges the logs of all sections into this one file, with a header for
                each section.
            print_logs (bool, optional): The same as in the TomographyModel.recon() method.
            align_split_grid (bool, optional): This argument is accepted so that the interface
                matches the cone-beam method.  It has no effect for parallel beam.
            slices_per_section (int, optional): The number of slices in each section.  It must
                be at least ``2 * half_overlap``.  If None, the method chooses the sections from
                the memory of the GPUs.  If the value is at least the number of slices, the
                method uses one section.  Defaults to None.
            min_slices_per_section (int, optional): The smallest number of slices in a section
                when the method chooses the sections.  Defaults to 200.
            slices_per_part (int, optional): The deprecated name of ``slices_per_section``.

        Returns:
            Tuple[np.ndarray, dict]: The reconstructed volume as a numpy array, and a metadata
                dictionary.  The dictionary holds the recon parameters and model parameters of
                each section.  Its entry 'split_params' holds the overlap, the number of
                sections, the slice range of each section, the number of GPU groups, and the
                number of GPUs in each group.  If there is one section, the method calls
                recon() and returns the dictionary from recon().  The method issues a warning
                if the volume has too few slices to split.

        Raises:
            ValueError: If an input is missing, if the shapes do not agree, if half_overlap < 2,
                if ``slices_per_section`` gives a section with fewer than ``2 * half_overlap``
                slices, or if ``sino``, ``weights``, or ``init_recon`` is a sharded array.
            AssertionError: If an array has the wrong number of dimensions.

        Example:
            >>> sino = np.ones((180, 64, 64), dtype=np.float32)  # (views, rows, cols)
            >>> model = mbirtorch.ParallelBeamModel(sinogram_shape=sino.shape,
            ...                                     angles=np.linspace(0, np.pi, 180))
            >>> recon, recon_info = model.recon_split_sino(sino, half_overlap=4)
        """
        from . import _sharding
        from .utilities import copy_ct_model, stitch_arrays, merge_log_files

        if slices_per_part is not None:
            warnings.warn('slices_per_part is deprecated; use slices_per_section.',
                          DeprecationWarning, stacklevel=2)
            if slices_per_section is None:
                slices_per_section = slices_per_part
        if half_overlap < 2:
            raise ValueError('half_overlap must be >= 2.')
        if sino is None:
            raise ValueError("sino must be provided.")
        # Each section chooses its own GPUs, so the method refuses an input that is already on the GPUs.
        if (isinstance(sino, _sharding.Shards)
                or isinstance(weights, _sharding.Shards)
                or isinstance(init_recon, _sharding.Shards)):
            raise ValueError(
                'recon_split_sino does not accept a sinogram, weights, or an '
                'initial reconstruction in sharded form.  Pass the host (numpy '
                'or tensor) arrays.')
        if not (hasattr(sino, "ndim") and sino.ndim == 3):
            raise AssertionError("sino must be a 3D array shaped (num_views, num_rows, num_cols).")
        if weights is not None and getattr(weights, "shape", None) != sino.shape:
            raise AssertionError("weights, if provided, must have the same shape as sino.")

        # The method splits the sinogram in host memory, so the full sinogram is never on the GPUs.
        if isinstance(sino, torch.Tensor):
            sino = sino.detach().cpu().numpy()
        sino = np.asarray(sino)
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
            weights = np.asarray(weights)
        if init_recon is not None and isinstance(init_recon, torch.Tensor):
            init_recon = self._gather_recon(init_recon)

        num_rows = sino.shape[1]
        recon_rows, recon_cols = self.get_params('recon_shape')[:2]

        def _section_model(num_section_rows):
            """This function returns a copy of this model with ``num_section_rows`` detector rows."""
            model = copy_ct_model(self, new_num_det_rows=num_section_rows, no_warning=True)
            # Each section uses the regularization of this model, which is set from the full sinogram.
            model.set_params(no_warning=True, auto_regularize_flag=False)
            model.set_params(recon_shape=(recon_rows, recon_cols, num_section_rows))
            return model

        def _largest_section_rows(num_sections):
            """This function returns the number of rows in the largest section when there are
            ``num_sections`` sections.

            The count includes the extra rows on each side of a boundary.
            """
            biggest_kept = -(-num_rows // num_sections)
            if num_sections == 1:
                return biggest_kept
            if num_sections == 2:
                return biggest_kept + half_overlap
            return biggest_kept + 2 * half_overlap

        # Each section keeps at least 2 * half_overlap slices.
        max_sections = num_rows // (2 * half_overlap)
        weighted = weights is not None
        devices = self._split_devices()
        gpus_per_group = len(devices)
        estimated = False
        if slices_per_section is not None:
            if slices_per_section < 2 * half_overlap:
                raise ValueError(
                    f'slices_per_section must be at least 2 * half_overlap = {2 * half_overlap}; '
                    f'got {slices_per_section}.')
            num_sections = -(-num_rows // int(slices_per_section))
            if num_sections > 1 and num_sections > max_sections:
                raise ValueError(
                    f'slices_per_section={slices_per_section} gives {num_sections} sections of about '
                    f'{num_rows // num_sections} slices each, which is below 2 * half_overlap = '
                    f'{2 * half_overlap}; use at most {max_sections} sections, or a smaller '
                    f'half_overlap.')
            if num_sections > 1:
                section = _section_model(_largest_section_rows(num_sections))
                gpus_per_group = self._choose_gpus_per_group(
                    devices, lambda group: self._section_fits(section, group, weighted))
        elif max_sections < 2:
            # If there are fewer than 4 * half_overlap slices, the volume cannot be split.
            warnings.warn(
                "the volume has too few slices to split at this half_overlap; "
                "falling back to standard MBIR reconstruction.",
                UserWarning,
            )
            num_sections = 1
        else:
            min_rows = min(int(min_slices_per_section) + 2 * half_overlap, num_rows)
            smallest_section = _section_model(min_rows)
            gpus_per_group = self._choose_gpus_per_group(
                devices, lambda group: self._section_fits(smallest_section, group, weighted))
            num_sections = max_sections
            if gpus_per_group == len(devices):
                # With one group, choose the fewest sections whose largest section fits on the GPUs.
                for candidate in range(1, max_sections + 1):
                    if _section_model(_largest_section_rows(candidate))._fits_available_devices():
                        num_sections = candidate
                        break
            else:
                # With several groups, choose the fewest sections that fit on one group.  Then
                # use at least one section per group, if each section keeps min_slices_per_section slices.
                group = devices[:gpus_per_group]
                for candidate in range(1, max_sections + 1):
                    if self._section_fits(_section_model(_largest_section_rows(candidate)),
                                          group, weighted):
                        num_sections = candidate
                        break
                num_groups = len(devices) // gpus_per_group
                num_sections = max(num_sections,
                                   min(num_groups, num_rows // int(min_slices_per_section), max_sections))
            estimated = True

        if num_sections == 1:
            return self.recon(
                sino,
                weights=weights,
                init_recon=init_recon,
                max_iterations=max_iterations,
                stop_threshold_change_pct=stop_threshold_change_pct,
                first_iteration=first_iteration,
                logfile_path=logfile_path,
                print_logs=print_logs,
            )

        # The section slice ranges divide [0, num_rows) into nearly equal parts with no overlap.
        base, extra = divmod(num_rows, num_sections)
        section_ranges, start = [], 0
        for index in range(num_sections):
            stop = start + base + (1 if index < extra else 0)
            section_ranges.append((start, stop))
            start = stop

        # This sets the regularization from the full sinogram, as recon would do.
        self.auto_set_regularization_params(sino, weights=weights)

        def _recon_one_section(model_lo, model_hi, section_logfile_path, devices, rng):
            """This function reconstructs one section of detector rows on ``devices``.

            It returns (host_recon, recon_dict).  If ``devices`` is None, the section model
            chooses the GPUs.
            """
            model = _section_model(model_hi - model_lo)
            if devices is not None:
                model.configure_devices(devices=devices)

            sino_section = sino[:, model_lo:model_hi, :]
            weights_section = None if weights is None else weights[:, model_lo:model_hi, :]
            # Rows correspond to slices, so the initial reconstruction of the section uses the same index range.
            section_init = None if init_recon is None else init_recon[:, :, model_lo:model_hi]

            return model.recon(sino_section, weights=weights_section, init_recon=section_init,
                               max_iterations=max_iterations,
                               stop_threshold_change_pct=stop_threshold_change_pct,
                               first_iteration=first_iteration,
                               logfile_path=section_logfile_path,
                               print_logs=print_logs, rng=rng)

        # Each section writes its own log file.  The finally block merges the files.
        if logfile_path:
            log_path = os.path.expanduser(logfile_path)
            section_log_paths = [log_path + '.section{}'.format(index) for index in range(num_sections)]
        else:
            log_path, section_log_paths = None, [None] * num_sections

        def _job(index, lo, hi):
            """This function returns the job that reconstructs section ``index``."""
            model_lo, model_hi = max(lo - half_overlap, 0), min(hi + half_overlap, num_rows)
            return lambda group, rng: _recon_one_section(model_lo, model_hi, section_log_paths[index],
                                                         group, rng)

        jobs = [_job(index, lo, hi) for index, (lo, hi) in enumerate(section_ranges)]
        try:
            results = self._run_split_sections(jobs, devices, gpus_per_group)
        finally:
            if log_path:
                labels = ['recon_split_sino: section {} of {} (slices {}-{})'.format(
                    index + 1, num_sections, lo, hi - 1)
                    for index, (lo, hi) in enumerate(section_ranges)]
                merge_log_files(log_path, zip(labels, section_log_paths))
        section_recons = [recon for recon, _ in results]
        section_dicts = [recon_dict for _, recon_dict in results]

        # ramp_overlap is the number of blended slices.  It is even, at most 4, and at most half_overlap.
        ramp_overlap = 4
        ramp_overlap = min(ramp_overlap, half_overlap)
        ramp_overlap -= ramp_overlap % 2
        recon_full = stitch_arrays(section_recons, axis=2, overlap=2 * half_overlap,
                                   ramp_overlap=ramp_overlap)

        # The last three split_params entries are not used for parallel beam.  They are present
        # so that the fields match the cone-beam method.
        recon_full_dict = {'recon_params_sections': [d.get('recon_params') for d in section_dicts],
                           'recon_log_sections': [d.get('recon_log', '# Log info not saved.')
                                                  for d in section_dicts],
                           'notes_sections': [d.get('notes', '# No notes saved') for d in section_dicts],
                           'model_params_sections': [d.get('model_params') for d in section_dicts],
                           'split_params': {'half_overlap_sino': int(half_overlap),
                                            'half_overlap_recon': int(half_overlap),
                                            'num_sections': int(num_sections),
                                            'section_slice_ranges': [(int(lo), int(hi))
                                                                     for lo, hi in section_ranges],
                                            'slices_per_section': int(max(hi - lo for lo, hi
                                                                          in section_ranges)),
                                            'estimated': bool(estimated),
                                            'num_groups': len(devices) // gpus_per_group,
                                            'gpus_per_group': int(gpus_per_group),
                                            'align_split_grid': bool(align_split_grid),
                                            'grid_shift_alu': 0.0,
                                            'split_cut_mismatch_slices': 0.0}, }

        return recon_full, recon_full_dict


def recon_simple_parallel(sinogram, angles, weights=None, sharpness=1.0,
                          max_iterations=15):
    """
    This function performs a basic parallel-beam reconstruction in one call with the default
    geometry parameters.

    For other settings, create a :class:`ParallelBeamModel` and call
    :meth:`TomographyModel.recon`.

    Args:
        sinogram (numpy or tensor): The 3D sinogram, with shape
            (num_views, num_det_rows, num_det_channels).
        angles (numpy or tensor): A 1D array of projection angles in radians, one per view.
        weights (numpy or tensor, optional): The 3D positive weights, with the same shape as the
            sinogram.  If None, all weights are 1.  Defaults to None.
        sharpness (float, optional): Higher values give sharper edges and more noise.  Lower
            values give softer edges and less noise.  Defaults to 1.0.
        max_iterations (int, optional): The maximum number of iterations.  Defaults to 15.  If
            max_iterations is 0, the function returns a filtered back projection scaled to fit
            the data.

    Returns:
        (recon, recon_dict): The reconstructed volume and a dict.  The dict has the entries
        'recon_params', 'recon_log', 'notes', and 'model_params'.  'recon_params' holds the
        settings and the values recorded at each iteration.  'recon_log' holds the log text.
        'model_params' holds a copy of the model parameters.

    Example:
        >>> import numpy as np, mbirtorch
        >>> angles = np.linspace(0, np.pi, 180, endpoint=False)
        >>> recon, recon_dict = mbirtorch.recon_simple_parallel(sinogram, angles)
    """
    # The geometry comes from the shape of the full sinogram, so a sharded array is refused.
    from . import _sharding
    _sharding.reject_shards('recon_simple_parallel', sinogram=sinogram,
                            weights=weights)
    if torch.is_tensor(angles):
        angles = angles.detach().cpu().numpy()
    model = ParallelBeamModel(tuple(sinogram.shape), angles)
    model.set_params(sharpness=sharpness)
    return model.recon(sinogram, weights=weights, max_iterations=max_iterations)

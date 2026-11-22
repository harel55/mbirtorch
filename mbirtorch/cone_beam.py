"""This module defines ConeBeamModel.  The model supports flat and curved detectors, and it
supports circular and helical scans.

The cone-beam projection is computed in two separable steps.  The horizontal step maps a voxel
to detector channels as in parallel beam, with a magnification for each pixel.  The vertical
step maps each slice of a column of voxels to a range of detector rows.  This step uses the
affine map of :func:`_cone_vertical_affine`.  The forward and back projectors are exact
adjoints.  The order of the arithmetic in them must not be changed.
"""

import os

import numpy as np
import torch
import warnings

from .geometry_rules import channel_index, pixel_xy, rotate_about_z, row_index
from .horizontal_fan import fan_back_batch, fan_forward_batch
from .projectors import maybe_compile
from .qggmrf import qggmrf_slice_pair_sum
from .tomography_model import TomographyModel
from .vcd_utils import get_support_radius

_F32 = torch.float32

# The parameters (a, b, p, c) of the slice damping s_k = (c t^p + a b^p)/(t^p + b^p), where
# t_k = L |z_k| / (R dz).  None turns damping off.  The convergence tests depend on these values.
_DC_DAMPING_DEFAULT = (0.25, 100.0, 0.7, 0.5)


def _dc_damped_update_direction(forward_grad, prior_grad, forward_hess,
                                prior_hess, s_row, slice_sums):
    # d = -(g - (1 - s_k) gbar_k) / H.  gbar_k is the mean of g over the slice, weighted by H^-1.
    # The function also returns the slice term of the exact line search, or None if slice_sums
    # is None.
    g = forward_grad + prior_grad
    h_inv = 1.0 / (forward_hess + prior_hess)
    gbar = torch.sum(h_inv * g, dim=0) / torch.sum(h_inv, dim=0)
    slice_offsets = (1.0 - s_row) * gbar
    direction = -(g - slice_offsets[None, :]) * h_inv
    if slice_sums is None:
        return direction, None
    return direction, qggmrf_slice_pair_sum(slice_sums, slice_offsets)


def _cone_pixel_xy_mag(pixel_indices, angles, num_rows, num_cols, delta_voxel,
                       delta_voxel_row, magnification, source_detector_dist):
    """This function returns the rotated in-plane coordinates x and y and the magnification of
    each pixel.

    Each output has shape (Vb, P).  The function is also correct when source_detector_dist is
    infinite.
    """
    x_tilde, y_tilde = pixel_xy(pixel_indices, num_rows, num_cols, delta_voxel,
                                delta_voxel_row)
    return _cone_xy_mag(x_tilde, y_tilde, angles, magnification,
                        source_detector_dist)


def _cone_xy_mag(x_tilde, y_tilde, angles, magnification, source_detector_dist):
    """This function returns the rotated in-plane coordinates and the magnification of each
    position.  Each output has shape (Vb, P)."""
    cosine = torch.cos(angles)[:, None]
    sine = torch.sin(angles)[:, None]
    x, y = rotate_about_z(x_tilde, y_tilde, cosine, sine)
    pixel_mag = 1.0 / (1.0 / magnification - y / source_detector_dist)
    return x, y, pixel_mag


def _cone_channel_coordinate(x, y, pixel_mag, magnification,
                             source_detector_dist, use_curved_detector):
    """This function returns the detector coordinate u along the channels for rotated points.

    For a curved detector, u is the arc length along the cylinder, and u is positive toward +x.
    """
    if not use_curved_detector:
        return pixel_mag * x
    source_iso_dist = source_detector_dist / magnification
    return source_detector_dist * torch.atan2(x, source_iso_dist - y)


def _cone_row_coordinate(pixel_mag, z):
    """This function returns the detector coordinate v along the rows.

    The coordinate v is the height z of a point multiplied by its magnification.  This formula
    holds for both flat and curved detectors.
    """
    return pixel_mag * z


def _cone_horizontal_data(pixel_indices, angles, num_rows, num_cols, num_channels,
                          delta_voxel, delta_voxel_row, delta_det_channel,
                          det_channel_offset, magnification, source_detector_dist,
                          use_curved_detector):
    """This function computes the inputs of the horizontal projection step for one batch of
    views.

    It returns (n_p, centers, W_p_c, weight_scale, pixel_mag).  Each output has shape (Vb, P).
    ``pixel_mag`` is the magnification of each pixel.
    """
    x, y, pixel_mag = _cone_pixel_xy_mag(pixel_indices, angles, num_rows, num_cols,
                                         delta_voxel, delta_voxel_row,
                                         magnification, source_detector_dist)
    u = _cone_channel_coordinate(x, y, pixel_mag, magnification,
                                 source_detector_dist, use_curved_detector)
    if not use_curved_detector:
        theta = torch.atan2(u, torch.as_tensor(source_detector_dist, dtype=_F32,
                                               device=u.device))
    else:
        theta = u / source_detector_dist

    n_p = channel_index(u, delta_det_channel, det_channel_offset, num_channels)
    footprint_xy = torch.maximum((angles[:, None] - theta).cos().abs() * delta_voxel,
                                 (angles[:, None] - theta).sin().abs() * delta_voxel_row)
    if not use_curved_detector:
        # A flat detector needs this 1/cos(theta) factor for oblique rays.  A curved detector does not.
        W_p_c = pixel_mag * (footprint_xy / delta_det_channel) / torch.cos(theta)
    else:
        W_p_c = pixel_mag * (footprint_xy / delta_det_channel)
    weight_scale = (delta_voxel_row * delta_voxel) / footprint_xy
    centers = torch.round(n_p).to(torch.int32)
    return n_p, centers, W_p_c, weight_scale, pixel_mag


def _cone_vertical_affine(pixel_mag, z_shifts, num_slices, delta_voxel_slice,
                          delta_det_row, det_row_offset, recon_slice_offset,
                          num_rows_r):
    """This function returns the affine map m = m0 + W_p_r * l from slice index l to detector
    row m.

    The slice index l counts slices in the whole volume.

    Args:
        pixel_mag: (torch.Tensor) The magnification of each view and pixel, shape (Vb, P).
        z_shifts: (torch.Tensor) The helical z shift of each view, shape (Vb,).
        num_slices (int): The number of slices in the whole reconstruction.  This value does not
            depend on the range of slices being projected.
        delta_voxel_slice, delta_det_row, det_row_offset, recon_slice_offset: The scalar
            parameters of the vertical geometry.
        num_rows_r (int): The number of detector rows.

    Returns:
        (m0, W_p_r, z_offset): ``m0`` is the detector row of slice 0, with shape (Vb, P).
        ``W_p_r`` is the number of rows per slice, with shape (Vb, P).  ``z_offset`` is the z
        offset of each view, with shape (Vb,).
    """
    z_offset = recon_slice_offset - z_shifts
    W_p_r = pixel_mag * delta_voxel_slice / delta_det_row
    z_at_slice_0 = z_offset[:, None] - delta_voxel_slice * (num_slices - 1) / 2.0
    m0 = row_index(_cone_row_coordinate(pixel_mag, z_at_slice_0),
                   delta_det_row, det_row_offset, num_rows_r)
    return m0, W_p_r, z_offset


def _cone_forward_view_batch(values, pixel_indices, view_params_batch,
                             num_rows_r, num_channels, num_recon_rows,
                             num_recon_cols, num_slices, delta_voxel,
                             delta_voxel_row, delta_voxel_slice,
                             delta_det_channel, delta_det_row,
                             det_channel_offset, det_row_offset, recon_slice_offset,
                             magnification, source_detector_dist,
                             use_curved_detector, det_channel_radius, det_row_radius,
                             vox_slice_radius, slice_start=0, plan=None):
    """This function computes the cone-beam forward projection of one batch of views.

    The output has shape (Vb, R, C).  ``values`` can hold only some of the slices.  In that case
    ``values`` has shape (P, L) and holds the L slices that start at ``slice_start``.  The sum of
    the outputs over a set of slice ranges that covers all slices equals the full projection.
    The function ignores ``plan``.
    """
    angles = view_params_batch[:, 0]
    z_shifts = view_params_batch[:, 1]
    n_p, centers, W_p_c, weight_scale, pixel_mag = _cone_horizontal_data(
        pixel_indices, angles, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset,
        magnification, source_detector_dist, use_curved_detector)
    vb, num_pixels = n_p.shape
    dev = values.device

    m0, W_p_r, z_offset = _cone_vertical_affine(
        pixel_mag, z_shifts, num_slices, delta_voxel_slice, delta_det_row,
        det_row_offset, recon_slice_offset, num_rows_r)

    # The values are scaled by 1/cos(phi), where phi is the vertical cone angle of each voxel.
    band_len = values.shape[1]
    k = torch.arange(slice_start, slice_start + band_len, dtype=_F32, device=dev)
    z = (delta_voxel_slice * (k - (num_slices - 1) / 2.0))[None, None, :] \
        + z_offset[:, None, None]
    v_slices = pixel_mag.unsqueeze(-1) * z
    cos_phi = torch.cos(torch.atan2(v_slices, torch.as_tensor(
        source_detector_dist, dtype=_F32, device=dev)))
    scaled_values = values[None, :, :] / cos_phi

    # Detector row m maps to the fractional slice index k_m = (m - m0) / W_p_r.
    m = torch.arange(num_rows_r, dtype=_F32, device=dev)
    k_m = (m[None, None, :] - m0.unsqueeze(-1)) / W_p_r.unsqueeze(-1)
    k_center = torch.round(k_m).to(torch.int64)

    slope = W_p_r.unsqueeze(-1)
    L_max_r = torch.clamp(W_p_r, max=1.0).unsqueeze(-1)
    m_p = slope * (k_center.to(_F32) - k_m)

    det_col = torch.zeros((vb, num_pixels, num_rows_r), dtype=_F32, device=dev)
    for k_off in range(-vox_slice_radius, vox_slice_radius + 1):
        k_ind = k_center + k_off
        A = torch.clamp((slope + 1.0) / 2.0 - (m_p + slope * k_off).abs(), min=0.0)
        A = torch.minimum(A, L_max_r)
        A = A * ((k_ind >= slice_start)
                 & (k_ind < slice_start + band_len)).to(_F32)
        g = torch.gather(scaled_values, 2,
                         (k_ind - slice_start).clamp(0, band_len - 1))
        det_col = det_col + A * g

    acc = fan_forward_batch((n_p, centers, W_p_c, weight_scale), det_col,
                            num_channels, det_channel_radius)
    return acc.permute(0, 2, 1)


def _cone_back_view_batch(sino_batch, pixel_indices, view_params_batch,
                          num_rows_r, num_channels, num_recon_rows,
                          num_recon_cols, num_slices, delta_voxel,
                          delta_voxel_row, delta_voxel_slice,
                          delta_det_channel, delta_det_row, det_channel_offset,
                          det_row_offset, recon_slice_offset, magnification,
                          source_detector_dist, use_curved_detector, det_channel_radius,
                          det_row_radius, vox_slice_radius, coeff_power=1, slice_start=0,
                          band_slices=None, plan=None):
    """This function computes the cone-beam back projection of one batch of views.

    The function sums the result over the views in the batch.  The output has shape (P, S).  If
    ``band_slices`` is given, the output has shape (P, band_slices) and holds the slices that
    start at ``slice_start``.  The function ignores ``plan``.
    """
    angles = view_params_batch[:, 0]
    z_shifts = view_params_batch[:, 1]
    n_p, centers, W_p_c, weight_scale, pixel_mag = _cone_horizontal_data(
        pixel_indices, angles, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset,
        magnification, source_detector_dist, use_curved_detector)
    vb, num_pixels = n_p.shape
    dev = sino_batch.device

    sino_T = sino_batch.permute(0, 2, 1).contiguous()
    det_col = fan_back_batch(sino_T, (n_p, centers, W_p_c, weight_scale),
                             num_channels, det_channel_radius,
                             coeff_power=coeff_power, reduce_views=False)

    m0, W_p_r, z_offset = _cone_vertical_affine(
        pixel_mag, z_shifts, num_slices, delta_voxel_slice, delta_det_row,
        det_row_offset, recon_slice_offset, num_rows_r)

    band_len = num_slices if band_slices is None else band_slices
    k = torch.arange(slice_start, slice_start + band_len, dtype=_F32, device=dev)
    z = (delta_voxel_slice * (k - (num_slices - 1) / 2.0))[None, None, :] \
        + z_offset[:, None, None]
    v_slices = pixel_mag.unsqueeze(-1) * z
    sdd_t = torch.as_tensor(source_detector_dist, dtype=_F32, device=dev)
    cos_phi = torch.cos(torch.atan2(v_slices, sdd_t))
    slope = W_p_r.unsqueeze(-1)
    m_p = m0.unsqueeze(-1) + slope * k[None, None, :]
    m_center = torch.round(m_p).to(torch.int64)
    L_max_r = torch.clamp(slope, max=1.0)

    out = torch.zeros((num_pixels, band_len), dtype=_F32, device=dev)
    for m_off in range(-det_row_radius, det_row_radius + 1):
        mm = m_center + m_off
        L = torch.clamp((slope + 1.0) / 2.0 - (m_p - mm.to(_F32)).abs(), min=0.0)
        A = torch.minimum(L, L_max_r) / cos_phi
        A = A * ((mm >= 0) & (mm < num_rows_r)).to(_F32)
        if coeff_power != 1:
            A = A ** coeff_power
        g = torch.gather(det_col, 2, mm.clamp(0, num_rows_r - 1))
        out = out + torch.einsum("vps,vps->ps", A, g)
    return out


class ConeBeamModel(TomographyModel):
    """
    This class is used to compute forward and back projections in a cone-beam geometry.  It
    extends :class:`TomographyModel`.

    Args:
        sinogram_shape (tuple): (num_views, num_det_rows, num_det_channels).
        angles (ndarray): A 1D array of projection angles in radians.
        source_detector_dist (float): The distance from the source to the detector in ALU.
        source_iso_dist (float): The distance from the source to iso, the center of rotation,
            in ALU.
        helical_z_shifts (ndarray, optional): The z shift of each view for a helical scan.  If
            None, the scan is circular.  Defaults to None.
        use_curved_detector (bool): If False, the detector is flat.  If True, the detector is a
            cylinder with radius source_detector_dist.  Defaults to False.
        view_batch_size, compile_mode: The same as in ParallelBeamModel.

    Note:
        Two parameters are set with ``set_params`` after the model is built.

        **recon_slice_offset** (float, default 0.0): The position of the center of the
        reconstruction along the rotation axis, in ALU.  ``auto_set_recon_geometry`` centers
        the reconstruction on the part of the object that the detector sees.  A positive value
        moves the reconstruction toward higher detector row indices.

        **axial_pad_fraction** (float or (top, bottom) pair, default 0.0): This parameter sets
        how far ``auto_set_recon_geometry`` extends the reconstruction at each end of the
        rotation axis.  A value of 0 adds no slices.  A value of 1 extends the reconstruction to
        the farthest point that any measured ray reaches.  The top end has the low slice and
        detector row indices.
    """

    def __init__(self, sinogram_shape, angles, source_detector_dist, source_iso_dist,
                 helical_z_shifts=None, use_curved_detector=False,
                 view_batch_size=None, compile_mode='auto'):
        angles = np.asarray(angles, dtype=np.float32).flatten()
        if helical_z_shifts is None:
            helical_z_shifts = np.zeros_like(angles)
        else:
            helical_z_shifts = np.asarray(helical_z_shifts, dtype=np.float32).flatten()
        if helical_z_shifts.shape != angles.shape:
            raise ValueError("Incompatible view dependent vector lengths: all "
                             "view-dependent vectors must have the same length.")
        view_params_array = np.stack([angles, helical_z_shifts], axis=1)
        super().__init__(sinogram_shape,
                         view_batch_size=view_batch_size, compile_mode=compile_mode,
                         geometry_type='cone', view_params_name='view_params_array',
                         view_params_array=view_params_array,
                         source_detector_dist=source_detector_dist,
                         source_iso_dist=source_iso_dist,
                         recon_slice_offset=0.0, axial_pad_fraction=0.0,
                         use_curved_detector=use_curved_detector)

    _dc_damping = _DC_DAMPING_DEFAULT

    # This selects the cone-beam table in ``_min_problem_sizes``.  That table gives the
    # minimum sinogram size for each number of GPUs.
    _size_table_key = 'cone'

    def create_projectors(self):
        super().create_projectors()
        # The damping profile is built here.  If it were built on first use, the worker threads
        # of the different devices would race to build it.
        self._dc_damping_slice_profile()

    def _view_batch_bodies(self):
        # The environment variable MBIRTORCH_DISABLE_TRITON=1 turns off the Triton kernels.
        from .kernel_availability import (cone_back_kernel_usable,
                                          cone_forward_kernel_usable)
        if cone_back_kernel_usable(self)[0]:
            from .triton_cone import _cone_back_view_batch_triton
            back_body = _cone_back_view_batch_triton
        else:
            back_body = _cone_back_view_batch
        if cone_forward_kernel_usable(self)[0]:
            from .triton_cone import _cone_forward_view_batch_triton
            fwd_body = _cone_forward_view_batch_triton
        else:
            fwd_body = _cone_forward_view_batch
        return fwd_body, back_body

    def _view_batch_args(self):
        gp_names = ['delta_det_row', 'delta_det_channel', 'det_row_offset',
                    'det_channel_offset', 'source_detector_dist', 'delta_voxel',
                    'voxel_row_aspect', 'voxel_slice_aspect', 'recon_slice_offset',
                    'use_curved_detector']
        (ddr, ddc, dro, dco, sdd, dv, vra, vsa, rso, curved) = self.get_params(gp_names)
        sinogram_shape = self.get_params('sinogram_shape')
        recon_shape = self.get_params('recon_shape')
        det_channel_radius, det_row_radius, vox_slice_radius = self._get_psf_radii()
        return dict(num_rows_r=sinogram_shape[1], num_channels=sinogram_shape[2],
                    num_recon_rows=recon_shape[0], num_recon_cols=recon_shape[1],
                    num_slices=recon_shape[2], delta_voxel=dv,
                    delta_voxel_row=vra * dv, delta_voxel_slice=vsa * dv,
                    delta_det_channel=ddc, delta_det_row=ddr,
                    det_channel_offset=dco, det_row_offset=dro,
                    recon_slice_offset=rso, magnification=self.get_magnification(),
                    source_detector_dist=sdd, use_curved_detector=curved,
                    det_channel_radius=det_channel_radius,
                    det_row_radius=det_row_radius,
                    vox_slice_radius=vox_slice_radius)

    def _transient_cols(self, band_cols):
        # The cone-beam projection functions hold temporary arrays of shape (Vb, P, S) and
        # (Vb, P, R), even when they project only some of the slices.
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape',
                                                       'recon_shape'])
        return max(int(recon_shape[2]), int(sinogram_shape[1]))

    def _dc_damping_slice_profile(self):
        """This method returns the slice damping vectors and the compiled update functions, one of
        each per device.

        The method caches the result.  It returns None if damping is disabled.
        """
        cfg = self._dc_damping
        if cfg is None:
            return None
        recon_shape = self.get_params('recon_shape')
        dv, slice_aspect, oz = self.get_params(
            ['delta_voxel', 'voxel_slice_aspect', 'recon_slice_offset'])
        R = self.get_params('source_iso_dist')
        z_shifts = np.asarray(self.get_params('view_params_array'))[:, 1]
        rp = self.recon_placement
        key = (tuple(cfg), tuple(recon_shape), dv, slice_aspect, oz, R,
               float(z_shifts.min()), float(z_shifts.max()),
               tuple(str(d) for d in rp.devices), rp.axis_len)
        cache = getattr(self, '_dc_damping_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]

        a, b, p, c = cfg
        nz = recon_shape[2]
        L = recon_shape[0] * dv
        dz = slice_aspect * dv
        z = self.recon_slice_z()

        def profile(t):
            return (c * t ** p + a * b ** p) / (t ** p + b ** p)

        if z_shifts.max() - z_shifts.min() == 0:
            s_prof = profile(L * np.abs(z - z_shifts[0]) / (R * dz))
        else:
            t = L * np.abs(z[:, None] - z_shifts[None, :]) / (R * dz)
            s_prof = profile(t).mean(axis=1)
        profiles, fns = [], []
        # The number of slices is passed, because a placement on one device can have no axis_len.
        for i, (dev, (s0, s1)) in enumerate(rp.shard_ranges(nz)):
            profiles.append(torch.as_tensor(
                s_prof[s0:s1].astype(np.float32), device=dev))
            fns.append(maybe_compile(_dc_damped_update_direction,
                                     self.compile_enabled, instance_key=i))
        self._dc_damping_cache = (key, (profiles, fns))
        return profiles, fns

    def _get_update_direction(self, forward_grad, prior_grad, forward_hess,
                              prior_hess, pixel_indices, dev_index=0, slice_sums=None):
        prof = self._dc_damping_slice_profile()
        if prof is None:
            return super()._get_update_direction(forward_grad, prior_grad,
                                                 forward_hess, prior_hess,
                                                 pixel_indices,
                                                 dev_index=dev_index,
                                                 slice_sums=slice_sums)
        profiles, fns = prof
        prior_hess_t = (prior_hess if torch.is_tensor(prior_hess)
                        else torch.as_tensor(prior_hess, dtype=_F32,
                                             device=forward_grad.device))
        return fns[dev_index](forward_grad, prior_grad, forward_hess,
                              prior_hess_t, profiles[dev_index], slice_sums)

    def _project_points_batch(self, points, view_params):
        # points has shape (N, 3), float64, on the CPU.  view_params has shape (V, 2).
        # Its columns are the angle and the helical shift.
        (ddr, ddc, dro, dco, sdd, curved) = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'det_row_offset',
             'det_channel_offset', 'source_detector_dist', 'use_curved_detector'])
        _, num_rows_r, num_channels = self.get_params('sinogram_shape')
        magnification = self.get_magnification()
        angles = view_params[:, 0]
        z_shifts = view_params[:, 1]
        x, y, pixel_mag = _cone_xy_mag(points[:, 0], points[:, 1], angles,
                                       magnification, sdd)
        u = _cone_channel_coordinate(x, y, pixel_mag, magnification, sdd, curved)
        # In a helical view, the object moves toward -z by the helical shift.
        z = points[:, 2][None, :] - z_shifts[:, None]
        v = _cone_row_coordinate(pixel_mag, z)
        return (row_index(v, ddr, dro, num_rows_r),
                channel_index(u, ddc, dco, num_channels))

    def get_magnification(self):
        """This method returns the magnification source_detector_dist / source_iso_dist.

        If source_detector_dist is infinite, the method returns 1.
        """
        source_detector_dist, source_iso_dist = self.get_params(
            ['source_detector_dist', 'source_iso_dist'])
        if np.isinf(source_detector_dist):
            return 1
        return source_detector_dist / source_iso_dist

    def verify_valid_params(self):
        """This method checks that the parameters are valid for a reconstruction.

        It raises a ValueError if a parameter is not valid.  It issues a warning if the cone angle
        is more than 45 degrees.
        """
        super().verify_valid_params()
        sinogram_shape, view_params_array = self.get_params(
            ['sinogram_shape', 'view_params_array'])
        voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['voxel_row_aspect', 'voxel_slice_aspect'])
        num_views, num_det_rows = sinogram_shape[:2]
        if view_params_array is None:
            raise ValueError("view_params_array was not set.")
        if tuple(view_params_array.shape) != (num_views, 2):
            raise ValueError('Number of view dependent parameter vectors must '
                             'equal the number of views.')
        if voxel_row_aspect <= 0 or voxel_slice_aspect <= 0:
            raise ValueError('Voxel aspect ratios must be positive.')
        source_detector_dist, delta_det_row, det_row_offset = self.get_params(
            ['source_detector_dist', 'delta_det_row', 'det_row_offset'])
        half_detector_height = delta_det_row * num_det_rows / 2 + abs(det_row_offset)
        if half_detector_height > source_detector_dist:
            warnings.warn('Cone angle is more than 45 degrees.  This will likely '
                          'produce recon artifacts.')

    def _get_psf_radii(self):
        """This method returns the tap radii of the cone-beam projector loops as
        (det_channel_radius, det_row_radius, vox_slice_radius).

        The channel and row radii are in detector pixels at the largest magnification.  The
        channel radius uses only the in-plane voxel pitches.  The slice radius is in voxels of
        one detector pixel at the smallest magnification.
        """
        (delta_det_row, delta_det_channel, source_detector_dist, recon_shape,
         delta_voxel, voxel_row_aspect, voxel_slice_aspect) = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'source_detector_dist',
             'recon_shape', 'delta_voxel', 'voxel_row_aspect', 'voxel_slice_aspect'])
        magnification = self.get_magnification()
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel
        delta_det = min(delta_det_row, delta_det_channel)

        if np.isinf(source_detector_dist):
            max_magnification = min_magnification = 1
        else:
            source_to_iso_dist = source_detector_dist / magnification
            half_width_x = 0.5 * recon_shape[1] * delta_voxel
            half_width_y = 0.5 * recon_shape[0] * delta_voxel_row
            half_xy_extent = max(half_width_x, half_width_y)
            max_magnification = source_detector_dist / (source_to_iso_dist - half_xy_extent)
            min_magnification = source_detector_dist / (source_to_iso_dist + half_xy_extent)

        max_voxel_pitch = max(delta_voxel, delta_voxel_row, delta_voxel_slice)
        det_row_radius = int(np.ceil(np.ceil(max_voxel_pitch * max_magnification / delta_det)
                                     / 2))
        min_voxel_pitch = min(delta_voxel, delta_voxel_row, delta_voxel_slice)
        max_voxels_per_detector = delta_det / (min_magnification * min_voxel_pitch)
        vox_slice_radius = int(np.ceil(np.ceil(max_voxels_per_detector) / 2))
        max_in_plane_pitch = max(delta_voxel, delta_voxel_row)
        det_channel_radius = int(np.ceil(np.ceil(max_in_plane_pitch * max_magnification
                                                 / delta_det) / 2))
        return det_channel_radius, det_row_radius, vox_slice_radius

    def pixel_magnification_bounds(self, support_radius=None):
        """This method returns the smallest and largest pixel magnification over the support.

        If ``source_detector_dist`` is infinite, both values are 1.

        Args:
            support_radius (float, optional): The support radius in ALU.  If None, the method
                computes it from the recon geometry of the model.  Defaults to None.

        Returns:
            tuple of float: ``(min_magnification, max_magnification)``.  The largest value is
            ``inf`` if the support reaches the source.
        """
        source_detector_dist, source_iso_dist = self.get_params(
            ['source_detector_dist', 'source_iso_dist'])
        if np.isinf(source_detector_dist):
            return 1.0, 1.0
        if support_radius is None:
            recon_shape, delta_voxel, voxel_row_aspect, use_ror_mask = self.get_params(
                ['recon_shape', 'delta_voxel', 'voxel_row_aspect', 'use_ror_mask'])
            support_radius = get_support_radius(recon_shape, voxel_row_aspect * delta_voxel,
                                                delta_voxel, use_ror_mask=use_ror_mask)
        min_magnification = source_detector_dist / (source_iso_dist + support_radius)
        if source_iso_dist - support_radius <= 0:
            return float(min_magnification), float('inf')
        return float(min_magnification), float(source_detector_dist / (source_iso_dist - support_radius))

    @staticmethod
    def detector_mn_to_uv(m, n, delta_det_channel, delta_det_row, det_channel_offset,
                          det_row_offset, num_det_rows, num_det_channels):
        """This method converts fractional detector indices (m, n) to physical detector
        coordinates (u, v)."""
        det_center_row = (num_det_rows - 1) / 2.0
        det_center_channel = (num_det_channels - 1) / 2.0
        v = (m - det_center_row) * delta_det_row - det_row_offset
        u = (n - det_center_channel) * delta_det_channel - det_channel_offset
        return u, v

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """This method sets the recon shape, voxel size, and recon_slice_offset for a cone-beam
        reconstruction.

        The volume covers the field of view of the detector at iso.  For a helical scan, the volume
        also covers the full helical travel.  The method extends the volume at each end by
        ``axial_pad_fraction``.  The center of the volume moves with ``det_row_offset``.
        """
        delta_det_row, delta_det_channel = self.get_params(
            ['delta_det_row', 'delta_det_channel'])
        voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['voxel_row_aspect', 'voxel_slice_aspect'])
        magnification = self.get_magnification()

        delta_voxel = delta_det_channel / magnification
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel

        sinogram_shape = self.get_params('sinogram_shape')
        num_det_rows, num_det_channels = sinogram_shape[1:3]
        num_recon_rows = int(np.round(num_det_channels
                                      * ((delta_det_channel / delta_voxel_row) / magnification)))
        num_recon_cols = int(np.round(num_det_channels
                                      * ((delta_det_channel / delta_voxel) / magnification)))

        z_shifts = np.asarray(self.get_params('view_params_array'))[:, 1]
        z_min, z_max = float(np.min(z_shifts)), float(np.max(z_shifts))
        z_travel = z_max - z_min
        H_iso = num_det_rows * (delta_det_row / magnification)
        num_recon_slices = max(1, int(np.ceil((H_iso + z_travel) / delta_voxel_slice)))
        # At iso, the detector rows see a range of z centered at -det_row_offset / magnification.
        det_row_offset = self.get_params('det_row_offset')
        recon_slice_offset = 0.5 * (z_min + z_max) - det_row_offset / magnification

        # An edge ray at detector height v reaches |z| = |v| (SID + R) / SDD at the far side
        # of the support, where R is the support radius.
        source_detector_dist, det_channel_offset, use_ror_mask = \
            self.get_params(['source_detector_dist', 'det_channel_offset', 'use_ror_mask'])
        support_radius = get_support_radius((num_recon_rows, num_recon_cols),
                                            delta_voxel_row, delta_voxel,
                                            use_ror_mask=use_ror_mask)
        _, v_row_low = self.detector_mn_to_uv(-0.5, 0.0, delta_det_channel,
                                              delta_det_row, det_channel_offset,
                                              det_row_offset, num_det_rows,
                                              num_det_channels)
        _, v_row_high = self.detector_mn_to_uv(num_det_rows - 0.5, 0.0,
                                               delta_det_channel, delta_det_row,
                                               det_channel_offset, det_row_offset,
                                               num_det_rows, num_det_channels)
        v_bot = max(float(v_row_low), float(v_row_high))
        v_top = min(float(v_row_low), float(v_row_high))
        z_per_v_far_side = 1.0 / self.pixel_magnification_bounds(support_radius)[0]
        excess_bot = max(0.0, v_bot * (z_per_v_far_side - 1.0 / magnification))
        excess_top = max(0.0, -v_top * (z_per_v_far_side - 1.0 / magnification))

        axial_pad_fraction = self.get_params('axial_pad_fraction')
        if isinstance(axial_pad_fraction, (tuple, list)):
            if len(axial_pad_fraction) != 2:
                raise ValueError('axial_pad_fraction must be a float or a '
                                 '(top_fraction, bottom_fraction) pair; got '
                                 f'{axial_pad_fraction!r}.')
            pad_frac_top, pad_frac_bot = (float(f) for f in axial_pad_fraction)
        else:
            pad_frac_top = pad_frac_bot = float(axial_pad_fraction)
        if pad_frac_top < 0 or pad_frac_bot < 0:
            raise ValueError(f'axial_pad_fraction must be >= 0; got {axial_pad_fraction!r}.')
        num_slices_top = int(np.ceil(pad_frac_top * excess_top / delta_voxel_slice))
        num_slices_bot = int(np.ceil(pad_frac_bot * excess_bot / delta_voxel_slice))
        num_recon_slices += num_slices_top + num_slices_bot
        recon_slice_offset += 0.5 * (num_slices_bot - num_slices_top) * float(delta_voxel_slice)

        recon_shape = (num_recon_rows, num_recon_cols, num_recon_slices)
        self.set_params(no_compile=no_compile, no_warning=no_warning,
                        recon_shape=recon_shape, delta_voxel=delta_voxel,
                        recon_slice_offset=recon_slice_offset)

    def fdk_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """This method applies the FDK filter to a sinogram.

        The method multiplies each detector element by the FDK cosine weight.  It then applies
        the row filter and scales the result by a factor that depends on the voxel size.
        """
        sinogram = self._shard_sinogram(sinogram)
        # The sinogram can be a Shards container, so the shape comes from the parameters.
        _, num_rows, num_channels = (int(x) for x in
                                     self.get_params('sinogram_shape'))
        source_detector_dist = self.get_params('source_detector_dist')
        (delta_voxel, delta_det_row, delta_det_channel, voxel_row_aspect,
         voxel_slice_aspect) = self.get_params(
            ['delta_voxel', 'delta_det_row', 'delta_det_channel',
             'voxel_row_aspect', 'voxel_slice_aspect'])
        det_row_offset, det_channel_offset = self.get_params(
            ['det_row_offset', 'det_channel_offset'])
        voxel_volume = delta_voxel * (voxel_row_aspect * delta_voxel) \
            * (voxel_slice_aspect * delta_voxel)
        M_0 = self.get_magnification()

        # The FDK cosine weight is a (rows, channels) array, and it is the same for every view.
        m_grid, n_grid = np.meshgrid(np.arange(num_rows), np.arange(num_channels),
                                     indexing='ij')
        u_grid, v_grid = self.detector_mn_to_uv(m_grid, n_grid, delta_det_channel,
                                                delta_det_row, det_channel_offset,
                                                det_row_offset, num_rows, num_channels)
        weight_map = source_detector_dist / np.sqrt(
            source_detector_dist ** 2 + u_grid ** 2 + v_grid ** 2)
        weight_t = torch.as_tensor(weight_map.astype(np.float32),
                                   device=self.torch_device)

        alpha = delta_det_row / (voxel_volume * M_0)
        return self._apply_direct_recon_filter(sinogram, filter_name,
                                               filter_scale=alpha,
                                               output_sharded=output_sharded,
                                               row_weight=weight_t)

    def helical_fdk_z_weight(self, recon, sinogram):
        """This method scales each slice of a helical FDK reconstruction.

        The scale factor is the inverse of the fraction of views in which the detector sees the
        slice.
        """
        num_views, num_rows, num_channels = self.get_params('sinogram_shape')
        helical_z_shifts = np.asarray(self.get_params('view_params_array'))[:, 1]
        (delta_voxel, voxel_slice_aspect, recon_shape, recon_slice_offset,
         delta_det_row) = self.get_params(
            ['delta_voxel', 'voxel_slice_aspect', 'recon_shape',
             'recon_slice_offset', 'delta_det_row'])
        M_0 = self.get_magnification()
        delta_voxel_slice = voxel_slice_aspect * delta_voxel

        from . import _sharding
        if isinstance(recon, _sharding.Shards):
            # The weight is indexed by slice in the whole volume, not in one shard.
            w_full = self._helical_z_weight_row(sinogram)
            rp = recon.placement
            tensors = []
            for i, (dev, (s0, s1)) in enumerate(
                    rp.shard_ranges(int(recon_shape[2]))):
                w = torch.as_tensor(w_full[s0:s1].astype(np.float32), device=dev)
                tensors.append(recon.tensors[i] * w[None, None, :])
            return _sharding.Shards(tensors, rp)
        z_k = self.recon_slice_z()
        det_half_height_iso = 0.5 * num_rows * delta_det_row / M_0
        visible = np.abs(z_k[:, None] - helical_z_shifts[None, :]) <= det_half_height_iso
        coverage = np.sum(visible, axis=1)
        z_weight = np.where(coverage > 0, num_views / np.maximum(coverage, 1), 0.0)
        w = torch.as_tensor(z_weight.astype(np.float32), device=recon.device)
        return recon * w[None, None, :]

    def _helical_z_weight_row(self, sinogram):
        """This method returns the helical z weight of each slice as a numpy array of shape
        (num_slices,)."""
        num_views, num_rows, _ = self.get_params('sinogram_shape')
        helical_z_shifts = np.asarray(self.get_params('view_params_array'))[:, 1]
        (delta_voxel, voxel_slice_aspect, recon_shape, recon_slice_offset,
         delta_det_row) = self.get_params(
            ['delta_voxel', 'voxel_slice_aspect', 'recon_shape',
             'recon_slice_offset', 'delta_det_row'])
        M_0 = self.get_magnification()
        delta_voxel_slice = voxel_slice_aspect * delta_voxel
        z_k = self.recon_slice_z()
        det_half_height_iso = 0.5 * num_rows * delta_det_row / M_0
        visible = np.abs(z_k[:, None] - helical_z_shifts[None, :]) \
            <= det_half_height_iso
        coverage = np.sum(visible, axis=1)
        return np.where(coverage > 0, num_views / np.maximum(coverage, 1), 0.0)

    def recon_fdk(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        This method reconstructs a volume from a sinogram by the FDK algorithm.

        The method first filters the sinogram.  It then back projects the result with the exact
        adjoint of the forward projector.

        Args:
            sinogram (numpy or tensor): The 3D sinogram, with shape
                (num_views, num_det_rows, num_det_channels).
            filter_name (string, optional): The name of the filter.  Defaults to 'ramp'.
            output_sharded (bool, optional): If False, the method returns a numpy array.  If
                True, the method returns the result on the devices.  This result is a torch
                tensor for a model on one device, or a Shards container with one tensor per
                device for a model on several devices.  Defaults to False.

        Returns:
            recon (numpy or tensor): The reconstructed volume.

        Note:
            FDK assumes equally spaced views over the full angular range.  It applies no
            short-scan weighting.  For helical scans, FDK gives an approximate result.  The FDK
            result is useful as an initial value for ``recon()``.
        """
        # This call chooses the GPUs and checks memory for a direct reconstruction before any allocation.
        self._apply_device_policy(workload='direct')
        # The filter places the sinogram on the devices.  This method keeps no reference to the
        # device sinogram during the back projection.
        filtered_sinogram = self.fdk_filter(sinogram, filter_name=filter_name,
                                            output_sharded=True)
        recon = self._back_project(filtered_sinogram, output_sharded=True)
        helical_z_shifts = np.asarray(self.get_params('view_params_array'))[:, 1]
        if float(np.max(helical_z_shifts) - np.min(helical_z_shifts)) > 0:
            recon = self.helical_fdk_z_weight(recon, sinogram)
        return recon if output_sharded else self._gather_recon(recon)

    def recon_direct(self, sinogram, filter_name="ramp", output_sharded=False):
        """This method reconstructs a volume by the FDK algorithm.

        This method is the same as :meth:`recon_fdk`.  See :meth:`TomographyModel.recon_direct`
        for the arguments and return value.
        """
        return self.recon_fdk(sinogram, filter_name=filter_name,
                              output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        return self.fdk_filter(sinogram, filter_name=filter_name,
                               output_sharded=output_sharded)

    def recon_split_sino(self, sino, weights=None, half_overlap=5, init_recon=None, max_iterations=15, stop_threshold_change_pct=0.2,
                         first_iteration=0, compute_prior_loss=False, logfile_path='~/.mbirtorch/logs/recon.log', print_logs=True,
                         align_split_grid=False):
        """
        This method reconstructs a cone-beam volume in two halves.  The split reduces the GPU
        memory needed by about a factor of 2.  The method is meant for a volume that does not fit
        on the GPUs as a whole.

        The method divides the detector rows at the row where the central ray hits the detector.
        Each half includes a few extra rows beyond this row.  The method reconstructs each half
        with ordinary MBIR.  It then joins the halves with a linear blend across the overlap.  If
        each half fits on half of the GPUs, the two halves are reconstructed at the same time.

        This method is used like :meth:`recon`, and its result is approximately equal to the
        result of :meth:`recon`.  This method does not accept ``output_sharded``.

        Args:
            sino (numpy or tensor): The full sinogram, with shape (num_views, num_rows, num_cols).
                The method does not accept a sharded array.
            weights (numpy or tensor, optional): The sinogram weights, with the same shape as
                ``sino``.  The method does not accept a sharded array.
            half_overlap (int): The number of extra detector rows that each half includes beyond
                the iso row.  The method computes the overlap of the reconstructed halves from
                this value.
            init_recon (optional): The same as in the recon method.  The method does not accept a
                sharded array.
            max_iterations (int, optional): The same as in the recon method.
            stop_threshold_change_pct (float, optional): The same as in the recon method.
            first_iteration (int, optional): The same as in the TomographyModel.recon() method.
            compute_prior_loss (bool, optional): This argument is accepted so that the interface
                matches other methods.  The method does not use it.
            logfile_path (str, optional): The same as in the TomographyModel.recon() method.  The
                method merges the logs of the two halves into this one file, with a header for
                each half.
            print_logs (bool, optional): The same as in the TomographyModel.recon() method.
            align_split_grid (bool, optional): If True, the method shifts the recon slice grid by
                up to half a slice.  The shift aligns the boundary between the reconstructed
                halves with the boundary between the detector halves.  This alignment removes
                stripes at the boundary.  Defaults to False.

        Returns:
            Tuple[np.ndarray, dict]: The reconstructed volume as a numpy array, and a metadata
                dictionary.  The dictionary holds the recon parameters and model parameters of
                each half.  Its entry 'split_params' holds the overlaps, the grid shift, the
                number of GPU groups, and the number of GPUs in each group.  If either half
                would have too few slices, the method issues a warning and calls recon()
                instead.  In this case, the method returns the dictionary from recon(), which
                has no entries for the halves.

        Raises:
            ValueError: If an input is missing, if the shapes do not agree, if half_overlap < 2,
                if the geometry has nonzero helical z shifts, or if ``sino``, ``weights``, or
                ``init_recon`` is a sharded array.
            AssertionError: If an array has the wrong number of dimensions.

        Example:
            >>> sino = np.ones((180, 64, 64), dtype=np.float32)  # (views, rows, cols)
            >>> model = mbirtorch.ConeBeamModel(sinogram_shape=sino.shape,
            ...                                 angles=np.linspace(0, np.pi, 180),
            ...                                 source_detector_dist=1000.0,
            ...                                 source_iso_dist=500.0)
            >>> recon, recon_info = model.recon_split_sino(sino, half_overlap=4)
        """
        from . import _sharding
        from .utilities import copy_ct_model, stitch_arrays, merge_log_files

        if half_overlap < 2:
            raise ValueError('half_overlap must be >= 2.')
        helical_z_shifts = self.get_params('view_params_array')[:, 1]
        if any(helical_z_shifts != 0):
            raise ValueError('helical_z_shifts must be zero.')
        if sino is None:
            raise ValueError("sino must be provided.")
        # Each half chooses its own GPUs, so the method refuses an input that is already on the GPUs.
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

        num_views, full_num_rows, num_cols = sino.shape

        delta_det_row = self.get_params('delta_det_row')
        full_det_row_offset = self.get_params('det_row_offset')
        delta_voxel = self.get_params('delta_voxel')
        voxel_slice_aspect, voxel_row_aspect = self.get_params(['voxel_slice_aspect', 'voxel_row_aspect'])
        delta_voxel_slice = voxel_slice_aspect * delta_voxel
        full_recon_shape = self.get_params('recon_shape')
        full_recon_slice_offset = self.get_params('recon_slice_offset')
        source_iso_dist, use_ror_mask = self.get_params(['source_iso_dist', 'use_ror_mask'])
        magnification = self.get_magnification()

        full_recon_rows, full_recon_cols, full_recon_slices = full_recon_shape

        # half_overlap_sino is the number of extra detector rows in each half.
        # half_overlap_recon is the number of extra recon slices in each half.
        delta_detector_row_at_iso = max(delta_det_row / magnification, 1e-12)
        ratio_pixel_to_sino_pitch = delta_voxel_slice / delta_detector_row_at_iso
        if ratio_pixel_to_sino_pitch > 1:
            half_overlap_sino = int(round(half_overlap * ratio_pixel_to_sino_pitch))
        else:
            half_overlap_sino = half_overlap
        # Every slice that the rows of a half see must be in the half volume.  Otherwise a stripe
        # appears at the boundary.  rho is the number of slices per detector row at iso, and
        # (1 + R/SID) bounds the spread of the cone.
        rho = 1.0 / ratio_pixel_to_sino_pitch
        support_radius = get_support_radius(full_recon_shape, voxel_row_aspect * delta_voxel,
                                            delta_voxel, use_ror_mask=use_ror_mask)
        half_overlap_recon = int(np.ceil(
            half_overlap_sino * (1.0 + support_radius / float(source_iso_dist)) * rho)) + 2

        # The detector halves divide at the detector row nearest to iso.
        det_iso_row_float = ((full_num_rows - 1) / 2.0) + (full_det_row_offset / delta_det_row)
        det_iso_row_index = int(round(det_iso_row_float))

        if not (0 < det_iso_row_index < full_num_rows):
            raise ValueError(
                f"Computed det_iso_row_index={det_iso_row_index} is out of valid range (0, {full_num_rows-1}). "
            )

        # A shift of the recon grid by a fraction of a slice removes the stripe at the boundary.
        # The shift also moves the output grid, so the shift is off by default.
        grid_shift_alu = 0.0
        if align_split_grid:
            def _wrap_half(x):
                return (x + 0.5) % 1.0 - 0.5
            iso_float_unshifted = (full_recon_slices - 1) / 2.0 - full_recon_slice_offset / delta_voxel_slice
            split_off_unshifted = round(iso_float_unshifted) - iso_float_unshifted
            candidates = [c for c in range(det_iso_row_index - 2, det_iso_row_index + 3)
                          if 0 < c < full_num_rows]
            eps_by_cut = {c: _wrap_half(split_off_unshifted - (c - det_iso_row_float) * rho)
                          for c in candidates}
            det_iso_row_index = min(eps_by_cut, key=lambda c: abs(eps_by_cut[c]))
            grid_shift_alu = -float(eps_by_cut[det_iso_row_index]) * delta_voxel_slice
            full_recon_slice_offset = full_recon_slice_offset + grid_shift_alu

        # The recon halves divide at the recon slice nearest to iso.
        full_recon_iso_slice_index_float = (full_recon_slices - 1) / 2.0 - full_recon_slice_offset / delta_voxel_slice
        split_index = int(round(full_recon_iso_slice_index_float))
        top_num_slices = split_index + 1

        split_offset = split_index - full_recon_iso_slice_index_float

        # This is the misalignment, in slices, between the detector boundary and the recon boundary.
        # It is reported in split_params.
        split_cut_mismatch = float((split_offset - (det_iso_row_index - det_iso_row_float) * rho
                                    + 0.5) % 1.0 - 0.5)

        if (split_index < 1 or split_index > full_recon_slices - 2
                or top_num_slices < half_overlap_recon
                or full_recon_slices - top_num_slices < half_overlap_recon):
            warnings.warn(
                "split_index is too close to the volume boundary for the recon overlap; "
                "falling back to standard MBIR reconstruction.",
                UserWarning,
            )
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

        # These are the detector row ranges of the two halves.
        top_lo, top_hi = 0, min(det_iso_row_index + half_overlap_sino, full_num_rows)
        bot_lo, bot_hi = max(det_iso_row_index - half_overlap_sino, 0), full_num_rows

        top_recon_shape = (full_recon_shape[0], full_recon_shape[1], top_num_slices + half_overlap_recon)
        bot_recon_shape = (full_recon_shape[0], full_recon_shape[1], (full_recon_shape[2] - top_num_slices) + half_overlap_recon)
        top_recon_slice_offset = (+half_overlap_recon - (top_recon_shape[2]-1)/2 + 0 + split_offset) * delta_voxel_slice
        bot_recon_slice_offset = (-half_overlap_recon + (bot_recon_shape[2]-1)/2 + 1 + split_offset) * delta_voxel_slice

        full_det_center = (full_num_rows - 1) / 2.0

        # This sets the regularization from the full sinogram, as recon would do.
        self.auto_set_regularization_params(sino, weights=weights)

        def _half_model(lo, hi, recon_shape, recon_slice_offset):
            """This function returns the model for one half of the detector rows."""
            num_rows = hi - lo
            det_center = (num_rows - 1) / 2.0
            det_row_offset = full_det_row_offset + (full_det_center - (det_center + lo)) * delta_det_row

            model = copy_ct_model(self, new_num_det_rows=num_rows, no_warning=True)
            model.set_params(det_row_offset=det_row_offset)
            model.set_params(no_warning=True, auto_regularize_flag=False)
            model.set_params(recon_shape=recon_shape)
            model.set_params(recon_slice_offset=recon_slice_offset)
            return model

        def _recon_one_half(model, lo, hi, is_top, half_logfile_path, devices, rng):
            """This function reconstructs one half of the detector rows on ``devices``.

            It returns (host_recon, recon_dict).  If ``devices`` is None, the model of the half
            chooses the GPUs.
            """
            recon_shape = model.get_params('recon_shape')
            if devices is not None:
                model.configure_devices(devices=devices)

            sino_half = sino[:, lo:hi, :]
            weights_half = None if weights is None else weights[:, lo:hi, :]

            # The top half uses the first recon_shape[2] slices of init_recon.  The bottom half uses the last ones.
            half_init = None
            if init_recon is not None:
                half_init = init_recon[:, :, :recon_shape[2]] if is_top else init_recon[:, :, -recon_shape[2]:]

            recon_half, recon_dict = model.recon(sino_half, weights=weights_half, init_recon=half_init,
                                                 max_iterations=max_iterations,
                                                 stop_threshold_change_pct=stop_threshold_change_pct,
                                                 first_iteration=first_iteration,
                                                 logfile_path=half_logfile_path,
                                                 print_logs=print_logs, rng=rng)
            return recon_half, recon_dict

        # Each half writes its own log file.  The finally block merges the files.
        if logfile_path:
            log_path = os.path.expanduser(logfile_path)
            half_log_paths = (log_path + '.top', log_path + '.bot')
        else:
            log_path, half_log_paths = None, (None, None)
        top_model = _half_model(top_lo, top_hi, top_recon_shape, top_recon_slice_offset)
        bot_model = _half_model(bot_lo, bot_hi, bot_recon_shape, bot_recon_slice_offset)

        devices = self._split_devices()
        gpus_per_group = self._choose_gpus_per_group(
            devices, lambda group: all(self._section_fits(model, group, weights is not None)
                                       for model in (top_model, bot_model)), max_groups=2)
        jobs = [lambda group, rng: _recon_one_half(top_model, top_lo, top_hi, True, half_log_paths[0], group, rng),
                lambda group, rng: _recon_one_half(bot_model, bot_lo, bot_hi, False, half_log_paths[1], group, rng)]
        try:
            (recon_top_half, recon_top_dict), (recon_bot_half, recon_bot_dict) = \
                self._run_split_sections(jobs, devices, gpus_per_group)
        finally:
            if log_path:
                merge_log_files(log_path, zip(('recon_split_sino: top half', 'recon_split_sino: bottom half'),
                                              half_log_paths))

        # ramp_overlap is the number of blended slices.  It is even, at most 4, and at most half_overlap_recon.
        ramp_overlap = 4
        ramp_overlap = min(ramp_overlap, half_overlap_recon)
        ramp_overlap -= ramp_overlap % 2
        recon_full = stitch_arrays([recon_top_half, recon_bot_half], axis=2,
                                   overlap=2 * half_overlap_recon, ramp_overlap=ramp_overlap)

        recon_full_dict = {'recon_params_top': recon_top_dict.get('recon_params'),
                           'recon_params_bottom': recon_bot_dict.get('recon_params'),
                           'recon_log_top': recon_top_dict.get('recon_log', '# Log info not saved.'),
                           'recon_log_bottom': recon_bot_dict.get('recon_log', '# Log info not saved.'),
                           'notes_top': recon_top_dict.get('notes', '# No notes saved'),
                           'notes_bottom': recon_bot_dict.get('notes', '# No notes saved'),
                           'model_params_top': recon_top_dict.get('model_params'),
                           'model_params_bottom': recon_bot_dict.get('model_params'),
                           # The grid shift is in ALU.  The misalignment is in slices.
                           'split_params': {'half_overlap_sino': int(half_overlap_sino),
                                            'half_overlap_recon': int(half_overlap_recon),
                                            'align_split_grid': bool(align_split_grid),
                                            'grid_shift_alu': float(grid_shift_alu),
                                            'split_cut_mismatch_slices': split_cut_mismatch,
                                            'num_groups': len(devices) // gpus_per_group,
                                            'gpus_per_group': int(gpus_per_group)}, }

        return recon_full, recon_full_dict


def recon_simple_cone(sinogram, angles, source_detector_dist, source_iso_dist,
                      weights=None, sharpness=1.0, max_iterations=15):
    """
    This function performs a basic cone-beam reconstruction in one call with the default
    geometry parameters.

    For other settings, such as a helical scan or a curved detector, create a
    :class:`ConeBeamModel` and call :meth:`TomographyModel.recon`.

    Args:
        sinogram (numpy or tensor): The 3D sinogram, with shape
            (num_views, num_det_rows, num_det_channels).
        angles (numpy or tensor): A 1D array of projection angles in radians, one per view.
        source_detector_dist (float): The distance from the source to the detector in ALU.
        source_iso_dist (float): The distance from the source to iso, the center of rotation,
            in ALU.
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
        >>> angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        >>> recon, recon_dict = mbirtorch.recon_simple_cone(
        ...     sinogram, angles, source_detector_dist=600, source_iso_dist=400)
    """
    # The geometry comes from the shape of the full sinogram, so a sharded array is refused.
    from . import _sharding
    _sharding.reject_shards('recon_simple_cone', sinogram=sinogram,
                            weights=weights)
    if torch.is_tensor(angles):
        angles = angles.detach().cpu().numpy()
    model = ConeBeamModel(tuple(sinogram.shape), angles,
                          source_detector_dist=source_detector_dist,
                          source_iso_dist=source_iso_dist)
    model.set_params(sharpness=sharpness)
    return model.recon(sinogram, weights=weights, max_iterations=max_iterations)

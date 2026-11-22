"""This module defines ``MultiAxisParallelModel``.  This model is a parallel beam geometry in
which each view has an elevation (tilt) angle as well as an azimuth angle.  At zero elevation
the model is ordinary parallel beam.

The projectors compute the horizontal and vertical detector footprints separately, as the cone
beam projectors do.  The detector row coordinate is v = z*cos(el) + y*sin(el).
"""

import warnings

import numpy as np
import torch

from .geometry_rules import channel_index, pixel_xy, rotate_about_z, row_index
from .horizontal_fan import fan_back_batch, fan_forward_batch
from .tomography_model import TomographyModel

_F32 = torch.float32


def _multiaxis_horizontal_data(pixel_indices, azimuth, num_rows, num_cols,
                               num_channels, delta_voxel, delta_voxel_row,
                               delta_det_channel, det_channel_offset):
    """This function returns the horizontal projection terms for one batch of views.  It
    returns n_p and centers (int32) with shape (Vb, P), W_p_c and weight_scale with shape
    (Vb, 1), and the rotated depth y with shape (Vb, P).
    """
    x_tilde, y_tilde = pixel_xy(pixel_indices, num_rows, num_cols, delta_voxel,
                                delta_voxel_row)
    cosine = torch.cos(azimuth)[:, None]
    sine = torch.sin(azimuth)[:, None]
    x, y = rotate_about_z(x_tilde, y_tilde, cosine, sine)
    n_p = channel_index(x, delta_det_channel, det_channel_offset, num_channels)
    footprint_xy = torch.maximum(cosine.abs() * delta_voxel,
                                 sine.abs() * delta_voxel_row)       # (Vb, 1)
    W_p_c = footprint_xy / delta_det_channel
    weight_scale = (delta_voxel * delta_voxel_row) / footprint_xy
    centers = torch.round(n_p).to(torch.int32)
    return n_p, centers, W_p_c, weight_scale, y


def _multiaxis_row_coordinate(y, z, cos_el, sin_el):
    """This function returns the detector row coordinate v = z*cos(el) + y*sin(el).
    The arguments ``cos_el`` and ``sin_el`` have shape (Vb, 1)."""
    return z * cos_el + y * sin_el


def _multiaxis_vertical_terms(y, azimuth, elevation, num_slices,
                              delta_voxel, delta_voxel_row, delta_voxel_slice,
                              delta_det_row, det_row_offset, recon_slice_offset,
                              num_rows_r):
    """This function returns the vertical projection terms for one batch of views.  Slice k
    projects to the detector row m(k) = m0 + slope*k.  The total vertical weight equals
    delta_voxel_slice/delta_det_row at every elevation.

    Returns:
        tuple: m0 (Vb, P), slope (Vb, 1), W_p_r (Vb, 1), L_max (Vb, 1), and scaling (Vb, 1).
    """
    cos_el = torch.cos(elevation)[:, None]                           # (Vb, 1)
    sin_el = torch.sin(elevation)[:, None]
    z_0 = -delta_voxel_slice * (num_slices - 1) / 2.0 + recon_slice_offset
    v_0 = _multiaxis_row_coordinate(y, z_0, cos_el, sin_el)          # (Vb, P)
    m0 = row_index(v_0, delta_det_row, det_row_offset, num_rows_r)
    slope = (delta_voxel_slice * cos_el) / delta_det_row             # (Vb, 1)

    z_edge = delta_voxel_slice * cos_el.abs()
    x_edge = sin_el.abs() * torch.sin(azimuth)[:, None].abs() * delta_voxel
    y_edge = sin_el.abs() * torch.cos(azimuth)[:, None].abs() * delta_voxel_row
    W_p_r = torch.maximum(z_edge, torch.maximum(x_edge, y_edge)) / delta_det_row
    L_max = torch.clamp(W_p_r, max=1.0)
    scaling = (delta_voxel_slice / delta_det_row) / W_p_r
    return m0, slope, W_p_r, L_max, scaling


def _multiaxis_forward_view_batch(values, pixel_indices, view_params_batch,
                                  num_rows_r, num_channels, num_recon_rows,
                                  num_recon_cols, num_slices, delta_voxel,
                                  delta_voxel_row, delta_voxel_slice,
                                  delta_det_channel, delta_det_row,
                                  det_channel_offset, det_row_offset,
                                  recon_slice_offset, det_channel_radius,
                                  det_row_radius, vox_slice_radius,
                                  slice_start=0, plan=None):
    """This function returns the forward projection of one batch of views, with shape
    (Vb, R, C).  The argument ``values`` may hold a contiguous range of slices, with shape
    (P, L), that starts at ``slice_start``.  The function ignores ``vox_slice_radius`` and
    ``plan``.
    """
    azimuth = view_params_batch[:, 0]
    elevation = view_params_batch[:, 1]
    n_p, centers, W_p_c, weight_scale, y = _multiaxis_horizontal_data(
        pixel_indices, azimuth, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset)
    vb, num_pixels = n_p.shape
    dev = values.device

    m0, slope, W_p_r, L_max, scaling = _multiaxis_vertical_terms(
        y, azimuth, elevation, num_slices, delta_voxel, delta_voxel_row,
        delta_voxel_slice, delta_det_row, det_row_offset, recon_slice_offset,
        num_rows_r)

    band_len = values.shape[1]
    k = torch.arange(slice_start, slice_start + band_len, dtype=_F32, device=dev)
    m_p = m0.unsqueeze(-1) + slope.unsqueeze(-1) * k[None, None, :]  # (Vb, P, L)
    m_center = torch.round(m_p).to(torch.int64)

    # The values are multiplied by the amplitude.  Slices past num_slices are set to zero.
    valid_k = (k < num_slices).to(_F32)                              # (L,)
    scaled_values = values[None, :, :] * scaling.unsqueeze(-1) \
        * valid_k[None, None, :]                                     # (Vb, P, L)

    W_b = W_p_r.unsqueeze(-1)
    L_max_b = L_max.unsqueeze(-1)
    det_col = torch.zeros((vb, num_pixels, num_rows_r), dtype=_F32, device=dev)
    for m_off in range(-det_row_radius, det_row_radius + 1):
        m = m_center + m_off
        A = torch.clamp((W_b + 1.0) / 2.0 - (m_p - m.to(_F32)).abs(), min=0.0)
        A = torch.minimum(A, L_max_b)
        A = A * ((m >= 0) & (m < num_rows_r)).to(_F32)
        det_col.scatter_add_(2, m.clamp(0, num_rows_r - 1), scaled_values * A)

    acc = fan_forward_batch((n_p, centers, W_p_c, weight_scale), det_col,
                            num_channels, det_channel_radius)
    return acc.permute(0, 2, 1)


def _multiaxis_back_view_batch(sino_batch, pixel_indices, view_params_batch,
                               num_rows_r, num_channels, num_recon_rows,
                               num_recon_cols, num_slices, delta_voxel,
                               delta_voxel_row, delta_voxel_slice,
                               delta_det_channel, delta_det_row,
                               det_channel_offset, det_row_offset,
                               recon_slice_offset, det_channel_radius, det_row_radius,
                               vox_slice_radius, coeff_power=1,
                               slice_start=0, band_slices=None, plan=None):
    """This function computes the back projection of one batch of views, summed over the
    views.  The result has shape (P, S), or (P, band_slices) when ``band_slices`` is given.
    The function ignores ``vox_slice_radius`` and ``plan``.
    """
    azimuth = view_params_batch[:, 0]
    elevation = view_params_batch[:, 1]
    n_p, centers, W_p_c, weight_scale, y = _multiaxis_horizontal_data(
        pixel_indices, azimuth, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset)
    vb, num_pixels = n_p.shape
    dev = sino_batch.device

    sino_T = sino_batch.permute(0, 2, 1).contiguous()                # (Vb, C, R)
    det_col = fan_back_batch(sino_T, (n_p, centers, W_p_c, weight_scale),
                             num_channels, det_channel_radius,
                             coeff_power=coeff_power, reduce_views=False)

    m0, slope, W_p_r, L_max, scaling = _multiaxis_vertical_terms(
        y, azimuth, elevation, num_slices, delta_voxel, delta_voxel_row,
        delta_voxel_slice, delta_det_row, det_row_offset, recon_slice_offset,
        num_rows_r)

    band_len = num_slices if band_slices is None else band_slices
    k = torch.arange(slice_start, slice_start + band_len, dtype=_F32, device=dev)
    m_p = m0.unsqueeze(-1) + slope.unsqueeze(-1) * k[None, None, :]  # (Vb, P, S)
    m_center = torch.round(m_p).to(torch.int64)

    W_b = W_p_r.unsqueeze(-1)
    L_max_b = L_max.unsqueeze(-1)
    scale_pow = (scaling ** coeff_power).unsqueeze(-1)               # (Vb, 1, 1)
    out = torch.zeros((num_pixels, band_len), dtype=_F32, device=dev)
    for m_off in range(-det_row_radius, det_row_radius + 1):
        mm = m_center + m_off
        A = torch.clamp((W_b + 1.0) / 2.0 - (m_p - mm.to(_F32)).abs(), min=0.0)
        A = torch.minimum(A, L_max_b)
        A = A * ((mm >= 0) & (mm < num_rows_r)).to(_F32)
        if coeff_power != 1:
            A = A ** coeff_power
        A = A * scale_pow
        g = torch.gather(det_col, 2, mm.clamp(0, num_rows_r - 1))
        out = out + torch.einsum("vps,vps->ps", A, g)
    out = out * (k < num_slices).to(_F32)[None, :]
    return out


class MultiAxisParallelModel(TomographyModel):
    """
    This class is used to model a parallel beam geometry in which each view has an elevation
    (tilt) angle.  It extends :class:`TomographyModel`.  At zero elevation the model is the same
    as ParallelBeamModel.  Parallel beam laminography is a special case.

    Each view has two angles:
      - Azimuth: The rotation around the object's z-axis, as in ParallelBeamModel.
      - Elevation: The angle from the xy plane at which the source looks at the object.  The
        source is on the +y side.  A point (x, y, z) projects to the detector row coordinate
        v = z cos(elevation) + y sin(elevation).

    Args:
        sinogram_shape (tuple): The shape (num_views, num_det_rows, num_det_channels).
        angles (ndarray): An array with shape (num_views, 2).  The column angles[:, 0] holds
            the azimuth and angles[:, 1] holds the elevation, in radians.
        view_batch_size, compile_mode: The same as in ParallelBeamModel.
    """

    def __init__(self, sinogram_shape, angles, view_batch_size=None,
                 compile_mode='auto'):
        angles = np.asarray(angles, dtype=np.float32)
        if angles.ndim != 2 or angles.shape[1] != 2:
            raise ValueError(f"angles must have shape (num_views, 2). "
                             f"Got {angles.shape}.")
        if angles.shape[0] != sinogram_shape[0]:
            raise ValueError(f"Number of angle pairs ({angles.shape[0]}) must "
                             f"match number of views ({sinogram_shape[0]}).")
        if np.any(np.abs(angles[:, 1]) > np.pi / 4):
            warnings.warn("One or more elevation angles exceed 45 degrees. "
                          "This may degrade approximation quality.")
        super().__init__(sinogram_shape,
                         view_batch_size=view_batch_size, compile_mode=compile_mode,
                         geometry_type=str(type(self)),
                         view_params_name='angles', angles=angles,
                         recon_slice_offset=0.0)

    # This name selects the multiaxis table of minimum sinogram sizes in ``_min_problem_sizes``.
    _size_table_key = 'multiaxis'

    def _view_batch_bodies(self):
        # The forward and back Triton kernels are checked separately.  MBIRTORCH_DISABLE_TRITON=1
        # turns them off.  The import is local because triton_multiaxis imports this module.
        from .kernel_availability import (multiaxis_back_kernel_usable,
                                          multiaxis_forward_kernel_usable)
        if multiaxis_back_kernel_usable(self)[0]:
            from .triton_multiaxis import _multiaxis_back_view_batch_triton
            back_body = _multiaxis_back_view_batch_triton
        else:
            back_body = _multiaxis_back_view_batch
        if multiaxis_forward_kernel_usable(self)[0]:
            from .triton_multiaxis import _multiaxis_forward_view_batch_triton
            fwd_body = _multiaxis_forward_view_batch_triton
        else:
            fwd_body = _multiaxis_forward_view_batch
        return fwd_body, back_body

    def _project_points_batch(self, points, view_params):
        # points is (N, 3) float64 on the CPU.  view_params is (V, 2) with columns (azimuth, elevation).
        ddr, ddc, dro, dco = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'det_row_offset',
             'det_channel_offset'])
        _, num_rows_r, num_channels = self.get_params('sinogram_shape')
        azimuth = view_params[:, 0]
        elevation = view_params[:, 1]
        cosine = torch.cos(azimuth)[:, None]
        sine = torch.sin(azimuth)[:, None]
        x, y = rotate_about_z(points[:, 0], points[:, 1], cosine, sine)
        cos_el = torch.cos(elevation)[:, None]
        sin_el = torch.sin(elevation)[:, None]
        v = _multiaxis_row_coordinate(y, points[:, 2][None, :], cos_el, sin_el)
        return (row_index(v, ddr, dro, num_rows_r),
                channel_index(x, ddc, dco, num_channels))

    def _view_batch_args(self):
        gp_names = ['delta_det_row', 'delta_det_channel', 'det_row_offset',
                    'det_channel_offset', 'delta_voxel', 'voxel_row_aspect',
                    'voxel_slice_aspect', 'recon_slice_offset']
        (ddr, ddc, dro, dco, dv, vra, vsa, rso) = self.get_params(gp_names)
        sinogram_shape = self.get_params('sinogram_shape')
        recon_shape = self.get_params('recon_shape')
        det_channel_radius, det_row_radius, vox_slice_radius = self._get_psf_radii()
        return dict(num_rows_r=sinogram_shape[1], num_channels=sinogram_shape[2],
                    num_recon_rows=recon_shape[0], num_recon_cols=recon_shape[1],
                    num_slices=recon_shape[2], delta_voxel=dv,
                    delta_voxel_row=vra * dv, delta_voxel_slice=vsa * dv,
                    delta_det_channel=ddc, delta_det_row=ddr,
                    det_channel_offset=dco, det_row_offset=dro,
                    recon_slice_offset=rso,
                    det_channel_radius=det_channel_radius,
                    det_row_radius=det_row_radius,
                    vox_slice_radius=vox_slice_radius)

    def _transient_cols(self, band_cols):
        # The projectors hold (Vb, P, S) and (Vb, P, R) temporary arrays for any slice range.
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape',
                                                       'recon_shape'])
        return max(int(recon_shape[2]), int(sinogram_shape[1]))

    def get_magnification(self):
        """This method returns the magnification, which is 1 for parallel beam."""
        return 1.0

    def verify_valid_params(self):
        """This method checks that the parameters are consistent with the geometry."""
        super().verify_valid_params()
        sinogram_shape = self.get_params('sinogram_shape')
        angles = np.asarray(self.get_params('angles'))
        if angles.shape[0] != sinogram_shape[0]:
            raise ValueError(f"View mismatch: {angles.shape[0]} angles for "
                             f"{sinogram_shape[0]} views.")
        if angles.shape[1] != 2:
            raise ValueError("Each view requires exactly 2 angles: "
                             "[azimuth, elevation].")

    def _get_psf_radii(self):
        """This method returns the tap radii of the multiaxis projector loops as
        (det_channel_radius, det_row_radius, vox_slice_radius).  The channel and row radii both
        equal the larger of the horizontal and vertical radii over all views.  The slice radius
        is 0, because no loop reads it."""
        (delta_det_channel, delta_det_row, delta_voxel, voxel_row_aspect,
         voxel_slice_aspect) = self.get_params(
            ['delta_det_channel', 'delta_det_row', 'delta_voxel',
             'voxel_row_aspect', 'voxel_slice_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel

        max_in_plane_pitch = max(delta_voxel, delta_voxel_row)
        psf_radius_u = int(np.ceil(np.ceil(max_in_plane_pitch / delta_det_channel) / 2))

        # The vertical radius is the largest over all views.
        angles = self.get_params('angles')
        if angles is not None:
            elevations = np.asarray(angles)[:, 1]
            max_abs_cos_el = float(np.max(np.abs(np.cos(elevations))))
            max_abs_sin_el = float(np.max(np.abs(np.sin(elevations))))
        else:
            max_abs_cos_el, max_abs_sin_el = 1.0, 0.0
        z_edge = max_abs_cos_el * delta_voxel_slice
        in_plane_edge = max_abs_sin_el * max(delta_voxel, delta_voxel_row)
        vertical_footprint = max(z_edge, in_plane_edge)
        psf_radius_v = int(np.ceil(np.ceil(vertical_footprint / delta_det_row) / 2))

        radius = max(psf_radius_u, psf_radius_v)
        return radius, radius, 0

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """This method sets the reconstruction shape to the largest box that projects onto the
        detector at the given angles.  The box is centered on the range of z that the detector
        rows see."""
        sinogram_shape = self.get_params('sinogram_shape')
        num_views, num_det_rows, num_det_channels = sinogram_shape
        delta_det_channel, delta_det_row = self.get_params(
            ['delta_det_channel', 'delta_det_row'])

        max_u = (num_det_channels * delta_det_channel) / 2.0
        max_v = (num_det_rows * delta_det_row) / 2.0

        angles = np.asarray(self.get_params('angles'))
        elevations = angles[:, 1]

        max_R_xy = max_u
        # The volume covers every z that some view sees.  The cosine is clamped at 0.1, so a
        # view from straight above does not give infinite z.
        det_row_offset = self.get_params('det_row_offset')
        cos_el = np.maximum(np.abs(np.cos(elevations)), 0.1)
        z_low = float(np.min((-max_v - det_row_offset) / cos_el))
        z_high = float(np.max((max_v - det_row_offset) / cos_el))
        recon_slice_offset = 0.5 * (z_low + z_high)

        voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['voxel_row_aspect', 'voxel_slice_aspect'])
        delta_voxel = delta_det_channel
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel
        num_recon_cols = int(np.floor(2 * max_R_xy / delta_voxel))
        num_recon_rows = int(np.floor(2 * max_R_xy / delta_voxel_row))
        num_recon_slices = int(np.floor((z_high - z_low) / delta_voxel_slice))

        self.set_params(recon_shape=(num_recon_rows, num_recon_cols, num_recon_slices),
                        delta_voxel=delta_voxel, recon_slice_offset=recon_slice_offset,
                        no_compile=no_compile, no_warning=no_warning)

    def fbp_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """This method applies the FBP filter to the sinogram.  The filter is the standard 1-D
        ramp along the channels with a uniform angular weight pi/num_views.  Because the weight
        is uniform, the order of the angles does not matter.  The filter ignores the elevation.
        """
        delta_voxel, voxel_row_aspect = self.get_params(
            ['delta_voxel', 'voxel_row_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        scaling_factor = 1.0 / (delta_voxel * delta_voxel_row)
        return self._apply_direct_recon_filter(
            sinogram, filter_name, filter_scale=scaling_factor,
            output_sharded=output_sharded, row_weight=None)

    def recon_fbp(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        This method performs FBP reconstruction.  It filters the sinogram and then applies
        the exact adjoint of the forward projector as the back projection.

        Args:
            sinogram (numpy or tensor): 3D sinogram data with shape
                (num_views, num_det_rows, num_det_channels).
            filter_name (string, optional): The name of the filter to use.
                Defaults to 'ramp'.
            output_sharded (bool, optional): If False, the method returns a numpy
                array.  If True, the method returns a torch tensor on a single device
                model, or a ``Shards`` container with one tensor per device on a
                multi-device model.  Defaults to False.

        Returns:
            recon (numpy or tensor): The reconstructed volume.

        Note:
            The filter assumes equally spaced azimuths and ignores elevation.  Therefore
            this reconstruction is approximate.  It is meant as an initial value for
            ``recon()``.
        """
        # The GPUs must be chosen before the first large allocation.
        self._apply_device_policy(workload='direct')
        filtered_sinogram = self.fbp_filter(sinogram, filter_name=filter_name,
                                            output_sharded=True)
        recon = self._back_project(filtered_sinogram, output_sharded=True)
        return recon if output_sharded else self._gather_recon(recon)

    def recon_direct(self, sinogram, filter_name="ramp", output_sharded=False):
        """This method performs direct reconstruction by calling :meth:`recon_fbp`.
        :meth:`TomographyModel.recon_direct` describes the arguments and the return value."""
        return self.recon_fbp(sinogram, filter_name=filter_name,
                              output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """This method applies the filter that direct reconstruction needs.  It calls
        :meth:`fbp_filter`."""
        return self.fbp_filter(sinogram, filter_name=filter_name,
                               output_sharded=output_sharded)


# The docs and examples also use this name for the model.
MultiAxisParallelBeamModel = MultiAxisParallelModel

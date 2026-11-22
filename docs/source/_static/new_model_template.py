import numpy as np
import torch

from mbirtorch import TomographyModel


# The projection bodies are MODULE-LEVEL pure functions, not methods.  The
# driver compiles them with torch.compile, one instance per device; a bound
# method would pin the model in the compile cache, and parameter reads inside
# the compiled region would be traced.  All parameter VALUES arrive through
# the keyword arguments built by _view_batch_args below.

def _template_forward_view_batch(values, pixel_indices, view_params_batch,
                                 num_rows, num_cols, num_channels,
                                 delta_det_channel, det_channel_offset,
                                 delta_voxel, det_channel_radius,
                                 det_row_radius, vox_slice_radius,
                                 slice_start=0, plan=None):
    """Forward project a batch of voxel cylinders into a batch of views.

    Args:
        values (tensor): (num_pixels, num_slices) voxel values, where
            values[i, j] is the voxel in slice j at the pixel given by
            pixel_indices[i].
        pixel_indices (tensor of int): indices into the flattened
            (num_rows x num_cols) recon plane.
        view_params_batch (tensor): the view-dependent parameters for this
            batch of views (for parallel beam, the angles).
        slice_start (int): first slice of a slice band, for the slice-banded
            multi-device drivers.  A geometry whose detector rows track the
            recon slices can assert it is 0 (see parallel_beam.py).
        plan: memoization slot reserved by the driver; leave unused.

    Returns:
        tensor of shape (num_views_in_batch, num_det_rows, num_det_channels)
    """
    # TODO: compute each voxel's detector coordinates from the geometry, then
    # spread each voxel value over the detector cells it hits.  See
    # _parallel_forward_view_batch in parallel_beam.py for the worked example,
    # and horizontal_fan.py for the shared fan machinery it uses.
    num_views = view_params_batch.shape[0]
    num_det_rows = values.shape[1]
    return torch.zeros((num_views, num_det_rows, num_channels),
                       dtype=values.dtype, device=values.device)


def _template_back_view_batch(sino_batch, pixel_indices, view_params_batch,
                              num_rows, num_cols, num_channels,
                              delta_det_channel, det_channel_offset,
                              delta_voxel, det_channel_radius,
                              det_row_radius, vox_slice_radius,
                              coeff_power=1, slice_start=0, band_slices=None,
                              plan=None):
    """Back project a batch of sinogram views onto a batch of voxel cylinders,
    summed over the batch's views (the adjoint of the forward body).

    Args:
        sino_batch (tensor): (num_views_in_batch, num_det_rows,
            num_det_channels) sinogram views.
        coeff_power (int): back project with the projection coefficients
            raised to this power.  1 for the gradient; 2 for the Hessian
            diagonal.
        slice_start, band_slices: the slice band for the banded multi-device
            back projection; a row-aligned geometry can assert the defaults.
        plan: memoization slot reserved by the driver; leave unused.

    Returns:
        tensor of shape (num_pixels, num_slices)
    """
    # TODO: for each voxel cylinder, gather and weight the sinogram values
    # along its rays.  See _parallel_back_view_batch in parallel_beam.py.
    num_pixels = pixel_indices.shape[0]
    num_slices = sino_batch.shape[1]
    return torch.zeros((num_pixels, num_slices),
                       dtype=sino_batch.dtype, device=sino_batch.device)


class TemplateModel(TomographyModel):
    """Template for a new scanner geometry, extending TomographyModel.

    The class inherits reconstruction, parameter handling, and multi-device
    support from TomographyModel and overrides only the geometry-specific
    pieces.  Items to change for a particular geometry are marked with TODO.
    ParallelBeamModel and ConeBeamModel are the tested, canonical examples.

    Args:
        sinogram_shape (tuple): (views, rows, channels).
        view_dependent_vec (ndarray): a view-dependent parameter vector with
            one entry per view (for example the view angles).
        param1 (scalar): a view-independent geometry parameter.
    """

    # TODO: adjust the signature for the geometry.  Avoid **kwargs so that
    # set_params can keep checking for valid parameter names.
    def __init__(self, sinogram_shape, view_dependent_vec, param1,
                 view_batch_size=None, compile_mode='auto'):
        view_dependent_vec = np.asarray(view_dependent_vec, dtype=np.float32)
        super().__init__(sinogram_shape, view_batch_size=view_batch_size,
                         compile_mode=compile_mode,
                         param1=param1,
                         view_params_name='view_dependent_vec',
                         view_dependent_vec=view_dependent_vec)

    # TODO: set True only if detector row r maps one-to-one to recon slice r
    # (the parallel-beam case).  It lets the multi-device drivers stream row
    # bands instead of full detector rows.
    rows_track_slices = False

    def get_magnification(self):
        """Scale factor from a voxel at iso to its projection on the
        detector; 1.0 for parallel beam, parameter-dependent otherwise."""
        # TODO: adjust for the geometry.
        return 1.0

    def _get_psf_radii(self):
        """This method returns the tap radii of the projection loops as
        (det_channel_radius, det_row_radius, vox_slice_radius).  Each radius is
        the number of cells on each side of the center that one loop visits.
        A radius that no loop reads is 0."""
        # TODO: compute each radius from the voxel and detector pitches along
        # the axis of its loop.
        delta_det_channel, delta_voxel = self.get_params(
            ['delta_det_channel', 'delta_voxel'])
        det_channel_radius = int(np.ceil(np.ceil(delta_voxel / delta_det_channel) / 2))
        return det_channel_radius, 0, 0

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """Set recon_shape and delta_voxel to reasonable defaults from the
        sinogram shape and the detector spacings."""
        # TODO: implement for the geometry; see parallel_beam.py.
        raise NotImplementedError

    def verify_valid_params(self):
        """Check that all parameters are compatible for a reconstruction;
        raise ValueError otherwise."""
        super().verify_valid_params()
        sinogram_shape, vec = self.get_params(
            ['sinogram_shape', 'view_dependent_vec'])
        # TODO: add the geometry's own checks.
        if np.asarray(vec).shape[0] != sinogram_shape[0]:
            raise ValueError('Number of view dependent parameter vectors '
                             'must equal the number of views.')

    def _view_batch_bodies(self):
        """The geometry's (forward, back) projection bodies.  Return the
        module-level torch bodies; a geometry with hand-written kernels
        selects them here when the availability gates pass (see
        dev_projector_kernels in the docs)."""
        return _template_forward_view_batch, _template_back_view_batch

    def _view_batch_args(self):
        """The eager keyword arguments for the bodies.  Every parameter read
        happens here, per call, so a set_params change is always picked up."""
        # TODO: read the parameters the bodies need.
        delta_det_channel, det_channel_offset, delta_voxel = self.get_params(
            ['delta_det_channel', 'det_channel_offset', 'delta_voxel'])
        recon_shape = self.get_params('recon_shape')
        num_channels = self.get_params('sinogram_shape')[2]
        det_channel_radius, det_row_radius, vox_slice_radius = self._get_psf_radii()
        return dict(num_rows=recon_shape[0], num_cols=recon_shape[1],
                    num_channels=num_channels,
                    delta_det_channel=delta_det_channel,
                    det_channel_offset=det_channel_offset,
                    delta_voxel=delta_voxel,
                    det_channel_radius=det_channel_radius,
                    det_row_radius=det_row_radius,
                    vox_slice_radius=vox_slice_radius)

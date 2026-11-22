"""This module tests cone beam volumes whose slices are thicker than their in-plane voxels.

A thick slice covers many detector rows, so the back projection needs many row taps.  The
footprint of a voxel along the channels depends only on its in-plane pitches, so the number of
channel taps must not grow with the slice pitch.  When it did, a volume with a slice aspect of
2.2 back projected about 50 times slower than a volume of cubic voxels on the same detector.
"""

import time

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch.cone_beam import _cone_back_view_batch, _cone_forward_view_batch
from mbirtorch.triton_cone import (_cone_back_view_batch_triton,
                                   _cone_forward_view_batch_triton)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the Triton cone kernels need a CUDA device")

# The slice aspect of the thick slice volumes.  It gives many more row taps than
# channel taps.
THICK_SLICE_ASPECT = 6.0
# The largest allowed ratio of the thick slice back projection time to the time
# for cubic voxels on the same detector and over the same height.
MAX_THICK_SLICE_TIME_RATIO = 4.0


def _cone_model(slice_aspect, cell=(6, 40, 16), num_slices=None, device="cuda"):
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    model = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=4 * cell[2],
                                    source_iso_dist=2 * cell[2], compile_mode="off")
    model.set_params(no_warning=True, verbose=0)
    if num_slices is None:
        num_slices = max(2, int(round(model.get_params('recon_shape')[2] / slice_aspect)))
    recon_shape = model.get_params('recon_shape')
    model.set_params(no_warning=True, voxel_slice_aspect=slice_aspect,
                     recon_shape=(recon_shape[0], recon_shape[1], num_slices))
    model.configure_devices(devices=[device])
    return model


def _body_inputs(model, seed=0):
    device = model.torch_device
    pixel_indices = torch.as_tensor(mbirtorch.gen_full_indices(model.get_params('recon_shape')),
                                    dtype=torch.int64, device=device)
    generator = torch.Generator().manual_seed(seed)
    sinogram = torch.rand(tuple(model.get_params('sinogram_shape')), generator=generator).to(device)
    values = torch.rand((pixel_indices.shape[0], int(model.get_params('recon_shape')[2])),
                        generator=generator).to(device)
    view_params = torch.as_tensor(model.get_params('view_params_array'), dtype=torch.float32,
                                  device=device)
    return sinogram, values, pixel_indices, view_params, model._view_batch_args()


def _rel_max(out, ref):
    return float((out - ref).abs().max() / max(float(ref.abs().max()), 1e-30))


def test_channel_radius_leaves_out_the_slice_pitch():
    cubic_channel, cubic_row, _ = _cone_model(1.0, device="cpu")._get_psf_radii()
    assert cubic_channel == cubic_row

    thick_model = _cone_model(THICK_SLICE_ASPECT, device="cpu")
    thick_channel, thick_row, _ = thick_model._get_psf_radii()
    assert thick_channel == cubic_channel
    assert thick_channel < thick_row
    assert thick_model._view_batch_args()['det_channel_radius'] == thick_channel


def test_thick_slice_torch_bodies_are_adjoint():
    model = _cone_model(THICK_SLICE_ASPECT, device="cpu")
    sinogram, values, pixel_indices, view_params, args = _body_inputs(model)
    forward = _cone_forward_view_batch(values, pixel_indices, view_params, **args)
    back = _cone_back_view_batch(sinogram, pixel_indices, view_params, **args)
    lhs = float((forward.double() * sinogram.double()).sum())
    rhs = float((values.double() * back.double()).sum())
    assert abs(lhs - rhs) <= 1e-5 * abs(rhs)


@pytest.mark.parametrize("coeff_power", [1, 2])
def test_thick_slice_channel_taps_drop_only_zero_weights(coeff_power):
    # The channel taps past the channel radius carry zero weight, so the back
    # projection with the row radius on the channels gives the same values.
    model = _cone_model(THICK_SLICE_ASPECT, device="cpu")
    sinogram, _, pixel_indices, view_params, args = _body_inputs(model)
    reference = _cone_back_view_batch(sinogram, pixel_indices, view_params,
                                      coeff_power=coeff_power,
                                      **{**args, 'det_channel_radius': args['det_row_radius']})
    out = _cone_back_view_batch(sinogram, pixel_indices, view_params,
                                coeff_power=coeff_power, **args)
    assert _rel_max(out, reference) <= 1e-6


@requires_cuda
@pytest.mark.parametrize("coeff_power,tol", [(1, 1e-5), (2, 1e-4)])
def test_thick_slice_back_kernel_parity(coeff_power, tol):
    model = _cone_model(THICK_SLICE_ASPECT)
    sinogram, _, pixel_indices, view_params, args = _body_inputs(model)
    reference = _cone_back_view_batch(sinogram, pixel_indices, view_params,
                                      coeff_power=coeff_power, **args)
    out = _cone_back_view_batch_triton(sinogram, pixel_indices, view_params,
                                       coeff_power=coeff_power, **args)
    assert out.shape == reference.shape
    assert _rel_max(out, reference) <= tol


@requires_cuda
def test_thick_slice_forward_kernel_parity():
    model = _cone_model(THICK_SLICE_ASPECT)
    _, values, pixel_indices, view_params, args = _body_inputs(model)
    reference = _cone_forward_view_batch(values, pixel_indices, view_params, **args)
    out = _cone_forward_view_batch_triton(values, pixel_indices, view_params, **args)
    assert _rel_max(out, reference) <= 1e-5


@requires_cuda
@pytest.mark.performance
def test_thick_slice_back_projection_time():
    # A back projection over the same height with slices 2.2 times thicker must
    # not take much longer than with cubic voxels, since both read the same
    # detector rows.
    num_views, num_rows, num_channels = 64, 512, 384
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    sinogram = torch.rand((num_views, num_rows, num_channels), device="cuda")

    def back_projection_seconds(slice_aspect, num_slices):
        model = mbirtorch.ConeBeamModel((num_views, num_rows, num_channels), angles,
                                        source_detector_dist=150.0, source_iso_dist=100.0,
                                        compile_mode="off")
        model.set_params(no_warning=True, verbose=0, delta_det_channel=0.0388,
                         delta_det_row=0.0388, delta_voxel=0.0908,
                         voxel_slice_aspect=slice_aspect, recon_shape=(160, 160, num_slices),
                         use_ror_mask=False)
        model.configure_devices(devices=["cuda"])
        model.back_project(sinogram, output_sharded=True)        # compiles the kernel
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.back_project(sinogram, output_sharded=True)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    cubic = back_projection_seconds(1.0, 220)
    thick = back_projection_seconds(2.2, 100)
    print(f"back projection: thick slices {thick:.3f} s, cubic voxels {cubic:.3f} s")
    assert thick <= MAX_THICK_SLICE_TIME_RATIO * cubic, (thick, cubic)

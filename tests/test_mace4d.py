"""Tests for the 4D reconstruction model and the frame-axis filter.

The filter matrix is checked against the transform it replaces.  The model
is checked on a 24-view cone-beam scan with a smooth sinogram.  A four-frame
run with the filter on differs from the same run with it off, and a seeded
four-frame run gives the same image on two CPU workers as on one.  The smooth sinogram is
kept because a random one reconstructs to a volume with extreme values, on
which the qGGMRF line search computes zero over zero.
"""

import numpy as np
import pytest
import torch
from scipy.fft import dct, idct

import mbirtorch
from mbirtorch.mace4d import (MACE4DModel, apply_temporal_filter,
                              temporal_filter_matrix)

NUM_VIEWS = 24          # 24 views over 360 degrees, 15 degrees per view
DET_ROWS = 8
DET_COLS = 10


def _rel_max(out, ref):
    out = np.asarray(out.detach().cpu() if torch.is_tensor(out) else out, dtype=np.float64)
    ref = np.asarray(ref.detach().cpu() if torch.is_tensor(ref) else ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


def _small_model():
    """A small cone-beam model with evenly spaced angles over 360 degrees."""
    angles = np.radians(360.0 / NUM_VIEWS) * np.arange(NUM_VIEWS)
    model = mbirtorch.ConeBeamModel((NUM_VIEWS, DET_ROWS, DET_COLS), angles,
                                    source_detector_dist=100.0, source_iso_dist=50.0)
    model.set_params(no_warning=True, verbose=0)
    return model


def _smooth_sino(num_views=NUM_VIEWS, rows=DET_ROWS, cols=DET_COLS):
    """A smooth, positive sinogram: a Gaussian bump modulated slowly over the views."""
    v = np.linspace(-1.0, 1.0, rows)
    c = np.linspace(-1.0, 1.0, cols)
    base = np.exp(-(v[:, None] ** 2 + c[None, :] ** 2))
    return np.stack([base * (1.0 + 0.1 * np.sin(0.5 * a))
                     for a in range(num_views)]).astype(np.float32)


def _four_frame_model():
    """A four-frame model of the 24-view scan, at a period the frame count can
    carry, so the filter stays on without a warning."""
    model = MACE4DModel(_small_model(), frames_per_rotation=4, frame_overlap_factor=1.0)
    model.set_params(verbose=0)
    return model


# ── the frame-axis filter matrix ─────────────────────────────────────────────
def _dct_filter(x, period, harmonics=True, band_width=1):
    """The frame-axis filter written as transforms: the reference the matrix
    must reproduce.  Zeroes the DCT-I modes at the period and its harmonics,
    with band_width modes on each side, along axis 0."""
    x = np.asarray(x, dtype=np.float32)
    num_frames = x.shape[0]
    if harmonics is False:
        harmonic_list = [1]
    elif harmonics is True:
        harmonic_list = list(range(1, int(np.floor(period / 2)) + 1))
    else:
        harmonic_list = list(harmonics)
    coefficients = dct(x, type=1, norm='ortho', axis=0)
    for harmonic in harmonic_list:
        center = int(round(2 * (num_frames - 1) / (period / harmonic)))
        low, high = max(0, center - band_width), min(num_frames, center + band_width + 1)
        if low < high:
            coefficients[low:high, ...] = 0
    return idct(coefficients, type=1, norm='ortho', axis=0).astype(np.float32)


def test_filter_matrix_equals_the_transform_filter():
    """The matrix reproduces the transform filter on random inputs and on
    unit impulses at the first and last frames, where the end behavior of
    the transform shows.  The tolerance is float32 rounding relative to the
    largest input value.  At period 6 the filter removes every mode of a
    three-frame axis, one frame cannot be filtered and is refused, and
    applying the matrix along another axis equals moving that axis to the
    front and filtering there."""
    num_frames = 12
    matrix = temporal_filter_matrix(num_frames, period=6)
    assert matrix.shape == (num_frames, num_frames) and matrix.dtype == torch.float32
    rng = np.random.default_rng(num_frames)
    inputs = [rng.standard_normal((num_frames, 4, 5, 3)).astype(np.float32)]
    for frame in (0, num_frames - 1):
        impulse = np.zeros((num_frames, 1, 1, 1), dtype=np.float32)
        impulse[frame] = 1.0
        inputs.append(impulse)
    worst = 0.0
    for x in inputs:
        reference = _dct_filter(x, period=6)
        filtered = apply_temporal_filter(torch.as_tensor(x), matrix, axis=0).numpy()
        worst = max(worst, float(np.max(np.abs(filtered - reference)) / np.max(np.abs(x))))
    print(f"filter matrix vs transform filter, {num_frames} frames: {worst:.2e}")
    assert worst < 1e-5
    # A symmetric projection, to the same rounding.
    m = matrix.numpy().astype(np.float64)
    symmetry, idempotence = np.max(np.abs(m - m.T)), np.max(np.abs(m @ m - m))
    print(f"symmetry {symmetry:.2e}, idempotence {idempotence:.2e}")
    assert symmetry < 1e-5 and idempotence < 1e-5

    # Three frames at period 6: every mode removed.  The zero is exact, since
    # the transform of a zeroed coefficient array is zero.
    assert torch.equal(temporal_filter_matrix(3, period=6), torch.zeros(3, 3))
    with pytest.raises(ValueError, match='at least 2'):
        temporal_filter_matrix(1, period=6)

    # Along another axis: data movement, so the check is exact.
    torch.manual_seed(1)
    axis_matrix = temporal_filter_matrix(12, period=6)
    x = torch.randn(5, 12, 4, 3)
    along_axis_1 = apply_temporal_filter(x, axis_matrix, axis=1)
    moved = apply_temporal_filter(x.movedim(1, 0), axis_matrix, axis=0).movedim(0, 1)
    assert torch.equal(along_axis_1, moved)


def test_reconstructions_run_end_to_end_with_and_without_the_filter(device):
    """A four-frame run at a period the frame count can carry runs the filter
    path: two iterations give finite values of the 4D shape that differ from
    the filter-off run, and the denoiser strength, estimated from the
    filtered initial image, differs from the filter-off value.  Four frames
    per rotation with no overlap give four frames from the 24 views, and at
    four frames the period-4 filter keeps one mode, the zeroth cosine mode,
    so the filtered stack is not zero."""
    np.random.seed(0)
    filtered = _four_frame_model()
    assert filtered.num_frames == 4
    filtered.set_device_pool([device])
    shape = (4,) + filtered.recon_shape
    init = np.linspace(0.0, 0.1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    init += 0.01 * np.random.default_rng(4).standard_normal(shape).astype(np.float32)
    on, on_dict = filtered.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                                 stop_threshold_change_pct=0)
    assert on.shape == shape and np.all(np.isfinite(on))
    assert on_dict['recon_params']['iterations completed'] == 2
    filtered.set_params(dejitter=False)
    np.random.seed(0)
    off, off_dict = filtered.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                                   stop_threshold_change_pct=0)
    rel = _rel_max(on, off)
    print(f"filter on vs off: rel_max = {rel:.2e}")
    assert rel > 1e-2                        # the filter is applied, not merely recorded
    assert on_dict['recon_params']['denoiser sigma_x'] != off_dict['recon_params']['denoiser sigma_x']


def test_two_workers_reproduce_one_worker_on_a_seeded_run():
    """A seeded run derives every draw from the seed and the name of what the
    draw is for, so no draw depends on which worker makes it or when.  Two CPU
    workers must therefore reach the same image as one, to a relative maximum
    difference of 1e-3.  The gate allows for the float rounding of a folding
    update that adds task outputs in whatever order they arrive, which two
    consensus iterations amplify to about 1e-4; a run that drew on its worker
    instead differed by most of the image."""
    seed = 5
    one_worker = _four_frame_model()
    shape = (4,) + one_worker.recon_shape
    init = np.linspace(0.0, 0.1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    init += 0.01 * np.random.default_rng(4).standard_normal(shape).astype(np.float32)

    one_worker.set_device_pool(['cpu'])
    one, one_dict = one_worker.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                                     stop_threshold_change_pct=0, seed=seed)
    two_workers = _four_frame_model()
    two_workers.set_device_pool(['cpu', 'cpu'])
    two, _ = two_workers.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                               stop_threshold_change_pct=0, seed=seed)
    rel = _rel_max(two, one)
    print(f"two CPU workers vs one on seed {seed}: rel_max = {rel:.2e}")
    assert one_dict['recon_params']['seed'] == seed
    assert rel < 1e-3

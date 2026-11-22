"""Value gates for the Triton parallel-beam kernels (the parallel back and
forward bodies).

The cone battery's shape, applied to the degenerate geometry.  Each kernel is
an alternative view-batch BODY, so every gate here compares it against the
torch body it replaces at the same inputs: parity across geometry variants
that move the hfan contract (the projected footprint, the tap radius, the
detector offset) at every coefficient power the body takes, the adjointness
pairings -- kernel against the OTHER direction's torch body, and the two
kernels against each other -- and the poison-the-padding class (a pixel count that is not a multiple of the kernel's
pixel tile, where the padded lanes must contribute exactly nothing).

One more class sits beside those.  Each wrapper rounds its width argument up
to a multiple of 16 before the launch -- the back's sinogram row count, the
forward's value column count -- and pads the input it reads with zeros to
match, so a width that is not a multiple has extra columns that the wrapper
then slices off.  Those tests read the values and the returned view's stride,
which is the width the wrapper really allocated.

One thing differs from the cone battery, because the vertical fan is gone:
there is no rounding carve-out to absorb -- no atan2-vs-sqrt divisor, no round-vs-floor
tie -- so these kernels differ from their bodies by float summation order
alone.  The tolerances stay at the design's figures (rel 1e-5 on the gradient
path, 1e-4 at coeff_power 2) because what a value gate must catch is a
miscompile, not a ULP.

The forward kernel scatters with float atomics, so its sums are reordered from
launch to launch and it is not bit-reproducible; 1e-5 covers that too.

Every test here launches a kernel, so every one of them needs CUDA and skips
without it.
"""

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch._utils import padded_kernel_width
from mbirtorch.parallel_beam import (_parallel_back_view_batch,
                                     _parallel_forward_view_batch)
from mbirtorch.triton_parallel import (PARALLEL_BACK_BLOCK_P,
                                       PARALLEL_BACK_BLOCK_R,
                                       PARALLEL_FWD_BLOCK_P,
                                       PARALLEL_FWD_BLOCK_R,
                                       PARALLEL_SORTED_VIEW_CHUNK,
                                       PARALLEL_SORTED_WINDOW,
                                       _parallel_back_view_batch_triton,
                                       _parallel_forward_view_batch_triton)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the hand-written Triton kernels need a CUDA device")

# The three contract variants the parity tests sweep.  They are not cosmetic:
# voxel_row_aspect widens the projected footprint enough to raise det_channel_radius
# from 1 to 2 (a longer tap loop, and W_p_c > 1 so the min(1, W_p_c) clip
# binds), and det_channel_offset shifts n_p off the channel grid so the tap
# centers and their trapezoid weights all move.
VARIANTS = {"square": {},
            "wide_voxel": {"row_aspect": 3.0},
            "det_offset": {"det_offset": 0.7}}


def _parallel_model(shape=(6, 12, 12), row_aspect=1.0, det_offset=0.0,
                    device="cuda", compile_mode="off"):
    angles = np.linspace(0, np.pi, shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(shape, angles, 
                                        compile_mode=compile_mode)
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    if row_aspect != 1.0 or det_offset != 0.0:
        model.set_params(no_warning=True, voxel_row_aspect=row_aspect,
                         det_channel_offset=det_offset)
        model.auto_set_recon_geometry(no_warning=True)
    return model


def _body_inputs(model, num_pixels=None, seed=0):
    """(sinogram, pixel_indices, view_params, body kwargs) for ONE view batch
    covering every view -- the shape the driver hands a body."""
    device = model.torch_device
    recon_shape = model.get_params('recon_shape')
    pixel_indices = torch.as_tensor(mbirtorch.gen_full_indices(recon_shape),
                                    dtype=torch.int64, device=device)
    if num_pixels is not None:
        pixel_indices = pixel_indices[:num_pixels]
    # A private generator: the seeded recon gates read the global streams.
    generator = torch.Generator().manual_seed(seed)
    sinogram = torch.rand(tuple(model.get_params('sinogram_shape')),
                          generator=generator).to(device)
    view_params = torch.as_tensor(model.get_params('angles'),
                                  dtype=torch.float32, device=device)
    return sinogram, pixel_indices, view_params, model._view_batch_args()


def _voxel_values(model, pixel_indices, seed=3, num_cols=None):
    """(P, cols) voxel cylinders -- the shape the driver hands a forward
    body."""
    if num_cols is None:
        num_cols = int(model.get_params('recon_shape')[2])
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((int(pixel_indices.shape[0]), num_cols),
                      generator=generator).to(model.torch_device)


def _rel_max(out, ref):
    # An identically zero reference would make every relative reading a free
    # pass, so the ruler is checked before it is used.
    scale = float(ref.abs().max())
    assert scale > 0.0, "the reference output is identically zero"
    return float((out - ref).abs().max()) / scale


@requires_cuda
@pytest.mark.parametrize("variant,coeff_power,tol",
                         [(v, 1, 1e-5) for v in VARIANTS] + [("square", 2, 1e-4)])
def test_parallel_back_kernel_parity(variant, coeff_power, tol):
    model = _parallel_model(**VARIANTS[variant])
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    reference = _parallel_back_view_batch(sinogram, pixel_indices, view_params,
                                          coeff_power=coeff_power, **args)
    kernel_out = _parallel_back_view_batch_triton(sinogram, pixel_indices,
                                                  view_params,
                                                  coeff_power=coeff_power,
                                                  **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    rel = _rel_max(kernel_out, reference)
    print(f"parallel back triton parity ({variant}, "
          f"coeff_power={coeff_power}, "
          f"det_channel_radius={args['det_channel_radius']}): "
          f"rel_max = {rel:.2e}")
    assert rel <= tol


@requires_cuda
@pytest.mark.parametrize("direction", ["back", "forward"])
@pytest.mark.parametrize("band", [5, 16])
def test_parallel_kernel_pads_the_width_argument_to_a_multiple_of_16(
        direction, band):
    """Each wrapper launches its width argument rounded up to a multiple of 16.

    The back kernel's width argument is its SINOGRAM ROW count and the
    forward kernel's is its VALUE COLUMN count; in both directions that width
    is the launch's vector axis and the returned view's row stride.  A
    rounded-up launch would read past the last real row (or column), so each
    wrapper copies its input into a zero-padded array of the launched width,
    and a zero row or column contributes exactly zero.  The values must be
    the torch body's at the design's 1e-5 gate either way, and the returned
    view's stride must be the width the wrapper really allocated.

    A 32-row sinogram and a 32-column value array make both cases reachable:
    bands of 16 and 32 need no padding, and bands of 5 and 12 do.  The back
    case also compares each banded call against the unbanded call over the
    same rows, so the rows the rounding added changed nothing.
    """
    model = _parallel_model(shape=(6, 32, 20))
    if direction == "back":
        sinogram, pixel_indices, view_params, args = _body_inputs(model)
        num_rows = int(sinogram.shape[1])
        assert num_rows == 32
        unbanded = _parallel_back_view_batch_triton(sinogram, pixel_indices,
                                                    view_params, **args)
        for row_start in range(0, num_rows, band):
            rows_band = sinogram[:, row_start:row_start + band]
            rows = int(rows_band.shape[1])
            kernel_out = _parallel_back_view_batch_triton(rows_band,
                                                          pixel_indices,
                                                          view_params, **args)
            reference = _parallel_back_view_batch(rows_band, pixel_indices,
                                                  view_params, **args)
            assert kernel_out.shape == reference.shape
            assert bool(kernel_out.isfinite().all())
            assert _rel_max(kernel_out, reference) <= 1e-5
            window = unbanded[:, row_start:row_start + rows]
            assert _rel_max(kernel_out, window) <= 1e-6
            padded = padded_kernel_width(rows)
            assert kernel_out.stride(0) == padded, (rows, padded)
            assert kernel_out.is_contiguous() == (padded == rows)
        return

    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    num_cols = int(values.shape[1])
    assert num_cols == 32
    for col_start in range(0, num_cols, band):
        cols_band = values[:, col_start:col_start + band]
        cols = int(cols_band.shape[1])
        kernel_out = _parallel_forward_view_batch_triton(cols_band,
                                                         pixel_indices,
                                                         view_params, **args)
        reference = _parallel_forward_view_batch(cols_band, pixel_indices,
                                                 view_params, **args)
        assert kernel_out.shape == reference.shape
        assert bool(kernel_out.isfinite().all())
        assert _rel_max(kernel_out, reference) <= 1e-5
        # The transpose puts the channel-major row stride last.
        assert kernel_out.stride(2) == padded_kernel_width(cols), cols


@requires_cuda
@pytest.mark.parametrize("num_pixels", [1, 3 * PARALLEL_BACK_BLOCK_P + 7])
def test_parallel_back_kernel_pixel_padding(num_pixels):
    # Poison the padding: a pixel count that is not a multiple of the kernel's
    # pixel tile pads the last block with lanes that must contribute exactly
    # nothing.  Two independent statements of that -- parity against the torch
    # body, and the invariant that a pixel's output does not depend on which
    # lane of which block it landed in (the same pixels inside a LARGER subset
    # must give the same values).
    model = _parallel_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    subset = pixel_indices[:num_pixels]
    reference = _parallel_back_view_batch(sinogram, subset, view_params, **args)
    kernel_out = _parallel_back_view_batch_triton(sinogram, subset,
                                                  view_params, **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    assert _rel_max(kernel_out, reference) <= 1e-5

    full = _parallel_back_view_batch_triton(sinogram, pixel_indices,
                                            view_params, **args)
    assert _rel_max(kernel_out, full[:num_pixels]) <= 1e-6


@requires_cuda
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_parallel_forward_kernel_parity(variant):
    # As for the back kernel: the geometry reaches the forward kernel only
    # through the one eager builder it shares with the torch body, so all
    # three variants exercise the same kernel with different contract values.
    # The forward body takes no coeff_power.
    model = _parallel_model(**VARIANTS[variant])
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    reference = _parallel_forward_view_batch(values, pixel_indices,
                                             view_params, **args)
    kernel_out = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                     view_params, **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    rel = _rel_max(kernel_out, reference)
    print(f"parallel forward triton parity ({variant}, "
          f"det_channel_radius={args['det_channel_radius']}): rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
def test_parallel_kernel_pair_adjointness():
    # Two pairings, both <F x, a> == <x, B a>.  First the KERNEL forward
    # against the TORCH back body, the mirror of test_parallel_back_kernel_
    # adjointness, which holds the forward side fixed to the torch body
    # instead.  Then the pairing that actually ships once both kernels are on:
    # KERNEL forward against KERNEL back.  A convention that drifted in BOTH
    # kernels together would pass the two one-sided statements and fail the
    # kernel-pair one.
    model = _parallel_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    forward = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                  view_params, **args)
    torch_back = _parallel_back_view_batch(sinogram, pixel_indices,
                                           view_params, **args)
    lhs = float((forward * sinogram).sum())
    rhs = float((values * torch_back).sum())
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    print(f"parallel forward triton adjointness: lhs {lhs:.6f}, "
          f"rhs {rhs:.6f}, rel {rel:.2e}")
    assert rel <= 1e-4

    back = _parallel_back_view_batch_triton(sinogram, pixel_indices,
                                            view_params, **args)
    rhs = float((values * back).sum())
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    print(f"parallel kernel-pair adjointness: lhs {lhs:.6f}, rhs {rhs:.6f}, "
          f"rel {rel:.2e}")
    assert rel <= 1e-4


@requires_cuda
@pytest.mark.parametrize("num_pixels", [1, 3 * PARALLEL_FWD_BLOCK_P + 7])
def test_parallel_forward_kernel_pixel_padding(num_pixels):
    # Poison the padding: a pixel count that is not a multiple of the kernel's
    # pixel tile pads the last block with lanes whose atomics must be masked
    # off entirely.  Two independent statements of that -- parity against the
    # torch body, and additivity over a pixel SPLIT (the forward sums all
    # pixels into one sinogram, so a subset's output is a partial sum, and the
    # two parts must reassemble the whole however the blocks were padded).
    model = _parallel_model()
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    subset, rest = pixel_indices[:num_pixels], pixel_indices[num_pixels:]
    reference = _parallel_forward_view_batch(values[:num_pixels], subset,
                                             view_params, **args)
    kernel_out = _parallel_forward_view_batch_triton(values[:num_pixels],
                                                     subset, view_params,
                                                     **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    assert _rel_max(kernel_out, reference) <= 1e-5

    full = _parallel_forward_view_batch_triton(values, pixel_indices,
                                               view_params, **args)
    rest_out = _parallel_forward_view_batch_triton(values[num_pixels:], rest,
                                                   view_params, **args)
    assert _rel_max(kernel_out + rest_out, full) <= 1e-5


@requires_cuda
def test_parallel_kernels_span_several_row_chunks():
    # Every shape above is narrower than the kernels' row tile, so the row grid
    # axis holds exactly one program and a bug in its offset would not show.
    # _tile_size caps the tile at next_pow2(extent), so forcing a second chunk
    # takes more rows than the pinned tile: this shape gives the back kernel
    # several row chunks and the forward kernel at least two.
    shape = (4, 4 * PARALLEL_BACK_BLOCK_R, 16)
    model = _parallel_model(shape=shape)
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    assert int(sinogram.shape[1]) > PARALLEL_BACK_BLOCK_R
    assert int(values.shape[1]) > PARALLEL_FWD_BLOCK_R

    back_ref = _parallel_back_view_batch(sinogram, pixel_indices, view_params,
                                         **args)
    back_out = _parallel_back_view_batch_triton(sinogram, pixel_indices,
                                                view_params, **args)
    back_rel = _rel_max(back_out, back_ref)
    fwd_ref = _parallel_forward_view_batch(values, pixel_indices, view_params,
                                           **args)
    fwd_out = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                  view_params, **args)
    fwd_rel = _rel_max(fwd_out, fwd_ref)
    print(f"parallel triton multi-row-chunk ({sinogram.shape[1]} rows): "
          f"back rel_max = {back_rel:.2e}, forward rel_max = {fwd_rel:.2e}")
    assert back_rel <= 1e-5
    assert fwd_rel <= 1e-5


# ── the sorted-contraction forward route ─────────────────────────
# The wrapper routes through the sorted kernel by default, so every forward
# gate above already exercises it; this test pins the pieces the default
# path cannot reach on the small shapes -- the two kernels against each
# other, the sparse-set fallback, the view-chunk tail.


@requires_cuda
@pytest.mark.parametrize("case", ["tap_vs_sorted", "sparse_pixels",
                                  "view_chunk_tail"])
def test_parallel_forward_sorted_route(case, monkeypatch):
    """Three statements about the sorted forward route.

    tap_vs_sorted: the two routes compute the same sums in a different order,
    so they gate against each other at the same figure the kernels gate
    against their torch bodies.

    sparse_pixels: a sparse pixel set is the ordinary way a SORTED tile's
    channel span exceeds the window.  The small parity shapes never reach it
    (their whole detector is narrower than the window), so this shape is wide
    (64 channels) and the set keeps every 103rd pixel.  The sorted 32-pixel
    tile then spans most of the detector, the kernel takes its per-tap
    fallback, and the values must still match the torch body.

    view_chunk_tail: 21 views is one full 16-view chunk plus a 5-view tail, so
    the tail chunk's clamped iterations run and must write nothing; a defect
    there double-counts the last view and fails the parity by orders.
    """
    if case == "tap_vs_sorted":
        model = _parallel_model()
        _, pixel_indices, view_params, args = _body_inputs(model)
        values = _voxel_values(model, pixel_indices)
        monkeypatch.setenv("MBIRTORCH_SORTED_FORWARD", "0")
        tap_out = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                      view_params, **args)
        monkeypatch.setenv("MBIRTORCH_SORTED_FORWARD", "1")
        sorted_out = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                         view_params, **args)
        assert sorted_out.shape == tap_out.shape
        assert bool(sorted_out.isfinite().all())
        rel = _rel_max(sorted_out, tap_out)
        print(f"sorted vs tap forward kernels: rel_max = {rel:.2e}")
        assert rel <= 1e-5
        return

    if case == "sparse_pixels":
        model = _parallel_model(shape=(6, 12, 64))
        _, pixel_indices, view_params, args = _body_inputs(model)
        pixel_indices = pixel_indices[::103].contiguous()
        assert int(pixel_indices.shape[0]) > PARALLEL_SORTED_WINDOW
    else:
        assert 21 % PARALLEL_SORTED_VIEW_CHUNK != 0
        model = _parallel_model(shape=(21, 12, 12))
        _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    reference = _parallel_forward_view_batch(values, pixel_indices,
                                             view_params, **args)
    kernel_out = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                     view_params, **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    rel = _rel_max(kernel_out, reference)
    print(f"sorted forward route ({case}): rel_max = {rel:.2e}")
    assert rel <= 1e-5

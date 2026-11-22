"""Value gates for the Triton multiaxis-parallel back and forward kernels.

The cone and parallel batteries' shape, applied to the multiaxis geometry.  Each
kernel is an alternative view-batch BODY, so every gate here compares it against
the torch body it replaces
(:func:`mbirtorch.multiaxis_parallel._multiaxis_back_view_batch`,
:func:`mbirtorch.multiaxis_parallel._multiaxis_forward_view_batch`) at the same
inputs: parity across variants that move the geometry, the adjointness pairings
(each kernel against the opposite torch body, and the kernel PAIR against
itself), and the poison-the-padding class (a pixel count that is not a multiple
of the kernel's pixel tile, where the padded lanes must contribute exactly
nothing).

The variant axis here is ELEVATION, because that is the axis this geometry adds
to parallel beam and the axis that separates the slope of the slice-to-row map
from the vertical footprint that sets the trapezoid weight.  Zero elevation is
the degenerate parallel-beam case; the sweep and the strong constant tilt each
move the footprint, the clip bound and the mass-conserving amplitude.  Two more
variants move the contract without touching elevation: a nonzero det_row_offset
shifts the row anchor off the grid, and a wide voxel_row_aspect raises
det_channel_radius and det_row_radius from 1 to 2, lengthening both tap loops.

The FORWARD kernel adds a class of its own: COVERAGE.  It inverts the
slice-to-row map instead of mirroring the torch body's scatter, so it has to
enumerate the recon slices that reach each detector row, and at a tilted view
more of them reach one row than the 2 * det_row_radius + 1 taps the back kernel
uses.  The bound that sets the enumeration is checked directly against a brute
force count of the contributing slices (a CPU test, so it runs everywhere), and
a thin-slice shape where that bound provably exceeds det_row_radius carries the
parity statement on a GPU.  The five variants below do NOT reach that case --
each of them needs one slice tap on each side, which det_row_radius already covers
-- so the coverage tests use their own shapes and say so.

One more class sits beside those.  Each wrapper rounds a width up to a multiple
of 16 before the launch -- the back kernel its slice band, the forward kernel
its detector row count -- so a width that is not one is computed with extra
columns or rows the wrapper then slices off.  Those tests read the values and
the returned view's stride, which is the width the wrapper really allocated.

Tolerances follow the design's value gate -- rel 1e-5 on the gradient path,
1e-4 at coeff_power 2.  The back kernel has one rounding carve-out to absorb,
the floor(m + 0.5) row center against the torch body's round-half-to-even (see
the module docstring of mbirtorch/triton_multiaxis.py for why the tie is
inert); otherwise both kernels differ from their bodies by float summation
order alone.

Where the BACK kernel is compared against ITSELF -- banded concatenation, a
pixel subset inside a larger one -- the comparison is torch.equal rather than a
tolerance.  That kernel gathers into a register accumulator and stores each
output element once, with no atomic adds, and each element's sum order is fixed
by the loop nesting alone, independent of the tile shape and of which band or
pixel block the element landed in.  Bit equality is therefore the statement to
make there, and a tolerance would hide a real change.  The FORWARD kernel
scatters with atomic adds, whose order varies from launch to launch, so its
self-comparisons are tolerances and bit equality would be the wrong statement;
each such test says which one it is making and why.

The last test is a whole reconstruction: a seeded run taking the kernel route
must reproduce one on the torch bodies.

Everything that launches a kernel needs CUDA and skips without it; the
coverage bound is exercised on any machine.
"""

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import kernel_availability
from mbirtorch._utils import padded_kernel_width
from mbirtorch.multiaxis_parallel import (_multiaxis_back_view_batch,
                                          _multiaxis_forward_view_batch,
                                          _multiaxis_horizontal_data,
                                          _multiaxis_vertical_terms)
from mbirtorch.triton_multiaxis import (MULTIAXIS_BACK_BLOCK_P,
                                        MULTIAXIS_FWD_BLOCK_P,
                                        MULTIAXIS_FWD_BLOCK_R,
                                        MULTIAXIS_FWD_MAX_SLICE_RADIUS,
                                        _multiaxis_back_view_batch_triton,
                                        _multiaxis_forward_view_batch_triton,
                                        _multiaxis_slice_tap_radius)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the hand-written Triton kernels need a CUDA device")

# The five contract variants the parity tests sweep.  The first three move the
# ELEVATION alone: zero (the parallel-beam case, where the vertical footprint
# is the voxel slice pitch and the amplitude is 1), a moderate sweep, and a
# strong constant tilt that stays under the model's 45-degree warning.  The
# last two move the contract without touching elevation.
VARIANTS = {"zero_elevation": {"elevation": "zero"},
            "elevation_sweep": {},
            "strong_tilt": {"elevation": "tilt"},
            "det_row_offset": {"det_row_offset": 0.7},
            "wide_voxel": {"row_aspect": 3.0}}


def _ma_model(shape=(6, 12, 12), elevation="sweep", row_aspect=1.0,
              det_row_offset=0.0, slice_aspect=1.0, device="cuda",
              compile_mode="off"):
    """A small multiaxis model, built as tests/test_multiaxis.py's _small_ma
    builds one: azimuths spanning pi and an elevation set chosen per variant.

    ``slice_aspect`` thins the voxel slice pitch against the detector row
    pitch, which is what lowers the slice-to-row slope below the point where
    det_row_radius still covers the forward kernel's slice enumeration.  It is not
    part of the variant sweep; only the forward coverage tests set it."""
    num_views = shape[0]
    azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
    if elevation == "zero":
        tilt = np.zeros(num_views)
    elif elevation == "tilt":
        tilt = np.full(num_views, 0.7)
    else:
        tilt = np.linspace(-0.5, 0.5, num_views)
    model = mbirtorch.MultiAxisParallelModel(
        shape, np.stack([azimuth, tilt], axis=1), compile_mode=compile_mode)
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    if row_aspect != 1.0 or det_row_offset != 0.0 or slice_aspect != 1.0:
        model.set_params(no_warning=True, voxel_row_aspect=row_aspect,
                         voxel_slice_aspect=slice_aspect,
                         det_row_offset=det_row_offset)
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
    # Multiaxis carries its per-view parameters as the (num_views, 2) angles
    # array: column 0 the azimuth, column 1 the elevation.
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
                         [(v, 1, 1e-5) for v in VARIANTS] + [("elevation_sweep", 2, 1e-4)])
def test_multiaxis_back_kernel_parity(variant, coeff_power, tol):
    # The geometry reaches the kernel only through the two eager builders it
    # shares with the torch body, so all five variants exercise the same kernel
    # with different contract values.
    model = _ma_model(**VARIANTS[variant])
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    reference = _multiaxis_back_view_batch(sinogram, pixel_indices, view_params,
                                           coeff_power=coeff_power, **args)
    kernel_out = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                                   view_params,
                                                   coeff_power=coeff_power,
                                                   **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    rel = _rel_max(kernel_out, reference)
    print(f"multiaxis back triton parity ({variant}, "
          f"coeff_power={coeff_power}, "
          f"det_channel_radius={args['det_channel_radius']}, "
          f"det_row_radius={args['det_row_radius']}): "
          f"rel_max = {rel:.2e}")
    assert rel <= tol


@requires_cuda
@pytest.mark.parametrize("direction,case", [("back", 5), ("back", 16),
                                            ("forward", ((6, 12, 12), 12)),
                                            ("forward", ((6, 32, 20), 32))])
def test_multiaxis_kernel_pads_the_width_argument_to_a_multiple_of_16(
        direction, case):
    """Each wrapper launches its width argument rounded up to a multiple of 16.

    The back kernel's width argument is its SLICE BAND and the forward
    kernel's is its DETECTOR ROW count; in both directions that width is the
    returned view's row stride.  The values must be the torch body's at the
    design's 1e-5 gate whether or not the width was rounded up, and the
    returned view's stride must be the width the wrapper really ALLOCATED --
    the real width when it is already a multiple of 16, the path that has to
    stay exactly what it was.

    Back: this shape's volume is at least 32 slices, so bands of 16 and 32 need
    no padding and bands of 5 and 12 do.  Each banded call is also compared
    against the unbanded call over the same slices, bit for bit (the back
    kernel has no atomics), so the columns the rounding added changed nothing.
    The tail band of each tiling is shorter than the requested band, and its
    padded columns address slices past the end of the volume, which is the
    case the kernel's address clamps exist for.

    Forward: the 12-row detector is rounded up to 16 and the 32-row one is
    not, so both paths are exercised.  The returned shape is exactly
    (Vb, R, C) -- the padding never reaches a caller -- and the torch body
    returns the same permuted layout at the real row count, so the two differ
    by exactly that padding and nothing else.
    """
    if direction == "back":
        band_slices = case
        model = _ma_model(shape=(6, 32, 20))
        sinogram, pixel_indices, view_params, args = _body_inputs(model)
        num_slices = int(args['num_slices'])
        assert num_slices >= 32
        unbanded = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                                     view_params, **args)
        for slice_start in range(0, num_slices, band_slices):
            length = min(band_slices, num_slices - slice_start)
            banded = _multiaxis_back_view_batch_triton(
                sinogram, pixel_indices, view_params, slice_start=slice_start,
                band_slices=length, **args)
            reference = _multiaxis_back_view_batch(
                sinogram, pixel_indices, view_params, slice_start=slice_start,
                band_slices=length, **args)
            assert banded.shape == reference.shape
            assert bool(banded.isfinite().all())
            assert _rel_max(banded, reference) <= 1e-5
            window = unbanded[:, slice_start:slice_start + length]
            assert bool(torch.equal(banded, window))
            padded = padded_kernel_width(length)
            assert banded.stride(0) == padded, (length, padded)
            assert banded.is_contiguous() == (padded == length)
        return

    shape, detector_rows = case
    model = _ma_model(shape=shape)
    _, pixel_indices, view_params, args = _body_inputs(model)
    assert int(args['num_rows_r']) == detector_rows
    values = _voxel_values(model, pixel_indices)
    kernel_out = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                      view_params, **args)
    reference = _multiaxis_forward_view_batch(values, pixel_indices,
                                              view_params, **args)
    num_views = int(view_params.shape[0])
    assert kernel_out.shape == (num_views, detector_rows,
                                int(args['num_channels']))
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    assert _rel_max(kernel_out, reference) <= 1e-5
    # The permute leaves the channel-major row axis on the unit stride and the
    # channel axis on the allocated row count; the torch body's own permute of
    # a contiguous (Vb, C, R) does the same at the unpadded count.
    padded = padded_kernel_width(detector_rows)
    assert kernel_out.stride(1) == 1
    assert kernel_out.stride(2) == padded, (detector_rows, padded)
    assert reference.stride(1) == 1
    assert reference.stride(2) == detector_rows


@requires_cuda
@pytest.mark.parametrize("num_pixels", [1, 31])
def test_multiaxis_back_kernel_pixel_padding(num_pixels):
    # Poison the padding: a pixel count that is not a multiple of the kernel's
    # pixel tile pads the last block with lanes that must contribute exactly
    # nothing.  The counts straddle that tile (see MULTIAXIS_BACK_BLOCK_P and
    # the minimum tile the wrapper shrinks to).  Two independent statements --
    # parity against the torch body, and the invariant that a pixel's output
    # does not depend on which lane of which block it landed in (the same
    # pixels inside a LARGER subset must give the same values).
    assert MULTIAXIS_BACK_BLOCK_P > 1
    model = _ma_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    subset = pixel_indices[:num_pixels]
    reference = _multiaxis_back_view_batch(sinogram, subset, view_params,
                                           **args)
    kernel_out = _multiaxis_back_view_batch_triton(sinogram, subset,
                                                   view_params, **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    assert _rel_max(kernel_out, reference) <= 1e-5

    full = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                             view_params, **args)
    assert bool(torch.equal(kernel_out, full[:num_pixels]))


# The shape the forward COVERAGE tests use.  A thin voxel slice pitch against
# the detector row pitch lowers the slice-to-row slope without touching the
# vertical footprint, which is exactly what makes more slices reach one row
# than det_row_radius counts.  It is deliberately not one of the VARIANTS: those
# sweep the contract for the parity gates, and every one of them happens to
# need a single slice tap on each side.
THIN_SLICE_SHAPE = dict(elevation="tilt", slice_aspect=0.5)


def _vertical_terms(model):
    """The vertical fan's per-view terms for a model's full pixel set, built by
    the same eager builders the bodies and the kernels use: (m0, slope, W_p_r,
    args)."""
    device = model.torch_device
    recon_shape = model.get_params('recon_shape')
    pixel_indices = torch.as_tensor(mbirtorch.gen_full_indices(recon_shape),
                                    dtype=torch.int64, device=device)
    view_params = torch.as_tensor(model.get_params('angles'),
                                  dtype=torch.float32, device=device)
    args = model._view_batch_args()
    azimuth, elevation = view_params[:, 0], view_params[:, 1]
    _, _, _, _, y = _multiaxis_horizontal_data(
        pixel_indices, azimuth, args['num_recon_rows'], args['num_recon_cols'],
        args['num_channels'], args['delta_voxel'], args['delta_voxel_row'],
        args['delta_det_channel'], args['det_channel_offset'])
    m0, slope, w_p_r, _, _ = _multiaxis_vertical_terms(
        y, azimuth, elevation, args['num_slices'], args['delta_voxel'],
        args['delta_voxel_row'], args['delta_voxel_slice'],
        args['delta_det_row'], args['det_row_offset'],
        args['recon_slice_offset'], args['num_rows_r'])
    return m0, slope, w_p_r, args


def _slice_reach(model):
    """How far the forward kernel's slice enumeration REALLY has to reach on
    this model, counted rather than derived.

    Returns the largest |k - round(k_center)| over every (pixel, detector row,
    slice) triple whose vertical trapezoid weight is nonzero -- the exact
    quantity :func:`mbirtorch.triton_multiaxis._multiaxis_slice_tap_radius` has
    to bound.  Brute force over the whole geometry, so it is an independent
    ruler for that bound and not a second copy of it.
    """
    m0, slope, w_p_r, args = _vertical_terms(model)
    device = m0.device
    k = torch.arange(int(args['num_slices']), dtype=torch.float32,
                     device=device)
    m = torch.arange(int(args['num_rows_r']), dtype=torch.float32,
                     device=device)
    reach = 0
    for v in range(m0.shape[0]):
        rise, width = float(slope[v, 0]), float(w_p_r[v, 0])
        m_p = m0[v][:, None] + rise * k[None, :]                      # (P, S)
        live = (m_p[:, :, None] - m[None, None, :]).abs() < (width + 1.0) / 2.0
        k_center = torch.floor((m[None, :] - m0[v][:, None]) / rise + 0.5)
        dist = (k[None, :, None] - k_center[:, None, :]).abs()      # (P, S, R)
        if bool(live.any()):
            reach = max(reach, int(dist[live].max()))
    # Same order as _vertical_terms above, minus m0, so the two cannot be
    # unpacked into each other's names by accident.
    return reach, slope, w_p_r, args


@pytest.mark.parametrize("variant", ["strong_tilt", "thin_slice"])
def test_multiaxis_forward_slice_tap_radius_covers_every_contributing_slice(
        variant):
    """The forward kernel's one hard requirement, checked on any machine.

    The kernel gathers where the torch body scatters, so it enumerates slices
    k = round(k_center) + t for |t| <= radius and anything outside that window
    is silently dropped.  This counts the contributing triples directly and
    asserts the wrapper's bound covers them, which is the statement a parity
    test can only make indirectly (a dropped slice at the edge of a trapezoid
    carries a small weight and can hide under a relative tolerance).
    """
    kwargs = (THIN_SLICE_SHAPE if variant == "thin_slice"
              else VARIANTS[variant])
    model = _ma_model(device='cpu', **kwargs)
    reach, slope, w_p_r, args = _slice_reach(model)
    radius = _multiaxis_slice_tap_radius(w_p_r, slope)
    print(f"multiaxis forward slice-tap coverage ({variant}): "
          f"reach = {reach}, bound = {radius}, det_row_radius = "
          f"{args['det_row_radius']}")
    assert radius >= reach
    # The bound is not free to grow without limit either.  Two statements of
    # that: it stays under the wrapper's cap, so these shapes launch the kernel
    # rather than delegating, and the window it enumerates stays smaller than
    # the volume, so the gather never walks more slices than a scatter would
    # have touched.  A bound that inflated would show up here rather than only
    # in a sweep.  Neither is a tightness gate -- the bound carries a
    # deliberate margin, and zero_elevation is the case where that margin is
    # widest, because its boundary taps carry weight exactly zero.
    assert radius <= MULTIAXIS_FWD_MAX_SLICE_RADIUS
    assert 2 * radius + 1 <= int(args['num_slices'])


@requires_cuda
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_multiaxis_forward_kernel_parity(variant):
    # As for the back kernel: the geometry reaches the forward kernel only
    # through the two eager builders it shares with the torch body, so all five
    # variants exercise the same kernel with different contract values.  The
    # forward body takes no coeff_power.
    model = _ma_model(**VARIANTS[variant])
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    reference = _multiaxis_forward_view_batch(values, pixel_indices,
                                              view_params, **args)
    kernel_out = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                      view_params, **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    rel = _rel_max(kernel_out, reference)
    print(f"multiaxis forward triton parity ({variant}, "
          f"det_channel_radius={args['det_channel_radius']}): rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
@pytest.mark.parametrize("num_pixels", [1, 31])
def test_multiaxis_forward_kernel_pixel_padding(num_pixels):
    # Poison the padding: a pixel count that is not a multiple of the kernel's
    # pixel tile pads the last block with lanes whose atomics must be masked off
    # entirely.  The counts straddle that tile (see MULTIAXIS_FWD_BLOCK_P and
    # the minimum tile the wrapper shrinks to).  Two independent statements --
    # parity against the torch body, and additivity over a pixel SPLIT (the
    # forward sums all pixels into one sinogram, so a subset's output is a
    # partial sum and the two parts must reassemble the whole however the blocks
    # were padded).  The split is compared at a tolerance, not bit for bit:
    # the atomics reorder the sum over pixels, so the three launches add the
    # same terms in different orders by construction.
    assert MULTIAXIS_FWD_BLOCK_P > 1
    model = _ma_model()
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    subset, rest = pixel_indices[:num_pixels], pixel_indices[num_pixels:]
    reference = _multiaxis_forward_view_batch(values[:num_pixels], subset,
                                              view_params, **args)
    kernel_out = _multiaxis_forward_view_batch_triton(values[:num_pixels],
                                                      subset, view_params,
                                                      **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    assert _rel_max(kernel_out, reference) <= 1e-5

    full = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                view_params, **args)
    rest_out = _multiaxis_forward_view_batch_triton(values[num_pixels:], rest,
                                                    view_params, **args)
    assert _rel_max(kernel_out + rest_out, full) <= 1e-6


@requires_cuda
def test_multiaxis_kernel_pair_adjointness():
    # Three statements of <F x, a> == <x, B a>, the pairing the whole projector
    # contract rests on.  Two hold one side fixed to the torch body -- kernel
    # back against the torch forward, kernel forward against the torch back --
    # and would each catch a weight or index convention that drifted in that
    # one kernel.  The third is the pairing that actually ships once both
    # kernels are on: KERNEL forward against KERNEL back, which is what a
    # convention that drifted in BOTH kernels together would fail.  It matters
    # more for this geometry than for the others, because the two kernels do
    # not mirror each other on the vertical axis -- the back gathers slices
    # from rows and the forward gathers rows from slices, through the two
    # algebraic forms of the one affine map.  The elevation sweep is the point
    # throughout: at zero elevation the vertical fan is degenerate and would
    # not exercise the tilt terms.
    model = _ma_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    torch_forward = _multiaxis_forward_view_batch(values, pixel_indices,
                                                  view_params, **args)
    torch_back = _multiaxis_back_view_batch(sinogram, pixel_indices,
                                            view_params, **args)
    forward = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                   view_params, **args)
    back = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                             view_params, **args)

    lhs = float((torch_forward * sinogram).sum())
    rhs = float((values * back).sum())
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    print(f"multiaxis back triton adjointness: lhs {lhs:.6f}, rhs {rhs:.6f}, "
          f"rel {rel:.2e}")
    assert rel <= 1e-4

    lhs = float((forward * sinogram).sum())
    rhs = float((values * torch_back).sum())
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    print(f"multiaxis forward triton adjointness: lhs {lhs:.6f}, "
          f"rhs {rhs:.6f}, rel {rel:.2e}")
    assert rel <= 1e-4

    rhs = float((values * back).sum())
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    print(f"multiaxis kernel-pair adjointness: lhs {lhs:.6f}, rhs {rhs:.6f}, "
          f"rel {rel:.2e}")
    assert rel <= 1e-4


@requires_cuda
def test_multiaxis_kernels_span_several_row_chunks():
    # Every shape above is narrower than the forward kernel's row tile, so its
    # row grid axis holds exactly one program and a bug in that offset would
    # not show.  _tile_size caps the tile at next_pow2(extent), so forcing more
    # chunks takes more rows than the pinned tile: this shape gives the forward
    # kernel several detector row chunks and the back kernel several slice
    # chunks.
    shape = (4, 4 * MULTIAXIS_FWD_BLOCK_R, 16)
    model = _ma_model(shape=shape)
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    assert int(args['num_rows_r']) > MULTIAXIS_FWD_BLOCK_R

    fwd_ref = _multiaxis_forward_view_batch(values, pixel_indices, view_params,
                                            **args)
    fwd_out = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                   view_params, **args)
    fwd_rel = _rel_max(fwd_out, fwd_ref)
    back_ref = _multiaxis_back_view_batch(sinogram, pixel_indices, view_params,
                                          **args)
    back_out = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                                 view_params, **args)
    back_rel = _rel_max(back_out, back_ref)
    print(f"multiaxis triton multi-row-chunk ({args['num_rows_r']} rows): "
          f"forward rel_max = {fwd_rel:.2e}, back rel_max = {back_rel:.2e}")
    # 1e-4 here rather than the 1e-5 the small shapes gate at, and the reason
    # is the row count, not the chunking.  The trapezoid weight subtracts two
    # row coordinates of size about num_rows_r, and the kernel and the torch
    # body round the m0 + slope * k forming them differently (the compiler may
    # fuse the multiply-add), so the weight carries an absolute perturbation of
    # about num_rows_r times float32 eps -- about 6e-5 at these 512 rows, read
    # about 2e-5 on both directions when this gate first ran.  The small-shape
    # parity tests above hold the 1e-5 statement where the coordinates cannot
    # inflate it; what this test states is that the chunk offsets are right,
    # and an offset bug would miss by orders of magnitude, not by rounding.
    assert fwd_rel <= 1e-4
    assert back_rel <= 1e-4


# The tolerance a whole RECONSTRUCTION is compared at, where the projection
# tests above compare single calls at 1e-5.  A reconstruction is an iterative
# solver, so a per-call float difference is carried forward and reshaped by
# every later update; 5e-3 is the figure the other model-level gates on this
# geometry use (tests/test_multiaxis.py's sharded-vs-single recon, which
# records a measured spread of 9.4e-4 against that gate, and the
# kernel-times-sharding gate in tests/test_kernels_sharded.py).
RECON_TOLERANCE = 5e-3


@requires_cuda
def test_multiaxis_kernel_recon_matches_a_torch_bodies_recon(monkeypatch):
    # The composition a user on a CUDA machine actually gets: a whole seeded
    # reconstruction through the kernel route, against the same reconstruction
    # on the torch bodies.  The reference FORCES the torch bodies rather than
    # setting compile_mode, because selection is availability-driven and eager
    # does not mean unkernelled.
    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        shape = (12, 24, 20)
        model = _ma_model(shape=shape, compile_mode='off')
        # The arm check: this test exists to measure the KERNELS, so a silent
        # availability decline must fail loudly rather than compare torch with
        # torch and pass vacuously.
        fwd, back = model._view_batch_bodies()
        assert fwd is _multiaxis_forward_view_batch_triton
        assert back is _multiaxis_back_view_batch_triton

        recon_shape = model.get_params('recon_shape')
        phantom = mbirtorch.utilities._gen_translation_phantom(recon_shape, 'dots', None,
                                                    fill_rate=0.05)
        sinogram = np.asarray(model.forward_project(phantom), dtype=np.float32)
        np.random.seed(0)
        kernel_recon, _info = model.recon(sinogram, max_iterations=3,
                                          stop_threshold_change_pct=0.0,
                                          logfile_path=None)

        reference_model = _ma_model(shape=shape, compile_mode='off')
        reference_model._view_batch_bodies = lambda: (
            _multiaxis_forward_view_batch, _multiaxis_back_view_batch)
        reference_model.create_projectors()
        np.random.seed(0)
        torch_recon, _info = reference_model.recon(
            sinogram, max_iterations=3, stop_threshold_change_pct=0.0,
            logfile_path=None)

        kernel_recon = np.asarray(kernel_recon, dtype=np.float64)
        torch_recon = np.asarray(torch_recon, dtype=np.float64)
        rel = float(np.max(np.abs(kernel_recon - torch_recon))
                    / np.max(np.abs(torch_recon)))
        print(f"multiaxis triton recon vs torch-bodies recon: "
              f"rel_max = {rel:.2e}")
        assert rel < RECON_TOLERANCE
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()

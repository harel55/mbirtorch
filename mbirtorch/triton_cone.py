"""This module holds hand-written Triton kernels for the cone beam back and
forward projections.

Each wrapper function in this module projects one batch of views.  It replaces a torch
function with the same signature: :func:`mbirtorch.cone_beam._cone_back_view_batch` or
:func:`mbirtorch.cone_beam._cone_forward_view_batch`.  The results match the torch
functions to a relative error of 1e-5, or 1e-4 when coeff_power is 2.  The forward kernel
adds with atomic operations, so its results can differ from run to run by float rounding.
The module imports without triton.  Only a call to a wrapper needs triton.
"""

import contextlib
import math

import torch

from ._utils import padded_kernel_width
from .cone_beam import (_cone_back_view_batch, _cone_horizontal_data,
                        _cone_vertical_affine)
from .projectors import compile_serialized

_F32 = torch.float32

try:
    import triton
    import triton.language as tl
except ImportError:                        # Triton is missing, as on macOS or a CPU build.
    triton = None

    class _NoTritonLanguage:
        """This class is used in place of ``triton.language`` when triton is not
        installed.  The module reads only ``constexpr`` from it at import time."""
        constexpr = None

    tl = _NoTritonLanguage()


def _jit(fn):
    """This function applies ``triton.jit`` when triton is installed, and
    otherwise returns the function unchanged."""
    return fn if triton is None else triton.jit(fn)


def _tl_builtin(name):
    """This function returns the ``triton.language`` builtin function ``name``.  The
    function looks for the name in both ``tl`` and ``tl.math``.  It returns None when triton
    is not installed.
    """
    fn = getattr(tl, name, None)
    if fn is None:
        fn = getattr(getattr(tl, 'math', None), name, None)
    return fn


_tl_abs = _tl_builtin('abs')
_tl_floor = _tl_builtin('floor')
_tl_sqrt = _tl_builtin('sqrt')
# tl.static_range gives the tap loops a loop count fixed at compile time.  The code uses
# plain range when the installed triton has no static_range.
_tap_range = _tl_builtin('static_range') or range

# The number of GPU registers limits the tile sizes below.  The inner loop holds the
# (BLOCK_P, BLOCK_L) accumulator and about six other tiles.
CONE_BACK_BLOCK_P = 16
CONE_BACK_BLOCK_L = 64
CONE_BACK_NUM_WARPS = 4
CONE_BACK_NUM_STAGES = 1
# The wrapper pads a slice range or a pixel set that is smaller than this tile size.
CONE_BACK_MIN_TILE = 8
# The number of views in a batch when the model's view_batch_size is None.  The calling
# code may use a smaller batch to fit in memory.
CONE_BACK_VIEW_CHUNK = 128

CONE_FWD_BLOCK_P = 8
CONE_FWD_BLOCK_R = 128
CONE_FWD_NUM_WARPS = 8
CONE_FWD_NUM_STAGES = 1
CONE_FWD_MIN_TILE = 8
CONE_FWD_VIEW_CHUNK = 128

# This set holds the launch keys whose triton compilation has finished in this process.
# A key starts with the kernel name and holds the integer arguments and the constexprs.
_COMPILED_LAUNCH_KEYS = set()


@_jit
def _cone_back_kernel(n_p_ptr, centers_ptr, w_p_c_ptr, weight_scale_ptr,
                      m0_ptr, w_p_r_ptr, pixel_mag_ptr, z_offset_ptr,
                      sino_ptr, out_ptr,
                      num_views, num_pixels, num_channels, num_rows, band_len,
                      slice_start, sino_view_stride,
                      delta_voxel_slice, slice_center, inv_sdd, num_rows_f,
                      DET_CHANNEL_RADIUS: tl.constexpr, DET_ROW_RADIUS: tl.constexpr,
                      COEFF_POWER: tl.constexpr, ACCUMULATE: tl.constexpr,
                      BLOCK_P: tl.constexpr, BLOCK_L: tl.constexpr):
    """This function is the Triton back projection kernel.  Each Triton program computes
    one block of pixels and one block of slices of the output.

        out[p, l] = sum over views v, row taps tr, channel taps tc of
                    Wrow[v, p, l, tr] * Wchan[v, p, tc]
                    * sino[v, c(v, p) + tc, m(v, p, l) + tr]

    The kernel computes both sets of weights from arrays indexed by (view, pixel).  The
    output covers a contiguous range of slices.  ``band_len`` is the number of slices in this
    range, padded by the wrapper.  The kernel masks pixel indices at or beyond ``num_pixels``.
    When ``ACCUMULATE`` is True, the kernel adds its result to the values already in ``out``.
    """
    p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)     # (BLOCK_P,)
    l_offs = tl.program_id(1) * BLOCK_L + tl.arange(0, BLOCK_L)     # (BLOCK_L,)
    p_mask = p_offs < num_pixels
    l_mask = l_offs < band_len
    tile_mask = p_mask[:, None] & l_mask[None, :]

    # k is the slice index of each output column in the full volume.  The row map and z
    # are computed from slice 0 of the full volume, so a slice range only limits k.
    k = (slice_start + l_offs).to(tl.float32)                       # (BLOCK_L,)
    z_at_k = delta_voxel_slice * (k - slice_center)                 # (BLOCK_L,)

    acc = tl.zeros((BLOCK_P, BLOCK_L), dtype=tl.float32)
    for v in range(num_views):
        pix_base = v.to(tl.int64) * num_pixels + p_offs
        n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
        centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
        w_p_c = tl.load(w_p_c_ptr + pix_base, mask=p_mask, other=0.0)
        weight_scale = tl.load(weight_scale_ptr + pix_base, mask=p_mask,
                               other=0.0)
        m0 = tl.load(m0_ptr + pix_base, mask=p_mask, other=0.0)
        w_p_r = tl.load(w_p_r_ptr + pix_base, mask=p_mask, other=0.0)
        pixel_mag = tl.load(pixel_mag_ptr + pix_base, mask=p_mask, other=0.0)
        z_offset = tl.load(z_offset_ptr + v)

        # m is the detector row of slice k.  inv_cos_phi is the cone angle factor, computed
        # from the detector height v = pixel_mag * z.
        m = m0[:, None] + w_p_r[:, None] * k[None, :]        # (BLOCK_P, BLOCK_L)
        m_center = _tl_floor(m + 0.5)                        # Ties carry zero weight.
        # The center is clamped before the integer conversions in the tap loop, because
        # conversion from float to int is undefined outside the int32 range.
        m_center = tl.minimum(tl.maximum(m_center, -1.0 - DET_ROW_RADIUS),
                              num_rows_f + DET_ROW_RADIUS)
        v_det = pixel_mag[:, None] * (z_at_k[None, :] + z_offset)
        inv_cos_phi = _tl_sqrt(1.0 + (v_det * inv_sdd) * (v_det * inv_sdd))
        l_max_r = tl.minimum(w_p_r, 1.0)[:, None]
        sino_view_ptr = sino_ptr + v.to(tl.int64) * sino_view_stride

        for tr in _tap_range(0, 2 * DET_ROW_RADIUS + 1):
            m_tap = m_center + (tr - DET_ROW_RADIUS)
            m_tap_i = m_tap.to(tl.int32)
            w_row = tl.maximum((w_p_r[:, None] + 1.0) / 2.0
                               - _tl_abs(m - m_tap), 0.0)
            # coeff_power must be applied after the cone angle factor.
            w_row = tl.minimum(w_row, l_max_r) * inv_cos_phi
            w_row = tl.where((m_tap_i >= 0) & (m_tap_i < num_rows), w_row, 0.0)
            if COEFF_POWER == 2:
                w_row = w_row * w_row
            m_row = tl.minimum(tl.maximum(m_tap_i, 0), num_rows - 1)

            row_vals = tl.zeros((BLOCK_P, BLOCK_L), dtype=tl.float32)
            for tc in _tap_range(0, 2 * DET_CHANNEL_RADIUS + 1):
                # The weight is zero where the tap is outside the detector.  Then the
                # index is clamped into range.
                n_tap = centers + (tc - DET_CHANNEL_RADIUS)
                w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                                    - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
                w_chan = tl.minimum(w_chan, tl.minimum(w_p_c, 1.0)) * weight_scale
                w_chan = tl.where((n_tap >= 0) & (n_tap < num_channels),
                                  w_chan, 0.0)
                if COEFF_POWER == 2:
                    w_chan = w_chan * w_chan
                n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
                # The views have the channel axis before the row axis.  The row axis is
                # contiguous, so each tile reads nearly consecutive addresses.
                chan_ptr = sino_view_ptr + n_chan.to(tl.int64) * num_rows
                vals = tl.load(chan_ptr[:, None] + m_row, mask=tile_mask,
                               other=0.0)
                row_vals = row_vals + w_chan[:, None] * vals
            acc = acc + w_row * row_vals

    out_ptrs = out_ptr + p_offs.to(tl.int64)[:, None] * band_len + l_offs[None, :]
    if ACCUMULATE:
        # This float32 sum equals torch's add_ of the stored array and this result.
        acc = tl.load(out_ptrs, mask=tile_mask, other=0.0) + acc
    tl.store(out_ptrs, acc, mask=tile_mask)


def _next_pow2(n):
    return 1 << (max(int(n), 1) - 1).bit_length()


def _tile_size(cap, extent, min_tile):
    """This function returns the tile size for an axis of length ``extent``.  The tile size
    is a power of two.  It equals ``cap``, reduced for a shorter axis, but it is never less
    than ``min_tile``.
    """
    return max(int(min_tile), min(int(cap), _next_pow2(extent)))


def _padded_accumulator(accumulate_into, num_pixels, band_len, launch_band, device):
    """This function returns the (num_pixels, launch_band) array that holds
    ``accumulate_into``, so that a back kernel can add into it.

    A back kernel wrapper returns the first ``band_len`` columns of a (num_pixels,
    launch_band) array.  The function returns None when ``accumulate_into`` is not such an
    output.
    """
    a = accumulate_into
    if (a.dtype != _F32 or a.device != device
            or tuple(a.shape) != (int(num_pixels), int(band_len))
            or tuple(a.stride()) != (int(launch_band), 1)):
        return None
    needed_bytes = (a.storage_offset() + int(num_pixels) * int(launch_band)) * 4
    if a.untyped_storage().nbytes() < needed_bytes:
        return None
    return a.as_strided((int(num_pixels), int(launch_band)), (int(launch_band), 1))


@torch.compiler.disable
def _cone_back_view_batch_triton(sino_batch, pixel_indices, view_params_batch,
                                 num_rows_r, num_channels, num_recon_rows,
                                 num_recon_cols, num_slices, delta_voxel,
                                 delta_voxel_row, delta_voxel_slice,
                                 delta_det_channel, delta_det_row,
                                 det_channel_offset, det_row_offset,
                                 recon_slice_offset, magnification,
                                 source_detector_dist, use_curved_detector,
                                 det_channel_radius, det_row_radius,
                                 vox_slice_radius, coeff_power=1,
                                 slice_start=0, band_slices=None, plan=None,
                                 accumulate_into=None):
    """This function computes the cone beam back projection of one batch of views with
    Triton.  It replaces :func:`mbirtorch.cone_beam._cone_back_view_batch`.

    The function has the same signature and returns the same (P, band_slices) tensor.  Each
    call allocates a new output tensor, so the caller may add into it in place.  When the
    number of slices is not a multiple of 16, the output is a strided view.  torch.compile
    does not compile this function.  The function ignores ``vox_slice_radius`` and ``plan``.

    ``accumulate_into`` is None or the output of an earlier call with the same pixels and
    slices.  If it is given, the function adds this batch to it in place and returns it.
    """
    if triton is None:
        raise RuntimeError('the Triton cone back body was called without '
                           'triton installed; the selection in '
                           'ConeBeamModel._view_batch_bodies should have kept '
                           'the torch body (see kernel_availability).')
    # The kernel supports only coeff_power 1 and 2.
    if coeff_power not in (1, 2):
        block = _cone_back_view_batch(
            sino_batch, pixel_indices, view_params_batch, num_rows_r,
            num_channels, num_recon_rows, num_recon_cols, num_slices,
            delta_voxel, delta_voxel_row, delta_voxel_slice, delta_det_channel,
            delta_det_row, det_channel_offset, det_row_offset,
            recon_slice_offset, magnification, source_detector_dist,
            use_curved_detector, det_channel_radius, det_row_radius,
            vox_slice_radius, coeff_power=coeff_power, slice_start=slice_start,
            band_slices=band_slices, plan=plan)
        return block if accumulate_into is None else accumulate_into.add_(block)

    angles = view_params_batch[:, 0]
    z_shifts = view_params_batch[:, 1]
    n_p, centers, w_p_c, weight_scale, pixel_mag = _cone_horizontal_data(
        pixel_indices, angles, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset,
        magnification, source_detector_dist, use_curved_detector)
    m0, w_p_r, z_offset = _cone_vertical_affine(
        pixel_mag, z_shifts, num_slices, delta_voxel_slice, delta_det_row,
        det_row_offset, recon_slice_offset, num_rows_r)

    num_views, num_pixels = n_p.shape
    band_len = int(num_slices if band_slices is None else band_slices)
    # The kernel runs on a slice count rounded up to a multiple of 16.  Every
    # slice count passed to the kernel uses this value.
    launch_band = padded_kernel_width(band_len)
    # This copy puts the channel axis before the row axis, so each tile reads contiguous
    # memory.  The copy is not padded, because the kernel clamps every sinogram index.
    sino_t = sino_batch.permute(0, 2, 1).contiguous()
    contract = [t.contiguous() for t in (n_p, centers, w_p_c, weight_scale,
                                         m0, w_p_r, pixel_mag, z_offset)]
    target = (None if accumulate_into is None else _padded_accumulator(
        accumulate_into, num_pixels, band_len, launch_band, sino_batch.device))
    out = target if target is not None else torch.empty(
        (num_pixels, launch_band), dtype=_F32, device=sino_batch.device)

    block_p = _tile_size(CONE_BACK_BLOCK_P, num_pixels, CONE_BACK_MIN_TILE)
    block_l = _tile_size(CONE_BACK_BLOCK_L, launch_band, CONE_BACK_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-launch_band // block_l))
    inv_sdd = (0.0 if math.isinf(float(source_detector_dist))
               else 1.0 / float(source_detector_dist))
    launch_key = ('back', sino_batch.device.index, int(det_channel_radius),
                  int(det_row_radius), int(coeff_power), target is not None, block_p, block_l,
                  int(num_views), int(num_pixels), int(num_channels),
                  int(num_rows_r), launch_band, int(slice_start))
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    compile_lock = compile_serialized() if first_launch else contextlib.nullcontext()
    # A Triton launch runs on the current CUDA device of the thread, which is device 0 in a
    # worker thread.  The device context makes the launch run on the device of the tensors.
    with torch.cuda.device(sino_batch.device), compile_lock:
        _cone_back_kernel[grid](
            *contract, sino_t, out,
            int(num_views), int(num_pixels), int(num_channels),
            int(num_rows_r), launch_band, int(slice_start),
            int(num_channels) * int(num_rows_r),
            float(delta_voxel_slice), (int(num_slices) - 1) / 2.0, inv_sdd,
            float(num_rows_r),
            DET_CHANNEL_RADIUS=int(det_channel_radius),
            DET_ROW_RADIUS=int(det_row_radius), COEFF_POWER=int(coeff_power),
            ACCUMULATE=target is not None,
            BLOCK_P=block_p, BLOCK_L=block_l,
            num_warps=CONE_BACK_NUM_WARPS, num_stages=CONE_BACK_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    if target is not None:
        return accumulate_into
    # This slice removes the padded columns.  The result is a strided view, and each
    # row of the view is contiguous.
    block = out if launch_band == band_len else out[:, :band_len]
    # An accumulate_into without the layout of a kernel output receives the new result here.
    return block if accumulate_into is None else accumulate_into.add_(block)


# maybe_compile reads this attribute and does not compile the function.
_cone_back_view_batch_triton._mbirtorch_no_compile = True
# The projection loop reads this attribute and passes its running sum as accumulate_into.
_cone_back_view_batch_triton._adds_into_output = True


def _cone_back_view_batch_cost(num_pixels, num_band_rows, args):
    """This function returns the GPU memory, in bytes, that each view uses in one batch of
    the back kernel.  It also returns the default number of views in a batch.

    The calling code reads this function through the ``_view_batch_cost`` attribute.  Each
    view uses 48 bytes per pixel for geometry arrays.  Each view also uses a copy of its
    sinogram plane with the channel axis first.
    """
    plane_bytes = 4 * int(args['num_channels']) * int(num_band_rows)
    return 48 * int(num_pixels) + plane_bytes, CONE_BACK_VIEW_CHUNK


_cone_back_view_batch_triton._view_batch_cost = _cone_back_view_batch_cost


def _cone_back_view_peak_bytes(num_pixels, num_band_rows, args):
    """This function returns the peak memory in bytes that one view of a batch uses in the
    back kernel.  The memory estimate uses this value.  The view batch size uses
    :func:`_cone_back_view_batch_cost` instead.

    At its peak, ``_cone_horizontal_data`` holds 11 arrays with one 4 byte entry for
    each view and pixel.  Each view therefore uses 44 bytes per pixel.  Each view also uses
    its copy of the sinogram plane.
    """
    # With zero pixels, the cost function returns the bytes of the sinogram view alone.
    plane_bytes = _cone_back_view_batch_cost(0, num_band_rows, args)[0]
    return 44 * int(num_pixels) + plane_bytes


_cone_back_view_batch_triton._view_peak_bytes = _cone_back_view_peak_bytes


@_jit
def _cone_forward_kernel(n_p_ptr, centers_ptr, w_p_c_ptr, weight_scale_ptr,
                         m0_ptr, w_p_r_ptr, pixel_mag_ptr, z_offset_ptr,
                         values_ptr, out_ptr,
                         num_pixels, num_channels, num_rows, band_len,
                         slice_start, out_view_stride,
                         delta_voxel_slice, slice_center, inv_sdd,
                         k_center_lo, k_center_hi,
                         DET_CHANNEL_RADIUS: tl.constexpr,
                         VOX_SLICE_RADIUS: tl.constexpr,
                         BLOCK_P: tl.constexpr, BLOCK_R: tl.constexpr):
    """This function is the Triton forward projection kernel.  Each Triton program computes
    one block of pixels, one block of detector rows, and one view.

        out[v, c, m] += sum over pixels p, channel taps tc, slice taps tk of
                        Wchan[v, p, tc] * Wslice[v, p, m, tk] / cos_phi
                        * values[p, k(v, p, m) + tk]

    The kernel maps detector row m to slice ``k = (m - m0) / W_p_r``.  The output has
    shape (Vb, C, R), with the channel axis before the row axis.  The wrapper transposes it.
    ``num_rows`` is the row count padded by the wrapper.  The kernel masks pixel indices at or
    beyond ``num_pixels``, and the wrapper removes the padded rows.
    """
    p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)     # (BLOCK_P,)
    r_offs = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)     # (BLOCK_R,)
    v = tl.program_id(2)
    p_mask = p_offs < num_pixels
    r_mask = r_offs < num_rows
    tile_mask = p_mask[:, None] & r_mask[None, :]

    pix_base = v.to(tl.int64) * num_pixels + p_offs
    n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
    centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
    w_p_c = tl.load(w_p_c_ptr + pix_base, mask=p_mask, other=0.0)
    weight_scale = tl.load(weight_scale_ptr + pix_base, mask=p_mask, other=0.0)
    m0 = tl.load(m0_ptr + pix_base, mask=p_mask, other=0.0)
    # The fill value is 1.0, not 0.0.  A masked pixel divides by w_p_r, and 0/0 would give
    # a NaN in the float to int conversion below.
    w_p_r = tl.load(w_p_r_ptr + pix_base, mask=p_mask, other=1.0)
    pixel_mag = tl.load(pixel_mag_ptr + pix_base, mask=p_mask, other=0.0)
    z_offset = tl.load(z_offset_ptr + v)

    # This code maps each detector row to a fractional slice index.
    slope = w_p_r[:, None]                                          # (BLOCK_P, 1)
    k_m = (r_offs.to(tl.float32)[None, :] - m0[:, None]) / slope
    k_center = _tl_floor(k_m + 0.5)                          # Ties carry zero weight.
    # The center is clamped before the integer conversions in the tap loop, because
    # conversion from float to int is undefined outside the int32 range.
    k_center = tl.minimum(tl.maximum(k_center, k_center_lo), k_center_hi)
    # m_p is the offset of the center slice from the row, in units of detector rows.
    m_p = slope * (k_center - k_m)
    l_max_r = tl.minimum(w_p_r, 1.0)[:, None]

    det_col = tl.zeros((BLOCK_P, BLOCK_R), dtype=tl.float32)
    for tk in _tap_range(0, 2 * VOX_SLICE_RADIUS + 1):
        k_off = tk - VOX_SLICE_RADIUS
        k_ind = k_center + k_off
        k_ind_i = k_ind.to(tl.int32)
        w_slice = tl.maximum((slope + 1.0) / 2.0 - _tl_abs(m_p + slope * k_off),
                             0.0)
        w_slice = tl.minimum(w_slice, l_max_r)
        # The weight is zero where the tap is outside the slice range, and then the index
        # is clamped.  z is still computed from the slice count of the full volume.
        in_band = (k_ind_i >= slice_start) & (k_ind_i < slice_start + band_len)
        w_slice = tl.where(in_band, w_slice, 0.0)
        # The cone angle factor uses the slice of the tap, not the detector row.  The
        # detector height is v = pixel_mag * z, as in the back kernel.
        v_det = pixel_mag[:, None] * (delta_voxel_slice * (k_ind - slice_center)
                                      + z_offset)
        inv_cos_phi = _tl_sqrt(1.0 + (v_det * inv_sdd) * (v_det * inv_sdd))
        # The clamp keeps the index in range.  The load mask includes in_band, so the
        # kernel does not read taps that have zero weight.
        k_local = tl.minimum(tl.maximum(k_ind_i - slice_start, 0), band_len - 1)
        vals = tl.load(values_ptr + p_offs.to(tl.int64)[:, None] * band_len
                       + k_local, mask=tile_mask & in_band, other=0.0)
        det_col = det_col + w_slice * (vals * inv_cos_phi)

    # This loop adds each channel tap into the output with one atomic add.
    out_view_ptr = out_ptr + v.to(tl.int64) * out_view_stride
    for tc in _tap_range(0, 2 * DET_CHANNEL_RADIUS + 1):
        # This is the same trapezoid weight as in the back kernel.  The atomic mask skips
        # taps outside the detector.
        n_tap = centers + (tc - DET_CHANNEL_RADIUS)
        w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                            - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
        w_chan = tl.minimum(w_chan, tl.minimum(w_p_c, 1.0)) * weight_scale
        n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
        out_ptrs = (out_view_ptr + n_chan.to(tl.int64)[:, None] * num_rows
                    + r_offs[None, :])
        tl.atomic_add(out_ptrs, w_chan[:, None] * det_col,
                      mask=tile_mask & ((n_tap >= 0)
                                        & (n_tap < num_channels))[:, None])


@torch.compiler.disable
def _cone_forward_view_batch_triton(values, pixel_indices, view_params_batch,
                                    num_rows_r, num_channels, num_recon_rows,
                                    num_recon_cols, num_slices, delta_voxel,
                                    delta_voxel_row, delta_voxel_slice,
                                    delta_det_channel, delta_det_row,
                                    det_channel_offset, det_row_offset,
                                    recon_slice_offset, magnification,
                                    source_detector_dist, use_curved_detector,
                                    det_channel_radius, det_row_radius,
                                    vox_slice_radius, slice_start=0, plan=None):
    """This function computes the cone beam forward projection of one batch of views with
    Triton.  It replaces :func:`mbirtorch.cone_beam._cone_forward_view_batch`.

    The function has the same signature and returns the same (Vb, R, C) tensor.  ``values``
    has shape (P, L).  When L is less than num_slices, ``values`` holds the L slices that
    start at ``slice_start``.  The sum of the outputs over slice ranges that cover the volume
    is the full projection.  The function ignores ``det_row_radius`` and ``plan``.
    """
    if triton is None:
        raise RuntimeError('the Triton cone forward body was called without '
                           'triton installed; the selection in '
                           'ConeBeamModel._view_batch_bodies should have kept '
                           'the torch body (see kernel_availability).')
    angles = view_params_batch[:, 0]
    z_shifts = view_params_batch[:, 1]
    n_p, centers, w_p_c, weight_scale, pixel_mag = _cone_horizontal_data(
        pixel_indices, angles, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset,
        magnification, source_detector_dist, use_curved_detector)
    m0, w_p_r, z_offset = _cone_vertical_affine(
        pixel_mag, z_shifts, num_slices, delta_voxel_slice, delta_det_row,
        det_row_offset, recon_slice_offset, num_rows_r)

    num_views, num_pixels = n_p.shape
    band_len = int(values.shape[1])
    # The kernel runs on a detector row count rounded up to a multiple of 16.  The
    # geometry functions above use the true row count.
    launch_rows = padded_kernel_width(int(num_rows_r))
    values = values.contiguous()
    contract = [t.contiguous() for t in (n_p, centers, w_p_c, weight_scale,
                                         m0, w_p_r, pixel_mag, z_offset)]
    # The output has the channel axis before the row axis.  It starts at zero, because
    # the kernel adds into it with atomic operations.
    out = torch.zeros((num_views, num_channels, launch_rows), dtype=_F32,
                      device=values.device)

    block_p = _tile_size(CONE_FWD_BLOCK_P, num_pixels, CONE_FWD_MIN_TILE)
    block_r = _tile_size(CONE_FWD_BLOCK_R, launch_rows, CONE_FWD_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-launch_rows // block_r), num_views)
    inv_sdd = (0.0 if math.isinf(float(source_detector_dist))
               else 1.0 / float(source_detector_dist))
    # The kernel clamps the center slice to these bounds.  At the low bound, the highest
    # tap is slice_start - 1.  At the high bound, the lowest tap is slice_start + band_len.
    slice_radius = int(vox_slice_radius)
    k_center_lo = float(int(slice_start) - slice_radius - 1)
    k_center_hi = float(int(slice_start) + band_len + slice_radius)
    launch_key = ('fwd', values.device.index, int(det_channel_radius), slice_radius, block_p,
                  block_r, int(num_views),
                  int(num_pixels), int(num_channels), launch_rows,
                  band_len, int(slice_start))
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    compile_lock = compile_serialized() if first_launch else contextlib.nullcontext()
    # The launch runs on the device of the tensors.  The device index is part of the
    # launch key.
    with torch.cuda.device(values.device), compile_lock:
        _cone_forward_kernel[grid](
            *contract, values, out,
            int(num_pixels), int(num_channels), launch_rows, band_len,
            int(slice_start), int(num_channels) * launch_rows,
            float(delta_voxel_slice), (int(num_slices) - 1) / 2.0, inv_sdd,
            k_center_lo, k_center_hi,
            DET_CHANNEL_RADIUS=int(det_channel_radius), VOX_SLICE_RADIUS=slice_radius,
            BLOCK_P=block_p, BLOCK_R=block_r,
            num_warps=CONE_FWD_NUM_WARPS, num_stages=CONE_FWD_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    if launch_rows == int(num_rows_r):
        return out.permute(0, 2, 1)
    # This slice removes the padded detector rows before the transpose.
    return out[:, :, :int(num_rows_r)].permute(0, 2, 1)


# maybe_compile reads this attribute and does not compile the function.
_cone_forward_view_batch_triton._mbirtorch_no_compile = True


def _cone_forward_view_batch_cost(num_pixels, band_len, args):
    """This function returns the GPU memory, in bytes, that each view uses in one batch of
    the forward kernel.  It also returns the default number of views in a batch.

    Each view uses 48 bytes per pixel for geometry arrays, as in the back kernel.  Each view
    also uses its output plane, which includes the padded detector rows.
    """
    plane_rows = padded_kernel_width(int(args['num_rows_r']))
    plane_bytes = 4 * int(args['num_channels']) * plane_rows
    return 48 * int(num_pixels) + plane_bytes, CONE_FWD_VIEW_CHUNK


_cone_forward_view_batch_triton._view_batch_cost = \
    _cone_forward_view_batch_cost


def _cone_forward_view_peak_bytes(num_pixels, band_len, args):
    """This function returns the peak memory in bytes that one view of a batch uses in the
    forward kernel.  The memory estimate uses this value.  The view batch size uses
    :func:`_cone_forward_view_batch_cost` instead.

    Each view uses 44 bytes per pixel inside ``_cone_horizontal_data``, as in
    :func:`_cone_back_view_peak_bytes`.  Each view also uses its output plane.
    """
    # With zero pixels, the cost function returns the bytes of the sinogram view alone.
    plane_bytes = _cone_forward_view_batch_cost(0, band_len, args)[0]
    return 44 * int(num_pixels) + plane_bytes


_cone_forward_view_batch_triton._view_peak_bytes = _cone_forward_view_peak_bytes

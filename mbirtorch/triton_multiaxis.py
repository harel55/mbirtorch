"""This module contains Triton kernels for the multiaxis parallel-beam back projection and
forward projection.

Each kernel has a wrapper function that projects one batch of views.  Each wrapper has the
same signature as the torch function that it replaces.  The torch functions are
:func:`mbirtorch.multiaxis_parallel._multiaxis_back_view_batch` and
:func:`mbirtorch.multiaxis_parallel._multiaxis_forward_view_batch`.  Each view has two angles.
``view_params_batch[:, 0]`` is the azimuth, and ``view_params_batch[:, 1]`` is the elevation.
The back projection kernel gives bit-identical results on every run.  The forward projection
kernel uses atomic adds, so its results can differ from run to run by float rounding.
"""

import contextlib
import math

import torch

from ._utils import padded_kernel_width
from .multiaxis_parallel import (_multiaxis_back_view_batch,
                                 _multiaxis_forward_view_batch,
                                 _multiaxis_horizontal_data,
                                 _multiaxis_vertical_terms)
from .projectors import compile_serialized
# The Triton helper functions come from the cone-beam module.  All kernels share
# _COMPILED_LAUNCH_KEYS, and each key starts with the name of its kernel.
from .triton_cone import (_COMPILED_LAUNCH_KEYS, _jit, _padded_accumulator,
                          _tap_range, _tile_size, _tl_abs, _tl_floor, tl, triton)

_F32 = torch.float32

# The tile constants below are the same as in the cone-beam kernels.
MULTIAXIS_BACK_BLOCK_P = 16
MULTIAXIS_BACK_BLOCK_L = 64
MULTIAXIS_BACK_NUM_WARPS = 4
MULTIAXIS_BACK_NUM_STAGES = 1
# A tile is never smaller than this size.  Smaller inputs are padded up to it.
MULTIAXIS_BACK_MIN_TILE = 8
# The number of views in a batch when the model's view_batch_size is None.
# The caller can use a smaller batch to fit in memory.
MULTIAXIS_BACK_VIEW_CHUNK = 128

MULTIAXIS_FWD_BLOCK_P = 8
MULTIAXIS_FWD_BLOCK_R = 128
MULTIAXIS_FWD_NUM_WARPS = 8
MULTIAXIS_FWD_NUM_STAGES = 1
MULTIAXIS_FWD_MIN_TILE = 8
MULTIAXIS_FWD_VIEW_CHUNK = 128

# The largest slice tap radius for which the forward wrapper launches the kernel.
# For a larger radius, the wrapper calls the torch function instead.
MULTIAXIS_FWD_MAX_SLICE_RADIUS = 32

# The smallest slope used to convert detector rows to slices.  This lower bound
# prevents an infinity or a NaN when the slope is zero.
_SLOPE_FLOOR = 1e-6


def _multiaxis_slice_tap_radius(w_p_r, slope):
    """This function returns the slice tap radius of the forward kernel for one batch of views.

    The radius is the number of slices on each side of round(k_center) that can contribute to
    one detector row.  If the slope is zero or negative, the radius is larger than
    ``MULTIAXIS_FWD_MAX_SLICE_RADIUS``.

    Args:
        w_p_r: (torch.Tensor) The vertical footprint of each view in detector rows, shape (Vb, 1).
        slope: (torch.Tensor) The slope from slice index to detector row for each view, shape (Vb, 1).

    Returns:
        int: The radius.  The caller compares it with ``MULTIAXIS_FWD_MAX_SLICE_RADIUS``.
    """
    # These two scalar reads synchronize the device once per forward call.
    w_max = float(w_p_r.max())
    slope_min = max(float(slope.min()), _SLOPE_FLOOR)
    return int(math.floor((w_max + 1.0) / (2.0 * slope_min) + 0.5)) + 1


@_jit
def _multiaxis_back_kernel(n_p_ptr, centers_ptr, m0_ptr,
                           w_p_c_ptr, weight_scale_ptr,
                           slope_ptr, w_p_r_ptr, l_max_ptr, scale_pow_ptr,
                           sino_ptr, out_ptr,
                           num_views, num_pixels, num_channels, num_rows,
                           band_len, slice_start, num_slices,
                           sino_view_stride, num_rows_f,
                           DET_CHANNEL_RADIUS: tl.constexpr,
                           DET_ROW_RADIUS: tl.constexpr, COEFF_POWER: tl.constexpr,
                           ACCUMULATE: tl.constexpr,
                           BLOCK_P: tl.constexpr, BLOCK_L: tl.constexpr):
    """This function is the Triton back projection kernel.  Each program computes one block of
    pixels and one block of slices of the output.

        out[p, l] = sum over views v, row taps tr, channel taps tc of
                    Wrow[v, l, tr] * Wchan[v, p, tc]
                    * sino[v, c(v, p) + tc, m(v, p, l) + tr]

    The output is zero where the slice index slice_start + l is greater than or equal to
    num_slices.  ``band_len`` is the padded number of slices in the output.  The kernel masks
    pixels with index ``num_pixels`` or greater.  The wrapper returns only the unpadded slices.
    When ``ACCUMULATE`` is True, the kernel adds its result to the values already in ``out``.
    """
    p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)     # (BLOCK_P,)
    l_offs = tl.program_id(1) * BLOCK_L + tl.arange(0, BLOCK_L)     # (BLOCK_L,)
    p_mask = p_offs < num_pixels
    l_mask = l_offs < band_len
    tile_mask = p_mask[:, None] & l_mask[None, :]

    # k_global is the slice index in the whole volume for each output column.
    # The map from slice to row uses these whole-volume indices.
    k_global = slice_start + l_offs                                 # (BLOCK_L,)
    k = k_global.to(tl.float32)
    valid_k = k_global < num_slices

    acc = tl.zeros((BLOCK_P, BLOCK_L), dtype=tl.float32)
    for v in range(num_views):
        pix_base = v.to(tl.int64) * num_pixels + p_offs
        n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
        centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
        m0 = tl.load(m0_ptr + pix_base, mask=p_mask, other=0.0)
        # These are scalars for each view.  The horizontal footprint depends only on
        # the azimuth.  The vertical terms depend on the azimuth and elevation.
        w_p_c = tl.load(w_p_c_ptr + v)
        weight_scale = tl.load(weight_scale_ptr + v)
        clip_c = tl.minimum(w_p_c, 1.0)
        slope = tl.load(slope_ptr + v)
        w_p_r = tl.load(w_p_r_ptr + v)
        l_max_r = tl.load(l_max_ptr + v)
        scale_pow = tl.load(scale_pow_ptr + v)

        # m is the affine map from slice index to detector row.  The weight uses the
        # footprint w_p_r, not the slope.
        m = m0[:, None] + slope * k[None, :]                 # (BLOCK_P, BLOCK_L)
        m_center = _tl_floor(m + 0.5)                        # Ties carry zero weight.
        # The center is clamped before the conversions to int in the tap loop,
        # because float to int conversion is undefined outside the int32 range.
        m_center = tl.minimum(tl.maximum(m_center, -1.0 - DET_ROW_RADIUS),
                              num_rows_f + DET_ROW_RADIUS)
        sino_view_ptr = sino_ptr + v.to(tl.int64) * sino_view_stride

        for tr in _tap_range(0, 2 * DET_ROW_RADIUS + 1):
            m_tap = m_center + (tr - DET_ROW_RADIUS)
            m_tap_i = m_tap.to(tl.int32)
            # In the vertical trapezoid rule, the footprint sets the width, and l_max_r caps it.
            # A tap outside the detector gets weight zero, and its index is then clamped.
            w_row = tl.maximum((w_p_r + 1.0) / 2.0
                               - _tl_abs(m - m_tap), 0.0)
            w_row = tl.minimum(w_row, l_max_r)
            w_row = tl.where((m_tap_i >= 0) & (m_tap_i < num_rows), w_row, 0.0)
            if COEFF_POWER == 2:
                w_row = w_row * w_row
            # scale_pow multiplies the weight after the power is applied.  The wrapper
            # has already raised scale_pow to the same power.
            w_row = w_row * scale_pow
            m_row = tl.minimum(tl.maximum(m_tap_i, 0), num_rows - 1)

            row_vals = tl.zeros((BLOCK_P, BLOCK_L), dtype=tl.float32)
            for tc in _tap_range(0, 2 * DET_CHANNEL_RADIUS + 1):
                # A tap outside the detector gets weight zero, and its index is then clamped.
                n_tap = centers + (tc - DET_CHANNEL_RADIUS)
                w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                                    - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
                w_chan = tl.minimum(w_chan, clip_c) * weight_scale
                w_chan = tl.where((n_tap >= 0) & (n_tap < num_channels),
                                  w_chan, 0.0)
                if COEFF_POWER == 2:
                    w_chan = w_chan * w_chan
                n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
                # The sinogram views are in channel-major order, so the row axis is contiguous.
                chan_ptr = sino_view_ptr + n_chan.to(tl.int64) * num_rows
                vals = tl.load(chan_ptr[:, None] + m_row, mask=tile_mask,
                               other=0.0)
                row_vals = row_vals + w_chan[:, None] * vals
            acc = acc + w_row * row_vals

    # Padded slices, with index num_slices or greater, are set to zero.
    acc = tl.where(valid_k[None, :], acc, 0.0)
    out_ptrs = out_ptr + p_offs.to(tl.int64)[:, None] * band_len + l_offs[None, :]
    if ACCUMULATE:
        # This float32 sum equals torch's add_ of the stored array and this result.
        acc = tl.load(out_ptrs, mask=tile_mask, other=0.0) + acc
    tl.store(out_ptrs, acc, mask=tile_mask)


@torch.compiler.disable
def _multiaxis_back_view_batch_triton(sino_batch, pixel_indices,
                                      view_params_batch, num_rows_r,
                                      num_channels, num_recon_rows,
                                      num_recon_cols, num_slices, delta_voxel,
                                      delta_voxel_row, delta_voxel_slice,
                                      delta_det_channel, delta_det_row,
                                      det_channel_offset, det_row_offset,
                                      recon_slice_offset, det_channel_radius,
                                      det_row_radius, vox_slice_radius,
                                      coeff_power=1, slice_start=0,
                                      band_slices=None, plan=None,
                                      accumulate_into=None):
    """This function back projects one batch of views with the Triton kernel.

    The function replaces :func:`mbirtorch.multiaxis_parallel._multiaxis_back_view_batch`, and
    it has the same signature.  It returns a new array of shape (P, band_len) on each call, so
    the caller can add into the array in place.  If band_len is not a multiple of 16, the
    returned array is a strided view.  torch.compile does not compile this function.  The
    function ignores ``vox_slice_radius`` and ``plan``.

    ``accumulate_into`` is None or the output of an earlier call with the same pixels and
    slices.  If it is given, the function adds this batch to it in place and returns it.
    """
    if triton is None:
        raise RuntimeError('the Triton multiaxis back body was called without '
                           'triton installed; a caller must keep the torch '
                           'body where no triton is available.')
    # The kernel supports only coeff_power 1 and 2.
    if coeff_power not in (1, 2):
        block = _multiaxis_back_view_batch(
            sino_batch, pixel_indices, view_params_batch, num_rows_r,
            num_channels, num_recon_rows, num_recon_cols, num_slices,
            delta_voxel, delta_voxel_row, delta_voxel_slice, delta_det_channel,
            delta_det_row, det_channel_offset, det_row_offset,
            recon_slice_offset, det_channel_radius, det_row_radius,
            vox_slice_radius, coeff_power=coeff_power,
            slice_start=slice_start, band_slices=band_slices, plan=plan)
        return block if accumulate_into is None else accumulate_into.add_(block)

    azimuth = view_params_batch[:, 0]
    elevation = view_params_batch[:, 1]
    # These are the functions that the torch version uses, so the kernel uses the
    # same tap centers and weights.
    n_p, centers, w_p_c, weight_scale, y = _multiaxis_horizontal_data(
        pixel_indices, azimuth, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset)
    m0, slope, w_p_r, l_max, scaling = _multiaxis_vertical_terms(
        y, azimuth, elevation, num_slices, delta_voxel, delta_voxel_row,
        delta_voxel_slice, delta_det_row, det_row_offset, recon_slice_offset,
        num_rows_r)
    scale_pow = scaling ** coeff_power

    num_views, num_pixels = n_p.shape
    band_len = int(num_slices if band_slices is None else band_slices)
    # The kernel runs on a number of slices rounded up to a multiple of 16.
    # The kernel receives this padded value as its band_len argument.
    launch_band = padded_kernel_width(band_len)
    # The views are copied into channel-major order.  The copy is not padded, because
    # the kernel clamps every sinogram index.
    sino_t = sino_batch.permute(0, 2, 1).contiguous()
    # These three arrays have one entry for each (view, pixel), so they use 12 bytes per pixel per view.
    contract = [t.contiguous() for t in (n_p, centers, m0)]
    # The reshape converts these arrays to one scalar per view.  It fails if any
    # array has more than one value per view.
    contract += [t.reshape(num_views).contiguous()
                 for t in (w_p_c, weight_scale, slope, w_p_r, l_max,
                           scale_pow)]
    target = (None if accumulate_into is None else _padded_accumulator(
        accumulate_into, num_pixels, band_len, launch_band, sino_batch.device))
    out = target if target is not None else torch.empty(
        (num_pixels, launch_band), dtype=_F32, device=sino_batch.device)

    block_p = _tile_size(MULTIAXIS_BACK_BLOCK_P, num_pixels,
                         MULTIAXIS_BACK_MIN_TILE)
    block_l = _tile_size(MULTIAXIS_BACK_BLOCK_L, launch_band,
                         MULTIAXIS_BACK_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-launch_band // block_l))
    launch_key = ('maback', sino_batch.device.index, int(det_channel_radius),
                  int(det_row_radius), int(coeff_power), target is not None,
                  block_p, block_l,
                  int(num_views), int(num_pixels), int(num_channels),
                  int(num_rows_r), launch_band, int(slice_start),
                  int(num_slices))
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    compile_lock = compile_serialized() if first_launch else contextlib.nullcontext()
    # The kernel runs on the device of the input tensors.
    with torch.cuda.device(sino_batch.device), compile_lock:
        _multiaxis_back_kernel[grid](
            *contract, sino_t, out,
            int(num_views), int(num_pixels), int(num_channels),
            int(num_rows_r), launch_band, int(slice_start), int(num_slices),
            int(num_channels) * int(num_rows_r), float(num_rows_r),
            DET_CHANNEL_RADIUS=int(det_channel_radius),
            DET_ROW_RADIUS=int(det_row_radius), COEFF_POWER=int(coeff_power),
            ACCUMULATE=target is not None,
            BLOCK_P=block_p, BLOCK_L=block_l,
            num_warps=MULTIAXIS_BACK_NUM_WARPS,
            num_stages=MULTIAXIS_BACK_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    if target is not None:
        return accumulate_into
    # This slice removes the padded columns.  The result is a strided view, and each
    # of its rows is contiguous.
    block = out if launch_band == band_len else out[:, :band_len]
    # An accumulate_into without the layout of a kernel output receives the new result here.
    return block if accumulate_into is None else accumulate_into.add_(block)


# maybe_compile reads this attribute and does not compile the function.
_multiaxis_back_view_batch_triton._mbirtorch_no_compile = True
# The projection loop reads this attribute and passes its running sum as accumulate_into.
_multiaxis_back_view_batch_triton._adds_into_output = True


def _multiaxis_back_view_batch_cost(num_pixels, num_band_rows, args):
    """This function returns the memory in bytes that each view uses in one batch of the back
    projection kernel.  It also returns the default number of views in a batch.

    The caller gets this function from the ``_view_batch_cost`` attribute of
    :func:`_multiaxis_back_view_batch_triton`.  Each view uses 12 bytes per pixel and also one
    channel-major copy of its sinogram view.
    """
    plane_bytes = 4 * int(args['num_channels']) * int(num_band_rows)
    return 12 * int(num_pixels) + plane_bytes, MULTIAXIS_BACK_VIEW_CHUNK


_multiaxis_back_view_batch_triton._view_batch_cost = \
    _multiaxis_back_view_batch_cost


def _multiaxis_back_view_peak_bytes(num_pixels, num_band_rows, args):
    """This function returns the peak memory in bytes that one view of a batch uses in the
    back projection kernel.  The memory estimate uses this value.  The view batch size uses
    :func:`_multiaxis_back_view_batch_cost` instead.

    At its peak, the geometry code holds six arrays with one 4 byte entry for each
    view and pixel.  Each view therefore uses 24 bytes per pixel.  Each view also uses
    its copy of the sinogram view.
    """
    # With zero pixels, the cost function returns the bytes of the sinogram view alone.
    plane_bytes = _multiaxis_back_view_batch_cost(0, num_band_rows, args)[0]
    return 24 * int(num_pixels) + plane_bytes


_multiaxis_back_view_batch_triton._view_peak_bytes = _multiaxis_back_view_peak_bytes


@_jit
def _multiaxis_forward_kernel(n_p_ptr, centers_ptr, m0_ptr,
                              w_p_c_ptr, weight_scale_ptr,
                              slope_ptr, w_p_r_ptr, l_max_ptr, scaling_ptr,
                              values_ptr, out_ptr,
                              num_pixels, num_channels, num_rows,
                              out_row_stride, out_view_stride,
                              band_len, slice_start, num_slices,
                              slope_floor, k_center_lo, k_center_hi,
                              DET_CHANNEL_RADIUS: tl.constexpr,
                              SLICE_RADIUS: tl.constexpr,
                              BLOCK_P: tl.constexpr, BLOCK_R: tl.constexpr):
    """This function is the Triton forward projection kernel.  Each program computes one block of
    pixels, one block of detector rows, and one view of the sinogram.

        out[v, c, m] += sum over pixels p, channel taps tc, slices k of
                        Wchan[v, p, tc] * Wrow[v, p, k, m] * scaling[v]
                        * values[p, k - slice_start]

    ``SLICE_RADIUS`` limits the number of slices that contribute to each row.  See
    :func:`_multiaxis_slice_tap_radius`.  The output has channel-major shape (Vb, C, R), and it
    must be zero on entry.  The wrapper transposes the output.  ``num_rows`` and
    ``out_row_stride`` both equal the padded row count.  The kernel masks pixels with index
    ``num_pixels`` or greater.  The wrapper removes the padded rows.
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
    m0 = tl.load(m0_ptr + pix_base, mask=p_mask, other=0.0)
    # These are scalars for each view.  The horizontal footprint depends only on
    # the azimuth.  The vertical terms depend on the azimuth and elevation.
    w_p_c = tl.load(w_p_c_ptr + v)
    weight_scale = tl.load(weight_scale_ptr + v)
    clip_c = tl.minimum(w_p_c, 1.0)
    slope = tl.load(slope_ptr + v)
    w_p_r = tl.load(w_p_r_ptr + v)
    l_max_r = tl.load(l_max_ptr + v)
    scaling = tl.load(scaling_ptr + v)

    # This converts each detector row to a fractional slice index.  The lower bound
    # slope_floor on the slope prevents a NaN.
    m_f = r_offs.to(tl.float32)                                     # (BLOCK_R,)
    k_m = (m_f[None, :] - m0[:, None]) / tl.maximum(slope, slope_floor)
    k_center = _tl_floor(k_m + 0.5)
    # The center is clamped before the conversions to int, because float to int conversion
    # is undefined outside the int32 range.  At the lower bound, the highest tap is
    # slice_start - 1.  At the upper bound, the lowest tap is slice_start + band_len.
    k_center = tl.minimum(tl.maximum(k_center, k_center_lo), k_center_hi)

    det_col = tl.zeros((BLOCK_P, BLOCK_R), dtype=tl.float32)
    for tk in _tap_range(0, 2 * SLICE_RADIUS + 1):
        k_ind = k_center + (tk - SLICE_RADIUS)
        k_ind_i = k_ind.to(tl.int32)
        # This is the same vertical weight as in the torch version.  The
        # footprint sets the width, and l_max_r caps it.
        m_p = m0[:, None] + slope * k_ind
        w_row = tl.maximum((w_p_r + 1.0) / 2.0 - _tl_abs(m_p - m_f[None, :]),
                           0.0)
        w_row = tl.minimum(w_row, l_max_r)
        # A tap outside the slices in ``values`` gets weight zero, and its index is then
        # clamped.  The same test removes padded slices, with index num_slices or greater.
        live = ((k_ind_i >= slice_start) & (k_ind_i < slice_start + band_len)
                & (k_ind_i < num_slices))
        w_row = tl.where(live, w_row, 0.0)
        # The clamp keeps the address inside the array.  The load mask includes ``live``,
        # so the kernel does not read taps with zero weight.
        k_local = tl.minimum(tl.maximum(k_ind_i - slice_start, 0), band_len - 1)
        vals = tl.load(values_ptr + p_offs.to(tl.int64)[:, None] * band_len
                       + k_local, mask=tile_mask & live, other=0.0)
        # scaling multiplies the value, not the weight, as in the torch version.  This
        # order must match the torch version, so that back projection stays the adjoint.
        det_col = det_col + w_row * (vals * scaling)

    # Each channel tap adds its contribution to the output with one atomic add.
    out_view_ptr = out_ptr + v.to(tl.int64) * out_view_stride
    for tc in _tap_range(0, 2 * DET_CHANNEL_RADIUS + 1):
        # This applies the trapezoid rule in the channel direction.  The atomic add
        # mask excludes taps outside the detector.
        n_tap = centers + (tc - DET_CHANNEL_RADIUS)
        w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                            - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
        w_chan = tl.minimum(w_chan, clip_c) * weight_scale
        n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
        out_ptrs = (out_view_ptr + n_chan.to(tl.int64)[:, None] * out_row_stride
                    + r_offs[None, :])
        tl.atomic_add(out_ptrs, w_chan[:, None] * det_col,
                      mask=tile_mask & ((n_tap >= 0)
                                        & (n_tap < num_channels))[:, None])


@torch.compiler.disable
def _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                         view_params_batch, num_rows_r,
                                         num_channels, num_recon_rows,
                                         num_recon_cols, num_slices,
                                         delta_voxel, delta_voxel_row,
                                         delta_voxel_slice, delta_det_channel,
                                         delta_det_row, det_channel_offset,
                                         det_row_offset, recon_slice_offset,
                                         det_channel_radius, det_row_radius,
                                         vox_slice_radius, slice_start=0, plan=None):
    """This function forward projects one batch of views with the Triton kernel.

    The function replaces :func:`mbirtorch.multiaxis_parallel._multiaxis_forward_view_batch`,
    and it has the same signature.  It returns an array of shape (Vb, R, C).  ``values`` has
    shape (P, L), and its L columns are the slices to project.  For this reason, there is no
    ``band_slices`` argument.  If L is less than num_slices, the columns are the L slices that
    start at ``slice_start``.  The sum of the outputs over a set of slice ranges that covers all
    slices equals the full projection.  If the slice tap radius is larger than
    ``MULTIAXIS_FWD_MAX_SLICE_RADIUS``, the function calls the torch version instead.  The
    function computes its own slice tap radius, and it ignores ``det_row_radius``,
    ``vox_slice_radius``, and ``plan``.
    """
    if triton is None:
        raise RuntimeError('the Triton multiaxis forward body was called '
                           'without triton installed; a caller must keep the '
                           'torch body where no triton is available.')
    azimuth = view_params_batch[:, 0]
    elevation = view_params_batch[:, 1]
    # These are the functions that the torch version uses, so the kernel uses the
    # same tap centers and weights.
    n_p, centers, w_p_c, weight_scale, y = _multiaxis_horizontal_data(
        pixel_indices, azimuth, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset)
    m0, slope, w_p_r, l_max, scaling = _multiaxis_vertical_terms(
        y, azimuth, elevation, num_slices, delta_voxel, delta_voxel_row,
        delta_voxel_slice, delta_det_row, det_row_offset, recon_slice_offset,
        num_rows_r)

    # A large slice tap radius means that the slope is near zero.  The kernel
    # cannot handle this case, so the torch version runs instead.
    slice_radius = _multiaxis_slice_tap_radius(w_p_r, slope)
    if slice_radius > MULTIAXIS_FWD_MAX_SLICE_RADIUS:
        return _multiaxis_forward_view_batch(
            values, pixel_indices, view_params_batch, num_rows_r, num_channels,
            num_recon_rows, num_recon_cols, num_slices, delta_voxel,
            delta_voxel_row, delta_voxel_slice, delta_det_channel,
            delta_det_row, det_channel_offset, det_row_offset,
            recon_slice_offset, det_channel_radius, det_row_radius, vox_slice_radius,
            slice_start=slice_start, plan=plan)

    num_views, num_pixels = n_p.shape
    band_len = int(values.shape[1])
    # The output and the kernel use a detector row count rounded up to a multiple of 16.
    # The geometry functions above use the unpadded row count.
    launch_rows = padded_kernel_width(int(num_rows_r))
    values = values.contiguous()
    # These three arrays have one entry for each (view, pixel), so they use 12 bytes per pixel per view.
    contract = [t.contiguous() for t in (n_p, centers, m0)]
    # The reshape converts these arrays to one scalar per view.  It fails if any
    # array has more than one value per view.
    contract += [t.reshape(num_views).contiguous()
                 for t in (w_p_c, weight_scale, slope, w_p_r, l_max, scaling)]
    # The output is channel major.  It starts at zero, because the kernel adds into
    # it with atomic adds.  The return statement transposes each view.
    out = torch.zeros((num_views, num_channels, launch_rows), dtype=_F32,
                      device=values.device)

    block_p = _tile_size(MULTIAXIS_FWD_BLOCK_P, num_pixels,
                         MULTIAXIS_FWD_MIN_TILE)
    block_r = _tile_size(MULTIAXIS_FWD_BLOCK_R, launch_rows,
                         MULTIAXIS_FWD_MIN_TILE)
    # The grid covers the padded rows, because the kernel computes those rows too.
    grid = (-(-num_pixels // block_p), -(-launch_rows // block_r), num_views)
    # The kernel clamps the slice center to these bounds.
    k_center_lo = float(int(slice_start) - slice_radius - 1)
    k_center_hi = float(int(slice_start) + band_len + slice_radius)
    launch_key = ('mafwd', values.device.index, int(det_channel_radius), slice_radius,
                  block_p, block_r, int(num_views), int(num_pixels),
                  int(num_channels), int(num_rows_r), launch_rows, band_len,
                  int(slice_start), int(num_slices))
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    compile_lock = compile_serialized() if first_launch else contextlib.nullcontext()
    # The kernel runs on the device of the input tensors.
    with torch.cuda.device(values.device), compile_lock:
        _multiaxis_forward_kernel[grid](
            *contract, values, out,
            int(num_pixels), int(num_channels), launch_rows,
            launch_rows, int(num_channels) * launch_rows,
            band_len, int(slice_start), int(num_slices),
            _SLOPE_FLOOR, k_center_lo, k_center_hi,
            DET_CHANNEL_RADIUS=int(det_channel_radius), SLICE_RADIUS=slice_radius,
            BLOCK_P=block_p, BLOCK_R=block_r,
            num_warps=MULTIAXIS_FWD_NUM_WARPS,
            num_stages=MULTIAXIS_FWD_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    if launch_rows == int(num_rows_r):
        return out.permute(0, 2, 1)
    # This slice removes the padded detector rows before the transpose.
    return out[:, :, :int(num_rows_r)].permute(0, 2, 1)


# maybe_compile reads this attribute and does not compile the function.
_multiaxis_forward_view_batch_triton._mbirtorch_no_compile = True


def _multiaxis_forward_view_batch_cost(num_pixels, band_len, args):
    """This function returns the memory in bytes that each view uses in one batch of the forward
    projection kernel.  It also returns the default number of views in a batch.

    Each view uses 12 bytes per pixel and also its output view over the padded detector rows.
    """
    plane_rows = padded_kernel_width(int(args['num_rows_r']))
    plane_bytes = 4 * int(args['num_channels']) * plane_rows
    return 12 * int(num_pixels) + plane_bytes, MULTIAXIS_FWD_VIEW_CHUNK


_multiaxis_forward_view_batch_triton._view_batch_cost = \
    _multiaxis_forward_view_batch_cost


def _multiaxis_forward_view_peak_bytes(num_pixels, band_len, args):
    """This function returns the peak memory in bytes that one view of a batch uses in the
    forward projection kernel.  The memory estimate uses this value.  The view batch size uses
    :func:`_multiaxis_forward_view_batch_cost` instead.

    At its peak, the geometry code holds six arrays with one 4 byte entry for each
    view and pixel.  Each view therefore uses 24 bytes per pixel.  Each view also uses
    its output view.
    """
    # With zero pixels, the cost function returns the bytes of the sinogram view alone.
    plane_bytes = _multiaxis_forward_view_batch_cost(0, band_len, args)[0]
    return 24 * int(num_pixels) + plane_bytes


_multiaxis_forward_view_batch_triton._view_peak_bytes = _multiaxis_forward_view_peak_bytes

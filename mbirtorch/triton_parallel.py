"""This module holds hand-written Triton kernels for the parallel-beam back projection and
forward projection.

Each kernel projects one batch of views.  Each wrapper function here has the same signature as
the torch function it replaces.  The torch functions are
:func:`mbirtorch.parallel_beam._parallel_back_view_batch` and
:func:`mbirtorch.parallel_beam._parallel_forward_view_batch`.  In parallel beam, detector row r
is recon slice r.  The forward kernel adds its results with atomic operations.  For this reason,
its results agree with the torch results to float rounding, but not bit for bit.  The module
can be imported when triton is not installed.
"""

import contextlib
import os

import torch

from ._utils import padded_kernel_width
from .parallel_beam import _parallel_back_view_batch, _parallel_hfan_math
from .projectors import compile_serialized
# The Triton helpers come from the cone-beam module.  All four kernels share
# _COMPILED_LAUNCH_KEYS, and every key starts with the name of its kernel.
from .triton_cone import (_COMPILED_LAUNCH_KEYS, _jit, _padded_accumulator,
                          _tap_range, _tile_size, _tl_abs, tl, triton)

_F32 = torch.float32

PARALLEL_BACK_BLOCK_P = 8
PARALLEL_BACK_BLOCK_R = 256
PARALLEL_BACK_NUM_WARPS = 4
PARALLEL_BACK_NUM_STAGES = 1
# This is the smallest tile size.  An axis shorter than this still uses a tile of this size.
PARALLEL_BACK_MIN_TILE = 8
# This is the view batch size that this kernel requests when the model's view_batch_size is
# None.  The projector may use a smaller batch to fit in memory.
PARALLEL_BACK_VIEW_CHUNK = 128

PARALLEL_FWD_BLOCK_P = 8
PARALLEL_FWD_BLOCK_R = 128
PARALLEL_FWD_NUM_WARPS = 8
PARALLEL_FWD_NUM_STAGES = 1
PARALLEL_FWD_MIN_TILE = 8
PARALLEL_FWD_VIEW_CHUNK = 128

# These constants are for the forward kernel that sorts the pixels by channel.
PARALLEL_SORTED_BLOCK_P = 32
PARALLEL_SORTED_WINDOW = 16
PARALLEL_SORTED_BLOCK_R = 128
PARALLEL_SORTED_NUM_WARPS = 8
PARALLEL_SORTED_NUM_STAGES = 1
# tl.dot needs every dimension to be at least 16.
PARALLEL_SORTED_MIN_R = 16
PARALLEL_SORTED_VIEW_CHUNK = 16


def sorted_forward_enabled():
    """This function returns True when the parallel forward projection uses the
    kernel that sorts the pixels by channel.  This kernel is the default.  The function
    returns False when the projection uses the kernel that loops over the
    channel taps.

    Setting ``MBIRTORCH_SORTED_FORWARD=0`` selects the kernel that loops over
    the taps.  The function reads the variable on each call.  Both kernels
    compute the same sums in a different order.
    """
    return os.environ.get('MBIRTORCH_SORTED_FORWARD', '1').strip().lower() \
        not in ('0', 'false', 'no', 'off')


@_jit
def _parallel_back_kernel(n_p_ptr, centers_ptr, w_p_c_ptr, weight_scale_ptr,
                          sino_ptr, out_ptr,
                          num_views, num_pixels, num_channels, num_band_rows,
                          sino_view_stride,
                          DET_CHANNEL_RADIUS: tl.constexpr, COEFF_POWER: tl.constexpr,
                          ACCUMULATE: tl.constexpr,
                          BLOCK_P: tl.constexpr, BLOCK_R: tl.constexpr):
    """This function is the Triton back projection kernel.  It runs one program
    for each pair (pixel block, row block) of the output.  It computes

        out[p, r] = sum over views v, channel taps tc of
                    Wchan[v, p, tc] ** coeff_power * sino[v, c(v, p) + tc, r]

    The argument ``num_band_rows`` is the padded row count.  The kernel masks
    pixel indices beyond ``num_pixels``.  The wrapper removes the padded
    columns.  When ``ACCUMULATE`` is True, the kernel adds its result to the
    values already in ``out``.
    """
    p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)     # (BLOCK_P,)
    r_offs = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)     # (BLOCK_R,)
    p_mask = p_offs < num_pixels
    r_mask = r_offs < num_band_rows
    tile_mask = p_mask[:, None] & r_mask[None, :]

    acc = tl.zeros((BLOCK_P, BLOCK_R), dtype=tl.float32)
    for v in range(num_views):
        pix_base = v.to(tl.int64) * num_pixels + p_offs
        n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
        centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
        # These are scalars for each view.  In parallel beam, the projected
        # width of a voxel depends only on the view angle.
        w_p_c = tl.load(w_p_c_ptr + v)
        weight_scale = tl.load(weight_scale_ptr + v)
        clip = tl.minimum(w_p_c, 1.0)
        sino_view_ptr = sino_ptr + v.to(tl.int64) * sino_view_stride

        for tc in _tap_range(0, 2 * DET_CHANNEL_RADIUS + 1):
            # The weight is zero where the tap falls off the detector.  The
            # index is then clamped to the detector.
            n_tap = centers + (tc - DET_CHANNEL_RADIUS)
            w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                                - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
            w_chan = tl.minimum(w_chan, clip) * weight_scale
            w_chan = tl.where((n_tap >= 0) & (n_tap < num_channels),
                              w_chan, 0.0)
            if COEFF_POWER == 2:
                w_chan = w_chan * w_chan
            n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
            # The views are channel major.  As a result, the row axis is
            # contiguous, and each tile reads memory with unit stride.
            row_ptr = sino_view_ptr + n_chan.to(tl.int64) * num_band_rows
            vals = tl.load(row_ptr[:, None] + r_offs[None, :], mask=tile_mask,
                           other=0.0)
            acc = acc + w_chan[:, None] * vals

    out_ptrs = (out_ptr + p_offs.to(tl.int64)[:, None] * num_band_rows
                + r_offs[None, :])
    if ACCUMULATE:
        # This float32 sum equals torch's add_ of the stored array and this result.
        acc = tl.load(out_ptrs, mask=tile_mask, other=0.0) + acc
    tl.store(out_ptrs, acc, mask=tile_mask)


@torch.compiler.disable
def _parallel_back_view_batch_triton(sino_batch, pixel_indices,
                                     view_params_batch, num_rows, num_cols,
                                     num_channels, delta_det_channel,
                                     det_channel_offset, delta_voxel,
                                     delta_voxel_row, det_channel_radius,
                                     det_row_radius, vox_slice_radius,
                                     coeff_power=1, slice_start=0,
                                     band_slices=None, plan=None,
                                     accumulate_into=None):
    """This function computes the parallel-beam back projection of one batch of
    views with a Triton kernel.  It replaces
    :func:`mbirtorch.parallel_beam._parallel_back_view_batch`.

    The function has the same signature and returns the same (P, rows) array.
    Each call allocates a new output array, so the caller may add to it in
    place.  The function is never compiled by torch.compile.  The arguments
    ``slice_start`` and ``band_slices`` must keep their defaults, because the
    sinogram rows already select the slices.  The function ignores
    ``det_row_radius``, ``vox_slice_radius``, and ``plan``.

    ``accumulate_into`` is None or the output of an earlier call with the same
    pixels and rows.  If it is given, the function adds this batch to it in
    place and returns it.
    """
    if triton is None:
        raise RuntimeError('the Triton parallel back body was called without '
                           'triton installed; the selection in '
                           'ParallelBeamModel._view_batch_bodies should have '
                           'kept the torch body (see kernel_availability).')
    assert slice_start == 0 and band_slices is None
    # The kernel supports only the powers 1 and 2.
    if coeff_power not in (1, 2):
        block = _parallel_back_view_batch(
            sino_batch, pixel_indices, view_params_batch, num_rows, num_cols,
            num_channels, delta_det_channel, det_channel_offset, delta_voxel,
            delta_voxel_row, det_channel_radius, det_row_radius,
            vox_slice_radius, coeff_power=coeff_power,
            slice_start=slice_start, band_slices=band_slices, plan=plan)
        return block if accumulate_into is None else accumulate_into.add_(block)

    n_p, centers, w_p_c, weight_scale = _parallel_hfan_math(
        pixel_indices, view_params_batch, num_rows, num_cols, num_channels,
        delta_det_channel, det_channel_offset, delta_voxel, delta_voxel_row)

    num_views, num_pixels = n_p.shape
    num_band_rows = int(sino_batch.shape[1])
    # The kernel is launched with the row count rounded up to a multiple of 16.
    # Every use of the row count below uses this padded value.
    launch_rows = padded_kernel_width(num_band_rows)
    # The views are made channel major, so each tile reads contiguous memory.
    if launch_rows == num_band_rows:
        sino_t = sino_batch.permute(0, 2, 1).contiguous()
    else:
        # A padded row would read past the last real row.  For this reason, the
        # copy has the padded row count, with zeros in the extra rows.
        sino_t = torch.empty(
            (int(sino_batch.shape[0]), int(sino_batch.shape[2]), launch_rows),
            dtype=sino_batch.dtype, device=sino_batch.device)
        sino_t[:, :, :num_band_rows] = sino_batch.permute(0, 2, 1)
        sino_t[:, :, num_band_rows:] = 0.0
    contract = [t.contiguous() for t in (n_p, centers)]
    # The reshape converts these arrays to one scalar per view.  It fails if
    # they do not have one value per view.
    contract += [t.reshape(num_views).contiguous()
                 for t in (w_p_c, weight_scale)]
    target = (None if accumulate_into is None else _padded_accumulator(
        accumulate_into, num_pixels, num_band_rows, launch_rows, sino_batch.device))
    out = target if target is not None else torch.empty(
        (num_pixels, launch_rows), dtype=_F32, device=sino_batch.device)

    block_p = _tile_size(PARALLEL_BACK_BLOCK_P, num_pixels,
                         PARALLEL_BACK_MIN_TILE)
    block_r = _tile_size(PARALLEL_BACK_BLOCK_R, launch_rows,
                         PARALLEL_BACK_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-launch_rows // block_r))
    launch_key = ('pback', sino_batch.device.index, int(det_channel_radius),
                  int(coeff_power), target is not None, block_p, block_r,
                  int(num_views), int(num_pixels), int(num_channels),
                  launch_rows)
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    compile_lock = compile_serialized() if first_launch else contextlib.nullcontext()
    # The kernel runs on the device of the tensors.  The device index is part
    # of the launch key.
    with torch.cuda.device(sino_batch.device), compile_lock:
        _parallel_back_kernel[grid](
            *contract, sino_t, out,
            int(num_views), int(num_pixels), int(num_channels), launch_rows,
            int(num_channels) * launch_rows,
            DET_CHANNEL_RADIUS=int(det_channel_radius), COEFF_POWER=int(coeff_power),
            ACCUMULATE=target is not None,
            BLOCK_P=block_p, BLOCK_R=block_r,
            num_warps=PARALLEL_BACK_NUM_WARPS,
            num_stages=PARALLEL_BACK_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    if target is not None:
        return accumulate_into
    # The padded columns are removed.  The result is a strided view, and each
    # of its rows is contiguous.
    block = out if launch_rows == num_band_rows else out[:, :num_band_rows]
    # An accumulate_into without the layout of a kernel output receives the new result here.
    return block if accumulate_into is None else accumulate_into.add_(block)


# The function maybe_compile reads this attribute and does not compile this function.
_parallel_back_view_batch_triton._mbirtorch_no_compile = True
# The projection loop reads this attribute and passes its running sum as accumulate_into.
_parallel_back_view_batch_triton._adds_into_output = True


def _parallel_back_view_batch_cost(num_pixels, band_rows, args):
    """This function returns two values for the back projection kernel.  The
    first value is the memory in bytes that one view of a batch uses.  The
    second value is the view batch size that this kernel requests.

    The projector reads this function through the ``_view_batch_cost``
    attribute.  Each view uses 16 bytes per pixel for the arrays from
    ``_parallel_hfan_math``.  Each view also uses a channel major copy of its
    sinogram view with the padded row count.
    """
    plane_bytes = (4 * int(args['num_channels'])
                   * padded_kernel_width(band_rows))
    return 16 * int(num_pixels) + plane_bytes, PARALLEL_BACK_VIEW_CHUNK


_parallel_back_view_batch_triton._view_batch_cost = \
    _parallel_back_view_batch_cost


def _parallel_back_view_peak_bytes(num_pixels, band_rows, args):
    """This function returns the peak memory in bytes that one view of a batch uses in the
    back projection kernel.  The memory estimate uses this value.  The view batch size uses
    :func:`_parallel_back_view_batch_cost` instead.

    At its peak, ``_parallel_hfan_math`` holds five arrays with one 4 byte entry for
    each view and pixel.  Each view therefore uses 20 bytes per pixel.  Each view also uses
    its copy of the sinogram view.
    """
    # With zero pixels, the cost function returns the bytes of the sinogram view alone.
    plane_bytes = _parallel_back_view_batch_cost(0, band_rows, args)[0]
    return 20 * int(num_pixels) + plane_bytes


_parallel_back_view_batch_triton._view_peak_bytes = _parallel_back_view_peak_bytes


@_jit
def _parallel_forward_kernel(n_p_ptr, centers_ptr, w_p_c_ptr,
                             weight_scale_ptr, values_ptr, out_ptr,
                             num_pixels, num_channels, num_cols,
                             out_view_stride,
                             DET_CHANNEL_RADIUS: tl.constexpr, BLOCK_P: tl.constexpr,
                             BLOCK_R: tl.constexpr):
    """This function is the Triton forward projection kernel that loops over
    the channel taps.  It runs one program for each triple (pixel block,
    column block, view) of the sinogram.  It computes

        out[v, c, r] += sum over pixels p, channel taps tc of
                        Wchan[v, p, tc] * values[p, r]

    The output is channel major, with shape (Vb, C, R).  The wrapper
    transposes the output before it returns it.  The argument ``num_cols`` is
    the padded column count.  The kernel masks pixel indices beyond
    ``num_pixels``.  The wrapper removes the padded columns.
    """
    p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)     # (BLOCK_P,)
    r_offs = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)     # (BLOCK_R,)
    v = tl.program_id(2)
    p_mask = p_offs < num_pixels
    r_mask = r_offs < num_cols
    tile_mask = p_mask[:, None] & r_mask[None, :]

    pix_base = v.to(tl.int64) * num_pixels + p_offs
    n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
    centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
    # These are scalars for each view.
    w_p_c = tl.load(w_p_c_ptr + v)
    weight_scale = tl.load(weight_scale_ptr + v)
    clip = tl.minimum(w_p_c, 1.0)

    # The kernel reads the voxel values once and keeps them in registers for all taps.
    vals = tl.load(values_ptr + p_offs.to(tl.int64)[:, None] * num_cols
                   + r_offs[None, :], mask=tile_mask, other=0.0)

    out_view_ptr = out_ptr + v.to(tl.int64) * out_view_stride
    for tc in _tap_range(0, 2 * DET_CHANNEL_RADIUS + 1):
        # The weights use the same trapezoid formula as the back projection.  The
        # mask excludes taps outside the detector, so no atomic add occurs for them.
        n_tap = centers + (tc - DET_CHANNEL_RADIUS)
        w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                            - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
        w_chan = tl.minimum(w_chan, clip) * weight_scale
        n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
        out_ptrs = (out_view_ptr + n_chan.to(tl.int64)[:, None] * num_cols
                    + r_offs[None, :])
        tl.atomic_add(out_ptrs, w_chan[:, None] * vals,
                      mask=tile_mask & ((n_tap >= 0)
                                        & (n_tap < num_channels))[:, None])


@_jit
def _parallel_forward_sorted_kernel(n_p_ptr, centers_ptr, w_p_c_ptr,
                                    weight_scale_ptr, values_ptr, perm_ptr,
                                    out_ptr, num_views, num_pixels,
                                    num_channels, num_cols, out_view_stride,
                                    VIEW_CHUNK: tl.constexpr,
                                    WINDOW: tl.constexpr,
                                    DET_CHANNEL_RADIUS: tl.constexpr,
                                    BLOCK_P: tl.constexpr,
                                    BLOCK_R: tl.constexpr):
    """This function is the Triton forward projection kernel that sorts the
    pixels by channel.  It runs one program for each triple (pixel block,
    column block, view block).  In each view, the pixels are sorted by the
    channel of their center.

    The kernel computes each tile as a small dense matrix product,
    transpose(W) @ values.  Here W holds the (BLOCK_P, WINDOW) trapezoid
    weights of the tile.  The product uses full float32 precision.  If the
    channels of a tile span more than WINDOW channels, the kernel loops over
    the taps for that tile instead.
    """
    p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    r_offs = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    v0 = tl.program_id(2) * VIEW_CHUNK
    p_mask = p_offs < num_pixels
    r_mask = r_offs < num_cols
    tile_mask = p_mask[:, None] & r_mask[None, :]

    for dv in range(VIEW_CHUNK):
        v = v0 + dv
        v_ok = v < num_views
        # The clamp keeps every address in bounds in the last view block.  The
        # store masks include v_ok, so an iteration past the last view writes nothing.
        v_safe = tl.minimum(v, num_views - 1)
        pix_base = v_safe.to(tl.int64) * num_pixels + p_offs
        n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
        centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
        w_p_c = tl.load(w_p_c_ptr + v_safe)
        weight_scale = tl.load(weight_scale_ptr + v_safe)
        clip = tl.minimum(w_p_c, 1.0)
        out_view_ptr = out_ptr + v_safe.to(tl.int64) * out_view_stride
        row_idx = tl.load(perm_ptr + pix_base, mask=p_mask, other=0)
        vals = tl.load(values_ptr + row_idx.to(tl.int64)[:, None] * num_cols
                       + r_offs[None, :], mask=tile_mask, other=0.0)

        big = 2147483647
        c_lo = tl.min(tl.where(p_mask, centers, big)) - DET_CHANNEL_RADIUS
        c_hi = tl.max(tl.where(p_mask, centers, -big)) + DET_CHANNEL_RADIUS
        span = c_hi - c_lo + 1
        if span <= WINDOW:
            # The window weights use the same trapezoid formula at every channel of
            # the window.  Channels outside the taps get weights of exactly zero.
            j = tl.arange(0, WINDOW)
            c = c_lo + j
            w = tl.maximum(
                (w_p_c + 1.0) / 2.0
                - _tl_abs(n_p[:, None] - c.to(tl.float32)[None, :]), 0.0)
            w = tl.minimum(w, clip) * weight_scale
            w = tl.where(((c >= 0) & (c < num_channels))[None, :], w, 0.0)
            w = tl.where(p_mask[:, None], w, 0.0)
            out_window = tl.dot(tl.trans(w), vals,
                                input_precision="ieee")
            c_addr = tl.minimum(tl.maximum(c, 0), num_channels - 1)
            win_ptrs = (out_view_ptr
                        + c_addr.to(tl.int64)[:, None] * num_cols
                        + r_offs[None, :])
            # This mask stops channels beyond the span of the tile from doing
            # an atomic add.
            tl.atomic_add(win_ptrs, out_window,
                          mask=(v_ok
                                & ((j <= (c_hi - c_lo))
                                   & (c >= 0)
                                   & (c < num_channels))[:, None]
                                & r_mask[None, :]))
        else:
            for tc in _tap_range(0, 2 * DET_CHANNEL_RADIUS + 1):
                n_tap = centers + (tc - DET_CHANNEL_RADIUS)
                w_chan = tl.maximum(
                    (w_p_c + 1.0) / 2.0
                    - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
                w_chan = tl.minimum(w_chan, clip) * weight_scale
                n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
                out_ptrs = (out_view_ptr
                            + n_chan.to(tl.int64)[:, None] * num_cols
                            + r_offs[None, :])
                tl.atomic_add(out_ptrs, w_chan[:, None] * vals,
                              mask=(v_ok & tile_mask
                                    & ((n_tap >= 0)
                                       & (n_tap < num_channels))[:, None]))


@torch.compiler.disable
def _parallel_forward_view_batch_triton(values, pixel_indices,
                                        view_params_batch, num_rows, num_cols,
                                        num_channels, delta_det_channel,
                                        det_channel_offset, delta_voxel,
                                        delta_voxel_row, det_channel_radius,
                                        det_row_radius, vox_slice_radius,
                                        slice_start=0, plan=None):
    """This function computes the parallel-beam forward projection of one batch
    of views with a Triton kernel.  It replaces
    :func:`mbirtorch.parallel_beam._parallel_forward_view_batch`.

    The function has the same signature and returns the same (Vb, rows, C)
    array.  The argument ``values`` has shape (P, cols), where cols is the
    number of detector rows that this call produces.  The argument
    ``slice_start`` must be 0, because each slice maps to one detector row.
    The function ignores ``det_row_radius``, ``vox_slice_radius``, and ``plan``.
    """
    if triton is None:
        raise RuntimeError('the Triton parallel forward body was called '
                           'without triton installed; the selection in '
                           'ParallelBeamModel._view_batch_bodies should have '
                           'kept the torch body (see kernel_availability).')
    assert slice_start == 0
    n_p, centers, w_p_c, weight_scale = _parallel_hfan_math(
        pixel_indices, view_params_batch, num_rows, num_cols, num_channels,
        delta_det_channel, det_channel_offset, delta_voxel, delta_voxel_row)

    num_views, num_pixels = n_p.shape
    num_value_cols = int(values.shape[1])
    # The kernel is launched with the column count rounded up to a multiple of 16.
    # Every use of the column count below uses this padded value.
    launch_cols = padded_kernel_width(num_value_cols)
    if launch_cols == num_value_cols:
        values = values.contiguous()
    else:
        # A padded column would read past the last real column.  For this reason,
        # the copy has the padded width, with zeros in the extra columns.
        padded_values = torch.empty((int(values.shape[0]), launch_cols),
                                    dtype=values.dtype, device=values.device)
        padded_values[:, :num_value_cols] = values
        padded_values[:, num_value_cols:] = 0.0
        values = padded_values
    if sorted_forward_enabled():
        # In each view, the pixels are sorted by the channel of their center.  The
        # permutation maps each sorted position back to its row of ``values``.
        order = torch.argsort(n_p, dim=1)
        contract = [torch.gather(n_p, 1, order).contiguous(),
                    torch.gather(centers, 1, order).contiguous(),
                    w_p_c.reshape(num_views).contiguous(),
                    weight_scale.reshape(num_views).contiguous()]
        perm = order.to(torch.int32).contiguous()
        out = torch.zeros((num_views, num_channels, launch_cols), dtype=_F32,
                          device=values.device)
        block_p = PARALLEL_SORTED_BLOCK_P
        block_r = max(PARALLEL_SORTED_MIN_R,
                      _tile_size(PARALLEL_SORTED_BLOCK_R, launch_cols,
                                 PARALLEL_SORTED_MIN_R))
        grid = (-(-num_pixels // block_p), -(-launch_cols // block_r),
                -(-num_views // PARALLEL_SORTED_VIEW_CHUNK))
        launch_key = ('pfwd_sorted', values.device.index, int(det_channel_radius),
                      block_p, block_r, int(num_views),
                      int(num_pixels), int(num_channels), launch_cols)
        first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
        compile_lock = (compile_serialized() if first_launch
                        else contextlib.nullcontext())
        with torch.cuda.device(values.device), compile_lock:
            _parallel_forward_sorted_kernel[grid](
                *contract, values, perm, out,
                int(num_views), int(num_pixels), int(num_channels),
                launch_cols, int(num_channels) * launch_cols,
                VIEW_CHUNK=PARALLEL_SORTED_VIEW_CHUNK,
                WINDOW=PARALLEL_SORTED_WINDOW,
                DET_CHANNEL_RADIUS=int(det_channel_radius), BLOCK_P=block_p,
                BLOCK_R=block_r,
                num_warps=PARALLEL_SORTED_NUM_WARPS,
                num_stages=PARALLEL_SORTED_NUM_STAGES)
        _COMPILED_LAUNCH_KEYS.add(launch_key)
        if launch_cols == num_value_cols:
            return out.permute(0, 2, 1)
        return out[:, :, :num_value_cols].permute(0, 2, 1)

    contract = [t.contiguous() for t in (n_p, centers)]
    # The reshape converts these arrays to one scalar per view.  It fails if
    # they do not have one value per view.
    contract += [t.reshape(num_views).contiguous()
                 for t in (w_p_c, weight_scale)]
    # The output is channel major.  It starts at zero, because the atomic adds
    # accumulate into it.  The return statement transposes each view.
    out = torch.zeros((num_views, num_channels, launch_cols), dtype=_F32,
                      device=values.device)

    block_p = _tile_size(PARALLEL_FWD_BLOCK_P, num_pixels,
                         PARALLEL_FWD_MIN_TILE)
    block_r = _tile_size(PARALLEL_FWD_BLOCK_R, launch_cols,
                         PARALLEL_FWD_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-launch_cols // block_r),
            num_views)
    launch_key = ('pfwd', values.device.index, int(det_channel_radius), block_p,
                  block_r, int(num_views),
                  int(num_pixels), int(num_channels), launch_cols)
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    compile_lock = compile_serialized() if first_launch else contextlib.nullcontext()
    # The kernel runs on the device of the tensors.  The device index is part
    # of the launch key.
    with torch.cuda.device(values.device), compile_lock:
        _parallel_forward_kernel[grid](
            *contract, values, out,
            int(num_pixels), int(num_channels), launch_cols,
            int(num_channels) * launch_cols,
            DET_CHANNEL_RADIUS=int(det_channel_radius), BLOCK_P=block_p, BLOCK_R=block_r,
            num_warps=PARALLEL_FWD_NUM_WARPS,
            num_stages=PARALLEL_FWD_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    if launch_cols == num_value_cols:
        return out.permute(0, 2, 1)
    # The extra columns are removed before the transpose.
    return out[:, :, :num_value_cols].permute(0, 2, 1)


# The function maybe_compile reads this attribute and does not compile this function.
_parallel_forward_view_batch_triton._mbirtorch_no_compile = True


def _parallel_forward_view_batch_cost(num_pixels, num_value_cols, args):
    """This function returns two values for the forward projection kernel.  The
    first value is the memory in bytes that one view of a batch uses.  The
    second value is the view batch size that this kernel requests.

    Each view uses 16 bytes per pixel for the arrays from
    ``_parallel_hfan_math``.  Each view also uses its output sinogram view with
    the padded column count.  The kernel that sorts the pixels uses 20 more
    bytes per pixel in each view.
    """
    plane_bytes = (4 * int(args['num_channels'])
                   * padded_kernel_width(num_value_cols))
    per_view = 16 * int(num_pixels) + plane_bytes
    if sorted_forward_enabled():
        per_view += 20 * int(num_pixels)
    return per_view, PARALLEL_FWD_VIEW_CHUNK


_parallel_forward_view_batch_triton._view_batch_cost = \
    _parallel_forward_view_batch_cost


def _parallel_forward_view_peak_bytes(num_pixels, num_value_cols, args):
    """This function returns the peak memory in bytes that one view of a batch uses in the
    forward projection kernel.  The memory estimate uses this value.  The view batch size uses
    :func:`_parallel_forward_view_batch_cost` instead.

    The kernel that sorts the pixels peaks at 28 bytes per pixel in each view.  At that point
    it holds ``n_p`` and ``centers``, the sort order as int64 and int32 arrays, the two sorted
    arrays, and the output sinogram view.  The kernel that loops over the channel taps peaks at 20 bytes
    per pixel inside ``_parallel_hfan_math``.  Each view also uses its output sinogram view.
    """
    pixel_bytes = 28 if sorted_forward_enabled() else 20
    # With zero pixels, the cost function returns the bytes of the sinogram view alone.
    plane_bytes = _parallel_forward_view_batch_cost(0, num_value_cols, args)[0]
    return pixel_bytes * int(num_pixels) + plane_bytes


_parallel_forward_view_batch_triton._view_peak_bytes = \
    _parallel_forward_view_peak_bytes

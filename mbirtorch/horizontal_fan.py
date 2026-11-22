"""This module holds the horizontal fan projectors that all geometries use.

Each geometry supplies the tuple (n_p, centers, W_p_c, weight_scale) for every view and pixel.
n_p is the projected channel coordinate, and centers is n_p rounded to an int32.  W_p_c is the
projected voxel width in channels, and weight_scale is a scale factor set by the geometry.
Each detector channel n = center + offset near the projection is called a tap.  Tap n gets the
trapezoid weight

    A = weight_scale * clip((W_p_c + 1) / 2 - |n_p - n|, 0, min(1, W_p_c)).

A tap outside the detector gets weight zero.  The code also clamps the index of that tap into
range, because torch index operations fail on indices that are out of range.
"""

import torch

_F32 = torch.float32


def tap_weights(n_p, n, W_p_c, weight_scale, num_channels):
    """This function returns the trapezoid weight for tap ``n`` and the tap index clamped
    into range.  The weight is zero where the tap is outside the detector.

    Args:
        n_p: (Vb, P) tensor.  The continuous projected channel coordinate.  Vb is the number of
            views in the batch, and P is the number of pixels.
        n: (Vb, P) int64 tensor.  The integer channel of the tap, equal to center + offset.
        W_p_c: (Vb, P) tensor or a tensor that broadcasts to it.  The projected voxel width
            in channels.
        weight_scale: (Vb, P) tensor or a tensor that broadcasts to it.  The weight scale of
            the geometry, for example the in-plane voxel area divided by the footprint length.
        num_channels (int): The number of detector channels.

    Returns:
        (A, n_clamped): A is the (Vb, P) float32 weight.  A is zero where the tap is outside the
        detector.  n_clamped is the (Vb, P) int64 tap index clamped into range.
    """
    A = torch.clamp((W_p_c + 1.0) / 2.0 - (n_p - n.to(_F32)).abs(), min=0.0)
    A = torch.minimum(A, torch.clamp(W_p_c, max=1.0)) * weight_scale
    A = A * ((n >= 0) & (n < num_channels)).to(_F32)
    return A, n.clamp(0, num_channels - 1)


def fan_forward_batch(hfan_data, values, num_channels, det_channel_radius):
    """This function computes the forward horizontal fan projection for one batch of views.
    For each pixel, the function multiplies the row of the pixel by the tap weights of the
    pixel.  The function then adds these products into the detector channels.

    Args:
        hfan_data: The tuple (n_p, centers, W_p_c, weight_scale) for this batch of views.
            n_p is a (Vb, P) float32 tensor, and centers is a (Vb, P) int32 tensor.
        values: Either a (P, num_cols) tensor or a (Vb, P, num_cols) tensor.  Parallel beam
            uses one set of rows for all views.  Cone beam uses the output of the vertical fan
            projection, which has one set of rows for each view.
        num_channels (int): The number of detector channels.
        det_channel_radius (int): The number of channel taps on each side of the center.  The
            total number of taps is 2*det_channel_radius + 1.

    Returns:
        (Vb, num_channels, num_cols) tensor of partial views, with the channel axis before the
        column axis.  The caller transposes these views into the sinogram axis order.
    """
    n_p, centers, W_p_c, weight_scale = hfan_data
    centers = centers.to(torch.int64)
    vb, num_pixels = n_p.shape
    num_cols = values.shape[-1]
    dev = values.device
    # Row v*num_channels + n of the flat accumulator holds view v, channel n.
    acc = torch.zeros((vb * num_channels, num_cols), dtype=_F32, device=dev)
    row_base = torch.arange(vb, device=dev)[:, None] * num_channels
    for offset in range(-det_channel_radius, det_channel_radius + 1):
        A, n = tap_weights(n_p, centers + offset, W_p_c, weight_scale,
                           num_channels)
        idx = (row_base + n).reshape(-1)
        src = (A.unsqueeze(-1) * values).reshape(-1, num_cols)
        acc.index_add_(0, idx, src)
    return acc.view(vb, num_channels, num_cols)


def fan_back_batch(sino_batch_T, hfan_data, num_channels, det_channel_radius,
                   coeff_power=1, reduce_views=True):
    """This function computes the back projection, which is the adjoint of the horizontal fan
    projection, for one batch of views.  For each pixel, the function reads the detector
    channels near the projection of the pixel.  It then sums these channels with the tap
    weights.

    Args:
        sino_batch_T: (Vb, num_channels, num_rows) tensor of views, with the channel axis
            before the row axis.
        hfan_data: The same tuple as in :func:`fan_forward_batch`.
        num_channels (int): The number of detector channels.
        det_channel_radius (int): The number of channel taps on each side of the center.
        coeff_power (int): The power applied to each weight.  A value of 2 gives the Hessian
            diagonal.
        reduce_views (bool): If True, the function sums over the views in the batch.  If False,
            the function keeps the view axis.

    Returns:
        (P, num_rows) tensor if ``reduce_views`` is True, and (Vb, P, num_rows) tensor otherwise.
    """
    n_p, centers, W_p_c, weight_scale = hfan_data
    centers = centers.to(torch.int64)
    vb, num_pixels = n_p.shape
    dev = sino_batch_T.device
    v_idx = torch.arange(vb, device=dev)[:, None]
    num_rows = sino_batch_T.shape[2]
    if reduce_views:
        out = torch.zeros((num_pixels, num_rows), dtype=_F32, device=dev)
    else:
        out = torch.zeros((vb, num_pixels, num_rows), dtype=_F32, device=dev)
    for offset in range(-det_channel_radius, det_channel_radius + 1):
        A, n = tap_weights(n_p, centers + offset, W_p_c, weight_scale,
                           num_channels)
        if coeff_power != 1:
            A = A ** coeff_power
        gathered = sino_batch_T[v_idx, n]                    # (Vb, P, num_rows)
        if reduce_views:
            out += torch.einsum("vp,vpr->pr", A, gathered)
        else:
            out = out + A.unsqueeze(-1) * gathered
    return out

"""This module defines QGGMRFDenoiser, an image denoiser that computes the qGGMRF proximal map.

The denoiser runs the VCD reconstruction with the identity as the forward model, so it uses no
projectors.  Each iteration updates the subsets of one fixed pixel partition in order.  If the
partition is drawn with a seeded random generator, the result is deterministic.  On several
devices, the denoiser divides the image by slices.  :meth:`QGGMRFDenoiser.denoise_stack`
denoises a stack of volumes of the same shape on one device.
"""

import datetime
import itertools

import numpy as np
import torch

from . import _memory_estimate, _sharding

from . import qggmrf as _qggmrf
from . import vcd_utils
from ._memory_estimate import image_l1_norm, stack_l1_norm
from ._utils import _AUTO_REGULARIZATION_PARAM_NAMES, Param, recon_param_names
from .projectors import maybe_compile
from .tomography_model import TomographyModel

# Each denoiser takes the next number from this counter.  The number names its compiled
# functions.
_instance_counter = itertools.count(1)

_F32_EPS = float(np.finfo(np.float32).eps)
# The regularization estimate for a stack never sets sigma_x below this value.
_SIGMA_X_FLOOR = 1e-6
# The noise statistics read at most this many voxels.
_STATISTICS_POINT_BUDGET = 5_000_000

# The number of tiles per axis for the sampled statistics, and the shortest tile edge.
_TILE_GRID = 2
_MIN_TILE = 4


def _sample_tiles(num_rows, num_cols, num_leading, point_budget):
    """This function chooses tiles on the row and column axes that hold at most
    ``point_budget`` voxels in total.  The count includes all ``num_leading`` entries of a
    third axis, which is not tiled.  Each tile is contiguous, and the tiles are spread over
    both axes.

    Returns:
        (list of slice, list of slice): The row ranges and the column ranges of the tiles.
        If the whole array holds at most ``point_budget`` voxels, each list holds one slice
        that covers the whole axis.
    """
    num_rows, num_cols, num_leading = int(num_rows), int(num_cols), int(num_leading)
    total = num_leading * num_rows * num_cols
    if total <= point_budget or total <= 0:
        return [slice(0, num_rows)], [slice(0, num_cols)]
    side = max(2, int((point_budget / num_leading) ** 0.5))
    count = max(1, min(_TILE_GRID, side // _MIN_TILE))
    edge = max(2, side // count)

    def spread(extent):
        """This function returns the tiles of one axis.  The function divides the axis into
        equal parts and centers one tile in each part."""
        width = min(edge, extent)
        if count == 1 or width >= extent:
            start = (extent - width) // 2
            return [slice(start, start + width)]
        starts = sorted({min(max(0, round((index + 0.5) * extent / count) - width // 2),
                             extent - width)
                         for index in range(count)})
        return [slice(start, start + width) for start in starts]

    return spread(num_rows), spread(num_cols)


def vcd_subset_denoiser(flat_image, flat_error_image, pixel_indices,
                        fm_constant, qggmrf_params, max_alpha, positivity, image_shape,
                        pair_lists):
    """This function performs one VCD subset update for the identity forward model.  The
    function changes ``flat_image`` and ``flat_error_image`` in place.  It returns
    (flat_image, flat_error_image, l1, alpha), where ``l1`` is the L1 norm of the update.
    The step size alpha is at most ``max_alpha``.  If ``positivity`` is True, the update
    keeps every pixel non-negative.
    ``pair_lists`` holds the pair lists of the subset, as for
    :func:`qggmrf.qggmrf_exact_quadratic`.  The tests in tests/test_denoiser.py compare against
    stored values, so the order of operations must not change."""
    prior_grad, prior_hess, slice_b_tilde_2 = _qggmrf.qggmrf_terms_at_indices(
        flat_image, image_shape, pixel_indices, qggmrf_params)

    # The forward Hessian is fm_constant, because the forward model is the identity.
    cur_error_image = flat_error_image[pixel_indices]
    forward_grad = -fm_constant * cur_error_image
    forward_hess = fm_constant

    delta_recon_at_indices = -((forward_grad + prior_grad)
                               / (forward_hess + prior_hess))

    # These are delta^T grad Q(x_hat; x'=x_hat) and the exact quadratic term of the prior.  The
    # step size alpha uses both.
    prior_linear = torch.sum(prior_grad * delta_recon_at_indices)
    prior_quadratic = _qggmrf.qggmrf_exact_quadratic(
        prior_hess, slice_b_tilde_2, delta_recon_at_indices, flat_image, pixel_indices,
        image_shape, qggmrf_params, pair_lists).to(delta_recon_at_indices.dtype)

    delta_sinogram = delta_recon_at_indices
    forward_linear = fm_constant * torch.sum(cur_error_image * delta_sinogram)
    forward_quadratic = fm_constant * torch.sum(delta_sinogram * delta_sinogram)

    alpha_numerator = forward_linear - prior_linear
    alpha_denominator = forward_quadratic + prior_quadratic + _F32_EPS
    alpha = alpha_numerator / alpha_denominator
    alpha = torch.clamp(alpha, _F32_EPS, max_alpha)

    # The positivity constraint clips the update so that image + alpha * delta >= 0.
    if positivity:
        delta_recon_at_indices = torch.maximum(
            -flat_image[pixel_indices] / (alpha + _F32_EPS), delta_recon_at_indices)
        delta_sinogram = delta_recon_at_indices

    delta_recon_at_indices = alpha * delta_recon_at_indices
    flat_image.index_add_(0, pixel_indices, delta_recon_at_indices)

    cur_error_image = cur_error_image - alpha * delta_sinogram
    flat_error_image.index_copy_(0, pixel_indices, cur_error_image)
    l1_for_subset = torch.sum(torch.abs(delta_recon_at_indices))
    return flat_image, flat_error_image, l1_for_subset, alpha


def vcd_subset_denoiser_batched(flat_image, flat_error_image, pixel_indices,
                                fm_constant, qggmrf_params, max_alpha, positivity,
                                image_shape, active, pair_lists):
    """This function performs one VCD subset update for a stack of images with the identity
    forward model.  The function computes the same update as :func:`vcd_subset_denoiser`,
    with an added first axis that indexes the volumes.

    Each volume has its own step size, and the volumes do not interact.  ``active`` is a
    boolean tensor with one entry per volume.  A volume whose entry is False does not change.
    ``pair_lists`` holds the pair lists of the subset.
    The function changes ``flat_image`` and ``flat_error_image`` in place.  Both have shape
    (num_volumes, num_pixels, num_slices).

    Returns:
        (flat_image, flat_error_image, l1, alpha): ``flat_image`` and ``flat_error_image``
        are the updated tensors.  ``l1`` holds the L1 norm of the update of each volume,
        with shape (num_volumes,).  ``alpha`` holds the step size of each volume, with shape
        (num_volumes,).
    """
    prior_grad, prior_hess, slice_b_tilde_2 = _qggmrf.qggmrf_terms_batched(
        flat_image, image_shape, pixel_indices, qggmrf_params)

    # The forward Hessian is fm_constant, because the forward model is the identity.
    cur_error_image = flat_error_image[:, pixel_indices]
    forward_grad = -fm_constant * cur_error_image
    forward_hess = fm_constant

    delta_recon_at_indices = -((forward_grad + prior_grad)
                               / (forward_hess + prior_hess))

    prior_linear = torch.sum(prior_grad * delta_recon_at_indices, dim=(1, 2))
    prior_quadratic = _qggmrf.qggmrf_exact_quadratic(
        prior_hess, slice_b_tilde_2, delta_recon_at_indices, flat_image, pixel_indices,
        image_shape, qggmrf_params, pair_lists).to(delta_recon_at_indices.dtype)

    delta_sinogram = delta_recon_at_indices
    forward_linear = fm_constant * torch.sum(cur_error_image * delta_sinogram,
                                             dim=(1, 2))
    forward_quadratic = fm_constant * torch.sum(delta_sinogram * delta_sinogram,
                                                dim=(1, 2))

    alpha_numerator = forward_linear - prior_linear
    alpha_denominator = forward_quadratic + prior_quadratic + _F32_EPS
    alpha = alpha_numerator / alpha_denominator
    alpha = torch.clamp(alpha, _F32_EPS, max_alpha)
    # A volume that has stopped gets a step size of zero.
    alpha = torch.where(active, alpha, torch.zeros_like(alpha))

    # The positivity constraint clips the update so that image + alpha * delta >= 0.
    if positivity:
        delta_recon_at_indices = torch.maximum(
            -flat_image[:, pixel_indices] / (alpha[:, None, None] + _F32_EPS),
            delta_recon_at_indices)

    delta_scaled = alpha[:, None, None] * delta_recon_at_indices
    flat_image.index_add_(1, pixel_indices, delta_scaled)

    cur_error_image = cur_error_image - delta_scaled
    flat_error_image.index_copy_(1, pixel_indices, cur_error_image)
    l1_for_subset = torch.sum(torch.abs(delta_scaled), dim=(1, 2))
    return flat_image, flat_error_image, l1_for_subset, alpha


def _subset_pair_lists(partition, image_shape, device):
    """This function returns, for each subset of the partition, the tuple (row_pairs,
    row_valid, col_pairs, col_valid) on the device.  The pairs come from
    :func:`vcd_utils.gen_neighbor_pair_lists`.  A valid array holds 1 for each pair of the
    subset and 0 for each row that pads the list."""
    lists = []
    for pairs, counts in vcd_utils.gen_neighbor_pair_lists(partition, image_shape):
        rows = np.arange(pairs.shape[1])
        lists.append([(torch.as_tensor(pairs[k], device=device),
                       torch.as_tensor((rows < counts[k]).astype(np.float32)[:, None],
                                       device=device))
                      for k in range(pairs.shape[0])])
    return [row + col for row, col in zip(*lists)]


def _is_stack(image):
    """This function returns True when the image is a stack of volumes, and False when it
    is one volume."""
    if isinstance(image, _sharding.Shards):
        return False
    return len(np.shape(image)) == 4


def _mask_key(use_ror_mask):
    """This function returns the mask setting in a form that can be compared with ==.  A
    mask array becomes its bytes."""
    if use_ror_mask is True or use_ror_mask is False:
        return use_ror_mask
    return np.asarray(use_ror_mask).tobytes()


def _volume_shape(image):
    """This function returns the shape of a 3D volume given as a numpy array, a torch
    tensor, or a Shards."""
    if isinstance(image, _sharding.Shards):
        first = image.tensors[0]
        num_slices = sum(int(t.shape[-1]) for t in image.tensors)
        return int(first.shape[0]), int(first.shape[1]), num_slices
    if torch.is_tensor(image):
        return tuple(int(n) for n in image.shape)
    return tuple(int(n) for n in np.asarray(image).shape)


def _subsample_to_host(image, row_step=1, col_step=1, slice_step=1):
    """This function returns ``numpy.asarray(image)[::row_step, ::col_step, ::slice_step]``
    for a 3D volume given as a numpy array, a tensor, or a Shards.  The function subsamples
    on the device that holds the data, so it copies only the sampled elements to the host.
    """
    def block_to_host(tensor, slice_start):
        """This function returns the subsampled block of one tensor as a numpy array."""
        block = tensor[::row_step, ::col_step, slice_start::slice_step]
        return block.detach().contiguous().cpu().numpy()

    if isinstance(image, _sharding.Shards):
        placement = image.placement
        if placement.axis % 3 != 2:
            raise ValueError(
                'A denoiser image must be sharded on its last (slice) axis; '
                'got a placement on axis {}.'.format(placement.axis))
        num_slices = _volume_shape(image)[2]
        blocks = [block_to_host(tensor, (-start) % slice_step)
                  for tensor, (_dev, (start, _end))
                  in zip(image.tensors, placement.shard_ranges(num_slices))]
        return np.concatenate(blocks, axis=-1)
    if torch.is_tensor(image):
        return block_to_host(image, 0)
    return np.asarray(image)[::row_step, ::col_step, ::slice_step]


class QGGMRFDenoiser(TomographyModel):
    """
    This class is used to remove additive white Gaussian noise from a volume.  The
    denoiser uses the same qGGMRF prior as the reconstruction.  :meth:`denoise` denoises one
    volume, and :meth:`denoise_stack` denoises several.

    Each call estimates the noise level from the volume it is given, and then it sets the
    regularization.  To apply one fixed denoiser to a series of volumes, call
    ``initialize_denoiser`` once and pass ``do_initialization=False`` on every later call.

    Args:
        image_shape (tuple of int): The shape of the volumes to denoise, as three integers.
            For a 2D image, use (1, num_rows, num_cols).
        compile_mode (str, optional): If 'auto', the denoiser compiles the kernels with
            ``torch.compile``.  If 'off', it does not.  Defaults to 'auto'.
    """

    # This name selects the denoiser rows of the minimum size table in _min_problem_sizes.
    # For the denoiser, the sizes are in image voxels.
    _size_table_key = 'denoiser'

    def __init__(self, image_shape, compile_mode='auto'):
        if len(image_shape) != 3:
            raise ValueError('image_shape must be 3-dimensional. Got image_shape={}. '
                             'To denoise a 2D image, use shape (1, m, n).'.format(image_shape))
        super().__init__(image_shape, compile_mode=compile_mode,
                         view_params_name='None', sigma_noise=None)
        # sigma_noise has no recompile flag, so setting it does not clear the caches.
        self.params['sigma_noise'] = Param(self.get_params('sigma_noise'), False)
        # This dict holds the result of initialize_denoiser: the partition, the regularization
        # parameters, the noise level, and the settings used to compute them.
        self.denoise_data = None
        # This flag is True when the stored partition came from the caller.
        self._partition_supplied = False
        # This number names the compiled functions of this object.  It comes from a counter,
        # because Python reuses an object address after the object is freed.
        self._instance_key = next(_instance_counter)
        self.set_params(use_ror_mask=False)
        self.set_params(sharpness=0)
        # The denoiser uses one fixed partition.
        self.set_params(granularity=[16], partition_sequence=[0])

    def _invalidate_device_caches(self):
        """This method clears the denoiser cache and the caches of the base model."""
        super()._invalidate_device_caches()
        self.denoise_data = None

    def get_magnification(self):
        """This method returns 1.  The TomographyModel interface requires this method."""
        return 1.0

    def verify_valid_params(self):
        """This method checks that the parameters are consistent for denoising."""
        super().verify_valid_params()
        sinogram_shape = self.get_params('sinogram_shape')
        image_shape = self.get_params('recon_shape')
        if tuple(image_shape) != tuple(sinogram_shape):
            raise ValueError('image_shape and sinogram_shape must be the same. \n'
                             f'Got {image_shape} for image_shape and '
                             f'{sinogram_shape} for sinogram_shape')

    def create_projectors(self):
        # The identity forward model has no projectors.
        pass

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """This method sets ``recon_shape`` equal to the input image shape."""
        sinogram_shape = self.get_params('sinogram_shape')
        self.set_params(no_compile=no_compile, no_warning=no_warning,
                        recon_shape=sinogram_shape)

    def auto_set_sigma_y(self, sinogram, sino_indicator, weights=1):
        # For the identity forward model, sigma_y is the noise level.
        sigma_y = self.get_params('sigma_noise')
        self.set_params(no_warning=True, sigma_y=sigma_y, auto_regularize_flag=True)

    def _check_lateral_truncation(self, sino_indicator):
        """This method does nothing.  Image content at the edge of the image is normal for
        the denoiser."""
        return

    def estimate_image_noise_std(self, image):
        """
        This method estimates the noise standard deviation of the image from differences
        between neighboring voxels in the image support.  The method uses a subsample of at
        most about five million voxels.  It never copies a whole sharded image to the host.
        """
        num_rows, num_cols, num_slices = _volume_shape(image)
        num_elements = num_rows * num_cols * num_slices
        num_pts_to_use = np.minimum(5_000_000, num_elements)
        stride = round((num_elements / num_pts_to_use) ** (1 / 3))
        small_image = _subsample_to_host(image, stride, stride, stride)

        support_indicator = self._get_sino_indicator(small_image, sigma_noise=0.0)
        sigma_noise = self._get_estimate_of_recon_std(small_image, support_indicator)
        support_indicator = self._get_sino_indicator(small_image, sigma_noise=sigma_noise)
        sigma_noise = self._get_estimate_of_recon_std(small_image, support_indicator)
        return sigma_noise

    def _get_estimate_of_recon_std(self, noisy_image, support_indicator):
        """This method returns an estimate of the noise standard deviation.  For each voxel,
        the method computes the standard deviation of the voxel and its three previous
        neighbors, one along each axis.  The method returns the mean of these standard
        deviations over the support.
        """
        def views():
            """This function yields the image and the image shifted by one voxel along
            each of the three axes."""
            yield noisy_image
            for axis in range(3):
                yield np.roll(noisy_image, 1, axis=axis)

        count = 4
        mean = np.array(noisy_image, dtype=noisy_image.dtype, copy=True)
        for index, value in enumerate(views()):
            if index:
                mean += value
        mean /= count
        spread = np.zeros_like(mean)
        deviation = np.empty_like(mean)
        for value in views():
            np.subtract(value, mean, out=deviation)
            deviation *= deviation
            spread += deviation
        spread /= count
        np.sqrt(spread, out=spread)
        return np.mean(spread[support_indicator.astype(bool)])

    def _get_sino_indicator(self, noisy_image, sigma_noise=None, verbose=1):
        """This method returns a binary indicator of the support of the noisy image."""
        if sigma_noise is None:
            sigma_noise = self.get_params('sigma_noise')
        percent_noise_floor = 5.0
        threshold = (0.01 * percent_noise_floor) * np.mean(np.fabs(noisy_image)) + sigma_noise
        threshold = min(threshold, np.amax(noisy_image))
        return np.int8(noisy_image >= threshold)

    def recon(self, *args, **kwargs):
        raise NotImplementedError('recon is not implemented for QGGMRFDenoiser.  '
                                  'Use `denoise` instead.')

    # ── the stored initialization ─────────────────────────────────────────────
    def _denoise_signature(self):
        """This method returns the settings that ``denoise_data`` depends on."""
        recon_shape, granularity, partition_sequence, use_ror_mask = self.get_params(
            ['recon_shape', 'granularity', 'partition_sequence', 'use_ror_mask'])
        return (tuple(int(n) for n in recon_shape), tuple(granularity),
                tuple(partition_sequence), _mask_key(use_ror_mask),
                str(self.torch_device))

    def _draw_partition(self, rng=None):
        """This method returns a new pixel partition of the image."""
        image_shape, granularity, partition_sequence, use_ror_mask = self.get_params(
            ['recon_shape', 'granularity', 'partition_sequence', 'use_ror_mask'])
        return vcd_utils.gen_set_of_pixel_partitions(
            image_shape, [granularity[partition_sequence[0]]],
            device=self.torch_device, use_ror_mask=use_ror_mask, rng=rng)[0]

    def _validated_partition(self, partition):
        """This method checks a partition given by the caller and returns it as a contiguous
        int64 tensor on the model device.  The check confirms that the subsets divide the pixels
        that the mask allows, with no pixel repeated."""
        image_shape = tuple(int(n) for n in self.get_params('recon_shape'))
        values = (partition.detach().cpu().numpy() if torch.is_tensor(partition)
                  else np.asarray(partition))
        if values.ndim != 2:
            raise ValueError('a denoiser partition has two dimensions, (subsets, '
                             f'indices); got shape {values.shape}.')
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError('a denoiser partition holds integer pixel indices; got '
                             f'dtype {values.dtype}.')
        num_pixels = image_shape[0] * image_shape[1]
        if values.size == 0:
            raise ValueError('a denoiser partition holds at least one pixel index; '
                             'got an empty array.')
        if values.min() < 0 or values.max() >= num_pixels:
            raise ValueError(
                f'a denoiser partition indexes the {image_shape[0]} by {image_shape[1]} '
                f'pixel grid, so every value lies in [0, {num_pixels}); got values from '
                f'{int(values.min())} to {int(values.max())}.')
        for row in range(values.shape[0]):
            if np.unique(values[row]).size != values.shape[1]:
                raise ValueError(
                    f'subset {row} of a denoiser partition holds an index twice.')
        allowed = np.asarray(vcd_utils.gen_full_indices(
            image_shape, use_ror_mask=self.get_params('use_ror_mask')))
        present = np.unique(values)
        if not np.array_equal(present, np.sort(allowed)):
            raise ValueError(
                'the subsets of a denoiser partition together cover exactly the pixels '
                f'the mask allows, which are {allowed.size} of {num_pixels}; they cover '
                f'{present.size}.')
        return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.int64,
                               device=self.torch_device)

    def _get_regularization_params(self, image):
        """This method returns the regularization parameters for denoising.  When
        auto-regularization is on, the method estimates them from the image."""
        names = list(_AUTO_REGULARIZATION_PARAM_NAMES)
        if not self.get_params('auto_regularize_flag'):
            return dict(zip(names, [float(v) for v in self.get_params(names)]))
        if image is None:
            raise ValueError(
                'initialize_denoiser estimates the regularization parameters from an '
                'image, because auto_regularize_flag is on.  Pass image=, or set the '
                'parameters yourself and turn the flag off.')
        if _is_stack(image):
            return self.auto_set_regularization_params_from_stack(image)
        # subsample_views keeps every step_size-th entry.  Subsampling the rows with the
        # same step gives auto_set_regularization_params the rows it would select.
        num_rows = _volume_shape(image)[0]
        sampled_rows = self.subsample_views(np.arange(num_rows))
        row_step = int(sampled_rows[1] - sampled_rows[0]) if sampled_rows.size > 1 else 1
        small_image = _subsample_to_host(image, row_step=row_step)
        verbose = self.get_params('verbose')
        self.set_params(no_warning=True, verbose=0)
        params = self.auto_set_regularization_params(small_image)
        self.set_params(no_warning=True, verbose=verbose)
        return params

    def initialize_denoiser(self, image=None, sigma_noise=None, partition=None,
                            rng=None):
        """
        This method computes the noise level, the regularization parameters, and the pixel
        partition, and it stores them in ``denoise_data``.

        :meth:`denoise` and :meth:`denoise_stack` call this method themselves.  After a
        direct call, the denoiser is a fixed operator.  Every later call with
        ``do_initialization=False`` uses the same partition and the same parameters.  A
        change of the GPUs or of ``recon_shape`` clears ``denoise_data``.

        Args:
            image (numpy or tensor or Shards, optional): One volume, or a stack of volumes
                whose first axis indexes the volumes.  The method estimates the noise level
                and the regularization parameters from it.
            sigma_noise (float, optional): The noise level.  If None, the method estimates it
                from ``image``.  If there is also no image, the method keeps the current value
                of the model.  The method sets ``sigma_y`` equal to the noise level.
            partition (array or tensor, optional): The pixel partition used in every
                iteration, with shape (subsets, indices).  If None, the method draws a random
                partition.
            rng (numpy.random.Generator, optional): The random generator used to draw the
                partition.  Defaults to None, which uses the global np.random state.

        Returns:
            dict: ``denoise_data``, with entries 'partition', 'regularization_params',
            'sigma_noise', and 'signature'.

        Raises:
            ValueError: If no noise level is available, if ``auto_regularize_flag`` is on and
                no image is given, or if a given partition does not divide the pixels that the
                mask allows.

        Example:
            >>> denoiser.initialize_denoiser(image=volume, sigma_noise=0.1)
            >>> out, d = denoiser.denoise(volume, do_initialization=False)
        """
        # The GPUs must be chosen before the partition is created on the device.
        self._apply_device_policy(workload='denoise')
        image_shape = tuple(int(n) for n in self.get_params('recon_shape'))
        if sigma_noise is None and image is not None:
            # The noise estimate treats a stack as one volume, with the volumes joined along
            # the row axis.
            volume = (image.reshape(-1, image_shape[1], image_shape[2])
                      if _is_stack(image) else image)
            sigma_noise = self.estimate_image_noise_std(volume)
        if sigma_noise is None:
            sigma_noise = self.get_params('sigma_noise')
        if sigma_noise is None:
            raise ValueError(
                'initialize_denoiser needs a noise level.  Pass sigma_noise, pass an '
                'image to estimate it from, or set sigma_noise on the model.')
        # For the identity forward model, sigma_y equals sigma_noise.
        self.set_params(no_warning=True, sigma_noise=float(sigma_noise),
                        sigma_y=float(sigma_noise))
        regularization_params = self._get_regularization_params(image)
        if partition is None:
            partition_tensor = self._draw_partition(rng=rng)
            self._partition_supplied = False
        else:
            partition_tensor = self._validated_partition(partition)
            self._partition_supplied = True
        self.denoise_data = {'partition': partition_tensor,
                             'regularization_params': regularization_params,
                             'sigma_noise': float(sigma_noise),
                             'signature': self._denoise_signature()}
        return self.denoise_data

    def _denoise_setup(self, image, sigma_noise, do_initialization):
        """This method returns the pixel partition and the regularization parameters for the
        next denoising run.  The method calls ``initialize_denoiser`` when
        ``do_initialization`` is True or when ``denoise_data`` is None."""
        if do_initialization or self.denoise_data is None:
            if not do_initialization and self._partition_supplied:
                raise ValueError(
                    'the partition given to initialize_denoiser was dropped by a change '
                    'of the device layout, so this call has none to reuse.  Call '
                    'initialize_denoiser again with the partition.')
            data = self.initialize_denoiser(image=image, sigma_noise=sigma_noise)
            return data['partition'], dict(data['regularization_params'])

        data = self.denoise_data
        level = data['sigma_noise'] if sigma_noise is None else float(sigma_noise)
        self.set_params(no_warning=True, sigma_noise=level, sigma_y=level)
        data['sigma_noise'] = level
        signature = self._denoise_signature()
        if signature != data['signature']:
            if self._partition_supplied:
                raise ValueError(
                    'the partition given to initialize_denoiser was made for other '
                    'settings than this call uses.  Call initialize_denoiser again with '
                    'a partition for the current settings.')
            data['partition'] = self._draw_partition()
            data['signature'] = signature
            self.logger.info('The pixel partition was drawn again, because the '
                             'partition settings changed.')
        return data['partition'], dict(data['regularization_params'], sigma_y=level)

    def denoise(self, image, sigma_noise=None, use_ror_mask=None, init_image=None,
                max_iterations=15, stop_threshold_change_pct=0.2, first_iteration=0,
                logfile_path='~/.mbirtorch/logs/recon.log', print_logs=True,
                output_sharded=False, do_initialization=True):
        """
        This method denoises a volume.  The method assumes additive white Gaussian noise
        and uses the qGGMRF prior.

        Args:
            image (numpy array, tensor, or Shards): The volume to denoise.  Its shape must be
                the shape the denoiser was built for.
            sigma_noise (float, optional): The noise standard deviation.  If None, the method
                estimates it from ``image``.  Defaults to None.
            use_ror_mask (bool, array, or None, optional): The region to denoise.  False
                selects the whole volume.  True selects the inscribed ellipse.  A 2D array of
                0s and 1s selects the columns marked 1.  None keeps the setting of the model,
                which starts as False.  Defaults to None.
            init_image (numpy array, tensor, or Shards, optional): The starting point of the
                minimization.  Defaults to ``image``.
            max_iterations (int, optional): The maximum number of iterations.  Defaults to 15.
            stop_threshold_change_pct (float, optional): The method stops when the percent
                change of the volume in an iteration falls below this value.  A value of 0
                runs every iteration.  Defaults to 0.2.
            first_iteration (int, optional): The number that the log gives the first
                iteration.  Defaults to 0.
            logfile_path (str, optional): The log file.  If None or '', the method writes no
                log file.  Defaults to '~/.mbirtorch/logs/recon.log'.
            print_logs (bool, optional): If True, the method prints the log.  Defaults to True.
            output_sharded (bool, optional): If True, the method returns the volume in the
                form it had on the devices, not as a numpy array.  Defaults to False.
            do_initialization (bool, optional): If True, the method estimates the noise level
                and sets the regularization for this call.  If False, the method reuses the
                values set by the previous call or by ``initialize_denoiser``.  Defaults to
                True.

        Returns:
            tuple: ``(denoised_image, denoise_dict)``.  ``denoised_image`` is the denoised
            volume.  ``denoise_dict`` has the same entries as the dictionary that
            :meth:`~mbirtorch.TomographyModel.recon` returns.

        Example:
            >>> denoiser = mbirtorch.QGGMRFDenoiser(noisy_image.shape)
            >>> denoised_image, d = denoiser.denoise(noisy_image, sigma_noise=0.1)
        """
        self._log_run_header(first_iteration, logfile_path, print_logs)
        # The GPUs must be chosen before the image is copied to them.
        self._apply_device_policy(workload='denoise', init_recon=init_image)
        self._log_device_report()

        if use_ror_mask is not None:
            self.set_params(no_warning=True, use_ror_mask=use_ror_mask)
        self.logger.info('Initializing QGGMRFDenoiser')
        # The denoiser uses one fixed partition and updates its subsets in order.
        partition, regularization_params = self._denoise_setup(
            image, sigma_noise, do_initialization)

        image_shape, granularity = self.get_params(['recon_shape', 'granularity'])
        partition_sequence = self.get_params('partition_sequence')
        verbose = self.get_params('verbose')

        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(
            ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        b = _qggmrf.get_b_from_nbr_wts(qggmrf_nbr_wts)
        qggmrf_params = (b, sigma_x, p, q, T)
        max_alpha = self.get_params('max_alpha')
        positivity = bool(self.get_params('positivity_flag'))
        max_iters = max_iterations
        stop_thresh = stop_threshold_change_pct / 100.0

        image_t = self._shard_recon(image)
        init_t = image_t if init_image is None else self._shard_recon(init_image)

        self.logger.info('Starting VCD iterations')
        if isinstance(image_t, _sharding.Shards):
            flat_image, nmae_update, alpha_values, num_iters = self._denoise_sharded(
                image_t, init_t, partition, fm_constant, qggmrf_params, max_alpha,
                positivity, tuple(image_shape), max_iters, stop_thresh, first_iteration, verbose)
            denoised = _sharding.Shards(
                [t.reshape(s.shape) for t, s in zip(flat_image.tensors, image_t.tensors)],
                flat_image.placement)
        else:
            flat_image = init_t.clone().reshape((-1, image_shape[2])).contiguous()
            flat_error_image = (image_t.reshape((-1, image_shape[2]))
                                - flat_image).contiguous()
            subset_denoiser = maybe_compile(vcd_subset_denoiser, self.compile_enabled)
            pair_lists = _subset_pair_lists(partition, image_shape, flat_image.device)
            nmae_update = np.zeros(max_iters)
            alpha_values = np.zeros(max_iters)
            num_iters = 0
            with torch.no_grad():
                for i in range(max_iters):
                    l1_accum = 0.0
                    alpha_accum = 0.0
                    for k in range(partition.shape[0]):
                        flat_image, flat_error_image, l1_subset, alpha_subset = \
                            subset_denoiser(flat_image, flat_error_image, partition[k],
                                            fm_constant, qggmrf_params, max_alpha,
                                            positivity, tuple(image_shape), pair_lists[k])
                        l1_accum = l1_accum + l1_subset
                        alpha_accum = alpha_accum + alpha_subset

                    # An image of zeros gives nan, not ZeroDivisionError.
                    image_l1 = float(image_l1_norm(flat_image))
                    nmae = (float(l1_accum) / image_l1 if image_l1
                            else float('nan'))
                    nmae_update[i] = nmae
                    alpha_values[i] = float(alpha_accum) / partition.shape[0]
                    num_iters += 1
                    if verbose >= 1 and (i % 5) == 0:
                        self.logger.info('After iteration {} of a max of {}: Pct change={:.4f}'
                                         .format(i + first_iteration, max_iters, 100 * nmae))
                    if nmae < stop_thresh:
                        break
            denoised = flat_image.reshape(tuple(image_shape))

        recon_params = dict(zip(recon_param_names,
                                [int(num_iters), granularity, partition_sequence,
                                 None, None, regularization_params,
                                 [100 * float(v) for v in nmae_update[:num_iters]],
                                 [float(v) for v in alpha_values[:num_iters]],
                                 None]))
        # A later call reopens the log file.
        self.close_log_file()
        notes = 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now())
        denoiser_dict = self.get_recon_dict(recon_params, notes=notes)
        return (denoised if output_sharded else self._gather_recon(denoised)), denoiser_dict

    def _denoise_sharded(self, image_sh, init_sh, partition, fm_constant,
                         qggmrf_params, max_alpha, positivity, image_shape, max_iters,
                         stop_thresh, first_iteration, verbose):
        """This method runs the denoising iterations on several devices, with the image
        divided by slices.  The method returns (flat_image, nmae history, alpha history,
        num_iters), where ``flat_image`` is a Shards."""
        devices = image_sh.placement.devices
        n = len(devices)
        pl = image_sh.placement
        dev0 = devices[0]

        def combine_on_lead(parts):
            """This function sums one scalar tensor from each shard on the first device."""
            total = parts[0]
            for part in parts[1:]:
                total = total + _sharding.move_shard(part, dev0,
                                                     self.dev2dev_safe)
            return total

        # Each shard has shape (num_pixels, local_slices).  The code gives num_pixels to
        # reshape, because reshape cannot infer it for a shard with no slices.
        num_pixels = int(image_shape[0]) * int(image_shape[1])
        flat_image = _sharding.Shards(
            [t.reshape(num_pixels, t.shape[-1]).clone().contiguous()
             for t in init_sh.tensors], pl)
        flat_error = _sharding.Shards(
            [(a.reshape(num_pixels, a.shape[-1]) - b).contiguous()
             for a, b in zip(image_sh.tensors, flat_image.tensors)], pl)

        grad_hess = [maybe_compile(_qggmrf.qggmrf_terms_at_indices,
                                   self.compile_enabled, instance_key=i)
                     for i in range(n)]
        idx_per_dev = [[torch.as_tensor(partition[k], dtype=torch.int64).to(d)
                        for d in devices] for k in range(partition.shape[0])]
        pairs_per_dev = [_subset_pair_lists(partition, image_shape, d) for d in devices]
        halos = {'left': [None] * n, 'right': [None] * n}

        nmae_update = np.zeros(max_iters)
        alpha_values = np.zeros(max_iters)
        num_iters = 0
        # If the caller already set a thread pool, this method uses that pool.
        owns_pool = n > 1 and self._per_device_pool is None
        if owns_pool:
            self._per_device_pool = _sharding.device_pool(n)
        try:
            with torch.no_grad():
                for i in range(max_iters):
                    # The shards exchange their boundary slices once per iteration.
                    halos['left'], halos['right'] = _sharding.exchange_qggmrf_halos(
                        flat_image, self.dev2dev_safe)
                    l1_accum = 0.0
                    alpha_accum = 0.0
                    for k in range(partition.shape[0]):
                        idx = idx_per_dev[k]

                        def terms_worker(j, dev):
                            grad, hess, slice_b_tilde_2 = grad_hess[j](
                                flat_image.tensors[j], image_shape, idx[j],
                                qggmrf_params, left_halo=halos['left'][j],
                                right_halo=halos['right'][j])
                            cur_error = flat_error.tensors[j][idx[j]]
                            forward_grad = -fm_constant * cur_error
                            delta = -((forward_grad + grad) / (fm_constant + hess))
                            # The four sums are scalar tensors.  The code adds them on the
                            # first device.
                            return (delta,
                                    torch.sum(grad * delta),
                                    _qggmrf.qggmrf_exact_quadratic(
                                        hess, slice_b_tilde_2, delta, flat_image.tensors[j],
                                        idx[j], image_shape, qggmrf_params,
                                        pairs_per_dev[j][k], halos['left'][j] is None,
                                        halos['right'][j] is None),
                                    fm_constant * torch.sum(cur_error * delta),
                                    fm_constant * torch.sum(delta * delta))

                        results = _sharding.run_per_device(
                            devices, terms_worker, executor=self._per_device_pool)
                        deltas = [r[0] for r in results]
                        prior_linear = combine_on_lead([r[1] for r in results])
                        prior_quadratic = combine_on_lead([r[2] for r in results])
                        forward_linear = combine_on_lead([r[3] for r in results])
                        forward_quadratic = combine_on_lead([r[4] for r in results])
                        # Each device receives the first slice of the direction of the next
                        # device.  The pairs across each boundary belong to the device before it.
                        _, delta_right = _sharding.exchange_qggmrf_halos(
                            _sharding.Shards(deltas, pl), self.dev2dev_safe)
                        boundary_sums = [
                            _qggmrf.qggmrf_boundary_pair_sum(
                                flat_image.tensors[j][idx[j], -1], halos['right'][j][idx[j]],
                                deltas[j][:, -1], delta_right[j], qggmrf_params)
                            for j in range(n) if delta_right[j] is not None]
                        if boundary_sums:
                            prior_quadratic = prior_quadratic - combine_on_lead(boundary_sums)
                        prior_quadratic = prior_quadratic.to(torch.float32)
                        alpha = ((forward_linear - prior_linear)
                                 / (forward_quadratic + prior_quadratic + _F32_EPS))
                        alpha = torch.clamp(alpha, _F32_EPS, max_alpha)
                        # Each device needs its own copy of the step size.
                        alpha_per_device = (
                            [alpha] if n == 1 else
                            [_sharding.move_shard(alpha, dev, self.dev2dev_safe)
                             for dev in devices])

                        def apply_worker(j, dev):
                            delta = deltas[j]
                            # The positivity constraint clips the update so that
                            # image + alpha * delta >= 0.
                            if positivity:
                                delta = torch.maximum(
                                    -flat_image.tensors[j][idx[j]]
                                    / (alpha_per_device[j] + _F32_EPS), delta)
                            step = alpha_per_device[j] * delta
                            flat_image.tensors[j].index_add_(0, idx[j], step)
                            flat_error.tensors[j].index_add_(0, idx[j], -step)
                            return torch.sum(torch.abs(step))

                        l1_parts = _sharding.run_per_device(
                            devices, apply_worker, executor=self._per_device_pool)
                        l1_accum = l1_accum + combine_on_lead(l1_parts)
                        alpha_accum = alpha_accum + alpha

                    # This is the only host synchronization of the iteration.  An image of
                    # zeros gives nan.
                    image_l1 = float(combine_on_lead(
                        [image_l1_norm(t) for t in flat_image.tensors]))
                    nmae = (float(l1_accum) / image_l1 if image_l1
                            else float('nan'))
                    nmae_update[i] = nmae
                    alpha_values[i] = float(alpha_accum) / partition.shape[0]
                    num_iters += 1
                    if verbose >= 1 and (i % 5) == 0:
                        self.logger.info('After iteration {} of a max of {}: Pct change={:.4f}'
                                         .format(i + first_iteration, max_iters, 100 * nmae))
                    if nmae < stop_thresh:
                        break
        finally:
            if owns_pool:
                self._per_device_pool.shutdown(wait=True)
                self._per_device_pool = None
        return flat_image, nmae_update, alpha_values, num_iters

    def auto_set_regularization_params_from_stack(self, stack):
        """
        This method sets the regularization parameters sigma_y, sigma_x, and sigma_prox
        from about 20 evenly spaced volumes of a stack.  The method returns the parameters as
        a dict.  The parameters change only when ``auto_regularize_flag`` is True, as in
        :meth:`auto_set_regularization_params`.

        Args:
            stack (numpy or tensor): The volumes, with shape ``(num_volumes,) + image_shape``.
                The stack may be on any device.

        Returns:
            dict: The values of ``sigma_y``, ``sigma_x``, and ``sigma_prox``.

        Raises:
            TypeError: If ``stack`` is a Shards.
        """
        _sharding.reject_shards('auto_set_regularization_params_from_stack',
                                stack=stack)
        names = list(_AUTO_REGULARIZATION_PARAM_NAMES)
        if self.get_params('auto_regularize_flag'):
            if not torch.is_tensor(stack):
                stack = np.asarray(stack)
            num_volumes = int(stack.shape[0])
            d0, d1, d2 = (int(n) for n in stack.shape[1:])
            chosen = self.subsample_views(np.arange(num_volumes))
            step = int(chosen[1] - chosen[0]) if chosen.size > 1 else 1
            # The code crops the chosen volumes to at most _STATISTICS_POINT_BUDGET voxels
            # before it copies them to the host.
            rows, cols = _sample_tiles(d1, d2, int(chosen.size) * d0,
                                       _STATISTICS_POINT_BUDGET)
            blocks = []
            for row in rows:
                for col in cols:
                    tile = stack[:, :, row, col]
                    height, width = row.stop - row.start, col.stop - col.start
                    piece = _subsample_to_host(
                        tile.reshape(num_volumes, d0 * height, width), row_step=step)
                    blocks.append(piece.reshape(-1, height, width))
            merged = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)
            tiled = len(blocks) > 1 or merged.shape[1:] != (d1, d2)

            sino_indicator = self._get_sino_indicator(merged)
            if tiled:
                # The noise estimate wraps around at the end of each axis.  So the code removes
                # entry 0 of each axis and the planes where tiles meet from the support.
                sino_indicator[:, 0, :] = 0
                sino_indicator[:, :, 0] = 0
                sino_indicator[::merged.shape[0] // len(blocks)] = 0
            self.auto_set_sigma_y(merged, sino_indicator)
            recon_std = self._get_estimate_of_recon_std(merged, sino_indicator)
            if not np.isfinite(recon_std):
                recon_std = 0.0
            self.auto_set_sigma_x(recon_std)
            self.auto_set_sigma_prox(recon_std)
            # A sigma_x near zero would make the denoiser return NaN.
            sigma_x = self.get_params('sigma_x')
            if not np.isfinite(sigma_x) or sigma_x < _SIGMA_X_FLOOR:
                self.set_params(no_warning=True, sigma_x=_SIGMA_X_FLOOR)
        values = [float(v) for v in self.get_params(names)]
        return dict(zip(names, values))

    def auto_batch_size(self, volume_shape=None):
        """
        This method returns the number of volumes that :meth:`denoise_stack` processes at
        once when its ``batch_size`` is None.

        On a CUDA device, the result is the largest batch that fits in the free memory of the
        device.  On other devices, the result is None, which means the whole stack.

        Args:
            volume_shape (tuple of int, optional): The shape of one volume.  Defaults to the
                image shape of the denoiser.  This denoiser accepts only that shape.

        Returns:
            int or None: The number of volumes in each batch.  The value is None when the
            device does not report its free memory.

        Raises:
            ValueError: If ``volume_shape`` is not the image shape of the denoiser.
            MemoryCheckError: If a batch of one volume does not fit.
        """
        image_shape = tuple(int(n) for n in self.get_params('recon_shape'))
        if volume_shape is not None and \
                tuple(int(n) for n in volume_shape) != image_shape:
            raise ValueError(
                'auto_batch_size prices the sweep this denoiser runs, which is '
                f'over volumes of shape {image_shape}; got volume_shape '
                f'{tuple(volume_shape)}.  Build a QGGMRFDenoiser at that shape.')
        device = self.torch_device
        budget = _memory_estimate.device_budget_bytes(device)
        if budget is None:
            return None
        plan = _memory_estimate.plan_from_model(self, [device], workload='denoise')
        estimate = _memory_estimate.estimate_peak_device_bytes(plan)
        batch = _memory_estimate.largest_denoise_batch(
            estimate, budget, margin=self.memory_check_margin)
        if batch < 1:
            need = _memory_estimate.denoise_batch_peak_bytes(estimate, 1)
            raise _memory_estimate.MemoryCheckError(
                'a denoise sweep of one volume of shape {} is modeled at {:.2f} GB '
                'against {:.2f} GB free on {} (with a {:.0%} margin), so no batch '
                'fits.  Pass batch_size to denoise_stack to run anyway.'.format(
                    image_shape, need / 2 ** 30, budget / 2 ** 30, device,
                    self.memory_check_margin))
        return int(batch)

    def denoise_stack(self, stack, sigma_noise=None, init_stack=None,
                      max_iterations=15, stop_threshold_change_pct=0.2,
                      batch_size=None, overwrite_input=False,
                      do_initialization=True):
        """
        This method denoises a stack of volumes of the same shape.  The method denoises each
        volume as :meth:`denoise` would, with one noise level and one set of regularization
        parameters for the whole stack.

        Each volume has its own stopping test.  The result for each volume equals the result
        of :meth:`denoise` on that volume alone.  The method sets the regularization from a
        subsample of the stack.  It processes the stack in batches on one device.  It writes
        no log file.

        Args:
            stack (numpy array or tensor): The volumes, with shape (num_volumes,) + image_shape.
            sigma_noise (float, optional): The noise standard deviation of every volume.  If
                None, the method estimates it from the stack.  Defaults to None.
            init_stack (numpy array or tensor, optional): The starting point for each volume,
                with the shape of ``stack``.  Defaults to ``stack``.
            max_iterations (int, optional): The maximum number of iterations for each volume.
                Defaults to 15.
            stop_threshold_change_pct (float, optional): A volume stops when its percent change
                in an iteration falls below this value.  A value of 0 runs every iteration.
                Defaults to 0.2.
            batch_size (int, optional): The number of volumes processed at once.  If None, the
                method chooses the number from the device memory.  Defaults to None.
            overwrite_input (bool, optional): If True and ``stack`` is a float32 tensor on the
                device, the method writes into ``stack`` instead of copying it.  This saves
                memory, but the contents of ``stack`` are not valid afterward.  Defaults to False.
            do_initialization (bool, optional): The same as in :meth:`denoise`.  Defaults to
                True.

        Returns:
            tuple: ``(denoised_stack, info)``.  ``denoised_stack`` holds the denoised volumes.
            It is a numpy array for numpy input and a tensor for tensor input.  ``info`` is a
            dictionary with 'num_iterations' and 'nmae_pct' for each volume,
            'regularization_params', and 'batch_size'.

        Raises:
            ValueError: If ``stack`` or ``init_stack`` has the wrong shape, if ``batch_size``
                is less than 1, or if the denoiser is configured for more than one device.

        Example:
            >>> denoiser = mbirtorch.QGGMRFDenoiser(stack.shape[1:])
            >>> denoised, info = denoiser.denoise_stack(stack, sigma_noise=0.1)
        """
        _sharding.reject_shards('denoise_stack', stack=stack, init_stack=init_stack)
        if self.recon_placement.n_devices > 1:
            raise ValueError(
                'denoise_stack runs on one device, and this denoiser is '
                f'configured with {self.recon_placement.n_devices}.  Configure '
                'it with one device, or call denoise for the sharded sweep.')
        image_shape = tuple(int(n) for n in self.get_params('recon_shape'))
        stack_shape = tuple(int(n) for n in stack.shape)
        if len(stack_shape) != 4 or stack_shape[1:] != image_shape:
            raise ValueError(
                f'stack must have shape (num_volumes,) + {image_shape}; got '
                f'{stack_shape}.')
        if init_stack is not None and tuple(int(n) for n in init_stack.shape) != stack_shape:
            raise ValueError(
                f'init_stack must have the shape of stack, {stack_shape}; got '
                f'{tuple(init_stack.shape)}.')
        if init_stack is stack:
            # The default starting point is the input.  The in-place code below must not use one
            # buffer for both the image and the residual.
            init_stack = None
        num_volumes = stack_shape[0]
        num_pixels = image_shape[0] * image_shape[1]
        num_slices = image_shape[2]

        stack_is_tensor = torch.is_tensor(stack)
        out = None
        if batch_size is None:
            if stack_is_tensor:
                # The output must be allocated before auto_batch_size reads the free memory.
                out = torch.empty(stack_shape, dtype=torch.float32, device=stack.device)
            batch_size = self.auto_batch_size()
        if batch_size is None or int(batch_size) > num_volumes:
            batch_size = num_volumes
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f'batch_size must be at least 1; got {batch_size}.')
        # With one batch, the method returns the working image.  With several batches, the
        # method writes each batch into one output array.
        single = batch_size == num_volumes
        if single:
            out = None
        elif out is None:
            out = (torch.empty(stack_shape, dtype=torch.float32, device=stack.device)
                   if stack_is_tensor else np.empty(stack_shape, dtype=np.float32))

        # All volumes use the same partition of the pixel grid.
        partition, regularization_params = self._denoise_setup(
            stack, sigma_noise, do_initialization)
        # The device must be read after the GPUs are chosen.
        device = self.torch_device
        verbose = self.get_params('verbose')

        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(
            ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        qggmrf_params = (_qggmrf.get_b_from_nbr_wts(qggmrf_nbr_wts), sigma_x, p, q, T)
        max_alpha = self.get_params('max_alpha')
        positivity = bool(self.get_params('positivity_flag'))
        stop_thresh = stop_threshold_change_pct / 100.0
        # Each denoiser object gets its own compiled function, named by its own number.
        subset_denoiser = maybe_compile(vcd_subset_denoiser_batched,
                                        self.compile_enabled,
                                        instance_key=self._instance_key)
        pair_lists = _subset_pair_lists(partition, image_shape, device)

        def flat_on_device(block):
            """This function returns a block of volumes as a float32 tensor of shape
            (B, num_pixels, num_slices) on the device.  It also returns a flag that is True
            when the code may write into the tensor.  The code writes into a tensor of the
            caller only when ``overwrite_input`` is True.
            """
            tensor = block if torch.is_tensor(block) else torch.as_tensor(block)
            moved = tensor.to(device=device, dtype=torch.float32)
            flat = moved.reshape(moved.shape[0], num_pixels, num_slices)
            owned = (moved is not tensor or flat.data_ptr() != moved.data_ptr()
                     or (overwrite_input and torch.is_tensor(block)))
            return flat, owned

        def padded(flat, pad):
            """This function returns the block with its last volume repeated ``pad`` more
            times."""
            if pad == 0:
                return flat
            return torch.cat([flat, flat[-1:].expand(pad, -1, -1)])

        num_iterations = np.zeros(num_volumes, dtype=int)
        nmae_pct = [[] for _ in range(num_volumes)]
        with torch.no_grad():
            for b0 in range(0, num_volumes, batch_size):
                b1 = min(b0 + batch_size, num_volumes)
                pad = batch_size - (b1 - b0)
                flat, owned = flat_on_device(stack[b0:b1])
                if pad:
                    flat, owned = padded(flat, pad), True
                if init_stack is None:
                    # The update writes the image in place.  The code uses the input tensor
                    # only when the code may write into that tensor.
                    flat_image = flat if owned else flat.clone()
                    flat_error_image = torch.zeros_like(flat_image)
                else:
                    init_flat, init_owned = flat_on_device(init_stack[b0:b1])
                    if pad:
                        init_flat, init_owned = padded(init_flat, pad), True
                    flat_image = init_flat if init_owned else init_flat.clone()
                    flat_error_image = (flat.sub_(flat_image) if owned
                                        else flat - flat_image)
                flat_image = flat_image.contiguous()
                flat_error_image = flat_error_image.contiguous()
                del flat
                counts, history = self._sweep_stack(
                    flat_image, flat_error_image, partition, fm_constant,
                    qggmrf_params, max_alpha, positivity, image_shape, max_iterations,
                    stop_thresh, subset_denoiser, pair_lists)
                real = b1 - b0
                result = flat_image[:real].reshape((real,) + image_shape)
                if single:
                    out = result.to(stack.device) if stack_is_tensor else result.cpu().numpy()
                elif stack_is_tensor:
                    out[b0:b1] = result.to(out.device)
                else:
                    out[b0:b1] = result.cpu().numpy()
                num_iterations[b0:b1] = counts[:real]
                nmae_pct[b0:b1] = history[:real]

        if verbose >= 1:
            self.logger.info(
                'Denoised {} volumes in batches of {}: {} to {} iterations per '
                'volume.'.format(num_volumes, batch_size,
                                 int(num_iterations.min()), int(num_iterations.max())))
        info = dict(num_iterations=num_iterations, nmae_pct=nmae_pct,
                    regularization_params=regularization_params,
                    batch_size=batch_size)
        return out, info

    @staticmethod
    def _sweep_stack(flat_image, flat_error_image, partition, fm_constant,
                     qggmrf_params, max_alpha, positivity, image_shape, max_iters,
                     stop_thresh, subset_denoiser, pair_lists):
        """This method runs the denoising iterations in place on one batch of volumes.  Each
        volume stops when its change falls below ``stop_thresh`` or after ``max_iters``
        iterations.  ``pair_lists`` holds the pair lists of each subset.

        Returns:
            (num_iterations, nmae_pct): ``num_iterations`` holds the number of iterations of
            each volume.  ``nmae_pct`` holds one list for each volume, with the percent change
            at each of its iterations.
        """
        num_vols = int(flat_image.shape[0])
        device = flat_image.device
        active_host = np.ones(num_vols, dtype=bool)
        active = torch.ones(num_vols, dtype=torch.bool, device=device)
        counts = np.zeros(num_vols, dtype=int)
        history = [[] for _ in range(num_vols)]
        for i in range(max_iters):
            l1_accum = torch.zeros(num_vols, dtype=flat_image.dtype, device=device)
            for k in range(partition.shape[0]):
                flat_image, flat_error_image, l1_subset, _alpha = subset_denoiser(
                    flat_image, flat_error_image, partition[k], fm_constant,
                    qggmrf_params, max_alpha, positivity, tuple(image_shape), active,
                    pair_lists[k])
                l1_accum = l1_accum + l1_subset
            # The code reads from the device once per iteration.  A volume of zeros gives nan.
            stats = torch.stack([l1_accum, stack_l1_norm(flat_image)])
            stats = stats.cpu().numpy().astype(np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                nmae = stats[0] / stats[1]
            for j in np.flatnonzero(active_host):
                history[j].append(100.0 * float(nmae[j]))
                counts[j] = i + 1
            active_host &= ~(nmae < stop_thresh)
            if not active_host.any():
                break
            active = torch.tensor(active_host, device=device)
        return counts, history


def median_filter3d(x, max_block_gb=4.0, return_min_max=False):
    """
    This function applies a 3x3x3 median filter to a 3D array.  At the edges, the filter
    repeats the edge values.

    Args:
        x (numpy array or tensor): The array.
        max_block_gb (float, optional): The approximate memory, in GB, that the filter may use
            at once.  The function processes the array in blocks along its first axis.
            Defaults to 4.0.
        return_min_max (bool, optional): If True, the function also returns the minimum and
            the maximum of each 3x3x3 neighborhood.  Defaults to False.

    Returns:
        The filtered array, with the shape and type of ``x``.  If ``return_min_max`` is True,
        the function returns the tuple ``(median, min, max)``.
    """
    import torch.nn.functional as F
    from .tomography_model import _resolve_device

    # Each output voxel needs its 26 neighbors, so the function does not accept a Shards.
    _sharding.reject_shards('median_filter3d', x=x)
    was_numpy = not isinstance(x, torch.Tensor)
    if was_numpy:
        xt = torch.as_tensor(np.asarray(x), device=_resolve_device('auto'))
    else:
        xt = x
    d0, d1, d2 = xt.shape
    x_gb = xt.numel() * 4 / (1024**3)
    num_blocks = int(np.ceil(27 * x_gb / max_block_gb))
    block_size = max(d0 // max(num_blocks, 1), 1)

    # Padding by 1 on every axis repeats the edge values.
    xp = F.pad(xt[None, None], (1, 1, 1, 1, 1, 1), mode='replicate')[0, 0]

    # This pads the end of axis 0 to a multiple of block_size.
    n_blocks = (d0 + block_size - 1) // block_size
    padded_Z = n_blocks * block_size
    pad_extra = padded_Z - d0
    if pad_extra > 0:
        xp = F.pad(xp[None, None], (0, 0, 0, 0, 0, pad_extra), mode='replicate')[0, 0]

    med_blocks, min_blocks, max_blocks = [], [], []
    with torch.no_grad():
        for i in range(n_blocks):
            z0 = i * block_size
            block = xp[z0:z0 + block_size + 2]

            patches = [
                torch.roll(block, shifts=(dz, dy, dx), dims=(0, 1, 2))
                for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            ]
            stacked = torch.stack(patches, dim=0)
            filtered = torch.median(stacked, dim=0).values
            med_blocks.append(filtered[1:-1, 1:-1, 1:-1])
            if return_min_max:
                min_blocks.append(torch.min(stacked, dim=0).values[1:-1, 1:-1, 1:-1])
                max_blocks.append(torch.max(stacked, dim=0).values[1:-1, 1:-1, 1:-1])

    def stitch(blocks):
        out = torch.cat(blocks, dim=0)[:d0]
        return out.cpu().numpy() if was_numpy else out

    if return_min_max:
        return stitch(med_blocks), stitch(min_blocks), stitch(max_blocks)
    return stitch(med_blocks)

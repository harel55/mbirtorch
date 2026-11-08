"""This module reads hyperspectral neutron data from TIFF stacks and hsnt HDF5 files.  It also
checks the data and converts the data to the hsnt HDF5 format.

A TIFF stack is a directory with one image per wavelength bin.  A multi-view stack is a
directory with one such subdirectory per view.  A stack of counts needs an open beam stack to
compute the transmission ratio.  The module reads the data a block of bins at a time.  It can
apply two optional corrections to each block.  The first correction smooths the open beam.  The
second correction calibrates the background from boxes that contain no sample.
"""
import json
import logging
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .denoise import _clean_transmission, _warn_cleaning
from .io import (ALLOWED_KEYS, _create_hyperspectral, _data_selection, _decode, _find_data_group, _write_metadata,
                 _written_atomically)

log = logging.getLogger(__name__)

INPUT_TYPES = ("counts", "transmission", "attenuation")

# An open beam value below this fraction of the median of its bin is treated as zero counts.
_OPEN_BEAM_FLOOR = 1e-6

# This dict holds the presets for background_boxes.  Each box is (y0, y1, x0, x1) in full-resolution pixels and must
# contain no sample.  The tiles (rows, cols) split the detector into regions calibrated separately.
_BACKGROUND_PRESETS = {
    # ORNL SNAP has four 256 x 256 chips.  Each chip is one tile with a 100 x 100 box in its outer corner.
    "ornl-snap": dict(boxes=((10, 110, 10, 110), (10, 110, 410, 510), (410, 510, 10, 110), (410, 510, 410, 510)),
                      tiles=(2, 2), shape=(512, 512)),
}


class InputError(ValueError):
    """This class is used to report bad input data or options in a one-line message."""


@dataclass
class Check:
    level: str          # 'ok', 'warn', or 'error'.
    message: str


@dataclass
class Dataset:
    """This class is used to hold a loaded dataset and the information reported about it."""
    T: np.ndarray                       # Transmission ratio, (pixels, bins), float32.
    dataset_type: str                   # Type of the source data: counts, transmission, or attenuation.
    spatial_shape: tuple                # (views, rows, cols) after view selection and downsampling.
    bin_indices: np.ndarray             # Source spectral index of the first bin in each column of T.
    dose: float | None                  # Open beam counts per pixel and bin, or None if unknown.
    source: str
    checks: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    dose_per_bin: np.ndarray | None = None      # Median open beam count in each bin, if measured.
    bin_weights: np.ndarray | None = None       # Given open beam count of each bin over its mean, or None.
    open_beam_observations: float = 0           # Effective number of open beam observations, or 0 if none.
    metadata: dict = field(default_factory=dict)    # hsnt metadata of an HDF5 input, cut to the selection.

    @property
    def pixels(self):
        return self.T.shape[0]

    @property
    def bins(self):
        return self.T.shape[1]


def _frac(mask):
    return float(np.mean(mask))


def _sample_stride(P, K):
    """This function returns the row stride that samples about 4e6 entries of a (P, K) matrix.
    The samples are whole rows."""
    return max(1, -(-P * K // 4_000_000))


def _stats(s):
    """This function returns the minimum, maximum, mean, and median of the finite values of s."""
    finite = s if np.isfinite(s).all() else s[np.isfinite(s)]
    have = finite.size > 0
    return dict(min=float(finite.min()) if have else float("nan"), max=float(finite.max()) if have else float("nan"),
                mean=float(finite.mean(dtype=np.float64)) if have else float("nan"),
                median=float(np.median(finite)) if have else float("nan"))


def infer_input_type(a, source_dtype=None):
    """This function infers whether an array holds counts, transmissions, or attenuations.  It
    returns the tuple (type, reason).

    Integer values are counts.  Data with negative values are attenuations.  Data with values up
    to 1.05 are transmissions.  For other nonnegative non-integer data, the function raises
    ``InputError``, which is a subclass of ValueError.
    """
    s = np.asarray(a).reshape(-1)[:: max(1, a.size // 200_000)]
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise InputError("the data have no finite values to infer the input type from; pass --input-type")
    lo, hi, med = float(s.min()), float(s.max()), float(np.median(s))
    if source_dtype is not None and np.dtype(source_dtype).kind in "iu":
        return "counts", f"integer source dtype {source_dtype}: counts"
    if lo >= 0 and np.array_equal(s, np.round(s)):
        return "counts", f"nonnegative integer values up to {hi:.3g}: counts"
    if lo < 0:
        return "attenuation", f"negative values (min {lo:.3g}): attenuation"
    if hi <= 1.05:
        return "transmission", f"values in [{lo:.3g}, {hi:.3g}]: transmission ratios"
    raise InputError(f"cannot tell whether the values (min {lo:.3g}, median {med:.3g}, max {hi:.3g}) are "
                     "transmissions or attenuations: pass --input-type transmission or --input-type attenuation")


def _check_elems(budget_mib):
    """This function returns the number of entries per block in the data checks for a memory
    budget in MiB.  The result is at least 2^20."""
    return max(2**20, int(budget_mib * 2**20) // 4)


def _summary_from_T(T, dose, spatial_shape, background=None, n_obs=0, chunk_elems=2**24, dose_per_bin=None,
                    smoothing=None):
    """This function computes the quantities that the data checks need from a transmission matrix
    in memory."""
    P, K = T.shape
    rows = max(1, chunk_elems // max(K, 1))
    pos_px, mean_px = np.empty(P, dtype=np.int64), np.empty(P, dtype=np.float64)
    pos_bin = np.zeros(K, dtype=np.int64)
    bin_min, bin_max = np.full(K, np.inf, dtype=T.dtype), np.full(K, -np.inf, dtype=T.dtype)
    nonfinite = negative = zero = above = 0
    for i in range(0, P, rows):
        block = T[i:i + rows]
        nonfinite += block.size - np.count_nonzero(np.isfinite(block))
        negative += np.count_nonzero(block < 0)
        zero += np.count_nonzero(block == 0)
        above += np.count_nonzero(block > 1)
        pos = block > 0
        pos_px[i:i + rows] = pos.sum(1)
        pos_bin += pos.sum(0)
        np.minimum(bin_min, block.min(0), out=bin_min)
        np.maximum(bin_max, block.max(0), out=bin_max)
        mean_px[i:i + rows] = block.mean(1, dtype=np.float64)
    n, stride = max(T.size, 1), _sample_stride(P, K)
    st = dict(_stats(T[::stride]), nonfinite=nonfinite / n, negative=negative / n, zero=zero / n, sampled=stride > 1)
    dead_bins = int((pos_bin == 0).sum())
    return dict(pixels=P, bins=K, nbytes=T.nbytes, stats=st, above_one=above / n, dead_px=_frac(pos_px == 0),
                dead_bins=dead_bins, const_bins=int((bin_max == bin_min).sum()) - dead_bins, dose=dose,
                bright=_bright_level(mean_px, spatial_shape, T[:: max(1, P // 20000)], dose, n_obs, dose_per_bin,
                                     chunk_elems, smoothing), background=background)


def _ratio_expectation(mu):
    """This function returns E[mu / S'] for S ~ Poisson(mu).  Here S' equals S, except that a zero
    is replaced by the median of S given S > 0.  The result is the mean bias that comes from
    dividing by a noisy open beam."""
    from scipy.special import expi
    mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-3)
    out = (1 + 1 / mu + 2 / mu ** 2 + 6 / mu ** 3).reshape(-1)
    m = mu.reshape(-1)
    exact, few = m <= 400, m < 40
    out[exact] = m[exact] * np.exp(-m[exact]) * (expi(m[exact]) - np.euler_gamma - np.log(m[exact]))
    p0 = np.exp(-m[few])
    cdf = p0[:, None] * (1 + np.cumsum(np.cumprod(m[few, None] / np.arange(1, 81), axis=1), axis=1))
    median = 1 + np.argmax(cdf >= (1 + p0[:, None]) / 2, axis=1)                  # Median of S given S > 0.
    out[few] += p0 * m[few] / median
    return out.reshape(mu.shape)


def _smoothed_ratio_expectation(m, width, side=256, seed=0):
    """This function returns E[m / O'], where O' is a smoothed open beam of Poisson(m) pixels.  It
    estimates the value by a simulation with a fixed seed."""
    m = np.maximum(np.asarray(m, dtype=np.float64).ravel(), 1e-3)
    rng = np.random.default_rng(seed)
    out = np.empty(m.size)
    for i in range(0, m.size, 8):
        mi = m[i:i + 8]
        ob = _smooth_open_beam(rng.poisson(mi, size=(1, side, side, mi.size)).astype(np.float32), width)
        ob = ob.reshape(-1, mi.size).astype(np.float64)
        med = np.nan_to_num(np.nanmedian(np.where(ob > 0, ob, np.nan), axis=0), nan=1.0)
        out[i:i + 8] = (mi / np.where(ob > 0, ob, med)).mean(0)
    return out


def _expected_ratio(dose_per_bin, n_obs, smoothing=None):
    """This function returns (low, high), the expected range of transmission in a region with no
    sample.  The range differs from 1 because the data are divided by a noisy open beam.  The function
    returns (1, 1) for an exact open beam or an unknown dose."""
    if not n_obs or dose_per_bin is None:
        return 1.0, 1.0
    medians, weight = np.unique(n_obs * np.asarray(dose_per_bin, dtype=np.float64).ravel(), return_counts=True)
    if smoothing is None:
        e = _ratio_expectation(medians[:, None] + np.linspace(-1, 1, 9))
    else:
        m = medians[:, None] * float((_smoothing_kernel(smoothing["width"]) ** 2).sum()) + np.linspace(-1, 1, 9)
        m = np.maximum(m, 1e-3)
        lo, hi = float(m.min()), float(m.max())
        grid = np.geomspace(lo, hi, int(np.clip(np.ceil(np.log(hi / lo) / np.log(1.1)), 1, 96)) + 1)
        e = np.interp(np.log(m), np.log(grid), _smoothed_ratio_expectation(grid, smoothing["width"]))
    return float(np.average(e.min(1), weights=weight)), float(np.average(e.max(1), weights=weight))


def _bright_level(pixel_means, spatial_shape, sub, dose=None, n_obs=0, dose_per_bin=None, chunk_elems=2**24,
                  smoothing=None):
    """This function returns the tuple (level, low, high).  The value level is the transmission of
    the most transparent regions.  The values low and high bound the expected transmission in a
    region with no sample, with a margin for noise."""
    V, rows, cols = spatial_shape
    b = 4 if rows >= 16 and cols >= 16 else 1
    m = pixel_means.reshape(V, rows, cols)[:, :rows // b * b, :cols // b * b]
    m = m.reshape(V, rows // b, b, cols // b, b).mean((2, 4)).ravel()
    if not m.size:
        return float("nan"), float("nan"), float("nan")
    level, noise = float(np.percentile(m, 95)), 0.0
    if sub.shape[0] and sub.shape[1] > 1:
        means = sub.mean(1)
        top = np.flatnonzero(means >= np.quantile(means, 0.9))
        step = max(1, chunk_elems // (8 * sub.shape[1]))
        per_row = []
        for i in range(0, top.size, step):
            d = np.diff(sub[top[i:i + step]].astype(np.float64), axis=1)
            per_row.append(np.mean(np.square(d, out=d), axis=1))
        sigma = np.sqrt(np.median(np.concatenate(per_row)) / 2)
        noise = float(sigma / np.sqrt(sub.shape[1] * b * b))
    lo, hi = _expected_ratio(dose_per_bin if dose_per_bin is not None or dose is None else [dose], n_obs, smoothing)
    margin = 0.03 + 5 * noise
    return level, lo - max(0.05, margin), hi + margin


def _checks_from_summary(sm, spatial_shape):
    """This function returns the standard data checks for a summary as a list of ``Check`` objects."""
    st, c = sm["stats"], []
    P, K = sm["pixels"], sm["bins"]
    c.append(Check("ok", f"{P:,} pixels x {K:,} bins ({spatial_shape[0]} view(s) x {spatial_shape[1]} x "
                         f"{spatial_shape[2]}), {sm['nbytes'] / 2**30:.2f} GiB as float32"))
    if st["nonfinite"] > 0:
        c.append(Check("error", f"{st['nonfinite']:.2%} of T is NaN or inf; the loader should have replaced these"))
    if st["negative"] > 0:
        c.append(Check("error", f"{st['negative']:.2%} of T is negative; the loader should have clipped these"))
    above, bg = sm["above_one"], sm.get("background")
    bright, low, high = sm.get("bright", (float("nan"),) * 3)
    if bg is not None:
        c.append(Check("ok", f"background {'calibrated at conversion' if bg.get('recorded') else 'boxes'} "
                             f"({bg['name']}, {len(bg['boxes'])} box(es), {bg['tiles'][0]} x {bg['tiles'][1]} tile(s)) "
                             f"read T = {bg['factor_median']:.4f} (tiles "
                             f"{', '.join(f'{x:.4f}' for x in bg['tile_medians'])}"
                             + (f"; views {min(bg['view_medians']):.4f} to {max(bg['view_medians']):.4f}"
                                if bg.get("view_medians") else "")
                             + "): each tile's data are divided by it, per view and bin, and the dose is the "
                             + ("views' mean" if bg.get("view_medians") else "sample's")))
        spread = bg.get("view_medians")
        if spread and max(spread) > 1.05 * min(spread):
            c.append(Check("warn", f"the views' exposures differ ({min(spread):.3f} to {max(spread):.3f} of the open "
                                   "beam's); each view is calibrated, but the dose, the chi-square and support "
                                   "selection use the views' mean"))
    elif bright > high:
        c.append(Check("warn", f"the most transparent regions read T = {bright:.3f}, above the {high:.3f} a matched "
                               "exposure reaches with this noise and open beam: the sample run and the open beam "
                               "differ in exposure (or the open beam is low); calibrate with --background-boxes"))
    elif bright < low:
        c.append(Check("warn", f"the most transparent regions read T = {bright:.3f}, below the {low:.3f} a matched "
                               "exposure reaches with this noise and open beam: if any pixels are free of the sample, "
                               "the sample run and the open beam differ in exposure; calibrate with "
                               "--background-boxes"))
    smooth = sm.get("smoothing")
    if smooth is not None:
        w, ratio, ind, measured = smooth["width"], smooth["variance_ratio"], smooth["independent_ratio"], \
            smooth["measured"]
        c.append(Check("ok" if measured else "warn",
                       f"open beam smoothed {w} x {w}{' at conversion' if smooth.get('recorded') else ''}: its "
                       f"variance x {ratio:.3f}"
                       + (f" across the observations (x {ind:.3f} if the pixels' noise were independent)" if measured
                          else " (independent pixels; one observation)")
                       + f", so it counts as {smooth['observations']:.3g} observations"
                       + ("; the detector's noise is correlated between neighboring pixels, so smoothing gains little"
                          if measured and ratio > 2 * ind else "")
                       + ("" if measured else ": one observation cannot measure the reduction, and on a detector whose "
                          "noise is correlated between neighboring pixels the independent-pixel value overstates the "
                          "observations")))
    if above > 0.5:
        c.append(Check("warn", f"{above:.1%} of T exceeds 1: the open beam may be too low or the sample missing"))
    elif above > 0:
        c.append(Check("ok", f"{above:.1%} of T exceeds 1"))
    if st["zero"] > 0.5:
        c.append(Check("warn", f"{st['zero']:.1%} of T is exactly zero: very low dose or a mostly opaque sample"))
    elif st["zero"] > 0:
        c.append(Check("ok", f"{st['zero']:.2%} of T is exactly zero (zero counts; the likelihood handles them)"))
    if sm["dead_px"] > 0:
        c.append(Check("warn", f"{sm['dead_px']:.2%} of pixels are zero in every bin (dead detector pixels or a mask)"))
    if sm["dead_bins"]:
        c.append(Check("warn", f"{sm['dead_bins']} bins are zero in every pixel; consider --wave-range to drop them"))
    if sm["const_bins"] > 0:
        c.append(Check("warn", f"{sm['const_bins']} bins are constant across pixels"))
    dose = sm["dose"]
    if dose is not None:
        if dose < 1:
            c.append(Check("warn", f"open-beam dose {dose:.3g} counts per pixel and bin is below 1: expect mostly "
                                   "zero counts"))
        else:
            c.append(Check("ok", f"dose {dose:.3g} open-beam counts per pixel and (binned) bin" +
                           (", scaled to the sample run's exposure" if bg is not None else "")))
    else:
        c.append(Check("warn", "dose unknown: --dose is needed for support selection"))
    if K > P:
        c.append(Check("warn", f"more bins ({K}) than pixels ({P}): the spectra are poorly determined; use "
                               "--downsample less or --wave-bin more"))
    return c


def _natural_key(name):
    """This function returns a sort key that orders runs of digits by value.  For example,
    img_2 comes before img_19."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _tif_names(directory):
    """This function lists the .tif and .tiff paths in a directory in natural order.  The
    extension match ignores case."""
    names = [f for f in os.listdir(directory) if f.lower().endswith((".tif", ".tiff"))]
    return [os.path.join(directory, f) for f in sorted(names, key=_natural_key)]


def _tif_files(directory):
    """This function returns (files, None) for a directory of TIFF files.  It returns
    (None, subdirs) for a directory of TIFF subdirectories."""
    files = _tif_names(directory)
    if not files:
        subdirs = [os.path.join(directory, d) for d in sorted(os.listdir(directory), key=_natural_key)
                   if os.path.isdir(os.path.join(directory, d)) and _tif_names(os.path.join(directory, d))]
        if subdirs:
            return None, subdirs
        raise FileNotFoundError(f"no .tif/.tiff files in {directory}")
    idx = [int(m.group(1)) if (m := re.search(r"(\d+)\.tiff?$", os.path.basename(f), re.IGNORECASE)) else None
           for f in files]
    if all(i is not None for i in idx):
        gaps = [(a, b) for a, b in zip(idx, idx[1:]) if b != a + 1]
        if gaps:
            warnings.warn(f"file indices are not consecutive in {directory}: {len(gaps)} gap(s), first at {gaps[0]}")
    else:
        warnings.warn(f"some file names in {directory} carry no trailing index; relying on sorted order")
    return files, None


def _open_beam_dirs(paths):
    """This function returns the open beam stacks in paths.  It replaces a directory of
    observation subdirectories by those subdirectories."""
    dirs = []
    for p in paths:
        files, subdirs = _tif_files(p)
        dirs += subdirs if files is None else [p]
    return dirs


class _TiffBlocks:
    """This class is used to read bins k0:k1 of a TIFF stack as float32 with shape (views, rows,
    cols, k1 - k0).  The stack has one image per bin.  With multi_view, each subdirectory is one
    view."""

    def __init__(self, directory, wave_range, downsample, workers, desc, views=None, multi_view=False):
        import tifffile
        files, subdirs = _tif_files(directory)
        if files is None and not multi_view:
            raise InputError(f"{directory} holds only subdirectories ({len(subdirs)}); pass one of them, or pass the "
                             "parent as the open beam to average them")
        dirs = [directory] if files is not None else subdirs
        chosen = dirs[slice(*views)] if views else dirs
        if not chosen:
            raise InputError(f"--views {views[0]}:{views[1]} selects none of the {len(dirs)} view(s) in {directory}")
        per_view = [files] if files is not None else [_tif_files(d)[0] for d in chosen]
        counts = sorted({len(f) for f in per_view})
        if len(counts) > 1:
            raise InputError(f"the view directories of {directory} hold different numbers of images ({counts})")
        self.view_files = [f[slice(*wave_range)] if wave_range else f for f in per_view]
        self.files = self.view_files[0]
        if not self.files:
            raise InputError(f"the wave range selects no files of the {counts[0]} in {directory}")
        self.first_index = per_view[0].index(self.files[0])
        self.columns = counts[0]                    # The number of bins in the input, before wave_range.
        with tifffile.TiffFile(self.files[0]) as t:
            page = t.pages[0]
            self.full_shape, self.source_dtype = tuple(page.shape), str(page.dtype)
        if len(self.full_shape) != 2:
            raise InputError(f"{self.files[0]}: expected a 2-D image per wavelength bin, got shape {self.full_shape}")
        self.downsample, self.workers, self.desc = downsample, workers, desc
        self.rows, self.cols = np.empty(self.full_shape, dtype=bool)[::downsample, ::downsample].shape
        self.views = len(self.view_files)
        self.file_type, self.file_dose, self.file_dose_per_bin, self.file_observations = None, None, None, 0
        self.file_wave_bin, self.file_background, self.file_smoothing = 1, None, None
        self.source_bins = self.first_index + np.arange(len(self.files))
        self.file_metadata = {}
        self.metadata = dict(files=len(self.files), first_file=os.path.basename(self.files[0]),
                             last_file=os.path.basename(self.files[-1]))
        if files is None:
            self.metadata["view_directories"] = [os.path.basename(d) for d in chosen]

    @property
    def bins(self):
        return len(self.files)

    def read(self, k0, k1, full=False):
        """This method returns bins k0:k1 as float32 with shape (views, rows, cols, bins).  It
        returns the downsampled pixels, or all pixels if full is True."""
        import tifffile
        stack = []
        for files in self.view_files:
            sel = files[k0:k1]
            arr = tifffile.imread(sel[0])[None] if len(sel) == 1 else tifffile.imread(sel, ioworkers=self.workers,
                                                                                       maxworkers=1)
            if arr.ndim != 3 or tuple(arr.shape[1:]) != self.full_shape:
                raise InputError(f"{self.desc}: images {k0}..{k1 - 1} have shape {arr.shape[1:]}, the first image "
                                 f"{self.full_shape}")
            stack.append(arr)
        arr = stack[0][None] if len(stack) == 1 else np.stack(stack)                  # (views, bins, rows, cols)
        if full:
            return np.moveaxis(arr.astype(np.float32, copy=False), 1, -1)
        arr = arr[:, :, ::self.downsample, ::self.downsample]
        return np.ascontiguousarray(np.moveaxis(arr, 1, -1), dtype=np.float32)

    def close(self):
        pass


class _Hdf5Blocks:
    """This class is used to read bins k0:k1 of an hsnt HDF5 dataset as float32 blocks."""

    def __init__(self, path, dataset, views, wave_range, downsample):
        import h5py
        self.f = h5py.File(path, "r")
        try:
            g, self.gname = _find_data_group(self.f, dataset)
        except (KeyError, ValueError) as e:
            self.f.close()
            raise InputError(f"{path}: {e.args[0]}") from None
        self.d = d = g["data"]
        self.file_type = _decode(g["dataset_type"][()]) if "dataset_type" in g else None
        self.sel, (self.views, self.rows, self.cols) = _data_selection(d.shape, views, downsample)
        self.sel_full = _data_selection(d.shape, views, 1)[0]
        self.full_shape = tuple(d.shape[1:3]) if d.ndim == 4 else tuple(d.shape[:2]) if d.ndim == 3 else (d.shape[0], 1)
        self.source_dtype = str(d.dtype)
        source = _file_source_bins(g, d.shape[-1])
        ks = _columns_in_range(source, wave_range)
        self.first_index, self.bins = (ks.start, len(ks)) if len(ks) else (0, 0)
        self.columns = len(source)                  # The number of bins in the input, before wave_range.
        self.source_bins = source[ks.start:ks.stop]
        attrs = dict(self.f.attrs, **g.attrs)
        per_bin = g["open_beam_dose"][()] if "open_beam_dose" in g and g["open_beam_dose"].shape == source.shape \
            else None
        self.file_dose_per_bin = None if per_bin is None else np.asarray(per_bin, dtype=np.float64)[ks.start:ks.stop]
        dose = attrs.get("dose")                                               # Per file column, or -1 if unknown.
        self.file_dose = float(dose) if dose is not None and float(dose) > 0 else None
        if self.file_dose is not None and self.file_dose_per_bin is not None and 0 < len(ks) < len(source):
            self.file_dose = float(np.median(self.file_dose_per_bin))          # The dose of the selected columns.
        self.file_observations = float(attrs.get("open_beam_observations", 0))
        self.file_wave_bin = max(1, int(attrs.get("wave_bin", 1)))                 # Source bins per column.
        self.file_background = _json_attr(attrs, "background")                     # Written by convert_to_hdf5.
        self.file_smoothing = _json_attr(attrs, "open_beam_smoothing")
        self.file_metadata = {k: g[k][()] for k in ALLOWED_KEYS
                              if k in g and k not in ("dataset_type", "dataset_modality")}
        self.desc = f"{path}:{self.gname}"
        scalars = {k: g[k][()] for k in g if isinstance(g[k], h5py.Dataset) and k not in ("data", "dataset_type")
                   and g[k].ndim == 0}
        self.metadata = dict(hdf5_scalar_metadata={k: str(v) for k, v in scalars.items()})
        log.info("HDF5 %s: dataset %s/data shape %s dtype %s chunks %s", path, self.gname.rstrip("/"), d.shape, d.dtype,
                 d.chunks)

    def read(self, k0, k1, full=False):
        """This method returns bins k0:k1 as float32 with shape (views, rows, cols, bins).  It
        returns the downsampled pixels, or all pixels if full is True."""
        a = self.d[(self.sel_full if full else self.sel) + (slice(self.first_index + k0, self.first_index + k1),)]
        if self.d.ndim == 3:
            a = a[None]
        elif self.d.ndim == 2:
            a = a[None, :, None, :]
        return np.ascontiguousarray(a, dtype=np.float32)

    def close(self):
        self.f.close()


def _json_attr(attrs, key):
    """This function returns a JSON-encoded attribute as a dict.  It returns None if the attribute
    is absent or is not a JSON dict."""
    try:
        v = json.loads(_decode(attrs[key])) if key in attrs else None
    except (TypeError, ValueError):
        return None
    return v if isinstance(v, dict) else None


def _file_source_bins(g, n_columns):
    """This function returns the source bin of each column.  It returns bin_indices if they are
    increasing.  Otherwise it returns the column indices."""
    if "bin_indices" in g and g["bin_indices"].shape == (n_columns,):
        b = np.asarray(g["bin_indices"][()], dtype=np.int64)
        if n_columns < 2 or bool(np.all(np.diff(b) > 0)):
            return b
        warnings.warn("bin_indices are not increasing; --wave-range indexes the file's columns")
    return np.arange(n_columns)


def _columns_in_range(source_bins, wave_range):
    """This function returns the range of columns whose source bin lies in wave_range.  If
    wave_range is None, it returns every column."""
    if not wave_range:
        return range(len(source_bins))
    lo, hi, _ = slice(*wave_range).indices(int(source_bins[-1]) + 1 if len(source_bins) else 0)
    return range(int(np.searchsorted(source_bins, lo)), int(np.searchsorted(source_bins, max(lo, hi))))


def _selected_metadata(meta, views, wave_cols, wave_bin, downsample):
    """This function returns the hsnt metadata adjusted to a selection of views and columns and to
    the downsampling."""
    out = {}
    for k, v in meta.items():
        v = _decode(v)
        if isinstance(v, np.ndarray) and v.shape == ():
            v = v.item()
        if k == "angles" and isinstance(v, np.ndarray) and v.ndim == 1:
            v = v[slice(*views)] if views else v
        elif k == "wavelengths" and isinstance(v, np.ndarray) and v.ndim == 1:
            v = v[wave_cols.start:wave_cols.stop] if wave_cols is not None else v
            v = _bin_spectral(v.astype(np.float64), wave_bin, "mean")
        elif k in ("delta_det_channel", "delta_det_row") and isinstance(v, (int, float)):
            v = v * downsample
        out[k] = v
    return out


def _open_source(path, dataset, views, wave_range, downsample, workers, open_beam):
    """This function returns the block reader for a TIFF directory or an HDF5 file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"input not found: {path}")
    if os.path.isdir(path):
        src = _TiffBlocks(path, wave_range, downsample, workers, desc="sample", views=views, multi_view=True)
    elif path.lower().endswith((".h5", ".hdf5", ".hdf")):
        if open_beam:
            warnings.warn("the open beam is ignored for HDF5 input")
        src = _Hdf5Blocks(path, dataset, views, wave_range, downsample)
        if src.bins == 0:
            src.close()
            raise InputError("the wave range selects no bins")
    else:
        raise InputError(f"{path}: not a directory of TIFFs and not an .h5/.hdf5 file")
    log.info("%s: %d view(s) x %d x %d pixels%s x %d bins, source dtype %s", src.desc, src.views, src.rows, src.cols,
             f" (every {downsample}th row and column of {src.full_shape[0]} x {src.full_shape[1]})" if downsample > 1
             else "", src.bins, src.source_dtype)
    return src


def _open_beam_blocks(paths, wave_range, downsample, workers, sample):
    """This function opens each open beam observation.  It checks that each observation has the
    same shape as the sample."""
    obs = [_TiffBlocks(d, wave_range, downsample, workers, desc=f"open beam {os.path.basename(d)}")
           for d in _open_beam_dirs(paths)]
    for o in obs:
        if (o.bins, o.rows, o.cols) != (sample.bins, sample.rows, sample.cols):
            raise InputError(f"{o.desc} has {o.bins} bins of {o.full_shape}, the sample {sample.bins} of "
                             f"{sample.full_shape}")
    return obs


def _open_beam_mean(obs, k0, k1):
    """This function returns the mean open beam of bins k0:k1 with shape (pixels, bins).  It
    returns None if there is no open beam."""
    ob = None
    for o in obs:
        blk = o.read(k0, k1)
        blk = blk.reshape(-1, blk.shape[-1])
        if ob is None:
            ob = blk
        else:
            ob += blk
    if ob is not None:
        ob /= len(obs)
    return ob


@dataclass
class _Background:
    """This class is used to hold the boxes with no sample that calibrate each detector tile in
    each bin."""
    name: str
    boxes: tuple                        # (y0, y1, x0, x1) in full-resolution pixels.
    tiles: tuple                        # (tile rows, tile cols).
    box_tile: np.ndarray                # The tile of each box.
    pixel_tile: np.ndarray              # The tile of each downsampled pixel of a view, in row-major order.


def _background(spec, tiles, src, downsample):
    """This function returns the background calibration from a preset name or from a list of
    boxes.  Each box is (y0, y1, x0, x1) in full-resolution pixels.  If spec is None, the function
    returns None.  Each tile of the detector must contain a box."""
    if spec is None:
        if tiles is not None:
            raise InputError("background tiles need background boxes (--background-boxes)")
        return None
    if len(src.full_shape) != 2 or src.full_shape[1] == 1:
        raise InputError("background boxes need image data (rows, cols), not a (pixels, bins) table")
    if isinstance(spec, str):
        if spec not in _BACKGROUND_PRESETS:
            raise InputError(f"unknown background preset {spec!r}; presets: {', '.join(_BACKGROUND_PRESETS)}")
        preset = _BACKGROUND_PRESETS[spec]
        if tuple(src.full_shape) != preset["shape"]:
            raise InputError(f"the {spec} preset is for {preset['shape'][0]} x {preset['shape'][1]} images; the "
                             f"input's are {src.full_shape[0]} x {src.full_shape[1]}")
        boxes, tiles, name = preset["boxes"], tuple(tiles or preset["tiles"]), spec
    else:
        boxes, tiles, name = tuple(tuple(int(v) for v in b) for b in spec), tuple(tiles or (1, 1)), "given"
    rows, cols = src.full_shape
    for y0, y1, x0, x1 in boxes:
        if not (0 <= y0 < y1 <= rows and 0 <= x0 < x1 <= cols):
            raise InputError(f"background box {y0}:{y1},{x0}:{x1} lies outside the {rows} x {cols} image")
    tr, tc = tiles
    if tr < 1 or tc < 1 or tr > rows or tc > cols:
        raise InputError(f"background tiles {tr} x {tc} do not fit a {rows} x {cols} image")
    y_edges, x_edges = np.linspace(0, rows, tr + 1).astype(int), np.linspace(0, cols, tc + 1).astype(int)

    def tile_of(y, x):
        return (np.searchsorted(y_edges, y, side="right") - 1) * tc + (np.searchsorted(x_edges, x, side="right") - 1)

    box_tile = np.array([tile_of((y0 + y1 - 1) // 2, (x0 + x1 - 1) // 2) for y0, y1, x0, x1 in boxes])
    missing = sorted(set(range(tr * tc)) - set(box_tile.tolist()))
    if missing:
        raise InputError(f"background tile(s) {missing} of the {tr} x {tc} grid hold no box")
    yy, xx = np.meshgrid(np.arange(0, rows, downsample), np.arange(0, cols, downsample), indexing="ij")
    return _Background(name, boxes, tiles, box_tile, tile_of(yy.ravel(), xx.ravel()))


def _box_sums(full, boxes, attenuation=False):
    """This function returns the float64 sum over each box of a full-resolution block.  The result
    has shape (views, boxes, bins)."""
    sums = []
    for y0, y1, x0, x1 in boxes:
        box = full[:, y0:y1, x0:x1]
        if attenuation:
            with np.errstate(over="ignore", invalid="ignore"):
                box = np.exp(-box)
        sums.append(np.nan_to_num(box, nan=0.0, posinf=0.0, neginf=0.0).sum((1, 2), dtype=np.float64))
    return np.stack(sums, 1)


def _smoothing_kernel(width):
    """This function returns the 2D window that smooths the open beam.  The window is the square
    root of the outer product of a Hamming window with itself, normalized to sum to 1."""
    h = np.hamming(width)
    k = np.sqrt(np.outer(h, h))
    return k / k.sum()


def _open_beam_total(obs, bins, block):
    """This function returns the open beam count summed over the selected bins, for each
    full-resolution pixel.  The result has shape (views, rows, cols)."""
    total = None
    for k0 in range(0, bins, block):
        for o in obs:
            s = np.clip(o.read(k0, min(k0 + block, bins), full=True), 0, None).sum(-1, dtype=np.float64)
            total = s if total is None else np.add(total, s, out=total)
    return total


def _dead_mask(total):
    """This function returns a mask of the dead detector pixels of an open beam, with shape (views,
    rows, cols, 1).  It returns None if no pixel is dead or if the counts are too low to tell."""
    if total is None:
        return None
    dead = total <= _OPEN_BEAM_FLOOR * np.median(total)
    if not dead.any():
        return None
    if np.median(total) < 20:
        log.info("open-beam smoothing: %d pixels hold no count, too few counts (median %.3g) to call them dead",
                 int(dead.sum()), float(np.median(total)))
        return None
    return dead[..., None]


def _smooth_open_beam(full, width, dead=None):
    """This function smooths a full-resolution open beam with shape (views, rows, cols, bins).  It
    smooths each image separately and leaves out the dead pixels."""
    from scipy.ndimage import correlate1d
    w = np.sqrt(np.hamming(width))
    w /= w.sum()
    order = np.argsort(full.strides, kind="stable")[::-1]              # The filter runs in memory order.
    rows, cols = int(np.flatnonzero(order == 1)[0]), int(np.flatnonzero(order == 2)[0])

    def smooth(a):
        return correlate1d(correlate1d(a.transpose(order), w, axis=rows, mode="mirror"), w, axis=cols,
                           mode="mirror").transpose(np.argsort(order))

    if dead is None or not dead.any():
        return smooth(full)
    # The weight has one value per pixel, shared by all bins.
    weight = smooth(np.broadcast_to(~dead, full.shape[:-1] + (1,)).astype(np.float32))
    out = smooth(np.where(dead, np.float32(0), full))
    return np.where(weight > 1e-6, out / np.maximum(weight, 1e-6), np.float32(0)).astype(np.float32)


def _smoothing_variances(images, width, dead=None):
    """This function returns (raw, smoothed), the variance across observations before and after
    smoothing.  The ratio smoothed / raw is the variance reduction from smoothing."""
    n = images.shape[0]
    x = images.astype(np.float32).reshape(n * images.shape[1], *images.shape[2:])
    scale = x.reshape(n, -1).mean(1, dtype=np.float64)
    x *= np.repeat(scale.mean() / np.where(scale > 0, scale, 1.0), images.shape[1]).astype(np.float32)[:, None, None,
                                                                                                        None]
    smoothed = _smooth_open_beam(x, width, None if dead is None else np.tile(dead, (n, 1, 1, 1))).reshape(n, -1)
    x = x.reshape(n, -1)
    return float(x.var(0, ddof=1, dtype=np.float64).sum()), float(smoothed.var(0, ddof=1, dtype=np.float64).sum())


def _read_block(src, obs, itype, k0, k1, downsample, smoothing, background, dead=None):
    """This function reads bins k0:k1 of the sample and of the mean open beam at the downsampled
    pixels.  It also returns the box sums and the smoothing variances when they are needed."""
    P = src.views * src.rows * src.cols
    if background is None:
        a, box_a = src.read(k0, k1).reshape(P, k1 - k0), None
    else:
        full = src.read(k0, k1, full=True)
        box_a = _box_sums(full, background.boxes, attenuation=itype == "attenuation")
        a = np.ascontiguousarray(full[:, ::downsample, ::downsample]).reshape(P, k1 - k0)
        del full
    if not obs:
        return a, None, box_a, None, None
    if not smoothing and background is None:
        return a, _for_views(_open_beam_mean(obs, k0, k1), src.views), box_a, None, None
    ob, first = None, []
    n_var = min(k1 - k0, max(1, 2**18 // int(np.prod(src.full_shape))))         # Bins per block used for the variances.
    for o in obs:
        blk = o.read(k0, k1, full=True)
        if smoothing and len(obs) > 1:
            first.append(np.array(blk[..., :n_var]))
        ob = blk if ob is None else np.add(ob, blk, out=ob)
    ob /= len(obs)
    if smoothing:
        ob = _smooth_open_beam(ob, smoothing, dead)
    box_ob = None if background is None else _box_sums(ob, background.boxes)
    variances = _smoothing_variances(np.stack(first), smoothing, dead) if first else None
    ob = _for_views(np.ascontiguousarray(ob[:, ::downsample, ::downsample]).reshape(-1, k1 - k0), src.views)
    return a, ob, box_a, box_ob, variances


def _for_views(ob, views):
    """This function repeats the open beam of one view, with shape (pixels, bins), for every view."""
    return ob if views == 1 else np.tile(ob, (views, 1))


def _tile_factors(box_a, box_ob, itype, wave_bin, background):
    """This function returns the transmission of the boxes of each tile in each grouped bin.  The
    result has shape (views, tiles, bins).  A bin whose boxes hold no signal gets factor 1."""
    n_tiles = background.tiles[0] * background.tiles[1]
    npix = np.array([(y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in background.boxes], dtype=np.float64)
    num = _bin_spectral(box_a, wave_bin, "sum")
    den = _bin_spectral(box_ob, wave_bin, "sum") if box_ob is not None else None
    V, _, Kb = num.shape
    tn, td = np.zeros((V, n_tiles, Kb)), np.zeros((V, n_tiles, Kb))
    for b, t in enumerate(background.box_tile):
        tn[:, t] += num[:, b]
        td[:, t] += den[:, b] if den is not None else npix[b] * wave_bin
    return np.where((td > 0) & (tn > 0), tn / np.maximum(td, 1e-300), 1.0)


def _apply_background(T, factors, background, views):
    """This function divides the transmission of each tile by its factor, in place.  It returns the
    mean factor in each bin, weighted by the number of pixels in each tile."""
    P, Kb = T.shape
    per_view = P // views
    tile = np.tile(background.pixel_tile, views)
    view = np.repeat(np.arange(views), per_view)
    T /= factors.astype(np.float32)[view, tile]
    counts = np.bincount(background.pixel_tile, minlength=factors.shape[1]).astype(np.float64)
    return (factors * counts[None, :, None]).sum((0, 1)) / (counts.sum() * views)


def _transmission_blocks(src, obs, itype, wave_bin, block, downsample, smoothing, background, dead=None):
    """This function yields (j0, j1, T, dose, info, factors) for successive blocks of grouped bins.
    It applies the open beam smoothing and the background calibration.  A second thread reads the
    next block while the caller processes the current one."""
    nb = src.bins
    starts = [k0 for k0 in range(0, nb - nb % wave_bin, block) if min(k0 + block, nb) // wave_bin * wave_bin > k0]

    def read(k0):
        k1 = min(k0 + block, nb) // wave_bin * wave_bin
        return k1, _read_block(src, obs, itype, k0, k1, downsample, smoothing, background, dead)

    pool = ThreadPoolExecutor(1)
    try:
        pending = pool.submit(read, starts[0]) if starts else None
        for n, k0 in enumerate(starts):
            k1, (a, ob, box_a, box_ob, variances) = pending.result()
            pending = pool.submit(read, starts[n + 1]) if n + 1 < len(starts) else None
            T, dose_b, info_b = _stack_to_transmission(a, itype, open_beam=ob, wave_bin=wave_bin)
            del a, ob
            if variances is not None:
                info_b["smoothing_variances"] = variances
            factors = None
            if background is not None:
                factors = _tile_factors(box_a, box_ob, itype, wave_bin, background)
                info_b["mean_factor"] = _apply_background(T, factors, background, src.views)
            yield k0 // wave_bin, k1 // wave_bin, T, dose_b, info_b, factors
    finally:
        pool.shutdown(wait=True)


def _check_smoothing(width, obs):
    """This function checks and returns the smoothing width.  The width must be 0, which means no
    smoothing, or an odd number of at least 3."""
    width = int(width or 0)
    if width and (width < 3 or width % 2 == 0):
        raise InputError(f"the open-beam smoothing width must be odd and at least 3, got {width}")
    if width and not obs:
        warnings.warn("open-beam smoothing is ignored: there is no open beam (the input is not counts)")
        return 0
    return width


def _background_summary(background, factors, mean_factor):
    """This function returns a summary of the calibration for the checks, the report, and the file
    attributes."""
    out = dict(name=background.name, boxes=[list(b) for b in background.boxes], tiles=list(background.tiles),
               factor_median=float(np.median(mean_factor)),
               tile_medians=[float(np.median(factors[:, t])) for t in range(factors.shape[1])],
               factor_range=[float(factors.min()), float(factors.max())])
    if factors.shape[0] > 1:
        out["view_medians"] = [float(np.median(f)) for f in factors]
    return out


def _resolve_input_type(input_type, src, probe, open_beam):
    """This function returns the input type and the reason for it.  The type is given, read from
    the file, or inferred from the data."""
    if input_type != "auto":
        if src.file_type and src.file_type != input_type:
            warnings.warn(f"the given input type {input_type} overrides the file's dataset_type {src.file_type!r}")
        itype, why = input_type, "given"
    elif src.file_type in ("attenuation", "transmission"):
        itype, why = src.file_type, "from the file's dataset_type"
    elif open_beam and isinstance(src, _TiffBlocks):
        itype, why = "counts", "an open beam was given"
    else:
        itype, why = infer_input_type(probe, src.source_dtype)
    if itype == "counts" and not isinstance(src, _TiffBlocks):
        raise InputError(f"the HDF5 input holds counts ({why}); an open beam is read only for TIFF stacks, so store "
                         "the transmission (counts / open beam) or its attenuation, or pass --input-type if the "
                         "values are already normalized")
    if itype == "counts" and not open_beam:
        raise InputError(f"the input holds counts ({why}) but no open beam was given: pass --open-beam DIR, or "
                         "--input-type transmission/attenuation if the values are already normalized")
    if open_beam and itype != "counts" and isinstance(src, _TiffBlocks):
        warnings.warn(f"the open beam is ignored: the input type is {itype}, not counts")
    return itype, why


def _bin_spectral(a, n, how):
    """This function combines each group of n adjacent bins.  It sums the group if how is 'sum'
    and averages it otherwise."""
    if n <= 1:
        return a
    K = a.shape[-1] // n * n
    g = a[..., :K].reshape(*a.shape[:-1], K // n, n)
    return g.sum(-1) if how == "sum" else g.mean(-1)


def _stack_to_transmission(a, input_type, open_beam=None, wave_bin=1):
    """This function converts (pixels, bins) values of the given type to a transmission ratio.  It
    groups adjacent bins if wave_bin > 1.  It returns the tuple (T, dose, info)."""
    info = {}
    if input_type == "counts":
        counts, ob = _bin_spectral(a, wave_bin, "sum"), _bin_spectral(open_beam, wave_bin, "sum")
        # The medians are computed on a subsample of pixels.
        stride = max(1, ob.shape[0] // 8192)
        bad = _no_open_beam_count(ob, stride)
        if bad.any():
            info["open_beam_zero_frac"] = _frac(bad)
            sub = np.where(bad[::stride], np.nan, ob[::stride])
            med = np.nanmedian(sub, axis=0)
            med = np.nan_to_num(med, nan=float(np.nanmedian(sub)) if np.isfinite(sub).any() else 1.0)
            ob = np.where(bad, med[None, :], ob)
        T = counts / ob
        dose = float(np.median(ob.reshape(-1)[:: max(1, ob.size // 1_000_000)]))
        info["dose_per_bin"] = np.median(ob[:: max(1, ob.shape[0] // 8192)], axis=0).astype(np.float64)
    elif input_type == "transmission":
        T, dose = _bin_spectral(a, wave_bin, "mean"), None
    elif input_type == "attenuation":
        with np.errstate(over="ignore", invalid="ignore"):
            T = np.exp(-a.astype(np.float32, copy=False))     # convert_to_hdf5 stores T = 0 as +inf.
        # T is cleaned before binning, so a NaN does not spread to its group.
        info.update(_clean_transmission(T))
        return _bin_spectral(T, wave_bin, "mean"), None, info
    else:
        raise InputError(f"unknown input type {input_type!r}")
    # This copy is needed because the block can be the reader's array.
    T = T.astype(np.float32)
    info.update(_clean_transmission(T))
    return T, dose, info


def _no_open_beam_count(ob, stride):
    """This function returns a mask of the open beam entries that count as zero.  An entry counts
    as zero if it is at most zero or below ``_OPEN_BEAM_FLOOR`` times the median of its bin."""
    bad = ob <= 0
    med = np.nanmedian(np.where(bad[::stride], np.nan, ob[::stride]), axis=0)
    return bad | (ob < _OPEN_BEAM_FLOOR * np.nan_to_num(med, nan=0.0)[None, :])


def _warn_conversion(info):
    if info.get("open_beam_zero_frac"):
        warnings.warn(f"the open beam holds no count (zero, negative, or below {_OPEN_BEAM_FLOOR:g} of the bin's "
                      f"median) in {100 * info['open_beam_zero_frac']:.3g}% of pixel-bins; those use the bin's median "
                      "open beam")
    _warn_cleaning(info)


def _check_host_memory(n_bytes, what):
    """This function raises MemoryError if an allocation exceeds the available CPU memory.  It
    warns if the allocation exceeds half of the available memory."""
    import psutil
    avail = psutil.virtual_memory().available
    if n_bytes > avail:
        raise MemoryError(f"{what} needs {n_bytes / 2**30:.1f} GiB of host memory but {avail / 2**30:.1f} GiB is "
                          "available: use --downsample, --wave-range or --wave-bin, or convert the data first "
                          "(`mbirtorch-hsnt convert` streams)")
    if n_bytes > 0.5 * avail:
        warnings.warn(f"{what} needs {n_bytes / 2**30:.1f} GiB of the {avail / 2**30:.1f} GiB of host memory available")


def _merge_dose(dose, estimate, src, wave_bin):
    """This function returns the dose per grouped bin.  The dose comes from the given dose, the open
    beam estimate, or the dose recorded in an HDF5 input."""
    if dose is None:
        return estimate if estimate is not None or src.file_dose is None else src.file_dose * wave_bin
    if estimate is None and src.file_dose is not None:
        _check_given_dose(dose, src)
    dose = dose * wave_bin * src.file_wave_bin
    if estimate is not None and abs(dose - estimate) / estimate > 0.5:
        warnings.warn(f"the given dose {dose:.3g} differs from the open-beam estimate {estimate:.3g} by more than 50%")
    return dose


def _split_dose(dose, src, wave_bin):
    """This function splits a given dose into a scalar dose and the weight of each grouped bin.  A
    scalar or None is returned unchanged with no weights.  An array must hold the open beam count
    per pixel of each bin of the input, before wave_range.  The function returns the mean of the
    selected bins as the dose and the grouped counts divided by their mean as the weights."""
    if dose is None or np.ndim(dose) == 0:
        return dose, None
    d = np.asarray(dose, dtype=np.float64)
    if d.ndim != 1 or d.shape[0] != src.columns:
        raise InputError(f"a per-bin dose must hold one open-beam count per bin of the input ({src.columns}), got "
                         f"shape {d.shape}")
    if not (np.isfinite(d).all() and (d > 0).all()):
        raise InputError("a per-bin dose must be finite and positive in every bin")
    selected = d[src.first_index:src.first_index + src.bins // wave_bin * wave_bin]
    grouped = _bin_spectral(selected, wave_bin, "sum")
    return float(selected.mean()), grouped / grouped.mean()


def _check_given_dose(dose, src):
    """This function warns if a given dose differs by more than 50% from the open beam dose recorded
    in an HDF5 input."""
    factor = src.file_background["factor_median"] if src.file_background is not None else 1.0
    open_beam = src.file_dose / factor / src.file_wave_bin                 # Per source bin, before the calibration.
    scaled = open_beam * factor
    note = (f" (the file's dose over its calibration factor {factor:.4f}): --dose is the open beam's count, which the "
            "calibration scales, so the scaled dose inspect prints would be scaled twice") if factor != 1.0 else ""
    if abs(dose - open_beam) > 0.5 * open_beam:
        warnings.warn(f"the given dose {dose:.3g} differs by more than 50% from the open-beam dose the file records, "
                      f"{open_beam:.3g} per source bin{note}")
    elif abs(dose - scaled) < abs(dose - open_beam):
        warnings.warn(f"the given dose {dose:.3g} is nearer the file's dose at the sample run's exposure, "
                      f"{scaled:.3g} per source bin, than its open-beam dose, {open_beam:.3g}{note}")


def _dose_per_bin(dose, estimate, src, wave_bin):
    """This function returns the open beam count per grouped bin.  It returns None if the dose is
    given."""
    if dose is not None:
        return None
    if estimate is not None:
        return estimate
    return None if src.file_dose_per_bin is None else _bin_spectral(src.file_dose_per_bin, wave_bin, "sum")


def _probe_type(input_type, src, open_beam):
    """This function returns the input type.  It takes the type from the arguments, the file, or the
    open beam, or it infers the type from 16 sampled bins."""
    known = input_type != "auto" or src.file_type in ("attenuation", "transmission") or \
        (open_beam and isinstance(src, _TiffBlocks))
    probe = None if known else np.concatenate(
        [src.read(k, k + 1) for k in np.unique(np.linspace(0, src.bins - 1, 16).astype(int))], axis=-1)
    itype, why = _resolve_input_type(input_type, src, probe, open_beam)
    log.info("input type: %s (%s)", itype, why if probe is None else f"{why}, from {probe.shape[-1]} bins across the "
             "range")
    return itype


def _open_beam_dose(per_bin):
    """This function returns the open beam dose, which is the median over bins of the per-bin
    median open beam.  It returns None if there is no open beam."""
    return None if per_bin is None else float(np.median(per_bin))


def _smoothing_dead_mask(obs, smoothing, src, wave_bin, block):
    """This function returns the mask of dead pixels that the open beam smoothing leaves out.  It
    returns None if there is no smoothing."""
    if not smoothing:
        return None
    t0 = time.perf_counter()
    dead = _dead_mask(_open_beam_total(obs, src.bins // wave_bin * wave_bin, block))
    log.info("open-beam smoothing: %d dead pixels, found in %.1f s", 0 if dead is None else int(dead.sum()),
             time.perf_counter() - t0)
    return dead


def _ratio_dose(per_bin, dose, src, itype, wave_bin):
    """This function returns the open beam dose per bin.  The check of the most transparent regions
    uses this dose to predict the bias from dividing by a noisy open beam."""
    if itype == "counts" or wave_bin == 1:
        return per_bin
    if per_bin is not None:
        return src.file_dose_per_bin[:len(per_bin) * wave_bin]
    return None if dose is None else np.array([dose / wave_bin])


def _effective_observations(n_obs, smoothing, variances, dead=None):
    """This function returns the effective number of open beam observations after smoothing.  It
    also returns a summary of the smoothing."""
    if not smoothing or not n_obs:
        return n_obs, None
    independent = float((_smoothing_kernel(smoothing) ** 2).sum())
    raw, smooth = variances if variances is not None else (0.0, 0.0)
    ratio = smooth / raw if raw > 0 and smooth > 0 else independent
    return n_obs / ratio, dict(width=smoothing, variance_ratio=ratio, independent_ratio=independent,
                               measured=bool(raw > 0 and smooth > 0), observations=n_obs / ratio,
                               dead_pixels=0 if dead is None else int(dead.sum()))


def _calibrated(dose, per_bin, n_eff, background, factor_parts, mean_factor):
    """This function returns the calibration summary, the dose, the per-bin dose, and the effective
    number of open beam observations.  When there is a background calibration, it scales these
    values to the exposure of the sample run."""
    if background is None:
        return None, dose, per_bin, n_eff
    bg = _background_summary(background, np.concatenate(factor_parts, axis=2), mean_factor)
    f = bg["factor_median"]
    return (bg, None if dose is None else dose * f, None if per_bin is None else per_bin * mean_factor,
            n_eff / f if n_eff else n_eff)


def _sum_variances(parts):
    return (sum(v[0] for v in parts), sum(v[1] for v in parts)) if parts else None


def load_dataset(path, open_beam=None, input_type="auto", dataset=None, dose=None, views=None, wave_range=None,
                 wave_bin=1, downsample=1, workers=None, background_boxes=None, background_tiles=None,
                 open_beam_smoothing=0, memory_budget_mib=512):
    """This function loads a TIFF stack or an hsnt HDF5 file as a transmission ratio and runs the
    data checks.

    The function reads the data in blocks of bins.  So the memory use is the size of the result
    plus about memory_budget_mib.

    Args:
        path (str): A TIFF directory or an HDF5 file in the hsnt format.  A TIFF directory holds
            one TIFF per wavelength bin, or one such subdirectory per view.
        open_beam (list of str, optional): The open beam TIFF stacks for a stack of counts.  The
            function averages the subdirectories of a directory of observations.  Defaults to
            None.
        input_type (str, optional): 'counts', 'transmission', 'attenuation', or 'auto'.  With
            'auto', the function uses the dataset_type in the file, or else infers the type from
            the values.  Defaults to 'auto'.
        dataset (str, optional): The HDF5 group that holds 'data'.  None means the root, or the
            only group that holds 'data'.  Defaults to None.
        dose (float or numpy.ndarray, optional): The open beam counts per pixel and source bin.
            This value replaces the estimate.  The function multiplies it by the number of source
            bins per group.  With a background calibration, the function also scales it to the
            exposure of the sample run.  A 1-D array gives the count of each bin of the input,
            before wave_range.  The dose is then the mean of the selected bins, and the result
            holds the weight of each grouped bin in ``bin_weights``.  Defaults to None.
        views (tuple, optional): The (start, stop) range of the views.  None means all views.
            Defaults to None.
        wave_range (tuple, optional): The (start, stop) range of the source bins.  None means all
            bins.  Defaults to None.
        wave_bin (int, optional): The number of adjacent bins in each group.  Defaults to 1.
        downsample (int, optional): The function keeps every n-th row and column.  Defaults to 1.
        workers (int, optional): The number of threads that decode TIFF images.  None means
            min(8, CPUs).  Defaults to None.
        background_boxes (str or list, optional): Boxes that contain no sample, or a preset name
            such as 'ornl-snap'.  Each box is (y0, y1, x0, x1) in full-resolution pixels.  The
            function divides the transmission of each detector tile by the transmission of its
            boxes.  This corrects for a sample run and an open beam with different exposures.
            Defaults to None.
        background_tiles (tuple, optional): The (rows, cols) of the detector tiles that are
            calibrated separately.  None means the tiles of the preset, or else (1, 1).  Defaults
            to None.
        open_beam_smoothing (int, optional): The width of the window that smooths the open beam
            in each bin.  The width must be odd and at least 3.  The smoothing leaves out dead
            pixels.  The value 0 means no smoothing.  Defaults to 0.
        memory_budget_mib (float, optional): The working memory in MiB.  Defaults to 512.

    Returns:
        Dataset: The transmission ratio with shape (pixels, bins), together with its geometry,
            dose, checks, and load information.
    """
    t0 = time.perf_counter()
    workers = workers or min(8, os.cpu_count() or 1)
    src = _open_source(path, dataset, views, wave_range, downsample, workers, open_beam)
    blocks = None
    try:
        V, rows, cols, nb = src.views, src.rows, src.cols, src.bins
        P = V * rows * cols
        wave_bin = max(1, wave_bin)
        K = nb // wave_bin
        if K == 0:
            raise InputError(f"a wave bin of {wave_bin} exceeds the {nb} selected bins")
        dose, bin_weights = _split_dose(dose, src, wave_bin)
        itype = _probe_type(input_type, src, open_beam)
        obs = _open_beam_blocks(open_beam, wave_range, downsample, workers, src) if itype == "counts" else []
        smoothing = _check_smoothing(open_beam_smoothing, obs)
        background = _background(background_boxes, background_tiles, src, downsample)
        full = bool(smoothing or background)
        block = _block_bins(src, len(obs), wave_bin, memory_budget_mib, None, full=full)
        _check_host_memory(P * K * 4 + block * _bytes_per_bin(src, len(obs), full),
                           f"loading {V} x {rows} x {cols} x {K} values")
        T = np.empty((P, K), dtype=np.float32)
        ob_zero, nonfinite, negative, factor_parts, variances = 0.0, 0.0, 0.0, [], []
        ob_per_bin = np.zeros(K) if itype == "counts" else None
        mean_factor = np.ones(K)
        dead = _smoothing_dead_mask(obs, smoothing, src, wave_bin, block)
        blocks = _transmission_blocks(src, obs, itype, wave_bin, block, downsample, smoothing, background, dead)
        for j0, j1, Tb, dose_b, info_b, factors in blocks:
            T[:, j0:j1] = Tb
            if "smoothing_variances" in info_b:
                variances.append(info_b["smoothing_variances"])
            if dose_b is not None:
                ob_per_bin[j0:j1] = info_b["dose_per_bin"]
            ob_zero += info_b.get("open_beam_zero_frac", 0.0) * (j1 - j0)
            nonfinite += info_b.get("nonfinite_frac", 0.0) * (j1 - j0)
            negative += info_b.get("negative_frac", 0.0) * (j1 - j0)
            if factors is not None:
                mean_factor[j0:j1] = info_b["mean_factor"]
                factor_parts.append(factors)
            del Tb
        info = {k: v / K for k, v in (("open_beam_zero_frac", ob_zero), ("nonfinite_frac", nonfinite),
                                      ("negative_frac", negative)) if v}
        _warn_conversion(info)
        per_bin = _dose_per_bin(dose, ob_per_bin, src, wave_bin)
        merged = _merge_dose(dose, _open_beam_dose(ob_per_bin), src, wave_bin)
        n_eff, sm_note = _effective_observations(len(obs), smoothing, _sum_variances(variances), dead)
        bg, merged, per_bin, n_eff = _calibrated(merged, per_bin, n_eff, background, factor_parts, mean_factor)
        if dose is not None and merged is not None and src.file_background is not None:
            merged *= src.file_background["factor_median"]          # Scales a given dose to the exposure in the file.
        if bg is not None:
            info["background"] = bg
        elif src.file_background is not None:
            bg = info["background"] = dict(src.file_background, recorded=True)
        if itype == "counts":
            info["open_beam_observations"] = n_eff
        if sm_note is not None or src.file_smoothing is not None:
            info["open_beam_smoothing"] = sm_note if sm_note is not None else dict(src.file_smoothing, recorded=True)
        info.update(src.metadata, load_seconds=round(time.perf_counter() - t0, 2))
        bin_indices = src.source_bins[:K * wave_bin:wave_bin]
        ds = Dataset(T=T, dataset_type=itype, spatial_shape=(V, rows, cols), bin_indices=bin_indices, dose=merged,
                     source=path if isinstance(src, _TiffBlocks) else src.desc, info=info, dose_per_bin=per_bin,
                     bin_weights=bin_weights,
                     open_beam_observations=n_eff if itype == "counts" else src.file_observations,
                     metadata=_selected_metadata(src.file_metadata, views, range(src.first_index,
                                                 src.first_index + nb), wave_bin, downsample))
    finally:
        if blocks is not None:
            blocks.close()                                  # This stops the read-ahead thread before the source closes.
        src.close()
    smoothed = ds.info.get("open_beam_smoothing")
    sm = dict(_summary_from_T(ds.T, ds.dose, ds.spatial_shape, bg, ds.open_beam_observations,
                              chunk_elems=_check_elems(memory_budget_mib),
                              dose_per_bin=_ratio_dose(ds.dose_per_bin, ds.dose, src, itype, wave_bin),
                              smoothing=smoothed), smoothing=smoothed)
    ds.info["T_stats"], ds.info["above_one"] = sm["stats"], sm["above_one"]
    ds.checks = _checks_from_summary(sm, ds.spatial_shape)
    return ds


def _bytes_per_bin(sample, n_obs, full=False):
    """This function returns the working memory in bytes for one bin of a block."""
    full_bytes = int(np.prod(sample.full_shape)) * sample.views * 4
    small = sample.rows * sample.cols * sample.views * 4
    return full_bytes * (2 + (3 if n_obs else 0) + (4 if full else 0)) + small * 4


def _block_bins(sample, n_obs, wave_bin, budget_mib, requested, full=False):
    """This function returns the number of bins per block.  The number is a multiple of wave_bin
    that fits within the memory budget."""
    if requested:
        block = max(wave_bin, requested // wave_bin * wave_bin)
    else:
        per_bin = _bytes_per_bin(sample, n_obs, full)
        if per_bin * wave_bin > budget_mib * 2**20:
            warnings.warn(f"one group of {wave_bin} bin(s) of the {sample.views} view(s) needs about "
                          f"{per_bin * wave_bin / 2**20:,.0f} MiB of working memory, more than the budget of "
                          f"{budget_mib:g} MiB; use --views ranges or --downsample to stay within it")
        block = max(wave_bin, int(budget_mib * 2**20 // per_bin) // wave_bin * wave_bin)
    return min(block, sample.bins // wave_bin * wave_bin) or wave_bin


def _converted_path(path):
    """This function returns the default output path of ``convert_to_hdf5``.  The path is
    <input stem>_converted.h5 in the directory of the input."""
    return os.path.splitext(os.path.normpath(path))[0] + "_converted.h5"


def convert_to_hdf5(path, output=None, open_beam=None, input_type="auto", dataset=None, dose=None, views=None,
                    wave_range=None, wave_bin=1, downsample=1, as_type="attenuation", block_bins=None,
                    memory_budget_mib=256, workers=None, progress=False, background_boxes=None,
                    background_tiles=None, open_beam_smoothing=0):
    """This function converts a TIFF stack or an HDF5 file to an hsnt HDF5 file.  It processes the
    data in blocks of bins.  The other arguments are the same as in :func:`load_dataset`.

    Args:
        output (str, optional): The output path, which must differ from the input.  None means
            <input stem>_converted.h5.  Defaults to None.
        as_type (str, optional): 'attenuation' or 'transmission'.  Defaults to 'attenuation'.
        block_bins (int, optional): The number of bins per block.  None means a number set by
            memory_budget_mib.  Defaults to None.
        memory_budget_mib (float, optional): The working memory in MiB.  Defaults to 256.
        progress (bool, optional): If True, the function shows a progress bar.  Defaults to
            False.

    Returns:
        (path, checks, info): The output path, the data checks, and the conversion statistics.
    """
    import h5py
    from tqdm import tqdm
    if dose is not None and np.ndim(dose) > 0:
        raise InputError("the converted file records a scalar dose only; give a per-bin dose when the data are fitted")
    t0 = time.perf_counter()
    workers = workers or min(8, os.cpu_count() or 1)
    src = _open_source(path, dataset, views, wave_range, downsample, workers, open_beam)
    V, rows, cols, nb = src.views, src.rows, src.cols, src.bins
    P = V * rows * cols
    blocks = None
    try:
        itype = _probe_type(input_type, src, open_beam)
        obs = _open_beam_blocks(open_beam, wave_range, downsample, workers, src) if itype == "counts" else []
        smoothing = _check_smoothing(open_beam_smoothing, obs)
        background = _background(background_boxes, background_tiles, src, downsample)
        wave_bin = max(1, wave_bin)
        K = nb // wave_bin
        if K == 0:
            raise InputError(f"a wave bin of {wave_bin} exceeds the {nb} selected bins")
        block = _block_bins(src, len(obs), wave_bin, memory_budget_mib, block_bins,
                            full=bool(smoothing or background))
        out = output or _converted_path(path)
        if os.path.exists(out) and any(os.path.exists(p) and os.path.samefile(out, p)
                                       for p in [path, *(open_beam or [])]):
            raise InputError(f"the output {out} is an input of the conversion; choose another output")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        log.info("streaming %d bins per block (%d blocks), %d TIFF reader threads -> %s (%s, %.2f GiB)", block,
                 -(-nb // block), workers, out, as_type, P * K * 4 / 2**30)
        tot = dict(n=0, nonfinite=0, negative=0, zero=0, above=0, inf_out=0)
        pos_px = np.zeros(P, dtype=np.int64)
        sum_px = np.zeros(P, dtype=np.float64)
        pos_bin = np.zeros(K, dtype=np.int64)
        bin_min = np.full(K, np.inf, dtype=np.float32)
        bin_max = np.full(K, -np.inf, dtype=np.float32)
        stride = _sample_stride(P, K)
        sample = np.empty((-(-P // stride), K), dtype=np.float32)
        ob_zero, nonfinite_in, negative_in, factor_parts, variances = 0.0, 0.0, 0.0, [], []
        ob_per_bin = np.zeros(K) if itype == "counts" else None
        mean_factor = np.ones(K)
        dead = _smoothing_dead_mask(obs, smoothing, src, wave_bin, block)
        blocks = _transmission_blocks(src, obs, itype, wave_bin, block, downsample, smoothing, background, dead)
        n_blocks = len(range(0, nb - nb % wave_bin, block))
        with _written_atomically(out) as tmp, h5py.File(tmp, "w") as f:
            d = _create_hyperspectral(f, (V, rows, cols, K), as_type, chunks=(1, min(rows, 128), cols, min(16, K)))
            for j0, j1, T, dose_b, info_b, factors in tqdm(blocks, desc="convert", unit="block", total=n_blocks,
                                                           disable=not progress, leave=False):
                if dose_b is not None:
                    ob_per_bin[j0:j1] = info_b["dose_per_bin"]
                ob_zero += info_b.get("open_beam_zero_frac", 0.0) * (j1 - j0)
                nonfinite_in += info_b.get("nonfinite_frac", 0.0) * (j1 - j0)
                negative_in += info_b.get("negative_frac", 0.0) * (j1 - j0)
                if "smoothing_variances" in info_b:
                    variances.append(info_b["smoothing_variances"])
                if factors is not None:
                    mean_factor[j0:j1] = info_b["mean_factor"]
                    factor_parts.append(factors)
                tot["n"] += T.size
                tot["nonfinite"] += int((~np.isfinite(T)).sum())
                tot["negative"] += int((T < 0).sum())
                tot["zero"] += int((T == 0).sum())
                tot["above"] += int((T > 1).sum())
                pos = T > 0
                pos_px += pos.sum(1)
                sum_px += T.sum(1, dtype=np.float64)
                pos_bin[j0:j1] = pos.sum(0)
                bin_min[j0:j1] = T.min(0)
                bin_max[j0:j1] = T.max(0)
                sample[:, j0:j1] = T[::stride]
                if as_type == "attenuation":
                    with np.errstate(divide="ignore"):
                        values = -np.log(T)
                    tot["inf_out"] += int(np.isinf(values).sum())
                else:
                    values = T
                d[:, :, :, j0:j1] = values.reshape(V, rows, cols, j1 - j0)
                del T, pos, values
            given_dose, dose = dose, _merge_dose(dose, _open_beam_dose(ob_per_bin), src, wave_bin)
            per_bin = _dose_per_bin(given_dose, ob_per_bin, src, wave_bin)
            n_eff, sm_note = _effective_observations(len(obs), smoothing, _sum_variances(variances), dead)
            bg, dose, per_bin, n_eff = _calibrated(dose, per_bin, n_eff, background, factor_parts, mean_factor)
            n_eff = n_eff or src.file_observations
            if given_dose is not None and dose is not None and src.file_background is not None:
                # This scales a given dose to the exposure recorded in the file.
                dose *= src.file_background["factor_median"]
            if bg is None and src.file_background is not None:
                bg = dict(src.file_background, recorded=True)
            if sm_note is None and src.file_smoothing is not None:
                sm_note = dict(src.file_smoothing, recorded=True)
            _warn_conversion(dict(open_beam_zero_frac=ob_zero / K, nonfinite_frac=nonfinite_in / K,
                                  negative_frac=negative_in / K))
            if tot["inf_out"]:
                warnings.warn(f"{tot['inf_out']} zero-transmission entries are inf in the attenuation output; the "
                              "solvers map them back to zero transmission")
            st = dict(_stats(sample), nonfinite=tot["nonfinite"] / tot["n"], negative=tot["negative"] / tot["n"],
                      zero=tot["zero"] / tot["n"], sampled=stride > 1)
            dead_bins = int((pos_bin == 0).sum())
            sm = dict(pixels=P, bins=K, nbytes=P * K * 4, stats=st, above_one=tot["above"] / tot["n"],
                      dead_px=_frac(pos_px == 0), dead_bins=dead_bins,
                      const_bins=int((bin_max == bin_min).sum()) - dead_bins, dose=dose,
                      bright=_bright_level(sum_px / K, (V, rows, cols), sample, dose, n_eff,
                                           _ratio_dose(per_bin, dose, src, itype, wave_bin),
                                           _check_elems(memory_budget_mib), smoothing=sm_note),
                      background=bg, smoothing=sm_note)
            checks = _checks_from_summary(sm, (V, rows, cols))
            f.create_dataset("bin_indices", data=src.source_bins[:K * wave_bin:wave_bin])
            if per_bin is not None:
                f.create_dataset("open_beam_dose", data=per_bin)
            columns = range(src.first_index, src.first_index + nb)
            _write_metadata(f, _selected_metadata(src.file_metadata, views, columns, wave_bin, downsample))
            attrs = dict(source=path if isinstance(src, _TiffBlocks) else src.desc, input_type=itype,
                         downsample=downsample, wave_bin=wave_bin, block_bins=block, mbirtorch_hsnt_cli="1",
                         checks=json.dumps([dict(level=c.level, message=c.message) for c in checks]))
            if dose is not None:
                attrs["dose"] = float(dose)
            if n_eff:
                attrs["open_beam_observations"] = float(n_eff)
            if sm_note is not None:
                attrs["open_beam_smoothing"] = json.dumps(sm_note)
            if bg is not None:
                attrs["background"] = json.dumps(bg)
            f.attrs.update(attrs)
    finally:
        if blocks is not None:
            blocks.close()                                  # This stops the read-ahead thread before the source closes.
        src.close()
    seconds = time.perf_counter() - t0
    info = dict(seconds=round(seconds, 2), block_bins=block, blocks=n_blocks, open_beam_observations=n_eff,
                stats=st, dose=dose)
    if sm_note is not None:
        info["open_beam_smoothing"] = sm_note
    if bg is not None:
        info["background"] = bg
    log.info("wrote %s: data %s %s, %.2f GiB, in %.1f s (%.0f bins/s)", out, (V, rows, cols, K), as_type,
             P * K * 4 / 2**30, seconds, nb / seconds)
    return out, checks, info

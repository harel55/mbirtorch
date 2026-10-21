"""Measure the noise of the counts on the rays that miss the sample.

Rays through air see only the open beam, so their transmission is 1 up to noise, and the model has nothing to get
wrong there. measure_noise finds those rays and returns two numbers for a fit:

- the dispersion: the variance of the transmission relative to Poisson counts at the given dose. Binned detector
  pixels whose counts are correlated within the bin, or counts corrected for detector dead time, have a dispersion
  above 1. A fit should use the dose divided by the dispersion, so that its loss has the scale of Poisson counts.
- the inflation: the variance of a sum over many pixels relative to the sum of their variances, from the correlation
  of the noise between neighbouring pixels (1 for independent pixels). The contamination thresholds use it.

tomo.warmup does both with noise='measure'. normalize_views corrects each view's exposure from its air rays.
"""
import numpy as np
import torch

from ._dist import all_sum, all_sum_float

MAX_OFFSET = 3          # neighbour offsets (pixels) up to which the correlation is measured
AIR_MARGIN = 2          # air rays lie at least this many pixels from any ray that sees the sample
NOISE_SIGMAS = 3.0      # a ray sees the sample if its mean transmission is this many standard errors low
AIR_QUANTILE = 0.9      # the open-beam level is the median of the rays above this quantile of mean transmission
SIGNIFICANCE = 3.0      # a correlation enters the inflation if it is this many standard errors from zero
CHUNK = 4096            # rays per chunk
VIEW_OUTLIER = 5.0      # a view's level is flagged when it is this many robust spreads from the median level
DISPERSION_BLOCK = 64   # adjacent bin pairs averaged for the per-bin dispersion


def _tile_slices(shape, tiles):
    """The (rows, channels) slices of each detector tile: the image split into tiles[0] x tiles[1] equal parts."""
    rs = np.linspace(0, shape[1], tiles[0] + 1).astype(int); cs = np.linspace(0, shape[2], tiles[1] + 1).astype(int)
    return [(slice(r0, r1), slice(c0, c1)) for r0, r1 in zip(rs[:-1], rs[1:]) for c0, c1 in zip(cs[:-1], cs[1:])]


def air_rays(T, shape, excluded=None, tiles=(1, 1)):
    """The rays that see only the open beam: True where a ray's mean transmission is within NOISE_SIGMAS standard
    errors of its image's open-beam level and the ray lies AIR_MARGIN pixels from any ray that falls short. T (rays,
    bins); shape (images, rows, channels), the rays' layout (one image per view); excluded (rays,) bool marks rays to
    leave out, such as those through a sample holder. The standard errors come from each ray's own differences between
    adjacent bins, so no noise level is assumed. The level is each image's own (its exposure may differ from the
    others'), and with tiles (rows, channels) each detector tile's own (a detector made of several sensors, such as a
    2 x 2 Timepix array, has a gain per sensor), from its brightest rays: at least 1 - AIR_QUANTILE of each image (or
    tile) must be air."""
    from scipy import ndimage as nd
    P, K = T.shape
    mt = torch.empty(P, dtype=torch.float64, device=T.device); se = torch.empty_like(mt)
    for s in range(0, P, CHUNK):
        x = T[s:s + CHUNK].double()
        mt[s:s + CHUNK] = x.mean(1)
        se[s:s + CHUNK] = torch.sqrt((x.diff(dim=1) ** 2).sum(1) / 2) / K
    mt, se = mt.cpu().numpy().reshape(shape), se.cpu().numpy().reshape(shape)
    sample = np.zeros(shape, bool)
    for rs, cs in _tile_slices(shape, tiles):                                      # each image's and tile's open beam
        m_, e_ = mt[:, rs, cs].reshape(shape[0], -1), se[:, rs, cs].reshape(shape[0], -1)
        bright = m_ > np.quantile(m_, AIR_QUANTILE, axis=1, keepdims=True) - NOISE_SIGMAS * np.median(e_, axis=1, keepdims=True)
        level = np.array([np.median(a_[b_]) for a_, b_ in zip(m_, bright)])[:, None]
        sample[:, rs, cs] = (m_ < level - NOISE_SIGMAS * e_).reshape(shape[0], rs.stop - rs.start, cs.stop - cs.start)
    sample = nd.binary_dilation(sample, structure=np.ones((1, 3, 3), bool), iterations=AIR_MARGIN)
    air = ~sample.reshape(-1)
    if excluded is not None:
        air &= ~np.asarray(excluded.cpu() if torch.is_tensor(excluded) else excluded, bool)
    return air


def noise_statistics(T, d, air, shape):
    """The dispersion, the noise correlation per neighbour offset and the inflation of the transmission T (rays,
    bins) at dose d (bins,), measured on the rays where air (rays,) is True; shape as for air_rays. The noise is taken
    from the differences between adjacent bins of each ray (independent noise; the differences also remove each ray's
    level), less the open beam's own difference; pairs of bins with a dose of zero (gaps) are left out. Only correlations more than SIGNIFICANCE standard errors from zero
    enter the inflation. With the rays split over processes, the sums are taken over all of them. Returns a dict:
    dispersion, inflation, inflation_se, correlations [(drow, dchannel, correlation, standard error, pairs)], air_rays,
    and dispersion_per_bin (bins,), the dispersion of each bin (_per_bin)."""
    dev = T.device; K = T.shape[1]
    d = torch.as_tensor(np.asarray(d, dtype=np.float64), device=dev)
    ids = torch.as_tensor(np.nonzero(air)[0], device=dev)
    n = all_sum_float(float(len(ids)), dev)
    sT = torch.zeros(K, dtype=torch.float64, device=dev); sD = torch.zeros(K - 1, dtype=torch.float64, device=dev)
    for s in range(0, len(ids), CHUNK):
        x = T[ids[s:s + CHUNK]].double()
        sT += x.sum(0); sD += (x[:, :-1] - x[:, 1:]).sum(0)
    m = all_sum(sT) / n; common = all_sum(sD) / n                      # open-beam level; its own difference
    ok = (d[:-1] > 0) & (d[1:] > 0); K1 = float(ok.sum())                # bin pairs with counts in both bins
    dd = d.clamp_min(torch.finfo(d.dtype).tiny)
    vd = ((m / dd)[:-1] + (m / dd)[1:])[ok]                              # Poisson variance of a difference per bin

    def z(rows):
        x = T[rows].double()
        return (x[:, :-1] - x[:, 1:] - common)[:, ok] / torch.sqrt(vd)

    pair_sum = torch.zeros(int(K1), dtype=torch.float64, device=dev)
    for s in range(0, len(ids), CHUNK):
        pair_sum += (z(ids[s:s + CHUNK]) ** 2).sum(0)
    pair_sum = all_sum(pair_sum)
    phi = float(pair_sum.sum()) / (n * K1)
    flat = np.arange(T.shape[0]).reshape(shape); A = np.asarray(air, bool).reshape(shape)
    rho = {}
    for dr in range(0, MAX_OFFSET + 1):
        for dc in range(-MAX_OFFSET, MAX_OFFSET + 1):
            if (dr, dc) <= (0, 0):
                continue
            r0, r1 = max(0, -dr), shape[1] - max(0, dr); c0, c1 = max(0, -dc), shape[2] - max(0, dc)
            pair = A[:, r0:r1, c0:c1] & A[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            ia = torch.as_tensor(flat[:, r0:r1, c0:c1][pair], device=dev)
            ib = torch.as_tensor(flat[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc][pair], device=dev)
            acc = 0.0
            for s in range(0, len(ia), CHUNK):
                acc += float((z(ia[s:s + CHUNK]) * z(ib[s:s + CHUNK])).sum())
            npair = all_sum_float(float(len(ia)), dev)
            if npair == 0:
                continue
            # adjacent differences share a bin (correlation -1/2), so the products in neighbouring bins correlate by 1/4
            rho[(dr, dc)] = (all_sum_float(acc, dev) / (npair * K1 * phi), float(np.sqrt(1.5 / (npair * K1))), int(npair))
    sig = [(r, s_) for r, s_, _ in rho.values() if abs(r) > SIGNIFICANCE * s_]
    corr = [(dr, dc, r, s_, npair) for (dr, dc), (r, s_, npair) in rho.items()]
    inflation = 1.0 + 2.0 * sum(r for r, _ in sig)                     # each offset counts in both directions
    inflation_se = 2.0 * float(np.sqrt(sum(s_ ** 2 for _, s_ in sig)))
    return dict(dispersion=phi, inflation=inflation, inflation_se=inflation_se, correlations=corr, air_rays=int(n),
                dispersion_per_bin=_per_bin(pair_sum.cpu().numpy() / n, ok.cpu().numpy(), K, phi))


def _per_bin(phi_pairs, ok, K, phi):
    """The dispersion of each bin, (K,): the dispersion of the valid bin pairs averaged over blocks of DISPERSION_BLOCK
    adjacent pairs, and each bin given the mean of its two pairs' block values.  A bin with no valid pair keeps the
    overall dispersion phi."""
    pair = np.full(K - 1, np.nan); idx = np.flatnonzero(ok)
    for b in range(0, len(idx), DISPERSION_BLOCK):
        pair[idx[b:b + DISPERSION_BLOCK]] = phi_pairs[b:b + DISPERSION_BLOCK].mean()
    both = np.stack([np.r_[np.nan, pair], np.r_[pair, np.nan]])
    have = np.isfinite(both); count = have.sum(0)
    return np.where(count > 0, np.where(have, both, 0.0).sum(0) / np.maximum(count, 1), phi)


def measure_noise(T, d, shape, excluded=None, tiles=(1, 1)):
    """The dispersion and the inflation of the transmission T (rays, bins) at dose d (bins,), measured on its air
    rays (air_rays; shape and excluded as there). Use it before a fit: the dose divided by the dispersion, and the
    inflation for the contamination thresholds. Returns noise_statistics' dict."""
    return noise_statistics(T, d, air_rays(T, shape, excluded, tiles), shape)


def view_levels(T, air, shape, tiles=(1, 1)):
    """Each image's (and each detector tile's) mean transmission over its air rays, relative to the median over all of
    them: the exposure and the tile's gain (1 where there are no air rays). T (rays, bins); air (rays,) bool; shape
    (images, rows, channels), one image per view; tiles as for air_rays. Returns the levels (images, number of tiles)
    and the flagged (image, tile) pairs (more than VIEW_OUTLIER robust spreads from the median, the spread from the
    median absolute deviation)."""
    per = shape[1] * shape[2]; air = np.asarray(air, bool).reshape(shape)
    idx = np.arange(per).reshape(shape[1], shape[2]); sl = _tile_slices(shape, tiles)
    lv = np.ones((shape[0], len(sl)))
    for i in range(shape[0]):
        for t, (rs, cs) in enumerate(sl):
            a = idx[rs, cs][air[i, rs, cs]]
            if len(a):
                lv[i, t] = float(T[torch.as_tensor(i * per + a, device=T.device)].double().mean())
    lv = lv / np.median(lv)
    spread = 1.4826 * np.median(np.abs(lv - 1.0))
    flagged = [(int(i), int(t)) for i, t in zip(*np.nonzero(np.abs(lv - 1.0) > VIEW_OUTLIER * max(spread, 1e-12)))]
    return lv, flagged


def normalize_views(T, shape, excluded=None, tiles=(1, 1)):
    """Divide each view's (and each detector tile's) transmission by its level (view_levels, measured on its air rays),
    in place: views measured with different exposures, and sensors with different gains, otherwise disagree by
    factors that no model term absorbs. shape, excluded and tiles as for air_rays. Returns the levels and the flagged
    (image, tile) pairs."""
    lv, flagged = view_levels(T, air_rays(T, shape, excluded, tiles), shape, tiles)
    per = shape[1] * shape[2]; idx = np.arange(per).reshape(shape[1], shape[2])
    with torch.no_grad():
        for i in range(shape[0]):
            for t, (rs, cs) in enumerate(_tile_slices(shape, tiles)):
                T[torch.as_tensor(i * per + idx[rs, cs].ravel(), device=T.device)] /= float(lv[i, t])
    return lv, flagged

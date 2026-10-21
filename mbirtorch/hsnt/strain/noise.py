"""Measure the noise of the counts on the rays that miss the sample.

Rays through air see only the open beam, so their transmission is 1 up to noise, and the model has nothing to get
wrong there. measure_noise finds those rays and returns two numbers for a fit:

- the dispersion: the variance of the transmission relative to Poisson counts at the given dose. Binned detector
  pixels whose counts are correlated within the bin, or counts corrected for detector dead time, have a dispersion
  above 1. A fit should use the dose divided by the dispersion, so that its loss has the scale of Poisson counts.
- the inflation: the variance of a sum over many pixels relative to the sum of their variances, from the correlation
  of the noise between neighbouring pixels (1 for independent pixels). The contamination thresholds use it.

tomo.warmup does both with noise='measure'.
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


def air_rays(T, shape, excluded=None):
    """The rays that see only the open beam: True where a ray's mean transmission is within NOISE_SIGMAS standard
    errors of the open-beam level and the ray lies AIR_MARGIN pixels from any ray that falls short. T (rays, bins);
    shape (images, rows, channels), the rays' layout (one image per view); excluded (rays,) bool marks rays to leave
    out, such as those through a sample holder. The standard errors come from each ray's own differences between
    adjacent bins, so no noise level is assumed."""
    from scipy import ndimage as nd
    P, K = T.shape
    mt = torch.empty(P, dtype=torch.float64, device=T.device); se = torch.empty_like(mt)
    for s in range(0, P, CHUNK):
        x = T[s:s + CHUNK].double()
        mt[s:s + CHUNK] = x.mean(1)
        se[s:s + CHUNK] = torch.sqrt((x.diff(dim=1) ** 2).sum(1) / 2) / K
    mt, se = mt.cpu().numpy(), se.cpu().numpy()
    bright = mt > np.quantile(mt, AIR_QUANTILE) - NOISE_SIGMAS * np.median(se)
    level = np.median(mt[bright])
    sample = (mt < level - NOISE_SIGMAS * se).reshape(shape)
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
    dispersion, inflation, inflation_se, correlations [(drow, dchannel, correlation, standard error, pairs)], air_rays."""
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

    acc = 0.0
    for s in range(0, len(ids), CHUNK):
        acc += float((z(ids[s:s + CHUNK]) ** 2).sum())
    phi = all_sum_float(acc, dev) / (n * K1)
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
    return dict(dispersion=phi, inflation=inflation, inflation_se=inflation_se, correlations=corr, air_rays=int(n))


def measure_noise(T, d, shape, excluded=None):
    """The dispersion and the inflation of the transmission T (rays, bins) at dose d (bins,), measured on its air
    rays (air_rays; shape and excluded as there). Use it before a fit: the dose divided by the dispersion, and the
    inflation for the contamination thresholds. Returns noise_statistics' dict."""
    return noise_statistics(T, d, air_rays(T, shape, excluded), shape)

"""Calibrate the resolution kernel from the open beam of the same measurement.

The open beam passes through beamline windows (usually aluminium) whose Bragg edges are blurred by the instrument's
wavelength resolution. Fitting those edges gives the kernel that the hybrid model (model.Hybrid2) applies to the
sample's transmission, so no separate calibration measurement is needed. Use fit_open_beam_kernel on the open beam's
counts summed over the detector; pass the returned kernel parameters to the sample fit and hold them fixed.
"""
import numpy as np
import torch

from .bragg import emg_median
from .model import Hybrid2, _emg_cdf

AL = ('fcc', 4.0495)                    # aluminium, the usual window material: lattice type and parameter (Angstrom)
WINDOW_HALF = 0.08                      # half width (Angstrom) of the fitting window around each edge
A_RANGE = 0.01                          # relative range of the window material's fitted lattice parameter
LM_ITERS = 60                           # Levenberg-Marquardt iterations
LM_DAMP0 = 1e-3                         # initial damping, relative to the diagonal of J^T J
MIN_EDGE_HEIGHT = 1e-6                  # heights start here (positive), so every edge can grow
KERNEL_START = (0.0, 0.0008, 0.003, 0.0, 0.1, 0.012, 0.0, 0.0)   # a moderate starting kernel (model.Hybrid2 slots)
FD_KERNEL = 1e-6                        # central-difference step for the kernel parameters (Angstrom, or weight)
FD_REL = 1e-6                           # relative central-difference step for the other parameters
FD_FLOOR = 1e-3                         # the relative step's floor on |parameter|
BASELINE_DEGREE = 3                     # the per-window baseline: exp(polynomial of this degree in wavelength)
TEXTURE_START = (1.0, 0.5)              # March-Dollase r (1 = random powder) and axis angle (radians) to start from


def fit_open_beam_kernel(counts, lam, material=AL, exclude=(), gaps=(), window_half=WINDOW_HALF, device='cpu',
                         kernel_start=KERNEL_START, edge_slope=True, texture=False, broad=True, fit_kernel=True):
    """Fit the resolution kernel to the window-material Bragg edges in an open-beam spectrum.

    counts: the open beam's counts per wavelength bin, summed over the detector (and observations); lam: the bins'
    wavelengths (Angstrom). material: (lattice type, lattice parameter) of the windows. exclude: wavelength intervals
    to leave out (other materials' edges, artifacts). gaps: bin indices after which the wavelength axis jumps (the
    windows do not cross them). Each window around an edge gets its own baseline (exp of a cubic); the edges' heights,
    the material's lattice parameter and the kernel are shared. With edge_slope, each edge also gets a slope below
    it; with texture, the material gets a texture (March-Dollase), as rolled window foils have. broad=False fits the
    kernel's core only (its broad component's weight held at zero). fit_kernel=False holds the kernel at kernel_start
    (to test a kernel from another measurement against these edges). Returns a dict: kernel (the 8 kernel parameters of
    model.Hybrid2), lattice (the fitted lattice parameter: the wavelength scale's check), heights, windows, the reduced
    chi-square per window, and the model and data in the windows."""
    lam = np.asarray(lam, dtype=np.float64); counts = np.asarray(counts, dtype=np.float64)
    lt, a0 = material
    M = Hybrid2(lam, np.ones_like(lam), [(lt, (a0 * (1 - A_RANGE), a0 * (1 + A_RANGE)))], n_spline=4, n_contam=1,
                device=device, two_scale=True, edge_slope=edge_slope, shape_model='texture' if texture else None)
    sl = M.slices[0]
    edges = (2 * a0 / torch.sqrt(M.mats[0]['ns'])).cpu().numpy()
    # windows around the edges inside the measured range, merged where they overlap, cut at the axis' gaps
    seg_id = np.zeros(lam.size, int)
    for g_ in sorted(gaps):
        seg_id[g_ + 1:] += 1
    keep = np.zeros(lam.size, bool)
    for e in edges:
        keep |= np.abs(lam - e) <= window_half
    for lo, hi in exclude:
        keep &= ~((lam >= lo) & (lam <= hi))
    win = np.zeros(lam.size, int) - 1; w = -1
    for k in range(lam.size):
        if keep[k] and (k == 0 or not keep[k - 1] or seg_id[k] != seg_id[k - 1]):
            w += 1
        if keep[k]:
            win[k] = w
    nwin = w + 1
    if nwin == 0:
        raise ValueError('no edge of the material lies in the measured range')
    used_edges = [float(e) for e in edges if any(np.abs(lam[win >= 0] - e) <= window_half)]
    idx = np.where(win >= 0)[0]
    y = torch.as_tensor(counts[idx], device=device, dtype=torch.float64)
    sd = torch.sqrt(y.clamp_min(1.0))
    lw = torch.as_tensor(lam[idx], device=device, dtype=torch.float64)
    wi = torch.as_tensor(win[idx], device=device)
    centre = torch.stack([lw[wi == j].mean() for j in range(nwin)])

    # parameters: the free global slots (kernel first, then lattice, heights, edge slopes, texture), then the
    # baseline coefficients of each window
    g0 = torch.zeros(M.n_global, dtype=torch.float64, device=device)
    g0[M.res_sl[0]:M.res_sl[1]] = torch.as_tensor(kernel_start, dtype=torch.float64)
    g0[sl['a']] = a0
    g0[sl['h'][0]:sl['h'][1]] = MIN_EDGE_HEIGHT
    if texture:
        g0[sl['x'][0]] = TEXTURE_START[0]; g0[sl['x'][0] + 1] = TEXTURE_START[1]
    free = lambda lo, hi: [i for i in range(lo, hi) if float(M.ub[i]) > float(M.lb[i])]
    res_free = free(*M.res_sl)
    if not broad:
        g0[M.res_sl[0] + 4] = 0.0
        res_free = [i for i in res_free if i not in (M.res_sl[0] + 4, M.res_sl[0] + 5)]
    n_k = len(res_free)
    gl_free = res_free + [sl['a']] + free(*sl['h']) + free(*sl['t']) + free(*sl['x'])
    gl_free_t = torch.as_tensor(gl_free, device=device)
    nb = BASELINE_DEGREE + 1
    inf = torch.full((nb * nwin,), np.inf, dtype=torch.float64, device=device)
    lb = torch.cat([M.lb[gl_free_t], -inf]); ub = torch.cat([M.ub[gl_free_t], inf])
    base0 = torch.zeros(nwin, nb, dtype=torch.float64, device=device)
    base0[:, 0] = torch.stack([torch.log(y[wi == j].mean()) for j in range(nwin)])
    p = torch.cat([g0[gl_free_t], base0.reshape(-1)])
    nh = sl['h'][1] - sl['h'][0]
    powers = torch.arange(nb, device=device, dtype=torch.float64)

    # the kernel's rows for the windows' bins only (the two-scale kernel of model.Hybrid2.band, without its detached
    # median: here the kernel is what is fitted, so its derivatives must include the centring)
    bidx = M.band_idx[idx]; bval = M.band_valid[idx]; lhi = M.lam_hi[idx][:, None]; llo = M.lam_lo[idx][:, None]
    lamj = M.lamf[bidx]

    def band_rows(g):
        s0, s1, t0, t1, wb, sb = [g[M.res_sl[0] + i] for i in range(6)]
        sigma = torch.sqrt(s0 * s0 + (s1 * lamj) ** 2).clamp_min(1e-6)
        tau = (t0 + t1 * lamj).clamp_min(1e-7)
        med = emg_median(sigma, tau).to(torch.float64)
        core = (_emg_cdf(lhi - lamj + med, sigma, tau) - _emg_cdf(llo - lamj + med, sigma, tau)).clamp_min(0)
        sbc = sb.clamp_min(1e-6)
        broad = 0.5 * (torch.erf((lhi - lamj) / (np.sqrt(2.0) * sbc)) - torch.erf((llo - lamj) / (np.sqrt(2.0) * sbc)))
        Rb = ((1 - wb) * core + wb * broad).clamp_min(0) * bval
        return Rb / Rb.sum(1, keepdim=True).clamp_min(1e-30)

    def model(p_):
        g = g0.clone(); g[gl_free_t] = p_[:len(gl_free)]
        X = M.spectra(g)[0]                                                    # (Kf,) the windows' Bragg attenuation
        T = (band_rows(g) * torch.exp(-X).to(torch.float64)[bidx]).sum(1)      # the kernel on the transmission
        b = p_[len(gl_free):].reshape(nwin, nb)[wi]
        dl = (lw - centre[wi])[:, None] ** powers
        return torch.exp((b * dl).sum(1)) * T

    def resid(p_):
        return (y - model(p_)) / sd

    # central-difference steps: absolute for the kernel widths and weights, relative for the rest
    hstep = torch.cat([torch.full((n_k,), FD_KERNEL, dtype=torch.float64, device=device),
                       torch.full((p.numel() - n_k,), FD_REL, dtype=torch.float64, device=device)])

    def jacobian(p_):
        cols = []
        for i in range(p_.numel()):
            h = float(hstep[i]) if i < n_k else float(hstep[i]) * max(abs(float(p_[i])), FD_FLOOR)
            pp, pm = p_.clone(), p_.clone(); pp[i] += h; pm[i] -= h
            cols.append((resid(pp) - resid(pm)) / (2 * h))
        return torch.stack(cols, 1)

    def lm(p, free, iters):
        """Levenberg-Marquardt on the parameters where `free` is True."""
        damp = LM_DAMP0
        r = resid(p); cost = float((r * r).sum())
        fi = torch.nonzero(free)[:, 0]
        for _ in range(iters):
            J = jacobian(p)[:, fi]                                              # (n_bins, n_free)
            A = J.T @ J; gr = J.T @ r
            while True:
                step = torch.linalg.solve(A + damp * torch.diag(torch.diagonal(A).clamp_min(1e-30)), -gr)
                p_new = p.clone(); p_new[fi] = p[fi] + step
                p_new = torch.maximum(torch.minimum(p_new, ub), lb)
                r_new = resid(p_new); c_new = float((r_new * r_new).sum())
                if c_new < cost:
                    p, r, cost = p_new, r_new, c_new; damp = max(damp / 3, 1e-9); break
                damp *= 10
                if damp > 1e8:
                    break
            if damp > 1e8:
                break
        return p

    # in stages: first the edge heights, the lattice and the baselines with the starting kernel and plain edges; then
    # the edge shapes; then the kernel too. A shape or kernel parameter freed while the heights are still near zero has
    # almost no effect on the data, takes a huge step and bends to imitate the edges.
    n_shape = len(free(*sl['t'])) + len(free(*sl['x']))
    n_gl = len(gl_free)
    stage = torch.ones(p.numel(), dtype=torch.bool, device=device)
    stage[:n_k] = False; stage[n_gl - n_shape:n_gl] = False
    p = lm(p, stage, LM_ITERS)
    if n_shape:
        stage[n_gl - n_shape:n_gl] = True
        p = lm(p, stage, LM_ITERS)
    if fit_kernel:
        p = lm(p, torch.ones_like(stage), LM_ITERS)
    g_fit = g0.clone(); g_fit[gl_free_t] = p[:len(gl_free)]
    kernel = g_fit[M.res_sl[0]:M.res_sl[1]]
    with torch.no_grad():
        m_ = model(p)
    chi = [float((((y - m_) / sd)[wi == j] ** 2).mean()) for j in range(nwin)]
    return dict(kernel=kernel.cpu().numpy(), lattice=float(g_fit[sl['a']]), heights=g_fit[sl['h'][0]:sl['h'][1]].cpu().numpy(),
                slopes=g_fit[sl['t'][0]:sl['t'][1]].cpu().numpy(), texture=g_fit[sl['x'][0]:sl['x'][1]].cpu().numpy(),
                edges=used_edges, windows=[(float(lw[wi == j].min()), float(lw[wi == j].max())) for j in range(nwin)],
                chi_square=chi, lam=lw.cpu().numpy(), data=y.cpu().numpy(), model=m_.cpu().numpy())

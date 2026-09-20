"""Hybrid spectral model: parametric Bragg-edge spectra plus free nonparametric components.

A polycrystalline material's linear attenuation over wavelength is, for an ideal powder, a sum of Bragg edges (each
lattice-plane family hkl contributes lambda^2 * height below lambda = 2 d_hkl and nothing above), absorption linear in
lambda, a near-constant incoherent term and a smooth inelastic term. This module parametrises a material by its
lattice parameter (which fixes every edge position), free nonnegative edge heights (which absorb structure factors,
multiplicity, Debye-Waller, texture and extinction), a linear absorption term, a constant, and a penalised B-spline
residual for whatever is smooth and unmodelled. Edges are smoothed steps whose width stands for the instrument's
resolution (a Gaussian kernel applied to the attenuation, exact only when the kernel is narrow relative to the edge
contrast; the polychromatic forward model that applies it to the transmission is a later stage).

hybrid_factorization fits R_bragg such materials plus R_free unconstrained rows to hyperspectral transmission data
under the NNAL, alternating the package's convex per-pixel solve for the maps with a Gauss-Newton step on the spectral
parameters that uses the same per-bin Newton statistics as the block solver, and the block Newton step for the free
rows. Because a mixture of Bragg spectra is not the spectrum of any lattice, the mixing that the bilinear model
leaves free is fixed by the physics, without pure pixels. See docs/hsnt_solver_notes.md (hybrid section) once merged.
"""
import math

import numpy as np
import torch

from ._loss import _nnal_prep, stable_nnal, stable_nnal_derivatives
from ._newton import block_newton_step, solve_W

LATTICES = {
    "fcc": lambda h, k, l: (h % 2 == k % 2 == l % 2),                       # all even or all odd
    "bcc": lambda h, k, l: (h + k + l) % 2 == 0,
    "diamond": lambda h, k, l: (h % 2 == k % 2 == l % 2 == 1) or (h % 2 == k % 2 == l % 2 == 0 and (h + k + l) % 4 == 0),
    "sc": lambda h, k, l: True,
}


def reflections(lattice, a, lam_min, lam_max, hmax=8):
    """Bragg-edge wavelengths 2 d_hkl in [lam_min, lam_max] for a cubic lattice of parameter a, ascending, one per
    distinct h^2 + k^2 + l^2 (families with equal spacing share an edge). Returns (n_values, edges) as arrays."""
    allowed = LATTICES[lattice]
    ns = sorted({h * h + k * k + l * l for h in range(hmax) for k in range(hmax) for l in range(hmax)
                 if (h, k, l) != (0, 0, 0) and allowed(h, k, l)})
    ns = np.array([n for n in ns if lam_min <= 2 * a / math.sqrt(n) <= lam_max], dtype=float)
    return ns, 2 * a / np.sqrt(ns)


def _bspline_basis(lam, n_knots, degree=3):
    """Cubic B-spline design matrix (K x n_basis) with equally spaced interior knots over the wavelength range."""
    from scipy.interpolate import BSpline
    lo, hi = float(lam.min()), float(lam.max())
    inner = np.linspace(lo, hi, n_knots)
    t = np.concatenate([[lo] * degree, inner, [hi] * degree])
    return np.asarray(BSpline.design_matrix(np.asarray(lam, dtype=float), t, degree).todense())


class BraggSpectrum:
    """One material's spectrum on a wavelength grid, as a differentiable function of its parameters.

    theta = (a, heights[n_edges_max], c0, c_abs, spline[n_spline]) as one torch vector. The edge count follows from
    the lattice parameter, so `n_edges_max` slots are allocated for the largest a in the search range and slots
    whose edge falls outside the grid are inactive. Heights, c0 and c_abs are kept nonnegative by squaring the raw
    parameters; the spline residual is free but ridge-penalised (see `penalty`).
    """

    def __init__(self, lam, lattice="fcc", a_range=(3.0, 4.6), n_spline=16, edge_width_rel=0.0, min_width=None):
        self.lam = torch.as_tensor(np.asarray(lam, dtype=np.float64))
        self.lattice = lattice
        self.a_range = a_range
        self.ns_all, _ = reflections(lattice, a_range[1], float(self.lam.min()), float(self.lam.max()) * a_range[1] / a_range[0] + 1e-9)
        self.n_edges = len(self.ns_all)
        self.S = torch.as_tensor(_bspline_basis(self.lam.numpy(), n_spline))           # (K, n_spline)
        self.n_spline = self.S.shape[1]
        self.edge_width_rel = edge_width_rel
        bin_w = float((self.lam[1:] - self.lam[:-1]).mean())
        self.min_width = 0.5 * bin_w if min_width is None else min_width
        self.lam_ref = float(self.lam.mean())

    @property
    def n_params(self):
        return 1 + self.n_edges + 2 + self.n_spline

    def split(self, theta):
        a = theta[0]
        h = theta[1:1 + self.n_edges] ** 2
        c0, c_abs = theta[1 + self.n_edges] ** 2, theta[2 + self.n_edges] ** 2
        spl = theta[3 + self.n_edges:]
        return a, h, c0, c_abs, spl

    def edges(self, a):
        ns = torch.as_tensor(self.ns_all, dtype=torch.float64, device=self.lam.device)
        return 2.0 * a / torch.sqrt(ns)                                               # (n_edges,)

    def __call__(self, theta, lam=None):
        lam = self.lam if lam is None else lam
        a, h, c0, c_abs, spl = self.split(theta)
        e = self.edges(a)                                                             # edge wavelengths
        width = torch.clamp(self.edge_width_rel * e, min=self.min_width)              # resolution width per edge
        # smoothed step: 1 below the edge, 0 above, Gaussian-blurred over `width`
        step = 0.5 * torch.erfc((lam[:, None] - e[None, :]) / (math.sqrt(2.0) * width[None, :]))   # (K, n_edges)
        coherent = (lam / self.lam_ref) ** 2 * (step @ h)
        smooth = c0 + c_abs * lam / self.lam_ref + (self.S.to(lam.device) if lam is not self.lam else self.S) @ spl
        return coherent + smooth

    def penalty(self, theta, ridge):
        """Ridge on the spline's second differences: keeps the residual smooth and small so it does not absorb the
        lambda^2 rise that belongs to the Bragg term."""
        spl = self.split(theta)[4]
        d2 = spl[2:] - 2 * spl[1:-1] + spl[:-2]
        return ridge * ((d2 * d2).sum() + 1e-2 * (spl * spl).sum())

    def design_t(self, a):
        """For a fixed lattice parameter the spectrum is LINEAR in (heights, c0, c_abs, spline): that design matrix
        (K x (n_edges + 2 + n_spline)) as a tensor on the model's device."""
        with torch.no_grad():
            e = self.edges(torch.as_tensor(float(a), dtype=torch.float64))
            width = torch.clamp(self.edge_width_rel * e, min=self.min_width)
            step = 0.5 * torch.erfc((self.lam[:, None] - e[None, :]) / (math.sqrt(2.0) * width[None, :]))
            coh = ((self.lam / self.lam_ref) ** 2)[:, None] * step
            lin = torch.stack([torch.ones_like(self.lam), self.lam / self.lam_ref], 1)
            return torch.cat([coh, lin, self.S], 1)

    def design(self, a):
        """`design_t` as numpy, for bounded least squares in `fit_spectrum`."""
        return self.design_t(a).cpu().numpy()

    def pack(self, a, lin):
        """theta from a lattice parameter and the linear coefficients (heights, c0, c_abs, spline), inverting the squares."""
        lin = np.asarray(lin, dtype=np.float64)
        nn = 2 + self.n_edges
        return torch.as_tensor(np.concatenate([[a], np.sqrt(np.clip(lin[:nn], 0, None)), lin[nn:]]), dtype=torch.float64)


def fit_spectrum(model, target, weights=None, a_grid=None, ridge=1e-3, refine_steps=3):
    """Fit a Bragg spectrum to a target spectrum by weighted least squares.

    The lattice parameter enters non-convexly (an edge more than a resolution width off sees no gradient toward its
    place), so it is scanned on a grid; at each value the remaining parameters are linear and solved by bounded least
    squares (heights, c0, c_abs >= 0; spline free with a ridge). The best grid point is refined by a golden-section
    search. Returns (theta, fit, a, residual_rms)."""
    from scipy.optimize import lsq_linear
    t = np.asarray(target, dtype=np.float64)
    w = np.ones_like(t) if weights is None else np.sqrt(np.asarray(weights, dtype=np.float64))
    nn = model.n_edges + 2
    ridge_rows = np.sqrt(ridge) * np.eye(model.n_spline)
    D2 = np.diff(np.eye(model.n_spline), 2, axis=0) * np.sqrt(ridge * 10)

    def solve(a):
        A = model.design(a)
        Aw = np.vstack([w[:, None] * A, np.hstack([np.zeros((model.n_spline, nn)), ridge_rows]), np.hstack([np.zeros((D2.shape[0], nn)), D2])])
        bw = np.concatenate([w * t, np.zeros(model.n_spline + D2.shape[0])])
        lb = np.concatenate([np.zeros(nn), -np.inf * np.ones(model.n_spline)])
        res = lsq_linear(Aw, bw, bounds=(lb, np.inf), lsmr_tol="auto", max_iter=200)
        fit = A @ res.x
        return float(np.sqrt(np.mean((w * (fit - t)) ** 2))), res.x, fit

    if a_grid is None:
        a_grid = np.linspace(model.a_range[0], model.a_range[1], int((model.a_range[1] - model.a_range[0]) / (0.002 * model.a_range[0])) + 1)
    scan = [solve(a)[0] for a in a_grid]
    i = int(np.argmin(scan)); lo, hi = a_grid[max(i - 1, 0)], a_grid[min(i + 1, len(a_grid) - 1)]
    for _ in range(refine_steps * 6):                                                 # golden section on the bracket
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if solve(m1)[0] < solve(m2)[0]:
            hi = m2
        else:
            lo = m1
    a = 0.5 * (lo + hi); rms, lin, fit = solve(a)
    return model.pack(a, lin), fit, float(a), rms


def _h_statistics(W, H, T, prep):
    """Gradient (R, K) and per-bin Hessian upper triangles (Q, K) of the NNAL with respect to H, W fixed."""
    X = W @ H
    G, Z = stable_nnal_derivatives(X, T, prep)
    R = W.shape[1]
    rows, cols = torch.triu_indices(R, R, device=W.device)
    return W.T @ G, (W[:, rows] * W[:, cols]).T @ Z, rows, cols


def _scan_correlation(Q, lam, lattice, a_values, n_spline, min_width, S_spline, device, chunk=192):
    """Canonical correlation between the row space (orthonormal Q, K x R) and the Bragg model space of `lattice` at
    each parameter in a_values, batched on `device`. The model space for a fixed a is the span of the design matrix
    (edge steps times lambda^2, 1, lambda, spline); its rank-revealing basis comes from a batched SVD so inactive edge
    slots do not inflate it. Returns (sigma_1 per a, the row-space direction per a as (G, K))."""
    lam_t = torch.as_tensor(lam, dtype=torch.float64, device=device); K = lam_t.numel()
    Qt = torch.as_tensor(Q, dtype=torch.float64, device=device)
    a_lo, a_hi = float(np.min(a_values)), float(np.max(a_values))                   # slots for every a in the batch
    ns_all = reflections(lattice, a_hi, float(lam.min()), float(lam.max()) * a_hi / a_lo + 1e-9)[0]
    if len(ns_all) == 0:
        return np.zeros(len(a_values)), np.zeros((len(a_values), K))
    ns_t = torch.as_tensor(ns_all, dtype=torch.float64, device=device)
    fixed = torch.cat([torch.ones(K, 1, dtype=torch.float64, device=device), lam_t[:, None] / lam_t[0],
                       torch.as_tensor(S_spline, dtype=torch.float64, device=device)], 1)                 # (K, 2 + n_spline)
    sig, dirs = [], []
    for c0 in range(0, len(a_values), chunk):
        a = torch.as_tensor(np.asarray(a_values[c0:c0 + chunk]), dtype=torch.float64, device=device)   # (G,)
        e = 2.0 * a[:, None] / torch.sqrt(ns_t)[None, :]                                                   # (G, n_edges)
        step = 0.5 * torch.erfc((lam_t[None, :, None] - e[:, None, :]) / (math.sqrt(2.0) * min_width))    # (G, K, n)
        coh = (lam_t / lam_t[0])[None, :, None] ** 2 * step
        D = torch.cat([coh, fixed[None].expand(coh.shape[0], -1, -1)], 2)                                 # (G, K, n_tot)
        # canonical correlations = singular values of Q^T D (D^T D)^(-1/2), with the Gram matrix's null space (inactive
        # or duplicate edge slots) removed: an n x n eigenproblem per a instead of a K x n SVD
        evals, evecs = torch.linalg.eigh(D.transpose(1, 2) @ D)
        keep = evals > 1e-13 * evals[:, -1:]
        scale = torch.where(keep, evals.clamp(min=1e-300).rsqrt(), torch.zeros_like(evals))
        M = (Qt.T[None] @ D) @ (evecs * scale[:, None, :])                                                 # (G, R, n_tot)
        Um, sm, _ = torch.linalg.svd(M, full_matrices=False)
        y = torch.einsum('kr,gr->gk', Qt, Um[:, :, 0])
        y = y * torch.sign(y.sum(1, keepdim=True))
        sig.append(sm[:, 0].cpu().numpy()); dirs.append(y.cpu().numpy())
    return np.concatenate(sig), np.concatenate(dirs)


def warm_start(H_rows, lam, n_bragg=None, lattice_types=("fcc", "bcc", "diamond"), a_range=(2.0, 7.0), lattices=None,
               edge_width_rel=0.0, ridge=1.0, n_spline=8, polish=2, free_ratio=4.0, free_floor=5e-3, device=None, verbose=False):
    """Bragg parameters for the crystalline components, discovered from the row space of a bilinear factorization.

    Nothing is assumed about the materials: each component's lattice type and parameter are found from the data.
    A bilinear row is a mixture of the pure spectra (the gauge is arbitrary), so a single-lattice fit to one row is
    biased by the other materials' edges. Up to the model's approximation error, a pure spectrum is the intersection
    of the bilinear row space with the model space of its lattice, which for a fixed (type, a) is linear (the span of
    the design matrix). Over a geometric grid of a for every lattice type, the closest pair of directions of the two
    spaces is found by canonical correlation (`_scan_correlation`); local maxima are refined with the sharp edge
    model, then components are chosen greedily by residual 1 - sigma^2 in the order of a BIC on the fit
    with NONNEGATIVE heights to each candidate's own direction (K ln rms^2 + n_edges ln K). The unconstrained
    correlation cannot tell a lattice from one whose edge set contains it (parameter 2a, sqrt(2)a across bcc/fcc, a
    superset type), which always correlates at least as well; with heights kept nonnegative the extra slots buy
    almost nothing and the edge-count penalty favours the smaller edge set. A candidate sharing most of its edges
    with a chosen one is skipped, and one whose edge set contains a smaller candidate's is dropped as an alias
    unless the edges it adds are significant in its own fit (height over standard error above 3) and not another
    smaller candidate's (a superset lattice can hold two materials' edges at once), for at least a third of them; a component whose fit rms exceeds `free_ratio` times the best component's (and `free_floor`, so that
    exact data cannot trip the ratio) is left nonparametric. Lattices whose edge sets are nested (fcc a and bcc a/sqrt2 share every even-index edge) are
    told apart only by the edges they do not share, so a material whose distinguishing edges are weak can be
    reported as the smaller edge set at low dose.

    Args:
        H_rows: rows of the bilinear factorization (R, bins).
        n_bragg: number of crystalline components to seek (default: all rows).
        lattices: optional explicit [(type, (a_lo, a_hi)), ...]; then only the parameter is searched, per component.
        device: torch device for the batched scan (default: cuda if available).

    Returns (thetas, lattices, residuals, free): thetas and lattices=[(type, (a_lo, a_hi))] for the chosen crystalline
    components, their residuals 1 - sigma^2, and `free` the number of components left nonparametric.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Hr = np.asarray(H_rows, dtype=np.float64)
    Hr = Hr / np.maximum(Hr.max(axis=1, keepdims=True), 1e-12)
    lam = np.asarray(lam, dtype=np.float64); K = lam.size; bin_w = float(np.median(np.diff(lam)))
    Q, _ = np.linalg.qr(Hr.T)                                                         # orthonormal basis of the row space
    n_bragg = Hr.shape[0] if n_bragg is None else int(n_bragg)
    S_spline = _bspline_basis(lam, n_spline)
    sharp_w = max(edge_width_rel * float(lam.mean()), 0.5 * bin_w)
    searches = [(lt, ar) for lt, ar in lattices] if lattices is not None else [(lt, tuple(a_range)) for lt in lattice_types]
    candidates = []
    for lt, (lo, hi) in searches:
        # coarse scan with edges smoothed to a few bins, so the correlation peaks are wider than the grid spacing
        grid = np.geomspace(lo, hi, int(np.log(hi / lo) / 0.001) + 2)
        sig, _ = _scan_correlation(Q, lam, lt, grid, n_spline, 3.0 * bin_w, S_spline, "cpu")   # float64 eigh: faster on the CPU
        peaks = [i for i in range(len(grid)) if sig[i] >= sig[max(i - 1, 0)] and sig[i] >= sig[min(i + 1, len(grid) - 1)]]
        peaks = sorted(peaks, key=lambda i: -sig[i])[:12 if lattices is None else 4]
        for i in peaks:                                                               # refine each peak with the sharp model
            lo_i, hi_i = max(lo, grid[i] * (1 - 0.0025)), min(hi, grid[i] * (1 + 0.0025))
            fine = np.linspace(lo_i, hi_i, 26)
            sf, _ = _scan_correlation(Q, lam, lt, fine, n_spline, sharp_w, S_spline, "cpu")
            j = int(np.argmax(sf)); lo_i, hi_i = fine[max(j - 1, 0)], fine[min(j + 1, len(fine) - 1)]
            cache = {}

            def sig_at(a):                                                            # golden-section maximisation of sigma(a):
                if a not in cache:                                                    # the direction is only right at the exact a
                    cache[a] = _scan_correlation(Q, lam, lt, np.array([a]), n_spline, sharp_w, S_spline, "cpu")
                return cache[a][0][0]
            x1, x2 = lo_i + 0.382 * (hi_i - lo_i), lo_i + 0.618 * (hi_i - lo_i)
            for _ in range(16):
                if sig_at(x1) > sig_at(x2):
                    hi_i = x2
                else:
                    lo_i = x1
                x1, x2 = lo_i + 0.382 * (hi_i - lo_i), lo_i + 0.618 * (hi_i - lo_i)
            a = max(cache, key=sig_at); s_a, y_a = cache[a]
            candidates.append(dict(type=lt, a=float(a), resid=max(1.0 - s_a[0] ** 2, 1e-15), y=y_a[0], search=(lo, hi),
                                   edges=reflections(lt, float(a), lam.min(), lam.max())[1]))
    candidates.sort(key=lambda c: c["resid"])
    # Score the plausible candidates by a BIC on the NONNEGATIVE-height fit to their own direction: the canonical
    # correlation is unconstrained, so a lattice whose edge set contains another's (parameter 2a, sqrt(2)a for
    # bcc/fcc, a superset type) always correlates at least as well; with heights kept >= 0 the extra slots buy
    # almost nothing, and the edge-count penalty n ln K then favours the smaller edge set.
    for c in candidates[:4 * n_bragg + 4]:
        m = BraggSpectrum(lam, lattice=c["type"], a_range=(c["a"] * 0.995, c["a"] * 1.005), n_spline=n_spline, edge_width_rel=edge_width_rel)
        target = np.clip(c["y"], 0, None); target /= max(target.max(), 1e-12)
        theta, fit, a, e = fit_spectrum(m, target, a_grid=np.linspace(c["a"] * (1 - 1e-4), c["a"] * (1 + 1e-4), 3), ridge=ridge, refine_steps=1)
        c["a"] = float(a); c["edges"] = reflections(c["type"], c["a"], lam.min(), lam.max())[1]
        c["fit_rms"] = max(e, 1e-12)
        c["bic"] = K * np.log(c["fit_rms"] ** 2) + len(c["edges"]) * np.log(K)
        # significance of each fitted edge: height over its standard error from the (unconstrained) normal equations
        A = m.design(a); cov = np.linalg.pinv(A.T @ A, rcond=1e-10) * c["fit_rms"] ** 2
        h = m.split(theta)[1].numpy(); se = np.sqrt(np.clip(np.diag(cov)[: m.n_edges], 1e-300, None))
        slot_edges = m.edges(theta[0]).numpy()
        c["t"] = {float(ed): float(hi / si) for ed, hi, si in zip(slot_edges, h, se) if lam.min() <= ed <= lam.max()}
    scored = [c for c in candidates if "bic" in c]

    def contains(d, c, frac=0.8, tol=1.5 * bin_w):
        """d's edge set reproduces at least `frac` of c's edges (c the smaller set)."""
        if len(d["edges"]) <= len(c["edges"]) or len(c["edges"]) == 0:
            return False
        return sum(np.min(np.abs(d["edges"] - e)) <= tol for e in c["edges"]) >= frac * len(c["edges"])

    def is_alias(d, c, tol=1.5 * bin_w, t_min=3.0, frac=1 / 3):
        """d contains c's edges; is d only re-describing c (spare slots fitting noise), or c plus another candidate's
        edges (a superset lattice can hold two materials' edges at once and then wins on correlation)? d is genuine
        only if at least `frac` of the edges it adds beyond c are significant AND not explained by any other
        smaller candidate."""
        extra = [(ed, t) for ed, t in d["t"].items() if np.min(np.abs(c["edges"] - ed)) > tol]
        if not extra:
            return True
        others = [o for o in scored if o is not d and o is not c and len(o["edges"]) < len(d["edges"])]
        own = [t > t_min and not any(np.min(np.abs(o["edges"] - ed)) <= tol for o in others) for ed, t in extra]
        return np.mean(own) < frac
    # a lattice containing a smaller candidate's edges (parameter 2a, sqrt2 a across bcc/fcc, a superset type) is an
    # alias of it, or of a mixture, unless the edges it adds are significant and its own
    candidates = sorted([d for d in scored if not any(contains(d, c) and is_alias(d, c) for c in scored if c is not d)],
                        key=lambda c: c["bic"])
    if verbose:
        for c in candidates[:3 * n_bragg]:
            print(f"    candidate {c['type']:8s} a = {c['a']:.4f} A  {len(c['edges']):2d} edges, correlation residual {c['resid']:.2e}, fit rms {c['fit_rms']:.2e}, BIC {c['bic']:.0f}", flush=True)

    def shares_edges(c, d, frac=0.8, tol=1.5 * bin_w):
        if len(c["edges"]) == 0 or len(d["edges"]) == 0:
            return False
        hit = sum(np.min(np.abs(d["edges"] - e)) <= tol for e in c["edges"])
        return hit >= frac * min(len(c["edges"]), len(d["edges"]))

    chosen = []
    for c in candidates:
        if len(chosen) >= n_bragg:
            break
        if any(shares_edges(c, d) or shares_edges(d, c) for d in chosen):
            continue                                                                  # the same edges again: not a new material
        if chosen and c["fit_rms"] > max(free_ratio * chosen[0]["fit_rms"], free_floor):
            break                                                                     # the rest are not crystalline
        chosen.append(c)
    thetas, out_lattices, residuals = [], [], []
    for c in chosen:
        a = c["a"]; lo, hi = c["search"]
        rng = (max(lo, a * 0.98), min(hi, a * 1.02))
        m = BraggSpectrum(lam, lattice=c["type"], a_range=rng, n_spline=n_spline, edge_width_rel=edge_width_rel)
        target = np.clip(c["y"], 0, None); target /= max(target.max(), 1e-12)
        theta, fit, a, e = fit_spectrum(m, target, a_grid=np.linspace(a * (1 - 0.002), a * (1 + 0.002), 9), ridge=ridge, refine_steps=2)
        for _ in range(polish):
            target = Q @ (Q.T @ fit); target = np.clip(target, 0, None) / max(target.max(), 1e-12)
            theta, fit, a, e = fit_spectrum(m, target, a_grid=np.linspace(a * (1 - 0.002), a * (1 + 0.002), 9), ridge=ridge, refine_steps=2)
        if verbose:
            print(f"  component {len(thetas)}: {c['type']} a = {a:.4f} A, {len(c['edges'])} edges in range, residual {c['resid']:.2e}, fit rms {e:.2e}", flush=True)
        thetas.append(theta); out_lattices.append((c["type"], rng)); residuals.append(c["resid"])
    return thetas, out_lattices, residuals, n_bragg - len(chosen)


def _quadratic_step(models, a_vals, g, Msym, H_old, Rb, ridge, P, dev):
    """Exact minimiser of the per-bin quadratic model of the loss (W fixed) over the linear spectral coefficients of
    all Bragg rows, for fixed lattice parameters: a bounded quadratic programme of ~30 unknowns per material.
    Returns (coefficient vectors, model value up to a constant)."""
    from scipy.optimize import lsq_linear
    D = [m.design_t(a).to(dev, torch.float64) for m, a in zip(models, a_vals)]
    sizes = [d.shape[1] for d in D]; n = int(sum(sizes)); off = np.cumsum([0] + sizes)
    Mb = Msym[:, :Rb, :Rb]                                                            # (K, Rb, Rb)
    c = g[:Rb] - torch.einsum('krs,sk->rk', Mb, H_old[:Rb])                          # gradient at H = 0 of the model
    A = torch.zeros(n, n, dtype=torch.float64, device=dev); b = torch.zeros(n, dtype=torch.float64, device=dev)
    for r in range(Rb):
        b[off[r]:off[r + 1]] = D[r].T @ c[r]
        for q in range(Rb):
            A[off[r]:off[r + 1], off[q]:off[q + 1]] = D[r].T @ (Mb[:, r, q, None] * D[q])
    for r, m in enumerate(models):                                                    # spline ridge, as in `penalty`
        ns = m.n_spline; i0 = off[r + 1] - ns; I = torch.eye(ns, dtype=torch.float64, device=dev)
        D2 = torch.diff(I, n=2, dim=0)
        A[i0:off[r + 1], i0:off[r + 1]] += 2.0 * P * ridge * (D2.T @ D2 + 1e-2 * I)
    A = A.cpu().numpy(); b = b.cpu().numpy()
    d = np.sqrt(np.clip(np.diag(A), 1e-300, None))
    As = A / d[:, None] / d[None, :]; bs = b / d
    L = np.linalg.cholesky(As + 1e-9 * np.eye(n))
    lb = np.concatenate([np.concatenate([np.zeros(m.n_edges + 2), -np.inf * np.ones(m.n_spline)]) for m in models])
    res = lsq_linear(L.T, -np.linalg.solve(L, bs), bounds=(lb, np.inf), method="bvls", tol=1e-12)
    x = res.x / d
    return [x[off[r]:off[r + 1]] for r in range(Rb)], float(0.5 * x @ A @ x + b @ x)


def _lattice_search(models, a_vals, r, bracket, *stats):
    """Golden-section search on material r's lattice parameter with the linear coefficients re-solved at each trial
    (variable projection). Returns (a, coefficients, model value)."""
    m = models[r]
    lo = max(m.a_range[0], a_vals[r] - bracket); hi = min(m.a_range[1], a_vals[r] + bracket)
    cache = {}

    def q(a):
        if a not in cache:
            trial = list(a_vals); trial[r] = a
            cache[a] = _quadratic_step(models, trial, *stats)
        return cache[a][1]
    x1, x2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
    for _ in range(8):
        if q(x1) < q(x2):
            hi = x2
        else:
            lo = x1
        x1, x2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
    best = min(list(cache) + [a_vals[r]], key=q)
    return best, cache[best][0], cache[best][1]


def _profiled_system(models, a_vals, W, H, G, Z, x_cur, Rb, ridge, dev, chunk=32768):
    """Newton system of the profiled objective L*(x) = min_W L(W, H(x)) over the linear spectral coefficients x of
    all Bragg rows (lattice parameters fixed), from the current W (assumed near-optimal for the current H).

    By the envelope theorem the gradient of L* is the partial gradient in x and its Hessian is the Schur complement
    A_xx - A_xW A_WW^-1 A_Wx of the joint (W, x) Hessian, accumulated over pixel chunks (A_WW is R x R per pixel,
    restricted to the maps' free set). Returns a dict for `_profiled_solve`."""
    P, K = G.shape; R = W.shape[1]
    D = [m.design_t(a).to(dev, torch.float64) for m, a in zip(models, a_vals)]
    sizes = [d.shape[1] for d in D]; n = int(sum(sizes)); off = np.cumsum([0] + sizes)
    Wd, Hd = W.double(), H.double()
    x0 = torch.as_tensor(np.concatenate(x_cur), dtype=torch.float64, device=dev)
    free = (Wd > 0).double()                                                          # maps' free set (P, R)
    A = torch.zeros(n, n, dtype=torch.float64, device=dev); g = torch.zeros(n, dtype=torch.float64, device=dev)
    S_red = torch.zeros(n, n, dtype=torch.float64, device=dev); g_red = torch.zeros(n, dtype=torch.float64, device=dev)
    WG = Wd.T @ G.double()                                                            # (R, K) gradient in H
    for r in range(Rb):
        g[off[r]:off[r + 1]] = D[r].T @ WG[r]
    Ainv_chunks, C_chunks, gW_chunks = [], [], []
    for p0 in range(0, P, chunk):
        sl = slice(p0, min(p0 + chunk, P)); Wc, Gc, Zc, fc = Wd[sl], G[sl].double(), Z[sl].double(), free[sl]
        Pc = Wc.shape[0]
        for r in range(Rb):
            for q in range(Rb):
                A[off[r]:off[r + 1], off[q]:off[q + 1]] += D[r].T @ (((Wc[:, r] * Wc[:, q]) @ Zc)[:, None] * D[q])
        Ap = torch.einsum('pk,rk,sk->prs', Zc, Hd, Hd)                                # per-pixel W Hessian
        mask = fc[:, :, None] * fc[:, None, :]
        Ap = Ap * mask + torch.diag_embed(1.0 - fc) + 1e-12 * torch.eye(R, dtype=torch.float64, device=dev)
        Ainv = torch.linalg.inv(Ap)
        C = torch.zeros(Pc, R, n, dtype=torch.float64, device=dev)                    # d^2 L / dW_pr dx
        for q in range(Rb):
            for r in range(R):
                C[:, r, off[q]:off[q + 1]] = Wc[:, q, None] * ((Zc * Hd[r][None, :]) @ D[q])
            C[:, q, off[q]:off[q + 1]] += Gc @ D[q]
        C = C * fc[:, :, None]
        gW = (Gc @ Hd.T) * fc                                                          # (Pc, R)
        S_red += torch.einsum('prs,pri,psj->ij', Ainv, C, C)
        g_red += torch.einsum('prs,ps,pri->i', Ainv, gW, C)
        Ainv_chunks.append(Ainv); C_chunks.append(C); gW_chunks.append(gW)
    for r, m in enumerate(models):                                                    # spline ridge, as in `penalty`
        ns = m.n_spline; i0 = off[r + 1] - ns; I = torch.eye(ns, dtype=torch.float64, device=dev)
        D2 = torch.diff(I, n=2, dim=0)
        A[i0:off[r + 1], i0:off[r + 1]] += 2.0 * P * ridge * (D2.T @ D2 + 1e-2 * I)
        g[i0:off[r + 1]] += 2.0 * P * ridge * ((D2.T @ D2 + 1e-2 * I) @ x0[i0:off[r + 1]])
    S = (A - S_red).cpu().numpy(); S = 0.5 * (S + S.T)
    lb = np.concatenate([np.concatenate([np.zeros(m.n_edges + 2), -np.inf * np.ones(m.n_spline)]) for m in models])
    return dict(S=S, g=(g - g_red).cpu().numpy(), x0=x0.cpu().numpy(), off=off, lb=lb, Rb=Rb, free=free, chunk=chunk,
                Ainv=Ainv_chunks, C=C_chunks, gW=gW_chunks, dev=dev, P=P, R=R)


def _profiled_solve(sys, damping):
    """Levenberg-damped bounded quadratic programme on the profiled system, and the maps' predicted response.
    Returns (coefficient vectors, dW, predicted decrease)."""
    from scipy.optimize import lsq_linear
    S, gt, x0, off, lb = sys["S"], sys["g"], sys["x0"], sys["off"], sys["lb"]
    n = S.shape[0]
    d = np.sqrt(np.clip(np.diag(S), 1e-300, None))
    Ss = S / d[:, None] / d[None, :] + damping * np.eye(n)
    jitter = 1e-10
    while True:
        try:
            L = np.linalg.cholesky(Ss + jitter * np.eye(n)); break
        except np.linalg.LinAlgError:
            jitter *= 10
    Sd = Ss * d[:, None] * d[None, :]
    b = (gt - Sd @ x0) / d
    res = lsq_linear(L.T, -np.linalg.solve(L, b), bounds=(lb, np.inf), method="bvls", tol=1e-12)
    x_new = res.x / d
    dx = x_new - x0
    decrease = float(-(gt @ dx + 0.5 * dx @ S @ dx))
    delta = torch.as_tensor(dx, dtype=torch.float64, device=sys["dev"])
    dW = torch.zeros(sys["P"], sys["R"], dtype=torch.float64, device=sys["dev"])
    for i, p0 in enumerate(range(0, sys["P"], sys["chunk"])):
        sl = slice(p0, min(p0 + sys["chunk"], sys["P"]))
        rhs = sys["gW"][i] + torch.einsum('pri,i->pr', sys["C"][i], delta)
        dW[sl] = -torch.einsum('prs,ps->pr', sys["Ainv"][i], rhs) * sys["free"][sl]
    return [x_new[off[r]:off[r + 1]] for r in range(sys["Rb"])], dW, decrease


def hybrid_factorization(T, lam, n_bragg=None, n_free=0, init_rows=None, init_maps=None, lattice_types=("fcc", "bcc", "diamond"),
                         a_range=(2.0, 7.0), lattices=None, thetas=None, H_free_init=None, W_init=None, max_outer=30,
                         rel_tol=1e-6, w_max_steps=100, ridge=1.0, n_spline=8, edge_width_rel=0.0, lattice_bracket=0.002,
                         free_ratio=4.0, verbose=False):
    """Fit n_bragg parametric Bragg spectra plus n_free nonparametric rows to T under the NNAL, assuming nothing about
    the materials: lattice types and parameters are discovered from the data (see `warm_start`).

    Args:
        T: transmission ratio (pixels, bins), on the device.
        lam: wavelength of each bin (bins,), in the units of the lattice parameters (Angstrom).
        n_bragg: number of crystalline components (default: the rows of init_rows minus n_free). A component whose
            spectrum no lattice explains (fit rms above `free_ratio` times the best component's) is made nonparametric instead.
        n_free: number of nonparametric components (amorphous, or anything the Bragg model cannot represent).
        init_rows, init_maps: rows and maps of a bilinear factorization of T (e.g. `nnal_factorization`): the warm
            start derives the lattices and spectra from the rows' span, the least-explained rows initialise the free
            components, and the maps are carried into the new gauge as the initial W (saves most of the first solve).
        lattice_types, a_range: the search space of the discovery (cubic types; a in Angstrom).
        lattices, thetas: optional explicit [(type, (a_lo, a_hi))] and initial parameters, bypassing the discovery.
        W_init: initial maps (pixels, R); default: convex solve with the initial spectra.
        ridge, n_spline: the spline residual's size and ridge (relative to the pixel count); the defaults were tuned on
            the three-metal phantom, where a looser residual trades map fidelity for a slightly lower loss.
        lattice_bracket: initial half-width of the lattice-parameter search, relative to a (shrinks adaptively).

    Each outer iteration: (1) a golden-section search on each lattice parameter and an exact projected Newton step on
    the linear spectral coefficients, both on the convex quadratic model of the loss in H with the maps fixed, Armijo
    checked; (2) the convex per-pixel map solve; (3) a Levenberg-damped Newton step on the PROFILED objective
    (maps' optimal response folded in through the Schur complement of the joint Hessian), which moves along the
    valley of near-equivalent factorizations that alternating steps crawl along; (4) a block Newton step on the free
    rows and a final map solve.

    Returns (W, H, thetas, info): H the assembled spectra (R, bins) in T's dtype; info['lattices'] the discovered
    (type, a) per crystalline component, info['loss'] per iteration.
    """
    dev, dt = T.device, T.dtype
    lam_np = np.asarray(lam, dtype=np.float64)
    lam_t = torch.as_tensor(lam_np, device=dev)
    discovered = None
    if lattices is None:
        if init_rows is None:
            raise ValueError("hybrid_factorization needs init_rows (a bilinear factorization) or explicit lattices")
        rows_np = np.asarray(init_rows, dtype=np.float64)
        n_bragg = rows_np.shape[0] - n_free if n_bragg is None else int(n_bragg)
        thetas, lattices, discovered, extra_free = warm_start(rows_np, lam_np, n_bragg=n_bragg, lattice_types=lattice_types,
                                                              a_range=a_range, edge_width_rel=edge_width_rel, ridge=ridge,
                                                              n_spline=n_spline, free_ratio=free_ratio, device=dev, verbose=verbose)
        n_free += extra_free
    models = [BraggSpectrum(lam_t.cpu(), lattice=l, a_range=ar, n_spline=n_spline, edge_width_rel=edge_width_rel) for l, ar in lattices]
    for m in models:
        m.lam = lam_t; m.S = m.S.to(dev)
    Rb = len(models); R = Rb + n_free
    if thetas is None and init_rows is not None:
        rows_np = np.asarray(init_rows, dtype=np.float64)
        thetas, _, _, _ = warm_start(rows_np, lam_np, lattices=lattices, edge_width_rel=edge_width_rel, ridge=ridge,
                                     n_spline=n_spline, device=dev, verbose=verbose)
        if n_free and H_free_init is None:
            # free components start from the rows least explained by the fitted Bragg spectra
            Hb0 = np.stack([m(th.to(lam_t.device)).cpu().numpy() for m, th in zip(models, thetas)])
            resid = rows_np - (np.linalg.lstsq(Hb0.T, rows_np.T, rcond=None)[0].T @ Hb0)
            worst = np.argsort(-np.linalg.norm(resid, axis=1) / np.linalg.norm(rows_np, axis=1))[:n_free]
            H_free_init = torch.as_tensor(rows_np[worst])
    if thetas is None:
        thetas = [m.pack(0.5 * (m.a_range[0] + m.a_range[1]), np.concatenate([0.05 * np.ones(m.n_edges), [0.05, 0.05], np.zeros(m.n_spline)])) for m in models]
    thetas = [th.to(dev, torch.float64).clone() for th in thetas]
    prep = _nnal_prep(T)
    P, K = T.shape

    def assemble(ths, H_free):
        rows = [m(th).to(dt) for m, th in zip(models, ths)]
        if n_free:
            rows.append(H_free.to(dt))
        return torch.cat([r.reshape(-1, K) for r in rows], 0).clamp_(min=0)

    H_free = None
    if n_free:
        H_free = (H_free_init.to(dev, dt) if H_free_init is not None else 0.1 * torch.rand(n_free, K, device=dev, dtype=dt))
    H = assemble(thetas, H_free)
    if W_init is None and init_maps is not None and init_rows is not None:
        Hb = torch.as_tensor(np.asarray(init_rows, dtype=np.float64), device=dev)
        M = torch.linalg.lstsq(H.double().T, Hb.T)[0].T                              # Hb ~ M @ H, so W_b Hb ~ (W_b M) H
        W_init = (torch.as_tensor(init_maps, device=dev).double() @ M).clamp(min=0).to(dt)
    W = solve_W(T, H, W_init.to(dev, dt) if W_init is not None else None, w_max_steps, 1e-10)
    loss = stable_nnal(W @ H, T, prep, dtype=torch.float64).item()
    info = dict(loss=[loss], a=[[float(th[0]) for th in thetas]], accepted=[], residuals=discovered,
                lattices=[(l, float(th[0])) for (l, _), th in zip(lattices, thetas)])
    brackets = [lattice_bracket * float(th[0]) for th in thetas]
    damping = 1e-3
    for outer in range(1, max_outer + 1):
        loss_start = loss
        # ---- spectral parameters, three moves per iteration.
        # (a) Lattice parameters: golden-section search on the W-fixed quadratic model of the loss in H (edge
        #     positions are local in wavelength, so the maps' response matters little there), and
        # (b) linear coefficients: exact minimiser of that convex model (projected Newton, W fixed), Armijo-checked.
        g, flat, rows, cols = _h_statistics(W, H, T, prep)
        Msym = torch.zeros(K, R, R, dtype=torch.float64, device=dev)
        Msym[:, rows, cols] = flat.T.double(); Msym[:, cols, rows] = flat.T.double()
        stats = (g.double(), Msym, H.double(), Rb, ridge, P, dev)
        a_vals = [float(th[0]) for th in thetas]
        coeffs, _ = _quadratic_step(models, a_vals, *stats)
        for r in range(Rb):
            a_new, coeffs, _ = _lattice_search(models, a_vals, r, brackets[r], *stats)
            brackets[r] = float(min(max(4.0 * abs(a_new - a_vals[r]), 2e-5 * a_new), lattice_bracket * a_new))
            a_vals[r] = a_new
        new_thetas = [m.pack(a, x).to(dev) for m, a, x in zip(models, a_vals, coeffs)]
        step = 1.0; accepted = False
        for _ in range(6):
            trial = [th + step * (nth - th) for th, nth in zip(thetas, new_thetas)]
            H_trial = assemble(trial, H_free)
            l_trial = stable_nnal(W @ H_trial, T, prep, dtype=torch.float64).item()
            if l_trial <= loss * (1 + 1e-12):
                thetas, H, loss, accepted = trial, H_trial, l_trial, True
                break
            step *= 0.5
        if accepted:
            W = solve_W(T, H, W, w_max_steps, 1e-10)
            loss = stable_nnal(W @ H, T, prep, dtype=torch.float64).item()
        # (c) Linear coefficients again, now as a Levenberg-damped Newton step on the PROFILED objective (the maps'
        #     optimal response folded in through the Schur complement): this is what moves along the valley of
        #     near-equivalent factorizations that the alternating moves crawl along.
        X = W @ H
        G, Z = stable_nnal_derivatives(X, T, prep)
        a_vals = [float(th[0]) for th in thetas]
        x_cur = [np.concatenate([v.cpu().numpy().ravel() for v in m.split(th)[1:]]) for m, th in zip(models, thetas)]
        system = _profiled_system(models, a_vals, W, H, G, Z, x_cur, Rb, ridge, dev)
        del G, Z, X                                                                   # keep the device memory for the solves
        for _ in range(6):
            coeffs, dW, predicted = _profiled_solve(system, damping)
            trial = [m.pack(a, x).to(dev) for m, a, x in zip(models, a_vals, coeffs)]
            H_trial = assemble(trial, H_free); W_trial = (W + dW.to(dt)).clamp_(min=0)
            l_trial = stable_nnal(W_trial @ H_trial, T, prep, dtype=torch.float64).item()
            if l_trial < loss and predicted > 0:
                ratio = (loss - l_trial) / predicted
                damping = max(damping / 3.0, 1e-8) if ratio > 0.5 else min(damping * 3.0, 1e3)
                thetas, H, W, loss = trial, H_trial, W_trial, l_trial
                W = solve_W(T, H, W, w_max_steps, 1e-10)
                loss = stable_nnal(W @ H, T, prep, dtype=torch.float64).item()
                break
            damping = min(damping * 10.0, 1e6)
        del system
        info["accepted"].append(accepted)
        # ---- free rows: one block Newton step on H (then the Bragg rows are re-imposed)
        if n_free:
            X = W @ H
            H_new, _, _ = block_newton_step(H.clone(), W, X, T, prep, 1)
            H_free = H_new[Rb:]
            H = assemble(thetas, H_free)
            W = solve_W(T, H, W, w_max_steps, 1e-10)
            loss = stable_nnal(W @ H, T, prep, dtype=torch.float64).item()
        info["loss"].append(loss); info["a"].append([float(th[0]) for th in thetas])   # W is optimal for H here
        if verbose:
            print(f"  outer {outer:2d}: loss {loss:.6f}  a = {[round(float(th[0]), 4) for th in thetas]}  theta step {'ok' if accepted else 'rejected'}", flush=True)
        if abs(loss_start - loss) <= rel_tol * abs(loss) and outer > 2:
            break
    info["outer"] = outer; info["lattices"] = [(l, float(th[0])) for (l, _), th in zip(lattices, thetas)]
    return W, H, thetas, info

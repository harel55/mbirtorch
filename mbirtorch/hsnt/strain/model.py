"""Hybrid Bragg model, version 2: open-beam artifacts, a measured wavelength axis, a wavelength-dependent resolution
kernel applied to the transmission, and per-pixel lattice strain.

Forward model for pixel p and measured bin k (transmission ratio = sample counts / open-beam counts):

    m_pk = s_k * n_pk + sum_j f_jk * beta_pj,      s_k = 1 - sum_j f_jk
    n_pk = sum_i Rt_ki exp(-X_pi)                       neutrons, through the instrument
    X_pi = sum_r W_pr mu_r(lam_i; eps_pr)               attenuation on a fine, uniform wavelength grid lam_i
    beta_pj = exp(-sum_r kappa_jr W_pr)                 what the sample passes of contamination j

- The fine grid spans the measured range and the capture gaps; each measured bin covers its own wavelength
  interval [lo_k, hi_k] (from the bin centres, never assuming a constant step).
- Rt is the resolution operator, banded: Rt_ki is proportional to Phi_i times the probability that a neutron of true
  wavelength lam_i is recorded in bin k, normalized over i; Phi is the incident flux (the measured open beam,
  interpolated across the gaps). Kernel: exponentially modified Gaussian centred on its median, sigma(lam)^2 =
  s0^2 + (s1 lam)^2, tau(lam) = t0 + t1 lam: generic time-of-flight terms, no beamline values.
- mu_r: a Bragg material, (lam / lam_ref)^2 sum_e h_e step(2 d_e (1 + eps) - lam) + c0 + c1 lam / lam_ref +
  spline(lam), heights >= 0. eps is a per-pixel lattice dilation (strain), exact; off unless asked for.
- Contamination j: an additive share f_jk >= 0 of bin k's open beam that does not follow exp(-X) (a prompt flash,
  delayed gammas, scatter, ...). Its sample transmission beta depends on the maps through one coefficient per
  material. f is free per bin and stays at zero unless the data ask for it: an L1 threshold of z standard
  deviations of its score at zero (z: Bonferroni level over the bins; standard deviation from the Fisher
  information and the fit's Pearson dispersion).

Stage A (StageA) fits free spectra H (R, K) on the measured bins with the same contamination terms, no resolution:
the contamination is separated from the neutron spectra before any lattice is sought. Stage B (Hybrid2 + FitB)
replaces H by Bragg materials. Loss: Poisson deviance of the counts, sum_pk d_k (m - T + T log(T / m)), d_k the open
beam's counts per pixel in bin k; plus the L1 on f and a ridge on the splines.
"""
import math

import numpy as np
import torch
from scipy.stats import norm

from .bragg import emg_median, reflections
from ._dist import all_sum, all_sum_float


def multiplicities(lattice, ns):
    """Number of (h, k, l), signs and permutations included, allowed by the lattice type for each h^2 + k^2 + l^2."""
    from .bragg import LATTICES
    allowed = LATTICES[lattice]
    out = []
    for n in ns:
        n = int(round(n)); hm = int(math.isqrt(n)) + 1
        out.append(sum(1 for h in range(-hm, hm + 1) for k in range(-hm, hm + 1) for l in range(-hm, hm + 1)
                       if h * h + k * k + l * l == n and allowed(abs(h), abs(k), abs(l))))
    return out


# A global parameter whose Fisher information is below this fraction of the largest is held in the global step.
INFO_FLOOR = 1e-12


def deviance_terms(m, T, d):
    m = m.clamp_min(1e-12)
    tlogt = torch.where(T > 0, T * torch.log(T.clamp_min(1e-30)), torch.zeros_like(T))
    return d * (m - T - T * torch.log(m) + tlogt)


def _l1_threshold(Fi0, dispersion, z):
    return z * torch.sqrt(dispersion * Fi0.clamp_min(0))


# ================================================================================================ stage A
class StageA:
    """Free nonnegative spectra H (R, K) and maps W (P, R) with contamination terms (f, kappa), on the measured bins.
    All arrays on the device at once (the cylinder test is 16k pixels)."""

    def __init__(self, T, d, W, H, n_contam=2, kappa=None, z=None, alpha=0.05, dtype=torch.float32):
        self.T, self.d = T, d.to(T.dtype)
        self.W, self.H = W.clone(), H.clone()
        self.R, self.K, self.J = H.shape[0], H.shape[1], n_contam
        self.f = torch.zeros(self.J, self.K, device=T.device, dtype=T.dtype)
        mu_bar = float(H.mean())
        self.kappa = (torch.as_tensor(kappa, device=T.device, dtype=T.dtype) if kappa is not None else
                      torch.stack([torch.full((self.R,), mu_bar * (0.1 + 0.4 * j)) for j in range(self.J)]).to(T.device, T.dtype)
                      if self.J else torch.zeros(0, self.R, device=T.device, dtype=T.dtype))
        self.z = norm.isf(alpha / (2 * self.K)) if z is None else z
        self.lam1 = torch.zeros_like(self.f)
        self.dispersion = 1.0

    def parts(self, W=None, H=None, f=None, kappa=None):
        W = self.W if W is None else W; H = self.H if H is None else H
        f = self.f if f is None else f; kappa = self.kappa if kappa is None else kappa
        e = torch.exp(-(W @ H))
        beta = torch.exp(-(W @ kappa.T)) if self.J else torch.zeros(W.shape[0], 0, device=W.device, dtype=W.dtype)
        s = 1 - f.sum(0)
        return s * e + beta @ f, e, beta, s

    def loss(self, **kw):
        m = self.parts(**kw)[0]
        f = kw.get('f', self.f)
        return float(deviance_terms(m, self.T, self.d).sum(dtype=torch.float64)) + float((self.lam1 * f).sum())

    def pixel_step(self):
        m, e, beta, s = self.parts()
        T, d, H, f, kap = self.T, self.d, self.H, self.f, self.kappa
        # dm/dW_pr = -s e_pk H_rk - sum_j f_jk beta_pj kappa_jr
        J = -(s * e)[:, :, None] * H.T[None] - torch.einsum('jk,pj,jr->pkr', f, beta, kap)
        mm = m.clamp_min(1e-12)
        grad = torch.einsum('pk,pkr->pr', d * (1 - T / mm), J).double()
        F = torch.einsum('pk,pka,pkb->pab', d / mm, J, J).double()
        self.W = _projected_newton_rows(self.W, grad, F, lambda Wn: deviance_terms(self.parts(W=Wn)[0], T, d).sum(1, dtype=torch.float64))

    def bin_step(self):
        """Per bin: (H_k, f_k) by projected Newton, f with the L1 threshold."""
        m, e, beta, s = self.parts()
        T, d, W, f = self.T, self.d, self.W, self.f
        R, Jc = self.R, self.J
        mm = m.clamp_min(1e-12)
        # derivatives: dm/dH_rk = -s_k e_pk W_pr ; dm/df_jk = beta_pj - e_pk; accumulated in pixel chunks
        n = R + Jc
        grad = torch.zeros(self.K, n, device=T.device, dtype=torch.float64)
        F = torch.zeros(self.K, n, n, device=T.device, dtype=torch.float64)
        Fi0 = torch.zeros(self.K, Jc, device=T.device, dtype=torch.float64)
        for i in range(0, T.shape[0], 2048):
            sl = slice(i, i + 2048)
            dH = -(s * e[sl])[:, :, None] * W[sl, None, :]
            dF = beta[sl, None, :] - e[sl, :, None]
            Jac = torch.cat([dH, dF], 2)
            grad += torch.einsum('pk,pkn->kn', d * (1 - T[sl] / mm[sl]), Jac).double()
            F += torch.einsum('pk,pka,pkb->kab', d / mm[sl], Jac, Jac).double()
            # the threshold: the score's standard deviation at f = 0 (current H), with the Pearson dispersion
            Fi0 += torch.einsum('pk,pkj->kj', d / e[sl].clamp_min(1e-12), dF * dF).double()
        self.dispersion = float((d * (T - m) ** 2 / mm).sum(dtype=torch.float64) / T.numel())
        self.lam1 = _l1_threshold(Fi0, self.dispersion, self.z).T.to(T.dtype)          # (J, K)
        grad[:, R:] += self.lam1.T.double()
        V = torch.cat([self.H.T, f.T], 1)                                             # (K, R + J)

        def per_bin_loss(Vn):
            Hn, fn = Vn[:, :R].T, Vn[:, R:].T
            mn = self.parts(H=Hn, f=fn)[0]
            return deviance_terms(mn, T, d).sum(0, dtype=torch.float64) + (self.lam1 * fn).sum(0).double()
        V = _projected_newton_rows(V, grad, F, per_bin_loss, cap=(R, 0.99))
        self.H, self.f = V[:, :R].T.contiguous(), V[:, R:].T.contiguous()

    def kappa_step(self):
        if not self.J or not bool((self.f > 0).any()):
            return
        m, e, beta, s = self.parts()
        T, d, W, f = self.T, self.d, self.W, self.f
        mm = m.clamp_min(1e-12)
        # dm/dkappa_jr = -f_jk beta_pj W_pr
        Jk = -(f[None, :, :, None] * (beta[:, :, None, None] * W[:, None, None, :]))   # (P, J, K, R)
        Jk = Jk.permute(0, 2, 1, 3).reshape(T.shape[0], self.K, -1)
        grad = torch.einsum('pk,pkn->n', d * (1 - T / mm), Jk).double()
        F = torch.einsum('pk,pka,pkb->ab', d / mm, Jk, Jk).double()
        x0 = self.kappa.flatten().double()
        l0 = self.loss(); lam = 1e-3
        for _ in range(8):
            step = torch.linalg.solve(F + lam * torch.diag(torch.diagonal(F).clamp_min(1e-12)), grad)
            k_new = (x0 - step).clamp_min(0).reshape(self.J, self.R).to(T.dtype)
            if self.loss(kappa=k_new) < l0:
                self.kappa = k_new; return
            lam *= 10

    def fit(self, rounds=20, verbose=True, tol=1e-7):
        hist = []
        prev = None
        for it in range(rounds):
            self.pixel_step(); self.bin_step(); self.kappa_step()
            l = self.loss()
            hist.append(dict(it=it, loss=l, active=[int((self.f[j] > 0).sum()) for j in range(self.J)],
                             kappa=self.kappa.flatten().tolist(), dispersion=self.dispersion))
            if verbose:
                print(f"  A {it}: loss {l:.2f} active {hist[-1]['active']} kappa {[round(v, 4) for v in hist[-1]['kappa']]} "
                      f"disp {self.dispersion:.3f}", flush=True)
            if prev is not None and abs(prev - l) <= tol * abs(l):
                break
            prev = l
        return hist


def _projected_newton_rows(V, grad, F, row_loss, cap=None):
    """One projected Newton step per row of V (rows independent): entries at zero with a positive gradient are held,
    the others take the Newton step, entries are clipped at zero, and each row backtracks on row_loss. cap = (start,
    max) bounds the sum of the entries from column `start` on (the contamination shares)."""
    dt = V.dtype
    n = V.shape[1]
    cur = V.double()
    active = (cur <= 0) & (grad > 0)
    keep = (~active).double()
    eye = torch.eye(n, device=V.device, dtype=torch.float64)
    jitter = 1e-10 * torch.diagonal(F, dim1=1, dim2=2).abs().mean(1).clamp_min(1e-30)[:, None, None] * eye
    Fm = (F + jitter) * keep[:, :, None] * keep[:, None, :] + torch.diag_embed(active.double())
    rhs = torch.where(active, torch.zeros_like(grad), grad)
    step = torch.linalg.solve(Fm, rhs[:, :, None])[:, :, 0]
    base = row_loss(V)
    t = torch.ones(V.shape[0], device=V.device, dtype=torch.float64)
    done = torch.zeros_like(t, dtype=torch.bool)
    best = cur.clone()
    for _ in range(12):
        new = (cur - t[:, None] * step).clamp_min(0)
        if cap is not None:
            c0, cmax = cap
            tot = new[:, c0:].sum(1, keepdim=True)
            new[:, c0:] = torch.where(tot > cmax, new[:, c0:] * cmax / tot.clamp_min(1e-30), new[:, c0:])
        l = row_loss(new.to(dt))
        ok = (l <= base) & ~done
        best[ok] = new[ok]; done |= ok
        if bool(done.all()):
            break
        t = torch.where(done, t, 0.5 * t)
    return best.to(dt)


# ================================================================================================ stage B
IC_OVERSAMPLE = 4          # the Ikeda-Carpenter kernel's offset grid: this many points per fine wavelength bin
IC_MARGIN = 20             # fine bins of offset grid beyond the band on each side


def bin_intervals(lam):
    lam = np.asarray(lam, dtype=np.float64)
    d = np.diff(lam); med = np.median(d)
    half_right = np.where(d > 1.5 * med, 0.5 * med, 0.5 * d)
    return lam - np.concatenate([[0.5 * med], half_right]), lam + np.concatenate([half_right, [0.5 * med]])


def _emg_cdf(x, sigma, tau):
    z = x / sigma
    r = sigma / tau
    return 0.5 * torch.erfc(-z / math.sqrt(2.0)) - torch.exp(-x / tau + 0.5 * r * r + torch.special.log_ndtr(z - r))


def bspline_basis(x, n_knots, lo, hi, degree=3):
    from scipy.interpolate import BSpline
    inner = np.linspace(lo, hi, n_knots)
    t = np.concatenate([[lo] * degree, inner, [hi] * degree])
    return np.asarray(BSpline.design_matrix(np.clip(np.asarray(x, dtype=float), lo, hi), t, degree).todense())


class Hybrid2:
    """Structure of the stage-B model: grids, the banded resolution operator, materials and the parameter layout.
    Global parameter vector g: per material [a, h (n_e), c0, c1, spline (n_s)], then res (s0, s1, t0, t1), then
    kappa (J x R)."""
    BAND = (-240, 220)                                   # fine-grid offsets of a bin's band: the tail is on the left

    def __init__(self, lam, dose_per_bin, lattices, n_spline=16, n_contam=2, device="cuda", dtype=torch.float32, pad=0.1,
                 edge_slope=False, two_scale=False, shape_model=None, kernel="two_scale"):
        self.dev, self.dt = device, dtype
        lam = np.asarray(lam, dtype=np.float64)
        lo, hi = bin_intervals(lam)
        hf = float(np.median(np.diff(lam)))
        lamf = np.arange(lam[0] - pad, lam[-1] + pad + 0.5 * hf, hf)
        phi = np.interp(lamf, lam, np.asarray(dose_per_bin, dtype=np.float64))
        f64 = dict(device=device, dtype=torch.float64)
        self.hf = hf
        self.lam = torch.as_tensor(lam, **f64); self.lamf = torch.as_tensor(lamf, **f64)
        self.lam_lo = torch.as_tensor(lo, **f64); self.lam_hi = torch.as_tensor(hi, **f64)
        self.phi = torch.as_tensor(phi / phi.max(), **f64)
        self.d = torch.as_tensor(np.asarray(dose_per_bin, dtype=np.float64), device=device, dtype=dtype)
        self.K, self.Kf = lam.size, lamf.size
        jk = np.rint((lam - lamf[0]) / hf).astype(np.int64)
        offs = np.arange(self.BAND[0], self.BAND[1] + 1)
        idx = np.clip(jk[:, None] + offs[None, :], 0, self.Kf - 1)
        self.band_idx = torch.as_tensor(idx, device=device)                         # (K, bw) fine indices
        self.band_valid = torch.as_tensor((jk[:, None] + offs[None, :] >= 0) & (jk[:, None] + offs[None, :] < self.Kf), device=device)
        self.lam_ref = float(lam.mean())
        self.B = torch.as_tensor(bspline_basis(lamf, n_spline, lamf[0], lamf[-1]), device=device, dtype=dtype)
        self.n_spline = self.B.shape[1]
        self.R = len(lattices); self.J = n_contam
        self.mats = []
        for lt, (a_lo, a_hi) in lattices:
            ns = reflections(lt, a_hi, float(lamf[0]) * a_lo / a_hi, float(lamf[-1]) * a_hi / a_lo)[0]
            self.mats.append(dict(lattice=lt, ns=torch.as_tensor(ns, device=device, dtype=torch.float64), a_range=(a_lo, a_hi),
                                  mult=torch.as_tensor(multiplicities(lt, ns), device=device, dtype=torch.float64)))
        self.shape_model = shape_model
        self.kernel = kernel
        self.slices, off = [], 0
        for m in self.mats:
            ne = len(m["ns"]); o = off + 1 + 2 * ne
            sl = dict(a=off, h=(off + 1, off + 1 + ne), t=(off + 1 + ne, o), x=(o, o + 2), c=(o + 2, o + 4),
                      s=(o + 4, o + 4 + self.n_spline))
            self.slices.append(sl); off = sl["s"][1]
        self.res_sl = (off, off + 8); off += 8
        self.kap_sl = (off, off + self.J * self.R); off += self.J * self.R
        self.n_global = off
        lb = np.full(off, -np.inf); ub = np.full(off, np.inf)
        for m, sl in zip(self.mats, self.slices):
            lb[sl["a"]], ub[sl["a"]] = m["a_range"]
            lb[sl["h"][0]:sl["h"][1]] = 0.0
            # per-reflection first-order shape (extinction, texture): t in [-1, 1] keeps the term >= 0 below the edge
            lb[sl["t"][0]:sl["t"][1]] = -1.0 if edge_slope else 0.0
            ub[sl["t"][0]:sl["t"][1]] = 1.0 if edge_slope else 0.0
            # linked shape models, one block per material: extinction (strength, unused) or texture (r, psi)
            if shape_model == "extinction":
                lb[sl["x"][0]], ub[sl["x"][0]] = 0.0, 1e4
                lb[sl["x"][0] + 1] = ub[sl["x"][0] + 1] = 0.0
            elif shape_model == "texture":
                lb[sl["x"][0]], ub[sl["x"][0]] = 0.3, 3.0
                lb[sl["x"][0] + 1], ub[sl["x"][0] + 1] = 0.0, math.pi / 2
            else:
                lb[sl["x"][0]:sl["x"][1]] = ub[sl["x"][0]:sl["x"][1]] = 0.0
        # widths within what the band holds: 5 sigma <= 48 fine bins, 5 tau <= ~240 at 4.4 A
        # (s0, s1, t0, t1) of the EMG core; (w_b, sigma_b) of a broader Gaussian component (two-scale kernel)
        # slots: (s0, s1, t0, t1) of the EMG core; (w0, b0, b1, w1) of the second component, by kernel:
        #   two_scale: broad Gaussian, weight w0, width b0 (b1 = w1 = 0)
        #   two_scale_lam: broad Gaussian, width sqrt(b0^2 + (b1 lam)^2), weight w0 + w1 (lam - 3 A)
        #   two_tail: a second exponential tail of decay b0 + b1 lam, weight w0 + w1 (lam - 3 A), on the same Gaussian
        #   ic: storage share exp(-(w0 / lam)^2), storage decay b0
        lb[self.res_sl[0]:self.res_sl[1]] = [0.0, 0.0, 1e-6, 0.0, 0.0, 0.004, 0.0, 0.0]
        ub[self.res_sl[0]:self.res_sl[1]] = [0.006, 0.002, 0.02, 0.008, 0.4 if two_scale else 0.0, 0.04 if two_scale else 0.004, 0.0, 0.0]
        if kernel in ("two_scale_lam", "two_tail"):
            lb[self.res_sl[0] + 4], ub[self.res_sl[0] + 4] = 0.0, 0.6
            lb[self.res_sl[0] + 5], ub[self.res_sl[0] + 5] = (0.001, 0.04) if kernel == "two_scale_lam" else (0.0, 0.04)
            lb[self.res_sl[0] + 6], ub[self.res_sl[0] + 6] = 0.0, 0.01
            lb[self.res_sl[0] + 7], ub[self.res_sl[0] + 7] = -0.3, 0.3
        if kernel == "ic":
            # moderator storage tail (Ikeda-Carpenter): last two slots = (lam_R, t_s): storage share
            # R(lam) = exp(-(lam_R / lam)^2), storage decay t_s (constant in time, so in wavelength)
            lb[self.res_sl[0] + 4], ub[self.res_sl[0] + 4] = 0.0, 10.0
            lb[self.res_sl[0] + 5], ub[self.res_sl[0] + 5] = 0.002, 0.04
        if kernel == "ikeda_carpenter":
            # slots (s0, s1, a0, a1, lam_R, t_s): Gaussian sqrt(s0^2 + (s1 lam)^2); the moderator's slowing-down pulse
            # gamma-shaped, x^2 exp(-x / tau) with tau = a0 + a1 lam; its storage part an exponential of length t_s with
            # the share R = exp(-(lam_R / lam)^2) (Ikeda and Carpenter 1985), all in wavelength units
            lb[self.res_sl[0]:self.res_sl[0] + 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0002]
            ub[self.res_sl[0]:self.res_sl[0] + 6] = [0.006, 0.002, 0.01, 0.004, 10.0, 0.04]
        lb[self.kap_sl[0]:self.kap_sl[1]] = 0.0
        self.lb = torch.as_tensor(lb, **f64); self.ub = torch.as_tensor(ub, **f64)

    # -- spectra and their parameter derivatives
    def spectra(self, g, eps=None):
        """mu (R, Kf), float64; or (Pc, R, Kf) with per-pixel dilations eps (Pc, R)."""
        x = self.lamf / self.lam_ref
        out = []
        for r, (m, sl) in enumerate(zip(self.mats, self.slices)):
            a = g[sl["a"]]; h = g[sl["h"][0]:sl["h"][1]]; t = g[sl["t"][0]:sl["t"][1]]
            xp = g[sl["x"][0]:sl["x"][1]]
            c = g[sl["c"][0]:sl["c"][1]]; s = g[sl["s"][0]:sl["s"][1]]
            e = 2.0 * a / torch.sqrt(m["ns"])
            smooth = c[0] + c[1] * x + self.B.double() @ s
            if eps is None:
                step = ((e[None, :] - (self.lamf[:, None] - 0.5 * self.hf)) / self.hf).clamp(0, 1)
                shape = 1 + t[None, :] * (self.lamf[:, None] - e[None, :]) / e[None, :]          # (Kf, n_e)
                shape = shape * self.linked_shape(self.lamf[:, None], e[None, :], h[None, :], m["mult"][None, :], xp)
                out.append(x * x * ((step * shape) @ h) + smooth)
            else:
                ee = e[None, :] * (1 + eps[:, r:r + 1].double())
                step = ((ee[:, :, None] - (self.lamf[None, None, :] - 0.5 * self.hf)) / self.hf).clamp(0, 1)
                shape = 1 + t[None, :, None] * (self.lamf[None, None, :] - ee[:, :, None]) / ee[:, :, None]
                shape = shape * self.linked_shape(self.lamf[None, None, :], ee[:, :, None], h[None, :, None],
                                                  m["mult"][None, :, None], xp)
                out.append(x * x * torch.einsum('pek,e->pk', step * shape, h) + smooth)
        return torch.stack(out, 0) if eps is None else torch.stack(out, 1)

    def edge_terms(self, g, eps, r):
        """Material r's edge terms under per-pixel dilations eps (Pc, R): (Pc, n_edges, Kf), each the spectrum's
        derivative with respect to that edge's height when there is no linked shape model."""
        m, sl = self.mats[r], self.slices[r]
        with torch.no_grad():
            a = g[sl["a"]]; t = g[sl["t"][0]:sl["t"][1]]
            x = self.lamf / self.lam_ref
            ee = (2.0 * a / torch.sqrt(m["ns"]))[None, :] * (1 + eps[:, r:r + 1].double())
            step = ((ee[:, :, None] - (self.lamf[None, None, :] - 0.5 * self.hf)) / self.hf).clamp(0, 1)
            shape = 1 + t[None, :, None] * (self.lamf[None, None, :] - ee[:, :, None]) / ee[:, :, None]
            return x * x * step * shape

    def linked_shape(self, lam, edge, h, mult, xp):
        """Per-reflection shape factor below each edge from a few shared parameters (shapes broadcast; lam the
        wavelength, edge = 2 d, h the reflection's fitted height, mult its multiplicity).

        extinction (Sabine 1988): E = E_L cos^2(theta) + E_B sin^2(theta), E_B = 1 / sqrt(1 + x), E_L its series
        (x <= 1) or asymptotic form, with x = strength * (lam / lam_ref)^3 * |F|^2 / sin(2 theta) and |F|^2 taken from
        the fitted height (h is proportional to mult * |F|^2 * d), so no structure factor is assumed.
        texture (March-Dollase, axis at angle psi from the beam): the pole density
        (r^2 cos^2 alpha + sin^2 alpha / r)^(-3/2) averaged over the cone of plane normals that diffract at the
        Bragg angle (normals at 90 deg - theta from the beam); r = 1 is a random powder."""
        if self.shape_model not in ("extinction", "texture"):
            return torch.ones((), dtype=torch.float64, device=self.dev)
        sin_t = (lam / edge).clamp(1e-6, 1 - 1e-9)                                     # sin(theta) = lam / (2 d)
        cos_t = torch.sqrt(1 - sin_t * sin_t)
        if self.shape_model == "extinction":
            F2 = h / (mult * edge / self.lam_ref).clamp_min(1e-30)
            xx = xp[0] * (lam / self.lam_ref) ** 3 * F2 / (2 * sin_t * cos_t).clamp_min(1e-6)
            EB = 1 / torch.sqrt(1 + xx)
            xs = xx.clamp(max=1.0); xl = xx.clamp(min=1.0)
            EL_small = 1 - xs / 2 + xs ** 2 / 4 - 5 * xs ** 3 / 48 + 7 * xs ** 4 / 192
            EL_large = torch.sqrt(2 / (math.pi * xl)) * (1 - 1 / (8 * xl) - 3 / (128 * xl ** 2) - 15 / (1024 * xl ** 3))
            EL = torch.where(xx <= 1, EL_small, EL_large)
            return EL * cos_t ** 2 + EB * sin_t ** 2
        r_md, psi = xp[0], xp[1]
        phi = torch.linspace(0, 2 * math.pi, 49, dtype=torch.float64, device=self.dev)[:-1]
        # beam along z, axis a = (sin psi, 0, cos psi); normals n = (cos t cos phi, cos t sin phi, sin t)
        cos_a = (cos_t[..., None] * torch.cos(phi) * torch.sin(psi) + sin_t[..., None] * torch.cos(psi))
        P = (r_md ** 2 * cos_a ** 2 + (1 - cos_a ** 2) / r_md) ** -1.5
        return P.mean(-1)

    def band(self, g):
        """Banded operator values (K, bw), float64, rows normalized."""
        s0, s1, t0, t1, wb, sb, b1, w1 = [g[self.res_sl[0] + i] for i in range(8)]
        if self.kernel == "ikeda_carpenter":
            return self._band_ikeda_carpenter(s0, s1, t0, t1, wb, sb)
        lamj = self.lamf[self.band_idx]                                               # (K, bw)
        sigma = torch.sqrt(s0 * s0 + (s1 * lamj) ** 2).clamp_min(1e-6)
        tau = (t0 + t1 * lamj).clamp_min(1e-7)
        if self.kernel == "ic":
            return self._band_ic(sigma, tau, lamj, wb, sb)
        if self.kernel in ("two_scale_lam", "two_tail"):
            w = (wb + w1 * (lamj - 3.0)).clamp(0.0, 0.8)
            if self.kernel == "two_tail":
                return self._band_mix(sigma, tau, lamj, w, (sb + b1 * lamj).clamp_min(1e-6))
            sb = torch.sqrt(sb * sb + (b1 * lamj) ** 2)
            wb = w
        med = emg_median(sigma.detach(), tau.detach()).to(torch.float64)
        hi = _emg_cdf(self.lam_hi[:, None] - lamj + med, sigma, tau)
        lo = _emg_cdf(self.lam_lo[:, None] - lamj + med, sigma, tau)
        core = (hi - lo).clamp_min(0)
        sbc = sb.clamp_min(1e-6)
        broad = 0.5 * (torch.erf((self.lam_hi[:, None] - lamj) / (math.sqrt(2.0) * sbc))
                       - torch.erf((self.lam_lo[:, None] - lamj) / (math.sqrt(2.0) * sbc)))
        Rb = ((1 - wb) * core + wb * broad).clamp_min(0) * self.phi[self.band_idx] * self.band_valid
        return Rb / Rb.sum(1, keepdim=True).clamp_min(1e-30)

    def _band_ikeda_carpenter(self, s0, s1, a0, a1, lam_R, t_s):
        """The Ikeda-Carpenter kernel: for each fine wavelength, the distribution of the measured minus the true
        wavelength is a Gaussian convolved with the moderator's emission (a gamma-shaped slowing-down pulse, and its
        convolution with an exponential storage decay, mixed with the storage share), built from its exact Fourier
        transform on an offset grid IC_OVERSAMPLE times finer than the wavelength grid, centred on its median, and
        integrated over each data bin."""
        dx = self.hf / IC_OVERSAMPLE
        lo = (self.BAND[0] - IC_MARGIN) * self.hf; n = int(round((self.BAND[1] - self.BAND[0] + 2 * IC_MARGIN) * self.hf / dx))
        x = lo + dx * torch.arange(n, device=self.dev, dtype=torch.float64)          # measured - true wavelength
        nfft = 2 * n
        w = 2 * math.pi * torch.fft.rfftfreq(nfft, d=dx).to(self.dev)                  # angular frequency
        lamf = self.lamf[:, None]
        sigma = torch.sqrt(s0 * s0 + (s1 * lamf) ** 2).clamp_min(1e-6)
        tau = (a0 + a1 * lamf).clamp_min(1e-7)
        R = torch.exp(-(lam_R / lamf) ** 2)
        iw = 1j * w[None, :]
        emit = (1 + iw * tau) ** -3 * ((1 - R) + R / (1 + iw * t_s.clamp_min(1e-7)))
        ft = emit * torch.exp(-0.5 * (sigma * w[None, :]) ** 2) * torch.exp(iw * lo)    # samples from the grid's start
        pdf = torch.fft.irfft(ft, n=nfft, dim=1)[:, :n] / dx                           # (Kf, n), a density on x
        cdf = torch.cumsum(pdf, 1) * dx
        cdf = cdf / cdf[:, -1:].clamp_min(1e-30)
        with torch.no_grad():                                                           # the median, held fixed for derivatives
            j = torch.searchsorted(cdf.detach().contiguous(), torch.full((cdf.shape[0], 1), 0.5, device=self.dev, dtype=torch.float64))
            j = j.clamp(1, n - 1)
            c0 = cdf.gather(1, j - 1); c1 = cdf.gather(1, j)
            med = (x[j - 1] + dx * (0.5 - c0) / (c1 - c0).clamp_min(1e-30))[:, 0]       # (Kf,)

        flat = cdf.reshape(-1)

        def at(xq, rows):                                                              # cdf of fine rows at offsets xq
            t = ((xq - lo) / dx).clamp(0, n - 1 - 1e-9)
            i0 = t.floor().long(); f = t - i0
            base = rows * n
            return (1 - f) * flat[base + i0] + f * flat[base + (i0 + 1).clamp(max=n - 1)]
        lamj = self.lamf[self.band_idx]
        medj = med[self.band_idx]
        hi = at(self.lam_hi[:, None] - lamj + medj, self.band_idx)
        loc = at(self.lam_lo[:, None] - lamj + medj, self.band_idx)
        Rb = (hi - loc).clamp_min(0) * self.phi[self.band_idx] * self.band_valid
        return Rb / Rb.sum(1, keepdim=True).clamp_min(1e-30)

    def sparse(self, Rb, dtype=None):
        """The band as a sparse (K, Kf) CSR matrix (duplicate indices at the grid's ends add up)."""
        dtype = self.dt if dtype is None else dtype
        rows = torch.arange(self.K, device=self.dev).repeat_interleave(Rb.shape[1])
        coo = torch.sparse_coo_tensor(torch.stack([rows, self.band_idx.reshape(-1)]), Rb.reshape(-1).to(dtype),
                                      (self.K, self.Kf)).coalesce()
        return coo.to_sparse_csr()

    def _band_mix(self, sigma, tau, lamj, w, taub):
        """Two exponential tails on one Gaussian: (1 - w) EMG(sigma, tau) + w EMG(sigma, taub), centred on the
        mixture's median (bisection on the fine grid, held fixed for derivatives)."""
        def cdf(x, sg, ta, tb, ww):
            return (1 - ww) * _emg_cdf(x, sg, ta) + ww * _emg_cdf(x, sg, tb)
        with torch.no_grad():
            idx = self.band_idx.reshape(-1)
            pick = lambda v: torch.zeros(self.Kf, dtype=torch.float64, device=self.dev).index_put_((idx,), v.detach().reshape(-1))
            sg_f, ta_f, tb_f, w_f = pick(sigma).clamp_min(1e-6), pick(tau).clamp_min(1e-7), pick(taub).clamp_min(1e-6), pick(w)
            lo = -6 * sg_f; hi = 6 * sg_f + 10 * torch.maximum(ta_f, tb_f)
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                below = cdf(mid, sg_f, ta_f, tb_f, w_f) < 0.5
                lo = torch.where(below, mid, lo); hi = torch.where(below, hi, mid)
            med = (0.5 * (lo + hi))[self.band_idx]
        hi_c = cdf(self.lam_hi[:, None] - lamj + med, sigma, tau, taub, w)
        lo_c = cdf(self.lam_lo[:, None] - lamj + med, sigma, tau, taub, w)
        Rb = (hi_c - lo_c).clamp_min(0) * self.phi[self.band_idx] * self.band_valid
        return Rb / Rb.sum(1, keepdim=True).clamp_min(1e-30)

    def _band_ic(self, sigma, tau, lamj, lam_R, ts):
        """Two-exponential moderator kernel: a shared Gaussian (sigma) with a fast tail (tau, slowing down) and a slow
        tail (ts, storage), the storage share R = exp(-(lam_R / lam)^2) (Ikeda-Carpenter's energy dependence), the
        mixture centred on its own median (found by bisection, held fixed for derivatives like emg_median)."""
        R = torch.exp(-(lam_R / lamj) ** 2)
        ts_ = ts.clamp_min(1e-6) * torch.ones_like(tau)

        def cdf(x):
            return (1 - R) * _emg_cdf(x, sigma, tau) + R * _emg_cdf(x, sigma, ts_)
        # the median depends only on the true wavelength: bisection on the fine grid (Kf points), then looked up
        with torch.no_grad():
            g_ = lambda v: v.detach()
            lf = self.lamf
            s0f = sigma.detach()
            # sigma, tau and R on the fine grid, from their values in the band (each fine point appears in the band)
            sig_f = torch.zeros(self.Kf, dtype=torch.float64, device=self.dev).index_put_((self.band_idx.reshape(-1),), g_(sigma).reshape(-1))
            tau_f = torch.zeros_like(sig_f).index_put_((self.band_idx.reshape(-1),), g_(tau).reshape(-1))
            R_f = torch.zeros_like(sig_f).index_put_((self.band_idx.reshape(-1),), g_(R).reshape(-1))
            ts_f = g_(ts).clamp_min(1e-6) * torch.ones_like(sig_f)
            sig_f = sig_f.clamp_min(1e-6); tau_f = tau_f.clamp_min(1e-7)
            cdf_f = lambda x: (1 - R_f) * _emg_cdf(x, sig_f, tau_f) + R_f * _emg_cdf(x, sig_f, ts_f)
            lo = -6 * sig_f; hi = 6 * sig_f + 10 * torch.maximum(tau_f, ts_f)
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                below = cdf_f(mid) < 0.5
                lo = torch.where(below, mid, lo); hi = torch.where(below, hi, mid)
            med = (0.5 * (lo + hi))[self.band_idx]
        hi_c = cdf(self.lam_hi[:, None] - lamj + med)
        lo_c = cdf(self.lam_lo[:, None] - lamj + med)
        Rb = (hi_c - lo_c).clamp_min(0) * self.phi[self.band_idx] * self.band_valid
        return Rb / Rb.sum(1, keepdim=True).clamp_min(1e-30)

    def band_matrix(self, Rb, dtype=None):
        """The band as a dense (K, Kf) matrix. A dense product is several times faster on a GPU than the sparse one
        (the band covers about a sixth of the fine grid), and the matrix is small (K x Kf)."""
        return self.sparse(Rb, dtype).to_dense()

    def apply_band(self, E, Rb):
        """sum_i Rt_ki E_pi for E (Pc, Kf); Rb the band values (K, bw), its sparse matrix, or its dense matrix."""
        if Rb.layout == torch.sparse_csr:
            return (Rb @ E.T).T
        S = Rb if Rb.shape == (self.K, self.Kf) else self.band_matrix(Rb, E.dtype)
        return E @ S.T.to(E.dtype)

    def kappa(self, g):
        return g[self.kap_sl[0]:self.kap_sl[1]].reshape(self.J, self.R)


class FitB:
    """Stage-B fit: per-pixel maps (and strain), global parameters by Levenberg-Marquardt Fisher scoring with
    derivatives by the chain rule through the banded operator, contamination per bin."""

    def __init__(self, model, T, g, W, f, eps=None, chunk=1024, ridge=None, z=None, alpha=0.05, inflation=1.0,
                 ray_mask=None):
        """ray_mask (rays,) bool: only the rays where it is True enter the likelihood (use it to leave out detector
        regions the model does not describe, such as a sample holder); None uses every ray."""
        self.M = model; self.T = T; self.g = g.clone(); self.W = W.clone(); self.f = f.clone()
        self.ray_weight = None if ray_mask is None else ray_mask.to(T.device, T.dtype)
        self.n_points = int(all_sum_float((T.shape[0] if ray_mask is None else int(ray_mask.sum())) * T.shape[1], T.device))
        # variance inflation of pixel sums from spatially correlated counts (noise.py); scales score variances
        self.inflation = float(inflation)
        self.eps = None if eps is None else eps.clone()
        self.chunk = chunk
        self.ridge = (1e-3 * all_sum_float(T.shape[0], T.device) * float(model.d.mean())) if ridge is None else ridge
        self.z = norm.isf(alpha / (2 * model.K)) if z is None else z
        self.lam1 = torch.zeros_like(self.f); self.dispersion = 1.0
        self.history = []

    def dose(self, sl):
        """The counts per bin of the rays sl, (K,) or (Pc, K): zero for the rays left out by ray_mask."""
        if self.ray_weight is None:
            return self.M.d
        return self.M.d[None, :] * self.ray_weight[sl, None]

    def chunks(self):
        P = self.T.shape[0]
        return [slice(i, min(i + self.chunk, P)) for i in range(0, P, self.chunk)]

    def _state(self, g):
        M = self.M
        return M.spectra(g).to(M.dt), M.band_matrix(M.band(g)), M.kappa(g).to(M.dt)

    def mean(self, sl, g=None, W=None, f=None, eps=None, cache=None):
        M = self.M; g = self.g if g is None else g
        W = self.W[sl] if W is None else W; f = self.f if f is None else f
        if eps is None and self.eps is not None:
            eps = self.eps[sl]
        mu, Rb, kap = self._state(g) if cache is None else cache
        X = W @ mu if eps is None else torch.einsum('pr,prk->pk', W, M.spectra(g, eps).to(M.dt))
        E = torch.exp(-X)
        n = M.apply_band(E, Rb)
        beta = torch.exp(-W @ kap.T)
        s = 1 - f.sum(0)
        return s * n + beta @ f, E, n, beta, s

    def spline_penalty(self, g):
        """The spline's smoothness and magnitude penalty; with ``kappa_prior`` (a standard deviation) a weak prior on the
        contamination's attenuation toward 0; and, when ``scale_anchor`` is set, a stiff quadratic pin of
        each material's spectrum at one fine-grid index: (index, targets (R,), weight). The pin removes the scale that
        maps and spectra otherwise trade freely (X = W mu is unchanged by mu -> s mu, W -> W / s)."""
        tot = 0.0
        for sl in self.M.slices:
            sp = g[sl["s"][0]:sl["s"][1]]
            d2 = sp[2:] - 2 * sp[1:-1] + sp[:-2]
            tot = tot + self.ridge * ((d2 * d2).sum() + 1e-2 * (sp * sp).sum())
        kp = getattr(self, "kappa_prior", None)
        if kp is not None:
            # a weak Gaussian prior on the contamination's attenuation by the sample, centred on 0 (contamination
            # that passes the sample): kp is its standard deviation. It keeps kappa determined when a contamination
            # component is small, where the data say little about it
            kap = g[self.M.kap_sl[0]:self.M.kap_sl[1]]
            tot = tot + 0.5 * (kap * kap).sum() / kp ** 2
        anchor = getattr(self, "scale_anchor", None)
        if anchor is not None:
            k, targets, weight = anchor
            tot = tot + weight * ((self.M.spectra(g)[:, k] - targets) ** 2).sum()
        return tot

    def loss(self, g=None, W=None, f=None, eps=None):
        g = self.g if g is None else g; f = self.f if f is None else f
        cache = self._state(g); tot = 0.0
        with torch.no_grad():
            for sl in self.chunks():
                Ws = None if W is None else W[sl]; es = None if eps is None else eps[sl]
                m = self.mean(sl, g, Ws, f, es, cache)[0]
                tot += float(deviance_terms(m, self.T[sl], self.dose(sl)).sum(dtype=torch.float64))
        tot = all_sum_float(tot, self.T.device)
        pen_c = float((self.lam_c * self.c).sum()) if hasattr(self, "lam_c") else float((self.lam1 * f).sum())
        return tot + float(self.spline_penalty(g)) + pen_c

    # -- maps (and strain): per-pixel projected Newton
    def pixel_step(self):
        M = self.M; R = M.R
        cache = self._state(self.g); mu, Rb, kap = cache
        for sl in self.chunks():
            T = self.T[sl]; W = self.W[sl]
            m, E, n, beta, s = self.mean(sl, cache=cache)
            mm = m.clamp_min(1e-12)
            cols = []
            mus = mu if self.eps is None else M.spectra(self.g, self.eps[sl]).to(M.dt)
            for r in range(R):
                mur = mus[r] if self.eps is None else mus[:, r]
                cols.append(-s * M.apply_band(E * mur, Rb) - torch.einsum('jk,pj,j->pk', self.f, beta, kap[:, r]))
            if self.eps is not None:
                for r in range(R):
                    tg = torch.zeros_like(self.eps[sl]); tg[:, r] = 1
                    _, dmu = torch.func.jvp(lambda e_: M.spectra(self.g, e_), (self.eps[sl].double(),), (tg.double(),))
                    dX = W[:, r:r + 1] * dmu[:, r].to(M.dt)
                    cols.append(-s * M.apply_band(E * dX, Rb))
            Jac = torch.stack(cols, 2)
            grad = torch.einsum('pk,pkn->pn', self.dose(sl) * (1 - T / mm), Jac).double()
            F = torch.einsum('pk,pka,pkb->pab', self.dose(sl) / mm, Jac, Jac).double()
            V = W if self.eps is None else torch.cat([W, self.eps[sl]], 1)
            nonneg = torch.zeros(V.shape[1], dtype=torch.bool, device=V.device); nonneg[:R] = True

            def row_loss(Vn):
                Wn = Vn[:, :R]; en = None if self.eps is None else Vn[:, R:]
                return deviance_terms(self.mean(sl, W=Wn, eps=en, cache=cache)[0], T, self.dose(sl)).sum(1, dtype=torch.float64)
            V = _projected_newton_mixed(V, grad, F, row_loss, nonneg)
            self.W[sl] = V[:, :R]
            if self.eps is not None:
                self.eps[sl] = V[:, R:]

    def ray_system(self, sl, cache):
        """Gradient (Pc, 2R) and Fisher information (Pc, 2R, 2R) of the deviance of the rays sl with respect to
        (W, U), where U = W * eps is the projected strain (the ray's mean strain times its projected density). Used by
        a tomographic fit, where W and U are projections of volumes. Needs per-ray strain (eps not None)."""
        M = self.M
        Jac, m = self.ray_jacobian(sl, cache)
        mm = m.clamp_min(1e-12); T = self.T[sl]
        grad = torch.einsum('pk,pkn->pn', self.dose(sl) * (1 - T / mm), Jac).double()
        F = torch.einsum('pk,pka,pkb->pab', self.dose(sl) / mm, Jac, Jac).double()
        return grad, F

    def global_derivatives(self, idx):
        """The spectra's and the band's derivatives for the global parameters idx at the current state: (cache, dmu,
        dRb), the inputs of global_jacobian."""
        M = self.M
        g0 = self.g.clone()
        cache = self._state(g0)
        dmu = {}
        mat_idx = [i for i in idx if i < M.res_sl[0]]
        if mat_idx:
            jac = torch.func.jacfwd(lambda gg: M.spectra(gg))(g0)                    # (R, Kf, n_global)
            for i in mat_idx:
                dmu[i] = jac[..., i].to(M.dt)
        dRb = {}
        if any(M.res_sl[0] <= i < M.res_sl[1] for i in idx):
            r0, r1 = M.res_sl

            def band_of(res):
                return M.band(torch.cat([g0[:r0], res, g0[r1:]]))
            jb = torch.func.jacfwd(band_of)(g0[r0:r1])                                # (K, bw, 8)
            for q in range(r1 - r0):
                dRb[q] = M.band_matrix(jb[..., q])
        return cache, dmu, dRb

    def penalty_system(self, idx):
        """Gradient and exact Hessian of the global penalties (spline_penalty) for the parameters idx."""
        gp = self.g.clone().requires_grad_(True)
        pen = self.spline_penalty(gp)
        n = len(idx)
        if not torch.is_tensor(pen):
            z = torch.zeros(n, device=self.g.device, dtype=torch.float64)
            return z, torch.zeros(n, n, device=self.g.device, dtype=torch.float64)
        gpen, = torch.autograd.grad(pen, gp, create_graph=True)
        Hpen = torch.stack([torch.autograd.grad(gpen[i], gp, retain_graph=True)[0] for i in idx])[:, idx]
        return gpen.detach()[idx], Hpen.detach()

    def joint_system(self, idx, chunk=None):
        """For a joint step over the rays' (W, U) and the global parameters idx: per ray the gradient (P, 2R) and Fisher
        information (P, 2R, 2R) for (W, U) and their cross information with the global parameters (P, 2R, n); and the
        global gradient (n) and Fisher information (n, n), penalties included. Needs per-ray strain."""
        M = self.M; R = M.R
        cache, dmu, dRb = self.global_derivatives(idx)
        P = self.T.shape[0]; n = len(idx); dev = self.T.device
        gch = chunk or max(64, self.chunk // 4)
        G = torch.zeros(P, 2 * R, device=dev, dtype=torch.float64); Fw = torch.zeros(P, 2 * R, 2 * R, device=dev, dtype=torch.float64)
        C = torch.zeros(P, 2 * R, n, device=dev, dtype=torch.float32)
        gg = torch.zeros(n, device=dev, dtype=torch.float64); Fg = torch.zeros(n, n, device=dev, dtype=torch.float64)
        for sl in [slice(i, min(i + gch, P)) for i in range(0, P, gch)]:
            Jw, m = self.ray_jacobian(sl, cache)                                      # (Pc, K, 2R)
            Jg, _ = self.global_jacobian(sl, cache, idx, dmu, dRb)                     # (n, Pc, K)
            mm = m.clamp_min(1e-12); T = self.T[sl]
            r_ = self.dose(sl) * (1 - T / mm); w_ = self.dose(sl) / mm
            G[sl] = torch.einsum('pk,pkn->pn', r_, Jw).double()
            Fw[sl] = torch.einsum('pk,pka,pkb->pab', w_, Jw, Jw).double()
            C[sl] = torch.einsum('pk,pka,npk->pan', w_, Jw, Jg)
            gg += torch.einsum('pk,npk->n', r_, Jg).double()
            Fg += torch.einsum('pk,apk,bpk->ab', w_, Jg, Jg).double()
            del Jw, Jg
        all_sum(gg); all_sum(Fg)
        gp, Hp = self.penalty_system(idx)
        return G, Fw, C, gg + gp, Fg + Hp

    def ray_jacobian(self, sl, cache):
        """d m / d (W, U) of the rays sl, (Pc, K, 2R), and the mean m (Pc, K)."""
        M = self.M; R = M.R
        T = self.T[sl]; W = self.W[sl]; eps = self.eps[sl]
        m, E, n, beta, s = self.mean(sl, cache=cache)
        _, Rb, kap = cache
        mus = M.spectra(self.g, eps).to(M.dt)                                         # (Pc, R, Kf)
        cols_w, cols_u = [], []
        for r in range(R):
            tg = torch.zeros_like(eps); tg[:, r] = 1
            _, dmu = torch.func.jvp(lambda e_: M.spectra(self.g, e_), (eps.double(),), (tg.double(),))
            dmu = dmu[:, r].to(M.dt)                                                  # d mu_r / d eps, (Pc, Kf)
            # X = sum_r W_r mu_r(U_r / W_r): dX/dU_r = mu_r', dX/dW_r (U fixed) = mu_r - eps_r mu_r'
            cols_w.append(-s * M.apply_band(E * (mus[:, r] - eps[:, r:r + 1] * dmu), Rb)
                          - torch.einsum('jk,pj,j->pk', self.f, beta, kap[:, r]))
            cols_u.append(-s * M.apply_band(E * dmu, Rb))
        return torch.stack(cols_w + cols_u, 2), m

    def ray_loss(self, sl, W, eps, cache):
        """Deviance of each ray in sl at maps W and strains eps (no penalties)."""
        return deviance_terms(self.mean(sl, W=W, eps=eps, cache=cache)[0], self.T[sl], self.dose(sl)).sum(1, dtype=torch.float64)

    # -- contamination on a multiscale dictionary: f_j = A c_j, c >= 0, each atom switched on by its own score test
    def build_atoms(self, widths=(1, 8, 32, 128)):
        """Nonnegative atoms over the measured bins: single bins and cubic B-spline bumps of the given widths (in
        bins, spaced by half a width), each normalized to a peak of 1. Generic: any contamination shape is a sum of
        them, and the broad ones pool evidence over neighbouring bins."""
        M = self.M; K = M.K
        cols = []
        for w in widths:
            if w == 1:
                cols.append(torch.eye(K, device=M.dev, dtype=M.dt)); continue
            k = torch.arange(K, device=M.dev, dtype=torch.float64)
            centres = torch.arange(-w // 2, K + w // 2, max(1, w // 2), device=M.dev, dtype=torch.float64)
            u = (k[:, None] - centres[None, :]) / (w / 2)                             # cubic B-spline on [-2, 2]
            au = u.abs()
            bsp = torch.where(au < 1, 2 / 3 - au ** 2 + au ** 3 / 2, torch.where(au < 2, (2 - au) ** 3 / 6, torch.zeros_like(au)))
            bsp = bsp / (2 / 3)
            keep = bsp.sum(0) > 0
            cols.append(bsp[:, keep].to(M.dt))
        self.A = torch.cat(cols, 1)                                                   # (K, n_atoms)
        self.c = torch.zeros(M.J, self.A.shape[1], device=M.dev, dtype=M.dt)
        self.z_atoms = norm.isf(0.05 / (2 * self.A.shape[1] * max(M.J, 1)))
        self.freeze_threshold = False

    def contamination_atoms_step(self, iters=4):
        M = self.M; J = M.J
        if not J:
            return
        if not hasattr(self, "A"):
            self.build_atoms()
        cache = self._state(self.g)
        grad = torch.zeros(M.K, J, device=M.dev, dtype=torch.float64)
        Fd = torch.zeros(M.K, J, device=M.dev, dtype=torch.float64)
        Fi0 = torch.zeros(M.K, J, device=M.dev, dtype=torch.float64)
        pear = 0.0
        with torch.no_grad():
            for sl in self.chunks():
                T = self.T[sl]
                m, E, n, beta, s = self.mean(sl, cache=cache)
                mm = m.clamp_min(1e-12)
                Dm = beta[:, None, :] - n[:, :, None]
                grad += torch.einsum('pk,pkj->kj', self.dose(sl) * (1 - T / mm), Dm).double()
                Fd += torch.einsum('pk,pkj->kj', self.dose(sl) / mm, Dm * Dm).double()
                # the price of a unit share: the score's standard deviation at f = 0 against a fixed reference
                # contrast, a fully penetrating contaminant (beta = 1), so a contaminant that imitates the neutrons
                # (beta near n, a vanishing score variance) is not free
                Fi0 += (self.dose(sl) / n.clamp_min(1e-12) * (1 - n) ** 2).sum(0).double()[:, None].expand(-1, J)
                pear += float((self.dose(sl) * (T - m) ** 2 / mm).sum(dtype=torch.float64))
        all_sum(grad); all_sum(Fd); all_sum(Fi0); pear = all_sum_float(pear, M.dev)
        self.dispersion = pear / self.n_points
        A = self.A.double()
        if not (self.freeze_threshold and hasattr(self, "lam_c")):
            self.lam_c = (self.z_atoms * torch.sqrt(self.dispersion * self.inflation * ((A * A).T @ Fi0))).T.to(M.dt)   # (J, n_atoms)
        gc = (A.T @ grad).T                                                           # (J, n_atoms)
        Hc = ((A * A).T @ Fd).T.clamp_min(1e-30)
        c_old = self.c.double()
        c_new = (c_old - (gc + self.lam_c.double()) / Hc).clamp_min(0)
        l0 = self.loss_c(self.c)
        for _ in range(10):
            if self.loss_c(c_new.to(M.dt)) <= l0:
                self.c = c_new.to(M.dt); break
            c_new = 0.5 * (c_new + c_old)
        self.f = (self.c @ self.A.T)
        tot = self.f.sum(0)
        self.f = torch.where(tot[None] > 0.99, self.f * (0.99 / tot.clamp_min(1e-30))[None], self.f)

    def loss_c(self, c):
        f = c @ self.A.T
        tot = f.sum(0); f = torch.where(tot[None] > 0.99, f * (0.99 / tot.clamp_min(1e-30))[None], f)
        cache = self._state(self.g); dv = 0.0
        with torch.no_grad():
            for sl in self.chunks():
                dv += float(deviance_terms(self.mean(sl, f=f, cache=cache)[0], self.T[sl], self.dose(sl)).sum(dtype=torch.float64))
        dv = all_sum_float(dv, self.T.device)
        return dv + float((self.lam_c * c).sum())

    # -- contamination: per bin, projected Newton with the L1 threshold
    def contamination_step(self):
        M = self.M; J = M.J
        if not J:
            return
        cache = self._state(self.g)
        grad = torch.zeros(M.K, J, device=M.dev, dtype=torch.float64)
        F = torch.zeros(M.K, J, J, device=M.dev, dtype=torch.float64)
        Fi0 = torch.zeros(M.K, J, device=M.dev, dtype=torch.float64)
        pear = 0.0
        with torch.no_grad():
            for sl in self.chunks():
                T = self.T[sl]
                m, E, n, beta, s = self.mean(sl, cache=cache)
                mm = m.clamp_min(1e-12)
                Dm = beta[:, None, :] - n[:, :, None]                                  # (Pc, K, J)
                grad += torch.einsum('pk,pkj->kj', self.dose(sl) * (1 - T / mm), Dm).double()
                F += torch.einsum('pk,pka,pkb->kab', self.dose(sl) / mm, Dm, Dm).double()
                Fi0 += torch.einsum('pk,pkj->kj', self.dose(sl) / n.clamp_min(1e-12), Dm * Dm).double()
                pear += float((self.dose(sl) * (T - m) ** 2 / mm).sum(dtype=torch.float64))
        all_sum(grad); all_sum(F); all_sum(Fi0); pear = all_sum_float(pear, M.dev)
        self.dispersion = pear / self.n_points
        self.lam1 = _l1_threshold(Fi0, self.dispersion * self.inflation, self.z).T.to(M.dt)
        grad += self.lam1.T.double()

        def per_bin_loss(fT):
            tot = torch.zeros(M.K, device=M.dev, dtype=torch.float64)
            for sl in self.chunks():
                tot += deviance_terms(self.mean(sl, f=fT.T, cache=cache)[0], self.T[sl], self.dose(sl)).sum(0, dtype=torch.float64)
            all_sum(tot)
            return tot + (self.lam1 * fT.T).sum(0).double()
        self.f = _projected_newton_rows(self.f.T.contiguous(), grad, F, per_bin_loss, cap=(0, 0.99)).T.contiguous()

    # -- globals: Levenberg-Marquardt Fisher scoring, derivatives by the chain rule
    def global_jacobian(self, sl, cache, idx, dmu, dRb):
        """dm/dg for the indices idx, (len(idx), Pc, K): materials through mu, resolution through the band, kappa
        through beta."""
        M = self.M
        mu, Rb, kap = cache
        m, E, n, beta, s = self.mean(sl, cache=cache)
        W = self.W[sl]
        eps = None if self.eps is None else self.eps[sl]
        terms = {}                                                                   # per material: x^2 step shape
        out = []
        for i in idx:
            if i < M.res_sl[0]:                                                      # a material parameter
                r = next(rr for rr, slc in enumerate(M.slices) if slc["a"] <= i < slc["s"][1])
                slr = M.slices[r]
                if eps is not None and M.shape_model is None and slr["h"][0] <= i < slr["h"][1]:
                    # a height under per-pixel strain, without a linked shape model: its derivative is its own edge
                    # term, so one evaluation serves all heights of the material
                    if r not in terms:
                        terms[r] = M.edge_terms(self.g, eps, r)
                    dm = terms[r][:, i - slr["h"][0]]
                    out.append(-s * M.apply_band(E * (W[:, r:r + 1] * dm.to(M.dt)), Rb))
                elif eps is not None and i < M.slices[r]["c"][0]:
                    # an edge parameter (a, h, t, x) under per-pixel strain: its derivative moves with each pixel's
                    # edges (the smooth part is not dilated, so the unstrained derivative serves for c and s)
                    tg = torch.zeros_like(self.g); tg[i] = 1
                    _, dm = torch.func.jvp(lambda gg: M.spectra(gg, eps)[:, r], (self.g,), (tg,))
                    out.append(-s * M.apply_band(E * (W[:, r:r + 1] * dm.to(M.dt)), Rb))
                else:
                    out.append(-s * M.apply_band(E * (W[:, r:r + 1] * dmu[i][r][None, :]), Rb))
            elif i < M.res_sl[1]:
                out.append(s * M.apply_band(E, dRb[i - M.res_sl[0]]))
            else:
                jr = i - M.kap_sl[0]; j, r = divmod(jr, M.R)
                out.append(-(self.f[j][None, :] * (beta[:, j:j + 1] * W[:, r:r + 1])))
        return torch.stack(out, 0), m

    def global_step(self, damping=1e-3, which=None, max_tries=8, system=None):
        """system: (grad, F) already computed at the current state for these indices, to reuse."""
        M = self.M
        idx = list(range(M.n_global)) if which is None else list(which)
        idx = [i for i in idx if float(M.ub[i]) > float(M.lb[i])]                     # fixed parameters stay out
        if not idx:
            return self.loss(), damping
        grad, F = self.global_system(idx) if system is None else system
        g0 = self.g.clone()
        l0 = self.loss()
        lam_ = damping
        # parameters without information (an edge outside the measured range, the absorption of a contamination
        # component that is off) stay put: dividing their roundoff gradient by a zero curvature gave huge steps
        d = torch.diagonal(F)
        live = d > INFO_FLOOR * d.max()
        idx_t = torch.as_tensor(idx, device=self.g.device)
        for _ in range(max_tries):
            A = F[live][:, live] + lam_ * torch.diag(d[live])
            step = torch.linalg.solve(A, grad[live])
            g_new = g0.clone(); g_new[idx_t[live]] = g0[idx_t[live]] - step
            g_new = torch.maximum(torch.minimum(g_new, M.ub), M.lb)
            l1 = self.loss(g=g_new)
            if l1 < l0:
                self.g = g_new
                return l1, max(lam_ / 3, 1e-9)
            lam_ *= 10
        return l0, lam_

    def global_system(self, idx):
        """Gradient and Fisher information (with the spline penalty's exact Hessian) of the loss for the global
        parameters idx, at the current state. Used by global_step, and summed over datasets by a joint fit."""
        M = self.M
        g0 = self.g.clone()
        cache = self._state(g0)
        # parameter derivatives of the spectra (cheap: R x Kf outputs) and of the band (4 x K x bw)
        dmu = {}
        mat_idx = [i for i in idx if i < M.res_sl[0]]
        if mat_idx:
            jac = torch.func.jacfwd(lambda gg: M.spectra(gg))(g0)                    # (R, Kf, n_global)
            for i in mat_idx:
                dmu[i] = jac[..., i].to(M.dt)
        dRb = {}
        if any(M.res_sl[0] <= i < M.res_sl[1] for i in idx):
            # the band depends on the kernel slots only: differentiate those (memory ~ K x bw x 8, not x n_global)
            r0, r1 = M.res_sl

            def band_of(res):
                return M.band(torch.cat([g0[:r0], res, g0[r1:]]))
            jb = torch.func.jacfwd(band_of)(g0[r0:r1])                                # (K, bw, 8)
            for q in range(r1 - r0):
                dRb[q] = M.band_matrix(jb[..., q])
        grad = torch.zeros(len(idx), device=M.dev, dtype=torch.float64)
        F = torch.zeros(len(idx), len(idx), device=M.dev, dtype=torch.float64)
        P = self.T.shape[0]; gch = max(64, self.chunk // 4)
        for sl in [slice(i, min(i + gch, P)) for i in range(0, P, gch)]:
            Jg, m = self.global_jacobian(sl, cache, idx, dmu, dRb)
            mm = m.clamp_min(1e-12); T = self.T[sl]
            grad += torch.einsum('pk,npk->n', self.dose(sl) * (1 - T / mm), Jg).double()
            F += torch.einsum('pk,apk,bpk->ab', self.dose(sl) / mm, Jg, Jg).double()
            del Jg
        all_sum(grad); all_sum(F)
        gp = g0.clone().requires_grad_(True)
        pen = self.spline_penalty(gp)
        if torch.is_tensor(pen):
            gpen, = torch.autograd.grad(pen, gp, create_graph=True)
            Hpen = torch.stack([torch.autograd.grad(gpen[i], gp, retain_graph=True)[0] for i in idx])[:, idx]
            grad += gpen.detach()[idx]; F += Hpen.detach()
        return grad, F

    def lattice_scan(self, r, half_width=0.004, n=21):
        """Line search on material r's lattice parameter (non-convex: an edge sees no gradient far from its place):
        evaluate the loss on a grid of a about the current value and keep the best."""
        M = self.M; sl = M.slices[r]
        a0 = float(self.g[sl["a"]]); best = (self.loss(), a0)
        for a in np.linspace(a0 * (1 - half_width), a0 * (1 + half_width), n):
            gg = self.g.clone(); gg[sl["a"]] = a
            l = self.loss(g=gg)
            if l < best[0]:
                best = (l, a)
        self.g[sl["a"]] = best[1]
        return best


def _projected_newton_mixed(V, grad, F, row_loss, nonneg):
    """_projected_newton_rows with only the columns flagged in `nonneg` bounded at zero."""
    dt = V.dtype; n = V.shape[1]
    cur = V.double()
    active = (cur <= 0) & (grad > 0) & nonneg[None, :]
    keep = (~active).double()
    eye = torch.eye(n, device=V.device, dtype=torch.float64)
    jitter = 1e-10 * torch.diagonal(F, dim1=1, dim2=2).abs().mean(1).clamp_min(1e-30)[:, None, None] * eye
    Fm = (F + jitter) * keep[:, :, None] * keep[:, None, :] + torch.diag_embed(active.double())
    rhs = torch.where(active, torch.zeros_like(grad), grad)
    step = torch.linalg.solve(Fm, rhs[:, :, None])[:, :, 0]
    base = row_loss(V)
    t = torch.ones(V.shape[0], device=V.device, dtype=torch.float64)
    done = torch.zeros_like(t, dtype=torch.bool); best = cur.clone()
    for _ in range(12):
        new = cur - t[:, None] * step
        new = torch.where(nonneg[None, :], new.clamp_min(0), new)
        l = row_loss(new.to(dt))
        ok = (l <= base) & ~done
        best[ok] = new[ok]; done |= ok
        if bool(done.all()):
            break
        t = torch.where(done, t, 0.5 * t)
    return best.to(dt)


# ================================================================================================ starting values
def ls_init_material(M, g, r, target, a0, w_reg=None):
    """Material r's linear coefficients (heights >= 0, c0, c1, spline) by bounded least squares of its
    instrument-smoothed spectrum on `target` (a spectrum on the measured bins), at lattice parameter a0; second
    differences and a small magnitude ridge on the spline (it spans constants and slopes, which c0 and c1 carry).
    Writes into g (in place) and returns the residual rms."""
    from scipy.optimize import lsq_linear
    sl = M.slices[r]
    g[sl['a']] = a0
    with torch.no_grad():
        Rb = M.band(g).cpu().numpy(); idx = M.band_idx.cpu().numpy()
        Rt = np.zeros((M.K, M.Kf)); np.add.at(Rt, (np.repeat(np.arange(M.K), idx.shape[1]), idx.ravel()), Rb.ravel())
        lamf = M.lamf.cpu().numpy(); x = lamf / M.lam_ref; hf = M.hf
        e = 2 * a0 / np.sqrt(M.mats[r]['ns'].cpu().numpy())
        step = np.clip((e[None, :] - (lamf[:, None] - 0.5 * hf)) / hf, 0, 1)
        Dfine = np.concatenate([(x * x)[:, None] * step, np.ones((M.Kf, 1)), x[:, None], M.B.double().cpu().numpy()], 1)
        A = Rt @ Dfine
        ne = len(e); nsp = M.n_spline
        D2 = np.diff(np.eye(nsp), 2, axis=0); w_reg = 1e-1 * np.sqrt(M.K) * float(np.abs(target).mean()) if w_reg is None else w_reg
        reg = np.vstack([np.hstack([np.zeros((D2.shape[0], ne + 2)), w_reg * D2]),
                         np.hstack([np.zeros((nsp, ne + 2)), w_reg * np.eye(nsp)])])
        sol = lsq_linear(np.vstack([A, reg]), np.concatenate([np.asarray(target, dtype=np.float64), np.zeros(reg.shape[0])]),
                         bounds=(np.concatenate([np.zeros(ne), -np.inf * np.ones(2 + nsp)]), np.inf))
        g[sl['h'][0]:sl['h'][1]] = torch.as_tensor(sol.x[:ne])
        g[sl['c'][0]:sl['c'][1]] = torch.as_tensor(sol.x[ne:ne + 2])
        g[sl['s'][0]:sl['s'][1]] = torch.as_tensor(sol.x[ne + 2:])
        return float(np.sqrt(np.mean((A @ sol.x - target) ** 2)))

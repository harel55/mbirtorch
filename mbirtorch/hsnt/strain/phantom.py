"""A 3D Bragg-edge phantom with strain: multiple parallel-beam views of a volume holding three
crystalline materials of different lattice types, each with a nontrivial strain field, measured through a realistic
instrument (the VENUS wavelength axis with its capture gaps, an open-beam-shaped flux, the jointly calibrated
two-scale resolution kernel on the transmission, open-beam contamination) with Poisson noise.

Materials (lattice, a in A, mean attenuation per voxel at 3 A, shape, strain field):
  Cu-like fcc 3.6149, cylinder along the rotation axis, bending: eps = 1.5e-3 * x / R (x across the cylinder)
  Ni-like fcc 3.5240, sphere, radial (shrink fit): eps = 2e-3 * (1 - (r / R)^2)
  Al-like fcc 4.0495, block, eps = 5e-4 + 1e-3 * z / L (z along the rotation axis); its contrast ~ 1/10 of the others'
Spectra: the hybrid model's Bragg form, heights h_hkl proportional to multiplicity x d_hkl (a monatomic cubic
crystal's |F|^2 is the same for every allowed reflection), plus absorption and a constant, scaled to the mean
attenuation above. Exact projection of strained material: each material's voxels are sorted into strain levels
(`levels` per material), each level's indicator volume is projected, and the attenuation of a ray is the sum over
(material, level) of its projected length times the spectrum dilated by that level's strain. Truth returned per
pixel: projected density of each material, its path-averaged strain, and the strain spread along the path.
Counts are spatially independent (the measured detector correlates neighbours; not modelled here)."""
import math
import os
import sys

import numpy as np
import torch
from scipy import ndimage

from .model import Hybrid2, multiplicities
from .bragg import reflections

AXIS_RANGE = (1.9, 4.41)        # wavelength range (Angstrom) of the default synthetic axis
AXIS_BINS = 2782                # bins of the default synthetic axis


def default_axis():
    """A synthetic wavelength axis (evenly spaced over AXIS_RANGE) with a flat relative flux."""
    lam = np.linspace(AXIS_RANGE[0], AXIS_RANGE[1], AXIS_BINS)
    return lam, np.ones_like(lam)

MATERIALS = [dict(name='Cu', lattice='fcc', a=3.6149, mu=1.10),
             dict(name='Ni', lattice='fcc', a=3.5240, mu=1.00),
             dict(name='Al', lattice='fcc', a=4.0495, mu=0.10)]
# The scene is Ni-Cu-Al. bcc Fe is not used with fcc Al: in this window bcc Fe 2.8665's edges lie within 0.1%
# (1000 microstrain) of fcc Al 4.0495's.
JOINT_KERNEL = [0.0, 0.0008, 0.0036, 0.0, 0.21, 0.0169, 0.0, 0.0]        # a two-scale kernel fitted jointly to Ni measurements


# The scene, in units of the volume width n about its centre. The objects do not intersect.
# Every object lies inside the reconstruction circle (radius 0.5) across the rotation axis z, with gaps between them:
# sphere-block 0.025, cylinder-block 0.025, sphere-cylinder 0.04 (closest surfaces).
GEOMETRY = dict(cu_cylinder=dict(y=-0.15, x=-0.22, r=0.18, z_half=0.38),
                ni_sphere=dict(z=0.05, y=-0.18, x=0.20, r=0.20),
                al_block=dict(x=0.0, x_half=0.30, y=0.215, y_half=0.16, z_half=0.30))


def volume(n):
    """Material index volume (-1 empty) and strain volume, shape (n, n, n) indexed (z along the rotation axis, y, x).
    The objects of GEOMETRY do not intersect."""
    c = (n - 1) / 2.0
    z, y, x = np.meshgrid(np.arange(n) - c, np.arange(n) - c, np.arange(n) - c, indexing='ij')
    mat = -np.ones((n, n, n), np.int8); eps = np.zeros((n, n, n), np.float64)
    g = GEOMETRY['cu_cylinder']                                # Cu cylinder along z: bending across x
    R1 = g['r'] * n; cyl = ((y - g['y'] * n) ** 2 + (x - g['x'] * n) ** 2 <= R1 ** 2) & (np.abs(z) <= g['z_half'] * n)
    mat[cyl] = 0; eps[cyl] = 1.5e-3 * (x[cyl] - g['x'] * n) / R1
    g = GEOMETRY['ni_sphere']                                  # Ni sphere: radial shrink fit
    R2 = g['r'] * n; r = np.sqrt((z - g['z'] * n) ** 2 + (y - g['y'] * n) ** 2 + (x - g['x'] * n) ** 2); sph = r <= R2
    mat[sph] = 1; eps[sph] = 2e-3 * (1 - (r[sph] / R2) ** 2)
    g = GEOMETRY['al_block']                                   # Al block: strain gradient along z
    blk = (np.abs(x - g['x'] * n) < g['x_half'] * n) & (np.abs(y - g['y'] * n) < g['y_half'] * n) & (np.abs(z) < g['z_half'] * n)
    mat[blk] = 2; eps[blk] = 5e-4 + 1e-3 * z[blk] / (g['z_half'] * n)
    assert not (cyl & sph).any() and not (cyl & blk).any() and not (sph & blk).any(), 'the objects intersect'
    return mat, eps


# Elastic constants (Young's modulus in Pa, Poisson ratio) for the tensor phantom.
ELASTIC = dict(Cu=(117e9, 0.34), Ni=(200e9, 0.31), Al=(69e9, 0.33))
TENSOR_PEAK = 1.5e-3                        # the largest strain component of each object in the tensor phantom
# Maxwell stress-function amplitudes (phi_1, phi_2, phi_3) per object: unequal, so the residual stress is anisotropic
MAXWELL = dict(Cu=(1.0, 0.4, -0.7), Ni=(-0.6, 1.0, 0.5), Al=(0.8, -0.5, 1.0))
COMPONENTS = ('xx', 'yy', 'zz', 'xy', 'yz', 'xz')     # strain component order (tensor shear strains)


def strain_tensor(n):
    """Material index volume and a 3D strain tensor per voxel, shape (n, n, n) and (n, n, n, 6) in COMPONENTS order
    (tensor shear strains), indexed (z, y, x). Each object carries a residual stress built from Maxwell stress
    functions phi_i = S_i b^3 (S from MAXWELL), with b a smooth function that is zero on the object's surface: the
    rod b = (1 - r^2 / R^2)(1 - z^2 / H^2), the sphere b = 1 - r^2 / R^2, the block b = (1 - X^2 / a^2)(1 - Y^2 / b^2)
    (1 - Z^2 / c^2). The stress satisfies 3D equilibrium everywhere and vanishes on the surface (traction-free); the
    strain follows by isotropic Hooke's law, each object scaled so its largest component is TENSOR_PEAK. The second
    derivatives are evaluated analytically by automatic differentiation."""
    import torch as _t
    mat, _ = volume(n)
    c = (n - 1) / 2.0
    zz, yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, np.arange(n) - c, indexing='ij')
    pts = _t.tensor(np.stack([xx, yy, zz], -1).reshape(-1, 3), dtype=_t.float64)
    G = GEOMETRY

    def bump(name, p):
        x, y, z = p[0], p[1], p[2]
        if name == 'Cu':
            g = G['cu_cylinder']; R, H = g['r'] * n, g['z_half'] * n
            return (1 - ((x - g['x'] * n) ** 2 + (y - g['y'] * n) ** 2) / R ** 2) * (1 - z ** 2 / H ** 2)
        if name == 'Ni':
            g = G['ni_sphere']; R = g['r'] * n
            return 1 - ((x - g['x'] * n) ** 2 + (y - g['y'] * n) ** 2 + (z - g['z'] * n) ** 2) / R ** 2
        g = G['al_block']
        return (1 - (x - g['x'] * n) ** 2 / (g['x_half'] * n) ** 2) * (1 - (y - g['y'] * n) ** 2 / (g['y_half'] * n) ** 2) * \
            (1 - z ** 2 / (g['z_half'] * n) ** 2)

    eps = np.zeros((n, n, n, 6))
    for r, m in enumerate(MATERIALS):
        sel = (mat == r).reshape(-1)
        if not sel.any():
            continue
        S1, S2, S3 = MAXWELL[m['name']]
        Hs = []
        for S in (S1, S2, S3):
            f = lambda p, S=S: S * bump(m['name'], p) ** 3
            Hs.append(_t.func.vmap(_t.func.hessian(f))(pts[sel]))              # (k, 3, 3), axes (x, y, z)
        h1, h2, h3 = Hs
        sig = _t.stack([h2[:, 2, 2] + h3[:, 1, 1], h3[:, 0, 0] + h1[:, 2, 2], h1[:, 1, 1] + h2[:, 0, 0],
                        -h3[:, 0, 1], -h1[:, 1, 2], -h2[:, 0, 2]], -1).numpy()
        Em, nu = ELASTIC[m['name']]
        tr = sig[:, :3].sum(1, keepdims=True)
        e = np.concatenate([((1 + nu) * sig[:, :3] - nu * tr) / Em, (1 + nu) * sig[:, 3:] / Em], 1)
        eps.reshape(-1, 6)[sel] = e * TENSOR_PEAK / np.abs(e).max()
    return mat, eps


def project(ind, angles):
    """Parallel projections of an indicator volume (z, y, x) for rotations about z: (views, z, x') path lengths in
    voxel units (bilinear rotation of each z slice, summed along y)."""
    out = np.zeros((len(angles), ind.shape[0], ind.shape[2]))
    for v, th in enumerate(angles):
        rot = ndimage.rotate(ind.astype(np.float32), np.degrees(th), axes=(1, 2), reshape=False, order=1, mode='constant')
        out[v] = rot.sum(1)
    return out


def spectra_model(lam, dose_per_bin, device='cpu', materials=None):
    """A Hybrid2 instance for the three materials (on the measured axis), the parameter vector with the phantom's
    heights / absorption / kernel, and the material scale factors. Used both to generate and to score. materials:
    the scene's materials (default MATERIALS)."""
    MATERIALS_ = MATERIALS if materials is None else materials
    M = Hybrid2(lam, dose_per_bin, [(m['lattice'], (m['a'] * 0.99, m['a'] * 1.01)) for m in MATERIALS_], n_spline=8,
                n_contam=2, device=device, two_scale=True)
    g = torch.zeros(M.n_global, dtype=torch.float64, device=device)
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor(JOINT_KERNEL, dtype=torch.float64)
    for r, (m, sl) in enumerate(zip(MATERIALS_, M.slices)):
        g[sl['a']] = m['a']
        ns = M.mats[r]['ns'].cpu().numpy(); d = m['a'] / np.sqrt(ns)
        h = np.array(multiplicities(m['lattice'], ns), float) * d                       # j d (|F|^2 constant)
        g[sl['h'][0]:sl['h'][1]] = torch.as_tensor(h)
        g[sl['c'][0]] = 0.3; g[sl['c'][0] + 1] = 0.2                                     # constant + absorption
    lamf = M.lamf.cpu().numpy(); at3 = np.abs(lamf - 3.0).argmin()
    for r, (m, sl) in enumerate(zip(MATERIALS_, M.slices)):
        # coherent part just below the longest-wavelength edge = 0.8 x the smooth part there (real metals keep a large
        # absorption + incoherent attenuation past the last edge), then the whole spectrum scaled to the target at 3 A
        with torch.no_grad():
            g_coh = g.clone(); g_coh[sl['c'][0]:sl['c'][1]] = 0
            g_smo = g.clone(); g_smo[sl['h'][0]:sl['h'][1]] = 0
            coh = M.spectra(g_coh)[r].cpu().numpy(); smo = M.spectra(g_smo)[r].cpu().numpy()
        e_last = 2 * m['a'] / np.sqrt(M.mats[r]['ns'].cpu().numpy()).min()
        j = np.abs(lamf - (min(e_last, lamf[-1]) - 0.01)).argmin()
        g[sl['h'][0]:sl['h'][1]] *= 0.8 * smo[j] / max(coh[j], 1e-12)
        with torch.no_grad():
            mu = M.spectra(g)[r].cpu().numpy()
        sc = m['mu'] / mu[at3]
        g[sl['h'][0]:sl['h'][1]] *= sc; g[sl['c'][0]:sl['c'][1]] *= sc
    return M, g


def make(n=64, views=4, dose=50.0, levels=31, seed=1, device="cuda", contamination=True, voxel=0.08, axis=None):
    """Generate the phantom data. `voxel` scales path lengths (attenuation per voxel = mu * voxel), chosen so the
    thickest chords attenuate ~2-3. Returns a dict with T (pixels, K), lam, dose_per_bin and the truth."""
    lam, flux = default_axis() if axis is None else axis
    lam = np.asarray(lam); flux = np.asarray(flux) / np.mean(flux)
    d = dose * flux
    M, g = spectra_model(lam, d, device)
    mat, eps = volume(n)
    angles = np.linspace(0, np.pi, views, endpoint=False)
    proj = []; strain_levels = []
    for r in range(3):
        e = eps[mat == r]; lv = np.linspace(e.min(), e.max(), levels) if e.size else np.zeros(1)
        idx = np.abs(eps[..., None] - lv).argmin(-1)
        pr = np.stack([project((mat == r) & (idx == l), angles) for l in range(len(lv))])     # (L, views, z, x)
        proj.append(pr); strain_levels.append(lv)
    P = views * n * n
    W_true = np.stack([p.sum(0).reshape(P) for p in proj], 1) * voxel                 # (P, 3) projected density
    eps_bar = np.stack([(p * lv[:, None, None, None]).sum(0).reshape(P) / np.maximum(p.sum(0).reshape(P), 1e-9)
                        for p, lv in zip(proj, strain_levels)], 1)
    eps_sd = np.stack([np.sqrt(np.maximum((p * lv[:, None, None, None] ** 2).sum(0).reshape(P) / np.maximum(p.sum(0).reshape(P), 1e-9)
                                          - eps_bar[:, i] ** 2, 0)) for i, (p, lv) in enumerate(zip(proj, strain_levels))], 1)
    # attenuation on the fine grid, exact over strain levels
    with torch.no_grad():
        X = torch.zeros(P, M.Kf, dtype=torch.float32, device=device)
        for r in range(3):
            for l, e_l in enumerate(strain_levels[r]):
                w = torch.as_tensor(proj[r][l].reshape(P) * voxel, dtype=torch.float32, device=device)
                if float(w.max()) <= 0:
                    continue
                eps_t = torch.full((1, 3), 0.0, dtype=torch.float64, device=device); eps_t[0, r] = e_l
                mu_l = M.spectra(g, eps_t)[0, r].float()                              # dilated spectrum (Kf,)
                X += w[:, None] * mu_l[None, :]
        n_t = M.apply_band(torch.exp(-X), M.band_matrix(M.band(g)))
        f = torch.zeros(2, M.K, device=device)
        if contamination:                                                         # spike at 2.638 A, bump to the gap
            k_spike = int(np.abs(lam - 2.638).argmin()); f[0, k_spike] = 0.04
            bump = (lam > 2.655) & (lam < 2.77); f[1, torch.as_tensor(bump, device=device)] = torch.as_tensor(
                0.16 * np.exp(-0.5 * ((lam[bump] - 2.70) / 0.03) ** 2), dtype=torch.float32, device=device)
        beta = torch.ones(P, 2, device=device)                                    # contamination passes the sample
        m = (1 - f.sum(0)) * n_t + beta @ f
        gen = torch.Generator(device=device).manual_seed(seed)
        dd = torch.as_tensor(d, dtype=torch.float32, device=device)
        T = torch.poisson(dd * m, generator=gen) / dd
    with torch.no_grad():
        mu0 = M.spectra(g).cpu().numpy()
    return dict(T=T, lam=lam, dose_per_bin=d, n=n, views=views, angles=angles, W_true=W_true, eps_bar=eps_bar, eps_sd=eps_sd,
                mu_true=mu0, lamf=M.lamf.cpu().numpy(), g_true=g.cpu().numpy(), f_true=f.cpu().numpy(), materials=MATERIALS)


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    out = make(n=64, views=4, dose=50.0)
    T = out['T'].cpu().numpy(); n, V = out['n'], out['views']; lam = out['lam']
    print('pixels', T.shape, 'T range', T.min(), T.max(), 'mean attenuation at 3 A', float(-np.log(np.clip(T[:, np.abs(lam - 3).argmin()], 1e-3, None)).mean()))
    for i, m in enumerate(out['materials']):
        w = out['W_true'][:, i]; e = out['eps_bar'][:, i][w > 0.05 * w.max()]; s = out['eps_sd'][:, i][w > 0.05 * w.max()]
        print(f"{m['name']}: projected density max {w.max():.2f}; path-mean strain {1e6 * e.min():.0f} to {1e6 * e.max():.0f} microstrain; "
              f"spread along paths up to {1e6 * s.max():.0f}")
    fig, ax = plt.subplots(3, V + 1, figsize=(4 * (V + 1), 11), layout='constrained')
    for v in range(V):
        ax[0, v].imshow(-np.log(np.clip(T.reshape(V, n, n, -1)[v, :, :, np.abs(lam - 3.0).argmin()], 1e-3, None)), cmap='magma'); ax[0, v].set_title(f'view {v}: attenuation at 3 A')
        for i, c in zip(range(3), ('Cu', 'Ni', 'Al')):
            pass
        ax[1, v].imshow(out['W_true'].reshape(V, n, n, 3)[v] / out['W_true'].max(0), interpolation='nearest'); ax[1, v].set_title('projected Cu / Ni / Al (RGB)')
        es = out['eps_bar'].reshape(V, n, n, 3)[v]; ws = out['W_true'].reshape(V, n, n, 3)[v]
        e_show = np.where(ws.sum(-1) > 0, (es * ws).sum(-1) / np.maximum(ws.sum(-1), 1e-9), np.nan)
        h = ax[2, v].imshow(1e6 * e_show, cmap='RdBu_r', vmin=-2000, vmax=2000); ax[2, v].set_title('density-weighted path-mean strain (microstrain)')
    fig.colorbar(h, ax=ax[2, V - 1])
    for i, m in enumerate(out['materials']):
        ax[i, V].plot(out['lamf'], out['mu_true'][i], lw=1); ax[i, V].set_title(f"{m['name']} ({m['lattice']} {m['a']} A) spectrum")
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phantom3d.png'), dpi=60)
    print('written')


def make_tomo(n=16, views=25, dose=50.0, levels=31, seed=1, device="cuda", contamination=True, voxel=None, fine=2,
              tensor=False, series=('z',), axis=None):
    """Tomographic version of `make`: `views` parallel-beam views over [0, pi) through the package projector. The
    volume is built and projected on a grid `fine` times finer than the n^3 reconstruction grid, and the detector is
    binned back to n x n, so the data are not generated on the reconstruction grid. `voxel` is the attenuation scale
    per coarse voxel (default 2.56 / n, so the object's attenuation does not depend on n).
    Rays are ordered (view, detector row = z, channel), as the projector's sinogram. Returns the dict of `make` plus
    the coarse truth volumes V_true (n, n, n, 3) (densities, in the projector's (row, col, slice) layout) and E_true
    (n, n, n, 3) (each material's mean strain in a voxel, 0 where it is absent), and the angles. With tensor=True the
    strain is the 3D tensor of strain_tensor: E_true is (n, n, n, 3 materials, 6 components, COMPONENTS order) and each
    view sees n . eps . n with n its beam direction (beam_geometry), projected exactly over strain levels view by view.
    series: the rotation axes measured, each with the same angles (beam_geometry.PERMS); rays are series-major.
    A series about x or y sees the sample's other strain components, so three series see all six.
    The spike phantom has its own generator, make_spike."""
    import mbirtorch
    mats = MATERIALS
    voxel = 2.56 / n if voxel is None else voxel
    lam, flux = default_axis() if axis is None else axis
    lam = np.asarray(lam); flux = np.asarray(flux) / np.mean(flux)
    d = dose * flux
    M, g = spectra_model(lam, d, device, materials=mats)
    N = fine * n
    angles = np.linspace(0, np.pi, views, endpoint=False)
    blk = lambda a: a.reshape(n, fine, n, fine, n, fine).mean((1, 3, 5))
    from .geometry import PERMS, series_weights
    series = tuple(series)
    S = len(series)
    if not tensor:
        mat, eps = volume(N)                                                   # (z, y, x)
        mat = mat.transpose(1, 2, 0); eps = eps.transpose(1, 2, 0)            # projector layout (y, x, z)
        seen = [eps] * (S * views)                                             # every view sees the same scalar
    else:
        mat, et = strain_tensor(N)
        mat = mat.transpose(1, 2, 0); et = et.transpose(1, 2, 0, 3)          # (y, x, z, 6)
        # each view sees n . eps . n, with n its beam direction in the sample frame (all six components in general)
        wts = series_weights([(ax, angles) for ax in series])                  # (S views, 6)
        seen = [(et * w).sum(-1) for w in wts]
    strain_levels = []
    for r in range(3):
        e = np.concatenate([sv[mat == r] for sv in seen]); strain_levels.append(np.linspace(e.min(), e.max(), levels))
    L = 3 * levels
    proj = np.zeros((3, levels, S * views, n, n))
    for k, ax in enumerate(series):
        perm = PERMS[ax]
        mat_s = mat.transpose(perm)
        if not tensor:
            inds = []
            seen_s = seen[0].transpose(perm)
            for r in range(3):
                idx = np.abs(seen_s[..., None] - strain_levels[r]).argmin(-1)
                inds += [((mat_s == r) & (idx == l)).astype(np.float32) for l in range(levels)]
            model = mbirtorch.ParallelBeamModel((views, N * L, N), angles)
            model.set_params(use_ror_mask=False)                             # the objects reach past the circle in some series
            sino = model.forward_project(np.concatenate(inds, 2), output_sharded=True)      # (views, N L, N)
            proj[:, :, k * views:(k + 1) * views] = (sino.reshape(views, L, n, fine, n, fine).mean((3, 5)) / fine).reshape(
                views, 3, levels, n, n).permute(1, 2, 0, 3, 4).cpu().numpy()
        else:
            for v in range(views):                                             # each view sees its own n.eps.n
                inds = []
                seen_s = seen[k * views + v].transpose(perm)
                for r in range(3):
                    idx = np.abs(seen_s[..., None] - strain_levels[r]).argmin(-1)
                    inds += [((mat_s == r) & (idx == l)).astype(np.float32) for l in range(levels)]
                model = mbirtorch.ParallelBeamModel((1, N * L, N), angles[v:v + 1])
                model.set_params(use_ror_mask=False)                             # the objects reach past the circle in some series
                sino = model.forward_project(np.concatenate(inds, 2), output_sharded=True)  # (1, N L, N)
                proj[:, :, k * views + v] = (sino.reshape(L, n, fine, n, fine).mean((2, 4)) / fine).reshape(3, levels, n, n).cpu().numpy()
    proj = proj.reshape(3, levels, S * views * n * n)                          # (material, level, ray)
    P = S * views * n * n
    W_true = proj.sum(1).T * voxel
    eps_bar = np.stack([(proj[r] * strain_levels[r][:, None]).sum(0) / np.maximum(proj[r].sum(0), 1e-9) for r in range(3)], 1)
    V_true = np.stack([blk((mat == r).astype(np.float64)) for r in range(3)], -1) * voxel
    frac = lambda r: np.maximum(blk((mat == r).astype(np.float64)), 1e-12)
    if not tensor:
        E_true = np.stack([blk(np.where(mat == r, seen[0], 0.0)) / frac(r) for r in range(3)], -1)
    else:                                                                      # (n, n, n, 3 materials, 6 components)
        E_true = np.stack([np.stack([blk(np.where(mat == r, et[..., k], 0.0)) / frac(r) for k in range(6)], -1)
                           for r in range(3)], -2)
    with torch.no_grad():
        X = torch.zeros(P, M.Kf, dtype=torch.float32, device=device)
        for r in range(3):
            for l, e_l in enumerate(strain_levels[r]):
                w = torch.as_tensor(proj[r, l] * voxel, dtype=torch.float32, device=device)
                if float(w.max()) <= 0:
                    continue
                eps_t = torch.zeros((1, 3), dtype=torch.float64, device=device); eps_t[0, r] = e_l
                X += w[:, None] * M.spectra(g, eps_t)[0, r].float()[None, :]
        n_t = M.apply_band(torch.exp(-X), M.band_matrix(M.band(g)))
        f = torch.zeros(2, M.K, device=device)
        if contamination:
            k_spike = int(np.abs(lam - 2.638).argmin()); f[0, k_spike] = 0.04
            bump = (lam > 2.655) & (lam < 2.77); f[1, torch.as_tensor(bump, device=device)] = torch.as_tensor(
                0.16 * np.exp(-0.5 * ((lam[bump] - 2.70) / 0.03) ** 2), dtype=torch.float32, device=device)
        m = (1 - f.sum(0)) * n_t + torch.ones(P, 2, device=device) @ f
        gen = torch.Generator(device=device).manual_seed(seed)
        dd = torch.as_tensor(d, dtype=torch.float32, device=device)
        T = torch.poisson(dd * m, generator=gen) / dd
        mu0 = M.spectra(g).cpu().numpy()
    return dict(T=T, lam=lam, dose_per_bin=d, n=n, views=views, angles=angles, W_true=W_true, eps_bar=eps_bar,
                V_true=V_true, E_true=E_true, voxel=voxel, mu_true=mu0, lamf=M.lamf.cpu().numpy(), g_true=g.cpu().numpy(),
                f_true=f.cpu().numpy(), materials=mats)


# The spike phantom (user, 2026-10-20): a steel spike pressed through a copper washer into a nickel plate. Three
# materials with distinct Bragg edges (bcc Fe and fcc Al would nearly coincide), bonded interfaces carrying a contact
# pressure, curved faces off the voxel grid everywhere. Geometry in units of n about the volume's centre, z the spike's
# axis; the plate is a disk so the press-fit (Lame) solution is exact.
SPIKE_MATERIALS = [dict(name='Fe', lattice='bcc', a=2.8665, mu=0.90),
                   dict(name='Cu', lattice='fcc', a=3.6149, mu=1.10),
                   dict(name='Ni', lattice='fcc', a=3.5240, mu=1.00)]
SPIKE_ELASTIC = dict(Fe=(211e9, 0.29), Cu=(117e9, 0.34), Ni=(200e9, 0.31))
SPIKE = dict(radius=0.08, top=0.34, tip_start=-0.18, tip=-0.30,           # spike: cylinder, then a cone down to the tip
             washer_outer=0.20, washer_z=(0.10, 0.18),                    # washer: annulus on top of the plate
             plate_outer=0.30, plate_z=(-0.10, 0.10))                     # plate: disk
SPIKE_PEAK = 1.5e-3                         # the largest strain component (sets the interference)


SPIKE_SUPERSAMPLE = 4                       # points per axis inside each fine voxel for the parts' volume fractions


def _spike_geometry(yy, xx, zz):
    """Boolean membership of the spike, washer and plate at points (units of n about the centre)."""
    G = SPIKE
    r = np.sqrt(xx ** 2 + yy ** 2); a = G['radius']
    in_z = lambda zr: (zz >= zr[0]) & (zz < zr[1])
    cone = np.where(zz < G['tip_start'], a * (zz - G['tip']) / (G['tip_start'] - G['tip']), a)
    spike = (r < cone) & (zz >= G['tip']) & (zz < G['top'])
    washer = (r >= a) & (r < G['washer_outer']) & in_z(G['washer_z'])
    plate = (r >= a) & (r < G['plate_outer']) & in_z(G['plate_z'])
    return spike, washer, plate


def spike_fields(N, supersample=SPIKE_SUPERSAMPLE):
    """The spike phantom on an N^3 grid, projector layout (rows y, cols x, slices z): each part's volume fraction per
    voxel (N, N, N, 3), from supersample^3 points per voxel, so faces and interfaces are not staircased; and each
    part's strain tensor at the voxel centre (N, N, N, 3, 6), COMPONENTS order, wherever the part occupies the voxel.
    Press fit: the spike, radius a, sits in holes of radius a with the same radial interference; in each contact zone
    (the washer's and the plate's thickness) the contact pressure is p = delta / (a / E_h ((b^2 + a^2) / (b^2 - a^2) +
    nu_h) + a / E_s (1 - nu_s)), b the hub's outer radius (Lame, plane stress through the thickness). The spike carries
    sigma_rr = sigma_tt = -p there and no stress elsewhere; a hub carries sigma_rr = -p a^2 / (b^2 - a^2) (b^2 / r^2 - 1),
    sigma_tt = p a^2 / (b^2 - a^2) (b^2 / r^2 + 1). This stress satisfies equilibrium, carries the contact pressure
    across each interface, and is traction-free on every outer surface; delta is scaled so the largest strain
    component is SPIKE_PEAK."""
    G = SPIKE; a = G['radius']
    c = (N - 1) / 2.0
    ax = (np.arange(N) - c) / N
    frac = np.zeros((N, N, N, 3))
    sub = (np.arange(supersample) + 0.5) / supersample - 0.5                   # offsets inside a voxel, in voxels
    for oy in sub:
        for ox in sub:
            for oz in sub:
                yy, xx, zz = np.meshgrid(ax + oy / N, ax + ox / N, ax + oz / N, indexing='ij')
                for k, part in enumerate(_spike_geometry(yy, xx, zz)):
                    frac[..., k] += part
    frac /= supersample ** 3
    yy, xx, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    r = np.maximum(np.sqrt(xx ** 2 + yy ** 2), 1e-9); th = np.arctan2(yy, xx); cs, sn = np.cos(th), np.sin(th)
    in_z = lambda zr: (zz >= zr[0]) & (zz < zr[1])
    Es, nus = SPIKE_ELASTIC['Fe']

    def strains(srr, stt, E, nu):
        err = (srr - nu * stt) / E; ett = (stt - nu * srr) / E; ezz = -nu * (srr + stt) / E
        z0 = np.zeros_like(err)
        return np.stack([err * cs ** 2 + ett * sn ** 2, err * sn ** 2 + ett * cs ** 2, ezz, (err - ett) * sn * cs, z0, z0], -1)

    eps = np.zeros((N, N, N, 3, 6))
    for k, (hub, b, zr) in enumerate((('Cu', G['washer_outer'], G['washer_z']), ('Ni', G['plate_outer'], G['plate_z']))):
        Eh, nuh = SPIKE_ELASTIC[hub]
        p = 1.0 / (a / Eh * ((b * b + a * a) / (b * b - a * a) + nuh) + a / Es * (1 - nus))
        kk = p * a * a / (b * b - a * a); rh = np.maximum(r, a)               # a hub's field, at r >= a
        eps[..., k + 1, :] = strains(-kk * (b * b / rh ** 2 - 1), kk * (b * b / rh ** 2 + 1), Eh, nuh)
        zone = in_z(zr)
        eps[zone, 0, :] = strains(-p * np.ones_like(r), -p * np.ones_like(r), Es, nus)[zone]
    eps *= (frac > 0)[..., None]
    eps *= SPIKE_PEAK / np.abs(eps).max()
    return frac, eps


def spike_tensor(N):
    """The spike phantom as one material per voxel: the material index (N, N, N) of each voxel's largest part (-1
    where no part fills half the voxel) and that part's strain (N, N, N, 6). See spike_fields."""
    frac, eps = spike_fields(N)
    mat = frac.argmax(-1).astype(np.int8); mat[frac.max(-1) < 0.5] = -1
    sel = np.clip(mat, 0, 2)
    e = np.take_along_axis(eps, sel[..., None, None].repeat(6, -1), 3)[..., 0, :]
    return mat, e * (mat >= 0)[..., None]


def make_spike(n=32, views=51, dose=320.0, levels=31, seed=1, device="cuda", contamination=True, voxel=None, fine=2,
               series=('z', 'x', 'y'), axis=None, supersample=SPIKE_SUPERSAMPLE):
    """Data for the spike phantom (spike_fields), as make_tomo's tensor version: built on a grid `fine` times finer
    than the n^3 reconstruction grid with each part's volume fraction per fine voxel (so faces and interfaces are not
    staircased), projected view by view through the package projector over each part's strain levels n . eps . n,
    the detector binned back to n x n. Rays are series-major, then (view, row, channel). Returns make_tomo's dict:
    V_true (n, n, n, 3) densities, E_true (n, n, n, 3, 6) each part's mean strain in a voxel, mat_true (n, n, n) each
    voxel's largest part (-1 where no part fills half of it), and the materials."""
    import mbirtorch
    from .geometry import PERMS, series_weights
    voxel = 2.56 / n if voxel is None else voxel
    lam, flux = default_axis() if axis is None else axis
    lam = np.asarray(lam); flux = np.asarray(flux) / np.mean(flux)
    d = dose * flux
    mats = SPIKE_MATERIALS; R = len(mats)
    M, g = spectra_model(lam, d, device, materials=mats)
    N = fine * n
    angles = np.linspace(0, np.pi, views, endpoint=False)
    blk = lambda x: x.reshape(n, fine, n, fine, n, fine).mean((1, 3, 5))
    frac, eps = spike_fields(N, supersample)                                     # (N, N, N, R), (N, N, N, R, 6)
    S = len(series)
    wts = series_weights([(ax, angles) for ax in series])                        # (S views, 6)
    # each part's strain levels: the range of n . eps . n over its voxels and all views
    levels_r = []
    for r in range(R):
        e = eps[..., r, :][frac[..., r] > 0]
        seen = e @ wts.T                                                         # (voxels, S views)
        levels_r.append(np.linspace(seen.min(), seen.max(), levels))
    L = R * levels
    proj = np.zeros((R, levels, S * views, n, n))
    for k, ax in enumerate(series):
        perm = PERMS[ax]
        frac_s = frac.transpose(perm + (3,)); eps_s = eps.transpose(perm + (3, 4))
        for v in range(views):
            w = wts[k * views + v]
            inds = []
            for r in range(R):
                seen = eps_s[..., r, :] @ w
                idx = np.abs(seen[..., None] - levels_r[r]).argmin(-1)
                inds += [(frac_s[..., r] * (idx == l)).astype(np.float32) for l in range(levels)]
            model = mbirtorch.ParallelBeamModel((1, N * L, N), angles[v:v + 1])
            model.set_params(use_ror_mask=False)
            sino = model.forward_project(np.concatenate(inds, 2), output_sharded=True)
            proj[:, :, k * views + v] = (sino.reshape(L, n, fine, n, fine).mean((2, 4)) / fine).reshape(R, levels, n, n).cpu().numpy()
    proj = proj.reshape(R, levels, S * views * n * n)
    P = S * views * n * n
    V_true = np.stack([blk(frac[..., r]) for r in range(R)], -1) * voxel
    E_true = np.stack([np.stack([blk(frac[..., r] * eps[..., r, c]) / np.maximum(blk(frac[..., r]), 1e-12) for c in range(6)], -1)
                       for r in range(R)], -2)
    occ = np.stack([blk(frac[..., r]) for r in range(R)], -1)
    mat_true = occ.argmax(-1).astype(np.int8); mat_true[occ.max(-1) < 0.5] = -1
    with torch.no_grad():
        X = torch.zeros(P, M.Kf, dtype=torch.float32, device=device)
        for r in range(R):
            for l, e_l in enumerate(levels_r[r]):
                wl = torch.as_tensor(proj[r, l] * voxel, dtype=torch.float32, device=device)
                if float(wl.max()) <= 0:
                    continue
                eps_t = torch.zeros((1, R), dtype=torch.float64, device=device); eps_t[0, r] = e_l
                X += wl[:, None] * M.spectra(g, eps_t)[0, r].float()[None, :]
        n_t = M.apply_band(torch.exp(-X), M.band_matrix(M.band(g)))
        f = torch.zeros(2, M.K, device=device)
        if contamination:
            k_spike = int(np.abs(lam - 2.638).argmin()); f[0, k_spike] = 0.04
            bump = (lam > 2.655) & (lam < 2.77); f[1, torch.as_tensor(bump, device=device)] = torch.as_tensor(
                0.16 * np.exp(-0.5 * ((lam[bump] - 2.70) / 0.03) ** 2), dtype=torch.float32, device=device)
        m = (1 - f.sum(0)) * n_t + torch.ones(P, 2, device=device) @ f
        gen = torch.Generator(device=device).manual_seed(seed)
        dd = torch.as_tensor(d, dtype=torch.float32, device=device)
        T = torch.poisson(dd * m, generator=gen) / dd
        mu0 = M.spectra(g).cpu().numpy()
    return dict(T=T, lam=lam, dose_per_bin=d, n=n, views=views, angles=angles, V_true=V_true, E_true=E_true,
                mat_true=mat_true, voxel=voxel, mu_true=mu0, lamf=M.lamf.cpu().numpy(), g_true=g.cpu().numpy(),
                f_true=f.cpu().numpy(), materials=mats)


# The Fe cube phantom: one bcc Fe cube seen by three rotation series, each with its own mount and a holder rod.
FE = dict(name='Fe', lattice='bcc', a=2.8665)
FE_ELASTIC = (211e9, 0.29)                  # Young's modulus (Pa) and Poisson ratio of Fe
FE_MAXWELL = (1.0, -0.6, 0.8)               # Maxwell stress-function amplitudes for the cube's residual stress
FE_PEAK = 1.5e-3                            # the largest strain component in the cube
FE_CUBE = dict(half=19.0 / 64, centre=(-0.5 / 64, -0.5 / 64, 7.0 / 64))   # half side and centre (y, x, z), units of n
FE_ATTEN_3A = 0.1 * 64                      # the cube's attenuation per unit length at 3 Angstrom, times n per voxel
FE_AXIS = (1.035, 6.027, 2841)              # wavelength range (Angstrom) and bins of the default axis
FE_A_RANGE = 0.02                           # relative lattice-parameter range of the model (as the Fe cube fits use)
HOLDER = dict(radius=14.0 / 64, gap=1.0 / 64, atten=0.27)  # rod radius and gap past the cube (units of n); attenuation
#                                                            relative to the cube's mean attenuation


def fe_cube_tensor(N, offset=0.0):
    """Indicator (N, N, N) of the Fe cube and its strain tensor (N, N, N, 6), COMPONENTS order, in the projector
    layout (rows y, cols x, slices z). The residual stress comes from Maxwell stress functions S_i b^3 with b the
    product of (1 - u^2 / h^2) over the cube's three axes: it satisfies equilibrium and is traction-free. offset moves
    the cube along all three axes, as a fraction of the grid side."""
    c = (N - 1) / 2.0
    h = FE_CUBE['half'] * N
    cy, cx, cz = ((v + offset) * N for v in FE_CUBE['centre'])
    yy, xx, zz = np.meshgrid(np.arange(N) - c - cy, np.arange(N) - c - cx, np.arange(N) - c - cz, indexing='ij')
    cube = (np.abs(yy) < h) & (np.abs(xx) < h) & (np.abs(zz) < h)
    pts = torch.tensor(np.stack([xx, yy, zz], -1).reshape(-1, 3)[cube.reshape(-1)], dtype=torch.float64)
    bump = lambda p: (1 - p[0] ** 2 / h ** 2) * (1 - p[1] ** 2 / h ** 2) * (1 - p[2] ** 2 / h ** 2)
    Hs = [torch.func.vmap(torch.func.hessian(lambda p, S=S: S * bump(p) ** 3))(pts) for S in FE_MAXWELL]
    h1, h2, h3 = Hs
    sig = torch.stack([h2[:, 2, 2] + h3[:, 1, 1], h3[:, 0, 0] + h1[:, 2, 2], h1[:, 1, 1] + h2[:, 0, 0],
                       -h3[:, 0, 1], -h1[:, 1, 2], -h2[:, 0, 2]], -1).numpy()
    Em, nu = FE_ELASTIC
    tr = sig[:, :3].sum(1, keepdims=True)
    e = np.concatenate([((1 + nu) * sig[:, :3] - nu * tr) / Em, (1 + nu) * sig[:, 3:] / Em], 1)
    eps = np.zeros((N, N, N, 6))
    eps.reshape(-1, 6)[cube.reshape(-1)] = e * FE_PEAK / np.abs(e).max()
    return cube, eps


def fe_spectrum_model(lam, dose_per_bin, n, device='cpu', kernel='two_scale', kernel_values=None, texture=None):
    """A Hybrid2 model for bcc Fe on the axis lam and its parameter vector: Bragg heights j d, a smooth part, the
    resolution kernel, scaled so the attenuation per voxel at 3 Angstrom is FE_ATTEN_3A / n. kernel: the kernel's form
    (model.Hybrid2); kernel_values: its 8 parameters (default: JOINT_KERNEL, a two-scale kernel). texture: (March-Dollase
    strength, axis polar angle, axis azimuth) for a textured cube (shape model texture3d), or the 9 degree-4 coefficients
    of a cubic texture (shape model cubic4); the caller sets view_dirs."""
    M = Hybrid2(lam, dose_per_bin, [(FE['lattice'], (FE['a'] * (1 - FE_A_RANGE), FE['a'] * (1 + FE_A_RANGE)))], n_spline=8, n_contam=2,
                device=device, two_scale=True, kernel=kernel,
                shape_model=None if texture is None else ('cubic4' if len(texture) == 9 else 'texture3d'))
    g = torch.zeros(M.n_global, dtype=torch.float64, device=device)
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor(JOINT_KERNEL if kernel_values is None else kernel_values, dtype=torch.float64)
    sl = M.slices[0]
    g[sl['a']] = FE['a']
    ns = M.mats[0]['ns'].cpu().numpy(); d = FE['a'] / np.sqrt(ns)
    g[sl['h'][0]:sl['h'][1]] = torch.as_tensor(np.array(multiplicities(FE['lattice'], ns), float) * d)
    g[sl['c'][0]] = 0.3; g[sl['c'][0] + 1] = 0.2
    if texture is not None:
        g[sl['x'][0]:sl['x'][1]] = torch.as_tensor(np.asarray(texture, float))
    with torch.no_grad():
        mu = M.spectra(g)[0].cpu().numpy()
    k3 = np.abs(M.lamf.cpu().numpy() - 3.0).argmin()
    scale = (FE_ATTEN_3A / n) / mu[k3]
    g[sl['h'][0]:sl['h'][1]] *= scale; g[sl['c'][0]:sl['c'][1]] *= scale
    return M, g


def make_fe_cube(n=64, angles=None, dose=1919.0, inflation=6.0, mounts=None, offsets=None, levels=31, fine=2, seed=1,
                 device='cuda', axis=None, holder=True, contamination=True, kernel='two_scale', kernel_values=None,
                 cube_shift=0.0, texture=None):
    """Synthetic data shaped like the Fe cube measurement: a bcc Fe cube with a residual-stress strain tensor
    (fe_cube_tensor), seen by rotation series with their own mounts and a holder rod each.

    mounts: per series (perm, signs, shift, rot) as tomo.MultiProjector takes them (shift in voxels of n, rot in
    degrees); default: rotations about the three axes with no shifts. offsets: per series centre-of-rotation offsets
    (detector channels). The holder is a rod on each series' rotation axis past the cube's end, with a flat attenuation
    spectrum; it is not part of the sample, so a fit should leave its detector rows out (holder_rows). Counts are
    Poisson with their variance inflated by `inflation` (counts drawn at dose / inflation, scaled back), as the
    measured detector's correlated counts. The volume is generated on a grid `fine` times finer and the detector
    binned back to n x n. Rays are series-major, then (view, row, channel), as MultiProjector orders them.
    Returns a dict with T, lam, dose_per_bin, angles, mounts, offsets, V_true (n, n, n, 1), E_true (n, n, n, 1, 6),
    the true spectrum, and holder_rows (the first detector row of the holder in each series). kernel, kernel_values:
    the resolution kernel the data are generated with (see fe_spectrum_model). cube_shift moves the cube along all three
    axes by this many detector pixels (a multiple of 1 / fine), so that its faces can fall between the planes of a
    finer reconstruction grid. texture: (March-Dollase strength, axis polar angle, axis azimuth in the sample frame), or
    the 9 degree-4 coefficients of a cubic texture, for a textured cube: each view's Bragg edges carry that view's
    texture factor (model.Hybrid2.texture_factor)."""
    import mbirtorch
    from .geometry import series_weights
    from .tomo import MultiProjector, Projector, detector_directions
    angles = np.linspace(0, np.pi, 51) if angles is None else np.asarray(angles)
    mounts = [((0, 1, 2), (1, 1, 1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), ((1, 2, 0), (1, 1, 1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
              ((2, 0, 1), (1, 1, 1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))] if mounts is None else \
        [tuple(m) + ((0.0, 0.0, 0.0),) * (4 - len(m)) for m in mounts]
    offsets = [0.0] * len(mounts) if offsets is None else list(offsets)
    if axis is None:
        lam = np.linspace(*FE_AXIS[:2], FE_AXIS[2]); flux = np.ones_like(lam)
    else:
        lam, flux = np.asarray(axis[0]), np.asarray(axis[1])
    d = dose * flux / np.mean(flux)
    M, g = fe_spectrum_model(lam, d, n, device, kernel=kernel, kernel_values=kernel_values, texture=texture)
    N = fine * n; V_ = len(angles); S = len(mounts)
    cube, et = fe_cube_tensor(N, cube_shift / n)
    wts = series_weights([((m[0], m[1], m[2]), angles) for m in mounts])         # (S views, 6)
    if texture is not None:
        from .geometry import directions_from_weights
        M.view_dirs = directions_from_weights(wts)
    seen_all = [(et * w).sum(-1) for w in wts]
    lv = np.linspace(min(s[cube].min() for s in seen_all), max(s[cube].max() for s in seen_all), levels)
    proj = np.zeros((levels, S * V_, n, n))                                      # path length per strain level
    hold = np.zeros((S * V_, n, n))
    holder_rows = []
    zc = (N - 1) / 2.0
    for k, (perm, signs, shift, rot) in enumerate(mounts):
        tr = (tuple(perm), tuple(signs), tuple(float(v) * fine for v in shift), tuple(float(v) for v in rot))
        # the holder: a rod in the series frame on the rotation axis (slices), from just past the cube's far end
        cube_s = MultiProjector._to_series(torch.as_tensor(cube[..., None], dtype=torch.float32, device=device), tr)[..., 0]
        zs = torch.nonzero(cube_s.sum((0, 1)) > 0.5)[:, 0]
        z0 = int(zs.max()) + 1 + int(round(HOLDER['gap'] * N))
        yy, xx = np.meshgrid(np.arange(N) - zc, np.arange(N) - zc, indexing='ij')     # the axis is the grid's centre
        rod = np.zeros((N, N, N), np.float32)
        if holder and z0 < N:
            rod[(yy ** 2 + xx ** 2 < (HOLDER['radius'] * N) ** 2), z0:] = 1.0
        holder_rows.append(z0 // fine)
        # the sample moves with the mount (the fractional part of the shift on the detector, as MultiProjector does);
        # the holder rod stays on the rotation axis
        whole, frac = MultiProjector._split(tr[2])
        d_rows, d_cols = detector_directions(angles)
        rod_t = torch.as_tensor(rod, device=device)[..., None]
        for v in range(V_):
            idx = np.abs(seen_all[k * V_ + v][..., None] - lv).argmin(-1)
            inds = torch.as_tensor(np.stack([(cube & (idx == l)) for l in range(levels)], -1), dtype=torch.float32, device=device)
            vol = MultiProjector._to_series(inds, tr)
            pv = Projector(N, angles[v:v + 1], levels, float(offsets[k]) * fine, False,
                           np.array([frac[0] * d_rows[v] + frac[1] * d_cols[v]]), frac[2])
            ph_ = Projector(N, angles[v:v + 1], 1, float(offsets[k]) * fine, False)
            rays = torch.cat([pv.fwd(vol), ph_.fwd(rod_t)], 1)                         # (N N, levels + 1)
            b = (rays.T.reshape(levels + 1, n, fine, n, fine).mean((2, 4)) / fine).cpu().numpy()
            proj[:, k * V_ + v] = b[:levels]; hold[k * V_ + v] = b[levels]
    P = S * V_ * n * n
    proj = proj.reshape(levels, P); hold = hold.reshape(P)
    blk = lambda a: a.reshape(n, fine, n, fine, n, fine).mean((1, 3, 5))
    frac = np.maximum(blk(cube.astype(np.float64)), 1e-12)
    V_true = blk(cube.astype(np.float64))[..., None]
    E_true = np.stack([blk(np.where(cube, et[..., c], 0.0)) / frac for c in range(6)], -1)[..., None, :]
    with torch.no_grad():
        X = torch.zeros(P, M.Kf, dtype=torch.float32, device=device)
        for l, e_l in enumerate(lv):
            w = torch.as_tensor(proj[l], dtype=torch.float32, device=device)
            if float(w.max()) <= 0:
                continue
            eps_t = torch.full((1, 1), float(e_l), dtype=torch.float64, device=device)
            if texture is None:
                X += w[:, None] * M.spectra(g, eps_t)[0, 0].float()[None, :]
            else:                                   # each view's spectrum, with its own texture factor
                nv = S * V_; per = n * n
                spec = M.spectra(g, eps_t.expand(nv, 1), torch.arange(nv, device=device))[:, 0].float()   # (views, Kf)
                for v in range(nv):
                    X[v * per:(v + 1) * per] += w[v * per:(v + 1) * per, None] * spec[v][None, :]
        mu0 = M.spectra(g)[0]
        X += torch.as_tensor(hold, dtype=torch.float32, device=device)[:, None] * float(HOLDER['atten'] * mu0.mean())
        n_t = M.apply_band(torch.exp(-X), M.band_matrix(M.band(g)))
        f = torch.zeros(2, M.K, device=device)
        if contamination:
            k_spike = int(np.abs(lam - 2.638).argmin()); f[0, k_spike] = 0.04
            bump = (lam > 2.655) & (lam < 2.77); f[1, torch.as_tensor(bump, device=device)] = torch.as_tensor(
                0.16 * np.exp(-0.5 * ((lam[bump] - 2.70) / 0.03) ** 2), dtype=torch.float32, device=device)
        m = (1 - f.sum(0)) * n_t + torch.ones(P, 2, device=device) @ f
        gen = torch.Generator(device=device).manual_seed(seed)
        dd = torch.as_tensor(d, dtype=torch.float32, device=device)
        T = torch.poisson(dd * m / inflation, generator=gen) * inflation / dd
    return dict(T=T, lam=lam, dose_per_bin=d, angles=angles, mounts=mounts, offsets=offsets, V_true=V_true,
                E_true=E_true, mu_true=mu0.cpu().numpy(), lamf=M.lamf.cpu().numpy(), g_true=g.cpu().numpy(),
                f_true=f.cpu().numpy(), holder_rows=holder_rows, n=n, inflation=inflation)

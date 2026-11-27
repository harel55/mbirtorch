"""Tests of the experimental strain-tomography projectors in ``mbirtorch.hsnt.strain``."""
import math
import types
import warnings

import numpy as np
import pytest
import torch

from mbirtorch.hsnt.strain.model import FallbackWarning
from mbirtorch.hsnt.strain.tomo import MultiProjector

N = 12
ANGLES = np.linspace(0, np.pi, 5, endpoint=False)
MOUNTS = [('x', ANGLES),
          (((1, 2, 0), (1, 1, 1), (0.4, -2.3, 1.7), (-0.35, 1.93, -0.63)), ANGLES),
          (((2, 0, 1), (-1, 1, -1), (3.0, -0.5, 0.25)), ANGLES)]


@pytest.fixture(scope="module")
def proj():
    """Three series: one plain, one with a fractional shift and a small rotation, one with a fractional shift only."""
    return MultiProjector(N, MOUNTS, 2, channel_offsets=[0.0, 0.3, -0.2], ror_mask=False)


def test_adjoint(proj):
    """back is the adjoint of fwd, including fractional shifts and small rotations."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, N, N, 2, generator=g)
    y = torch.randn(sum(proj.counts), 2, generator=g)
    a = float((proj.fwd(x).cpu() * y).sum())             # back returns its result on the projector's device
    b = float((x * proj.back(y).cpu()).sum())
    assert abs(a - b) <= 1e-4 * abs(a)


def test_set_mount_round_trip(proj):
    """Setting a series' mount to new values and back restores its projection."""
    x = torch.rand(N, N, N, 2, generator=torch.Generator().manual_seed(1))
    before = proj.fwd(x)
    (perm, signs, shift, rot), part = proj.parts[1]
    off = part.channel_offset
    proj.set_mount(1, shift=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0), offset=off + 0.5)
    assert not torch.allclose(proj.fwd(x), before)
    proj.set_mount(1, shift=shift, rot=rot, offset=off)
    assert torch.allclose(proj.fwd(x), before)


@pytest.mark.parametrize("prior", ["quadratic", "qggmrf"])
def test_density_prior_gradient(prior):
    """The density prior's gradient (dens_lap at V) matches central differences of its value."""
    from mbirtorch.hsnt.strain.tomo import TomoFit
    tf = TomoFit.__new__(TomoFit)
    tf.R, tf.voxel_scale, tf.density_prior = 1, 0.7, prior
    tf.sigma_V = torch.tensor([0.05], dtype=torch.float64)
    tf.smooth_V = tf.voxel_scale / tf.sigma_V ** 2
    V = torch.rand(6, 6, 6, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) * 0.1
    V[3:] += 0.5                                          # a sharp jump
    tf.V = V
    g = tf.dens_lap(V, tf._dens_wts(V))
    h = 1e-6
    for idx in [(0, 0, 0, 0), (2, 3, 1, 0), (3, 3, 3, 0)]:
        Vp, Vm = V.clone(), V.clone(); Vp[idx] += h; Vm[idx] -= h
        num = (tf._dens_prior_value(Vp) - tf._dens_prior_value(Vm)) / (2 * h)
        assert abs(float(g[idx]) - num) <= 1e-5 * max(1.0, abs(num))


def test_ikeda_carpenter_kernel():
    """The Ikeda-Carpenter kernel's rows are normalized and a sharp edge keeps its 50% point (median centring)."""
    from mbirtorch.hsnt.strain.model import Hybrid2
    lam = np.linspace(1.9, 4.41, 1400)
    M = Hybrid2(lam, np.ones_like(lam), [('fcc', (3.52 * 0.99, 3.52 * 1.01))], smooth_degree=5, n_contam=1, device='cpu')
    g = torch.zeros(M.n_global, dtype=torch.float64)
    g[M.res_sl[0]:M.res_sl[0] + 6] = torch.tensor([0.0, 0.0008, 0.001, 0.0004, 1.6, 0.004], dtype=torch.float64)
    Rb = M.band(g)
    assert torch.allclose(Rb.sum(1), torch.ones(Rb.shape[0], dtype=torch.float64), atol=1e-9)
    for e in (2.5, 3.5):
        T = (Rb * (M.lamf > e).double()[M.band_idx]).sum(1).numpy()
        assert abs(lam[np.argmin(np.abs(T - 0.5))] - e) < 2 * (lam[1] - lam[0])


def test_traction_free_faces():
    """A uniform uniaxial strain along x gives traction only on the faces normal to x; zero strain gives none."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit
    n = 6
    tf = object.__new__(TomoFit)
    tf.elastic = [(211e9, 0.29)]
    inside = torch.zeros(n, n, n, 1, dtype=torch.bool); inside[1:-1, 1:-1, 1:-1] = True
    tf._eq_inside = inside
    E = torch.zeros(n, n, n, 1, 6, dtype=torch.float64)
    assert float(tf._traction(E).abs().max()) == 0.0
    E[..., 0] = 1e-3 * inside[..., None].double()[..., 0]                          # exx inside the sample
    t = tf._traction(E)[..., 0, :, :]                                               # (n, n, n, 3 normals, 3)
    nu = 0.29; lam_, mu_ = nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))
    x_face = t[2, 1, 2, 0]                                                          # volume axes (y, x, z): x = 1 is a face
    assert abs(float(x_face[0]) - (lam_ + 2 * mu_) * 1e-3) < 1e-12 and float(x_face[1:].abs().max()) == 0.0
    y_face = t[1, 2, 2, 1]                                                          # y face: normal (0, 1, 0)
    assert abs(float(y_face[1]) - lam_ * 1e-3) < 1e-12                               # sigma_yy from the Poisson term
    assert float(t[2, 2, 2].abs().max()) == 0.0                                     # an interior voxel


def test_recenter_full_zero_mean():
    """recenter_strain(full=True) leaves every strain component with zero density-weighted mean over the material."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit

    class _M:                                                                  # the lattice slot recenter_strain moves
        slices = [{'a': 0}]
        lb = torch.tensor([2.8]); ub = torch.tensor([2.9])

    class _Fit:
        M = _M(); g = torch.tensor([2.8665], dtype=torch.float64)

    n = 5
    tf = object.__new__(TomoFit)
    tf.fit, tf.R, tf.C = _Fit(), 1, 6
    tf.V = torch.ones(n, n, n, 1, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    tf.E = 1e-3 * torch.randn(n, n, n, 1, 6, generator=g, dtype=torch.float64) + torch.tensor([3e-4, 0, 0, 2e-4, -1e-4, 5e-4], dtype=torch.float64)
    tf.sync = lambda: None
    tf.recenter_strain(full=True)
    assert float(tf.E.mean((0, 1, 2)).abs().max()) < 1e-12


def test_mean_projector():
    """TomoFit._mean_projector removes the volume-fraction-weighted mean deviatoric strain over the material, keeps the
    mean isotropic strain, and applying it twice changes nothing."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit

    n = 5
    tf = object.__new__(TomoFit)
    tf.R, tf.C = 1, 6
    g = torch.Generator().manual_seed(1)
    tf.V = 0.5 + torch.rand(n, n, n, 1, generator=g, dtype=torch.float64)
    X = torch.randn(n, n, n, 1, 6, generator=g, dtype=torch.float64)
    P = tf._mean_projector(tf.V, torch.ones_like(X, dtype=torch.bool))
    Y = P(X)
    w = tf.V[..., 0][..., None] * tf.material_voxels()[..., 0][..., None].to(tf.V.dtype)
    m = (w * Y[..., 0, :]).sum((0, 1, 2))
    assert float((m[:3] - m[:3].mean()).abs().max()) < 1e-10 and float(m[3:].abs().max()) < 1e-10   # no deviatoric mean
    mX = (w * X[..., 0, :]).sum((0, 1, 2))
    assert abs(float(m[:3].mean() - mX[:3].mean())) < 1e-10                                            # isotropic mean kept
    assert float((P(Y) - Y).abs().max()) < 1e-12


def test_traction_normal_flat_face():
    """traction_model='normal' carries the traction to the surface: on a flat free face half a voxel beyond the boundary
    voxels' centres, a stress falling linearly to zero at that surface gives zero traction (the axis model does not)."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit

    n, nu = 8, 0.3
    tf = object.__new__(TomoFit)
    tf.R, tf.C, tf.elastic = 1, 6, [(1.0, nu)]
    ins = torch.zeros(n, n, n, 1, dtype=torch.bool); ins[1:4, 1:7, 1:7] = True          # volume axes (y, x, z)
    tf._eq_inside = ins
    tf.V = ins.to(torch.float64)
    y = torch.arange(n, dtype=torch.float64)[:, None, None].expand(n, n, n)
    sig = 3.5 - y                                                                      # sigma_yy, zero at y = 3.5
    E = torch.zeros(n, n, n, 1, 6, dtype=torch.float64)
    E[..., 0, 0] = -nu * sig; E[..., 0, 1] = sig; E[..., 0, 2] = -nu * sig            # uniaxial stress along y
    t = tf._traction_normal(E)[3, 3, 3, 0]                                             # a voxel on the face y = 3
    assert float(t.abs().max()) < 1e-9
    assert float(tf._traction(E)[3, 3, 3, 0].abs().max()) > 0.1


def test_plane_fit_normals_sphere_and_cube():
    """surface.plane_fit_normals gives a voxelized sphere's boundary voxels normals near the radial direction, and a
    cube's faces, edges and corners their exact normals."""
    import numpy as np
    from mbirtorch.hsnt.strain.surface import plane_fit_normals

    n, radius = 24, 9.3
    g = np.indices((n,) * 3).astype(float) - (n - 1) / 2
    N = plane_fit_normals((g ** 2).sum(0) <= radius ** 2)
    b = np.linalg.norm(N, axis=-1) > 0
    radial = np.stack([g[2], g[1], g[0]], -1)[b]
    cos = (N[b] * radial).sum(-1) / np.linalg.norm(radial, axis=-1)
    assert cos.min() > np.cos(np.radians(10)) and np.degrees(np.arccos(np.clip(cos, -1, 1))).mean() < 5
    m = np.zeros((12,) * 3, bool); m[2:9, 2:9, 2:9] = True
    N = plane_fit_normals(m)
    np.testing.assert_allclose(N[5, 5, 8], [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(N[5, 8, 8], [2 ** -0.5, 2 ** -0.5, 0], atol=1e-9)
    np.testing.assert_allclose(N[8, 8, 8], [3 ** -0.5] * 3, atol=1e-9)


def test_traction_planefit_flat_face():
    """traction_model='planefit' matches 'normal' on a flat, grid-aligned free face: both carry the traction to the
    surface, where a stress falling linearly to zero gives zero traction."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit

    n, nu = 8, 0.3
    tf = object.__new__(TomoFit)
    tf.R, tf.C, tf.elastic = 1, 6, [(1.0, nu)]
    ins = torch.zeros(n, n, n, 1, dtype=torch.bool); ins[1:4, 1:7, 1:7] = True          # volume axes (y, x, z)
    tf._eq_inside = ins
    tf.V = ins.to(torch.float64)
    y = torch.arange(n, dtype=torch.float64)[:, None, None].expand(n, n, n)
    sig = 3.5 - y
    E = torch.zeros(n, n, n, 1, 6, dtype=torch.float64)
    E[..., 0, 0] = -nu * sig; E[..., 0, 1] = sig; E[..., 0, 2] = -nu * sig
    tf.traction_model = 'planefit'
    tf._refresh_plane_fit_normal(ins[..., 0])
    nrm = tf._plane_fit_cache[1]
    torch.testing.assert_close(nrm[3, 3, 3], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
    assert float(tf._traction_normal(E)[3, 3, 3, 0].abs().max()) < 1e-9


def test_traction_cutcell_oblique_face():
    """traction_model='cutcell' on an oblique flat free face: a uniform stress along the face (in equilibrium and free of
    traction) leaves a small residual on the face's voxels, far below the axis model's traction, and a uniform pressure
    (which loads the face) does not."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit

    n, nu, sub = 16, 0.3, 6
    nrm = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64); nrm = nrm / nrm.norm()
    t = torch.tensor([2.0, -1.0, 0.0], dtype=torch.float64); t = t / t.norm()          # along the face
    g = (torch.arange(sub, dtype=torch.float64) + 0.5) / sub - 0.5
    off = torch.stack(torch.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    iy, ix, iz = torch.meshgrid(*(torch.arange(n, dtype=torch.float64),) * 3, indexing='ij')   # volume axes (y, x, z)
    centre = torch.stack([ix, iy, iz], -1) - (n - 1) / 2
    fill = (((centre[..., None, :] + off) @ nrm) <= 0).double().mean(-1)
    box = torch.zeros(n, n, n, dtype=torch.bool); box[2:-2, 2:-2, 2:-2] = True
    ins = (fill > 0.5) & box
    tf = object.__new__(TomoFit)
    tf.R, tf.C, tf.elastic = 1, 6, [(1.0, nu)]
    tf._eq_inside = ins[..., None]
    tf.V = (fill * box)[..., None]
    tf.traction_model = 'cutcell'
    tf._refresh_plane_fit_normal(ins)
    tf._refresh_cutcell_areas(ins[..., None])
    face = ins & ~torch.roll(ins, -1, 1) & (centre.abs().amax(-1) < n / 2 - 4)           # oblique face, away from the box

    def strain(S):                                     # isotropic, Young's modulus 1, COMPONENTS order
        e = (1 + nu) * S - nu * torch.trace(S) * torch.eye(3, dtype=torch.float64)
        E = torch.zeros(n, n, n, 1, 6, dtype=torch.float64)
        E[..., 0, :] = torch.stack([e[0, 0], e[1, 1], e[2, 2], e[0, 1], e[1, 2], e[0, 2]])
        return E

    E = strain(torch.outer(t, t))
    cut = tf._traction_cutcell(E)[..., 0, :][face].norm(dim=-1)
    axis = tf._traction(E)[..., 0, :, :][face].norm(dim=(-2, -1))
    assert int(face.sum()) > 10
    assert float(cut.mean()) < 0.1 * float(axis.mean())
    press = tf._traction_cutcell(strain(torch.eye(3, dtype=torch.float64)))[..., 0, :][face].norm(dim=-1)
    assert float(press.mean()) > 5 * float(cut.mean())


def test_traction_sd_curvature():
    """traction_sd='curvature': the surface curvature from the plane-fit normals is near 2 / R on a voxelized sphere and
    near 0 on a cube's faces, and the traction's standard deviation on a flat face is the surface stress RMS times
    TRACTION_FLAT_REL."""
    import numpy as np
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit, TRACTION_FLAT_REL

    n, R = 32, 11.0
    g = torch.arange(n, dtype=torch.float64) - (n - 1) / 2
    yy, xx, zz = torch.meshgrid(g, g, g, indexing='ij')
    tf = object.__new__(TomoFit)
    tf.V = torch.zeros(n, n, n, 1, dtype=torch.float64)
    sph = (xx ** 2 + yy ** 2 + zz ** 2) <= R ** 2
    tf._refresh_plane_fit_normal(sph)
    k = tf._surface_curvature(sph)
    has = tf._plane_fit_cache[1].abs().sum(-1) > 0
    assert abs(float(np.median(k[has].numpy())) - 2 / R) < 0.35 * 2 / R
    cube = (xx.abs() <= 9) & (yy.abs() <= 9) & (zz.abs() <= 9)
    tf._plane_fit_cache = None
    tf._refresh_plane_fit_normal(cube)
    k = tf._surface_curvature(cube)
    assert float(k[n // 2, n // 2 + 9, n // 2]) < 1e-9                  # the middle of the face x = 9 (volume axes y, x, z)
    tf.R, tf.C, tf.elastic, tf.eq_sd, tf.D = 1, 6, [(1.0, 0.3)], 1e-5, None
    E = torch.zeros(n, n, n, 1, 6, dtype=torch.float64); E[..., 0, 0] = 1e-3
    tf.E = E
    tf._refresh_traction_sd(cube[..., None])
    s = torch.stack(TomoFit._stress(E[..., 0, :], 0.3), -1)
    rms = float(s[n // 2, n // 2 + 9, n // 2].norm())
    assert abs(float(tf._traction_sd_map[n // 2, n // 2 + 9, n // 2, 0]) - TRACTION_FLAT_REL * rms) < 1e-9


def test_refresh_mask():
    """TomoFit.refresh_mask replaces a frozen material mask that overlaps the current volume fractions' mask poorly, and
    keeps one that agrees."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit

    n = 8
    tf = object.__new__(TomoFit)
    tf.mask_method, tf.mask_freeze, tf._mask = 'max', True, None
    V = torch.zeros(n, n, n, 2); V[2:6, 2:6, 2:6, 0] = 1.0; V[1:4, 1:4, 1:4, 1] = 1.0
    tf.V = V
    tf.material_voxels()                                                     # freezes the masks
    tf._mask[..., 1] = True                                                  # material 1's mask now reaches everywhere
    out = tf.refresh_mask()
    assert [r for r, _ in out] == [1]
    assert bool((tf._mask[..., 1] == (V[..., 1] > 0.5)).all()) and bool((tf._mask[..., 0] == (V[..., 0] > 0.5)).all())


def test_qggmrf_strain_prior_gradient():
    """The q-GGMRF strain prior's gradient (strain_lap with its weights at E) matches finite differences of its value,
    and it is far weaker than the quadratic prior across a jump."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit

    n = 6
    tf = object.__new__(TomoFit)
    tf.R, tf.C, tf.voxel_scale, tf.smooth_sd = 1, 6, 1.0, 8e-4
    tf.smooth_E = 1.0 / tf.smooth_sd ** 2
    g = torch.Generator().manual_seed(2)
    E = 3e-4 * torch.randn(n, n, n, 1, 6, generator=g, dtype=torch.float64)
    E[3:] += 2e-3                                                            # a jump across the first axis
    tf.E = E
    tf.strain_prior = 'qggmrf'
    grad = tf.strain_lap(E, tf._strain_wts(E))
    i = (2, 1, 3, 0, 4); h = 1e-9
    Ep, Em = E.clone(), E.clone(); Ep[i] += h; Em[i] -= h
    fd = (tf._strain_prior_value(Ep) - tf._strain_prior_value(Em)) / (2 * h)
    assert abs(fd - float(grad[i])) < 1e-4 * max(abs(fd), 1.0)
    q = tf._strain_prior_value(E); tf.strain_prior = 'quadratic'
    assert q < 0.5 * tf._strain_prior_value(E)


def test_qggmrf_strain_prior_shape():
    """The strain q-GGMRF with its exponent q and threshold T changed (strain_q, strain_T): the gradient still matches
    finite differences of the value, and a smaller q penalizes a jump less."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit

    n = 6
    tf = object.__new__(TomoFit)
    tf.R, tf.C, tf.voxel_scale, tf.smooth_sd = 1, 6, 1.0, 8e-4
    tf.smooth_E = 1.0 / tf.smooth_sd ** 2
    g = torch.Generator().manual_seed(2)
    E = 3e-4 * torch.randn(n, n, n, 1, 6, generator=g, dtype=torch.float64)
    E[3:] += 2e-3
    tf.E, tf.strain_prior = E, 'qggmrf'
    tf.strain_q, tf.strain_T = 1.05, 0.5
    grad = tf.strain_lap(E, tf._strain_wts(E))
    i = (2, 1, 3, 0, 4); h = 1e-9
    Ep, Em = E.clone(), E.clone(); Ep[i] += h; Em[i] -= h
    fd = (tf._strain_prior_value(Ep) - tf._strain_prior_value(Em)) / (2 * h)
    assert abs(fd - float(grad[i])) < 1e-4 * max(abs(fd), 1.0)
    low = tf._strain_prior_value(E)
    tf.strain_q = 1.2
    assert low < tf._strain_prior_value(E)


def test_qggmrf_strain_prior_curvature():
    """The curvature weights give the q-GGMRF strain prior's exact Hessian (finite differences of its gradient), and they
    equal the surrogate weights where the differences are small."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit

    n = 6
    tf = object.__new__(TomoFit)
    tf.R, tf.C, tf.voxel_scale, tf.smooth_sd = 1, 6, 1.0, 8e-4
    tf.smooth_E = 1.0 / tf.smooth_sd ** 2
    tf.strain_prior = 'qggmrf'
    g = torch.Generator().manual_seed(3)
    E = 3e-4 * torch.randn(n, n, n, 1, 6, generator=g, dtype=torch.float64)
    E[3:] += 2e-3                                                            # a jump across the first axis
    v = torch.randn(E.shape, generator=g, dtype=torch.float64)
    grad = lambda x: tf.strain_lap(x, tf._strain_wts(x))
    h = 1e-9
    fd = (grad(E + h * v) - grad(E - h * v)) / (2 * h)
    hv = tf.strain_lap(v, tf._strain_wts(E, curvature=True))
    assert float((fd - hv).norm()) < 1e-4 * float(hv.norm())
    for small in (1e-4 * E, torch.zeros_like(E)):
        for s_, c_ in zip(tf._strain_wts(small), tf._strain_wts(small, curvature=True)):
            assert torch.allclose(s_, c_, rtol=1e-2), \
                f'largest relative difference {float(((s_ - c_).abs() / c_.abs()).max()):.3g}'


@pytest.mark.parametrize("prior", ["quadratic", "qggmrf"])
def test_second_order_strain_prior(prior):
    """With smooth_order=2 the strain smoothness prior costs nothing for a linear strain field, its gradient
    (strain_lap with its weights at E) matches finite differences of its value, the curvature weights give its Hessian,
    and strain_diag is the diagonal of strain_lap."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit

    n = 6
    tf = object.__new__(TomoFit)
    tf.R, tf.C, tf.voxel_scale, tf.smooth_sd, tf.smooth_order = 1, 6, 0.5, 8e-4, 2
    tf.smooth_E = tf._smooth_vs() / tf.smooth_sd ** 2
    tf.strain_prior = prior
    z, y, x = torch.meshgrid(*[torch.arange(n, dtype=torch.float64)] * 3, indexing='ij')
    lin = (1e-4 * (z + 2 * y - x) + 3e-4)[..., None, None].expand(n, n, n, 1, 6).clone()
    tf.E = lin
    assert abs(tf._strain_prior_value(lin)) < 1e-8 * tf._strain_prior_value(lin + 1e-4 * torch.rand(lin.shape, dtype=torch.float64))
    g = torch.Generator().manual_seed(4)
    E = lin + 3e-4 * torch.randn(n, n, n, 1, 6, generator=g, dtype=torch.float64)
    E[3:] += 2e-3                                                            # a jump across the first axis
    tf.E = E
    grad = tf.strain_lap(E, tf._strain_wts(E))
    i = (2, 1, 3, 0, 4); h = 1e-9
    Ep, Em = E.clone(), E.clone(); Ep[i] += h; Em[i] -= h
    fd = (tf._strain_prior_value(Ep) - tf._strain_prior_value(Em)) / (2 * h)
    assert abs(fd - float(grad[i])) < 1e-4 * max(abs(fd), 1.0)
    v = torch.randn(E.shape, generator=g, dtype=torch.float64)
    gradf = lambda X: tf.strain_lap(X, tf._strain_wts(X))
    fdh = (gradf(E + 1e-9 * v) - gradf(E - 1e-9 * v)) / 2e-9
    cw = tf._strain_wts(E, curvature=True)
    hv = tf.strain_lap(v, cw)
    assert float((fdh - hv).norm()) < 1e-4 * float(hv.norm())
    diag = tf.strain_diag(cw) if prior == 'qggmrf' else torch.as_tensor(tf.strain_diag(None)).expand_as(E)
    k = (2, 3, 2, 0, 1)                                                      # an interior unknown
    unit = torch.zeros_like(E); unit[k] = 1.0
    assert abs(float(tf.strain_lap(unit, cw)[k]) - float(diag[k])) < 1e-6 * abs(float(diag[k]))


def test_prior_scale_per_material():
    """With a per-material prior scale, the strain smoothness prior's value is each material's, scaled, and its gradient
    (strain_lap) still matches finite differences of the value (quadratic and q-GGMRF)."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit

    n = 5
    tf = object.__new__(TomoFit)
    tf.R, tf.C, tf.voxel_scale, tf.smooth_sd = 2, 6, 1.0, 8e-4
    tf.smooth_E = 1.0 / tf.smooth_sd ** 2
    g = torch.Generator().manual_seed(3)
    E = 5e-4 * torch.randn(n, n, n, 2, 6, generator=g, dtype=torch.float64); tf.E = E
    for prior in ('quadratic', 'qggmrf'):
        tf.strain_prior, tf.prior_scale = prior, None
        tf.prior_scale = torch.tensor([1.0, 0.25], dtype=torch.float64)
        grad = tf.strain_lap(E, tf._strain_wts(E))
        for i in ((2, 1, 3, 0, 4), (1, 3, 2, 1, 0)):
            h = 1e-10; Ep, Em = E.clone(), E.clone(); Ep[i] += h; Em[i] -= h
            fd = (tf._strain_prior_value(Ep) - tf._strain_prior_value(Em)) / (2 * h)
            assert abs(fd - float(grad[i])) < 1e-4 * max(abs(fd), 1.0), (prior, i, fd, float(grad[i]))
    tf.strain_prior, tf.prior_scale = 'quadratic', None
    a = tf._strain_prior_value(E)
    tf.prior_scale = torch.tensor([1.0, 0.25], dtype=torch.float64)
    E0 = E.clone(); E0[..., 1, :] = 0
    E1 = E.clone(); E1[..., 0, :] = 0
    tf.prior_scale = None
    v0, v1 = tf._strain_prior_value(E0), tf._strain_prior_value(E1)
    tf.prior_scale = torch.tensor([1.0, 0.25], dtype=torch.float64)
    assert abs(tf._strain_prior_value(E) - (v0 + 0.25 * v1)) < 1e-9 * a


def test_cg_solve_and_evidence_update():
    """TomoFit._cg_solve solves a positive diagonal system, and evidence_update sets each material's scale to
    gamma / (E^T L E) with gamma = M - scale tr(L H^-1), clamped to EVIDENCE_MAX_STEP per update."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit, EVIDENCE_MAX_STEP

    tf = object.__new__(TomoFit)
    tf.R, tf.C = 2, 6
    d = torch.linspace(1.0, 5.0, 2 * 6 * 8, dtype=torch.float64).reshape(2, 2, 2, 2, 6)
    H = lambda xV, xE: (xV, d * xE)
    prec = lambda rV, rE: (rV, rE)
    b = torch.ones_like(d)
    _, x = tf._cg_solve(H, prec, torch.zeros(2, 2, 2, 2, dtype=torch.float64), b, 60)
    assert float((x - 1 / d).abs().max()) < 1e-10
    tf.E = 1e-3 * torch.ones(2, 2, 2, 2, 6, dtype=torch.float64); tf.prior_scale = None
    L = lambda X: X                                                           # unit-scale curvature: identity
    fE = torch.ones_like(tf.E, dtype=torch.bool)
    tf.evidence_update(torch.tensor([40.0, 10.0], dtype=torch.float64), L, fE)
    M = 8 * 6 - 6; energy = 8 * 6 * 1e-6
    want = torch.tensor([(M - 40.0) / energy, (M - 10.0) / energy]).clamp(max=EVIDENCE_MAX_STEP)
    assert torch.allclose(tf.prior_scale, want.to(tf.prior_scale.dtype))


def test_uncertainty_sampling_and_posterior():
    """The uncertainty probes give the diagonal of H^-1 F H^-1 (kind 'sampling') or of H^-1 (kind 'posterior'); with
    a diagonal H = F + P one probe gives either exactly."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit

    tf = object.__new__(TomoFit)
    tf.R, tf.C = 2, 6
    tf.V = torch.ones(2, 2, 2, 2, dtype=torch.float64)
    tf.E = torch.zeros(2, 2, 2, 2, 6, dtype=torch.float64)
    tf.prior_scale = None; tf.voxel_scale = 1.0; tf.lam_E = 0.0
    fd = torch.linspace(1.0, 5.0, 2 * 6 * 8, dtype=torch.float64).reshape(tf.E.shape)    # the data's curvature
    pd = torch.linspace(3.0, 0.5, 2 * 6 * 8, dtype=torch.float64).reshape(tf.E.shape)    # the prior's
    H = lambda xV, xE: (xV, (fd + pd) * xE)
    H_data = lambda xV, xE: (torch.zeros_like(xV), fd * xE)
    prec = lambda rV, rE: (rV, rE)
    fE = torch.ones_like(tf.E, dtype=torch.bool)
    for kind, want in (('sampling', fd / (fd + pd) ** 2), ('posterior', 1 / (fd + pd))):
        tf.request_uncertainty(1, kind)
        tf._run_probes(H, prec, lambda X: X, fE, None, False, H_data)
        assert torch.allclose(tf.strain_sd ** 2, want, rtol=1e-8)
        assert tf.last_probe_cost['solves'] == (2 if kind == 'sampling' else 1)
    # block averages of 2 x 2 x 2 voxels: with a constant diagonal in each block, the average's variance is diag / 8
    fd = fd.reshape(2 * 2 * 2 * 2, 6)[:2].repeat(8, 1).reshape(tf.E.shape)                  # one value per block
    pd = torch.full_like(fd, 0.5)
    H = lambda xV, xE: (xV, (fd + pd) * xE)
    tf.request_uncertainty(1, 'sampling', block=2)
    tf._run_probes(H, prec, lambda X: X, fE, None, False, lambda xV, xE: (torch.zeros_like(xV), fd * xE))
    want = (fd / (fd + pd) ** 2)[:1, :1, :1] / 8
    assert tf.strain_sd.shape == (1, 1, 1, 2, 6) and torch.allclose(tf.strain_sd ** 2, want, rtol=1e-8)


@pytest.mark.parametrize("order", [1, 2])
def test_fft_preconditioner_symbol(order):
    """The strain priors' Fourier symbol (TomoFit._prior_symbol) reproduces their Hessian on a field away from the faces,
    for the first- and second-order smoothness priors."""
    from mbirtorch.hsnt.strain.tomo import TomoFit
    n = 12
    tf = object.__new__(TomoFit)
    tf.R, tf.C = 1, 6
    tf.V = torch.ones(n, n, n, 1, dtype=torch.float64)
    tf.elastic = [(211e9, 0.29)]
    tf.lam_E, tf.smooth_E, tf.eq_w, tf.smooth_order = 0.3, 2.0, 5.0, order
    g = torch.Generator().manual_seed(0)
    X = torch.zeros(n, n, n, 1, 6, dtype=torch.float64)
    X[4:8, 4:8, 4:8] = torch.randn(4, 4, 4, 1, 6, generator=g, dtype=torch.float64)
    full = torch.ones(n, n, n, 1, dtype=torch.bool)
    div = lambda Y: tf._divergence(Y, full).reshape(*Y.shape[:3], -1)
    direct = 2 * tf.lam_E * X + tf.smooth_E * tf._smooth_apply(X) + tf.eq_w * torch.func.vjp(div, X)[1](div(X))[0]
    sym = tf._prior_symbol()
    via = torch.fft.ifftn(torch.einsum('xyzrab,xyzrb->xyzra', sym, torch.fft.fftn(X, dim=(0, 1, 2))), dim=(0, 1, 2)).real
    assert torch.allclose(via, direct, atol=1e-10 * float(direct.abs().max()))


# The largest relative difference allowed between the fft_compact output and the default 'fft' output in double
# precision.  The two paths round differently.  The equilibrium prior's symbol entries are about eq_w = 1e10, so their
# rounding is about 1e-6.  The softest blocks are about 3e4 (at zero frequency, where the ridge, the data and the
# damping are all that is left).  The outputs therefore differ by about eq_w * eps / 3e4, which is about 1e-10.  This
# bound leaves a margin of about 100, and a real error is far larger.
FFT_COMPACT_REL_TOL = 1e-8


class _RayProjector:
    """A test projector whose back is the exact adjoint of fwd: each of P rays sums RAY_VOXELS random voxels with random
    positive weights, in each of Q channels."""
    RAY_VOXELS = 40

    def __init__(self, n, P, Q, gen, dtype):
        self.n, self.Q = n, Q
        self.idx = torch.randint(0, n ** 3, (P, self.RAY_VOXELS), generator=gen)
        self.wt = torch.rand(P, self.RAY_VOXELS, generator=gen, dtype=dtype)

    def fwd(self, vols):
        return (vols.reshape(-1, vols.shape[-1])[self.idx] * self.wt[..., None]).sum(1)

    def back(self, rays):
        out = torch.zeros(self.n ** 3, rays.shape[-1], dtype=rays.dtype)
        out.index_add_(0, self.idx.reshape(-1), (self.wt[..., None] * rays[:, None, :]).reshape(-1, rays.shape[-1]))
        return out.reshape(self.n, self.n, self.n, -1)


def _fft_fit(n, R, stencil='corner', order=2, elastic=True, dtype=torch.float64, seed=0, proj=None, C=6):
    """A TomoFit with only what the 'fft' preconditioner reads: a cube of material inside an empty volume, the package's
    prior strengths, and random ray information and view weights.  proj None gives a _RayProjector.  Returns the fit and
    the preconditioner's arguments (F, lam_, pV, dE_, fV, fE), on the projector's device.  C is the number of strain
    components, 6 for the tensor and 1 for one strain."""
    from mbirtorch.hsnt.strain.tomo import TomoFit, EQUILIBRIUM_SD, STRAIN_SMOOTH2_SD, STRAIN_PRIOR_SD
    gen = torch.Generator().manual_seed(seed)
    tf = object.__new__(TomoFit)
    tf.R, tf.C, tf.precond = R, C, 'fft'
    tf.proj = proj if proj is not None else _RayProjector(n, 4 * n * n, R * (1 + C), gen, dtype)
    rays = tf.proj.fwd(torch.zeros(n, n, n, tf.proj.Q, dtype=dtype))
    P, dev = rays.shape[0], rays.device                           # a package projector returns its rays on its device
    V = torch.zeros(n, n, n, R, dtype=dtype)
    a, b = n // 4, 3 * n // 4
    V[a:b, a:b, a:b] = 0.5 + torch.rand(b - a, b - a, b - a, R, generator=gen, dtype=dtype)
    tf.V = V.to(dev)
    tf.lam_E, tf.smooth_E, tf.smooth_order = 0.5 / STRAIN_PRIOR_SD ** 2, 1.0 / STRAIN_SMOOTH2_SD ** 2, order
    tf.elastic = [(200e9, 0.29), (70e9, 0.33)][:R] if elastic else None
    tf.eq_w, tf.eq_stencil = 1.0 / EQUILIBRIUM_SD ** 2, stencil
    tf.prior_scale = torch.tensor([1.0, 0.4][:R], dtype=dtype, device=dev)
    tf.mask_method, tf.mask_freeze, tf._mask = 'median', True, None
    tf.w_ray = torch.randn(P, C, generator=gen, dtype=dtype).to(dev)
    F = (1e4 * torch.rand(P, 2 * R, 2 * R, generator=gen, dtype=dtype)).to(dev)
    dE_ = (1e6 * (1 + torch.rand(n, n, n, R, C, generator=gen, dtype=dtype))).to(dev)
    pV = torch.ones(n, n, n, R, dtype=dtype, device=dev)
    fV, fE = torch.ones_like(pV, dtype=torch.bool), torch.ones(n, n, n, R, C, dtype=torch.bool, device=dev)
    return tf, (F, 1e-3, pV, dE_, fV, fE)


def _both_preconditioners(tf, args):
    """The 'fft' preconditioner without and with fft_compact."""
    tf.fft_compact = False
    old = tf._fft_preconditioner(*args)
    tf.fft_compact = True
    return old, tf._fft_preconditioner(*args)


@pytest.mark.parametrize("n,R", [(15, 1), (16, 2), (24, 1)])
@pytest.mark.parametrize("stencil", ["corner", "central"])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("elastic", [True, False])
def test_fft_compact_matches_full(n, R, stencil, order, elastic):
    """In double precision, fft_compact gives the 'fft' preconditioner's output to rounding, and it stores the priors'
    symbol as real numbers.  The test covers odd and even sides, one and two materials, both equilibrium stencils,
    both smoothness orders, and fits with and without the equilibrium prior."""
    tf, args = _fft_fit(n, R, stencil, order, elastic)
    old, new = _both_preconditioners(tf, args)
    g = torch.Generator().manual_seed(1)
    rV = torch.randn(n, n, n, R, generator=g, dtype=torch.float64)
    rE = torch.randn(n, n, n, R, 6, generator=g, dtype=torch.float64)
    (zV0, zE0), (zV1, zE1) = old(rV, rE), new(rV, rE)
    assert torch.equal(zV0, zV1)
    rel = float((zE1 - zE0).norm() / zE0.norm())
    assert rel < FFT_COMPACT_REL_TOL, rel
    assert tf.fft_compact_storage == 'real', tf.symbol_imag_rel


SCALAR_TEST_N = 16               # test_fft_compact_scalar_strain's volume has this side.
SCALAR_TEST_SEED = 3             # test_fft_compact_scalar_strain draws its random inputs from this seed.
SCALAR_STRAIN_COMPONENTS = 1     # A scalar strain has one component per voxel.


@pytest.mark.parametrize("R", [1, 2])
def test_fft_compact_scalar_strain(R):
    """With one strain per voxel (C = 1) and no equilibrium prior, fft_compact gives the 'fft' preconditioner's output
    to the double-precision tolerance.  fft_compact is on by default, so it must also hold for a scalar strain fit."""
    n, C = SCALAR_TEST_N, SCALAR_STRAIN_COMPONENTS
    tf, args = _fft_fit(n, R, elastic=False, C=C)
    old, new = _both_preconditioners(tf, args)
    g = torch.Generator().manual_seed(SCALAR_TEST_SEED)
    rV = torch.randn(n, n, n, R, generator=g, dtype=torch.float64)
    rE = torch.randn(n, n, n, R, C, generator=g, dtype=torch.float64)
    (zV0, zE0), (zV1, zE1) = old(rV, rE), new(rV, rE)
    assert torch.equal(zV0, zV1)
    rel = float((zE1 - zE0).norm() / zE0.norm())
    assert rel < FFT_COMPACT_REL_TOL, rel


def test_fft_compact_cg_iterates():
    """Conjugate gradients give the same iterates and residuals with the compact preconditioner as with the default
    one.  The residual's tolerance is looser, because a small residual carries the rounding of the first one."""
    n, R = 16, 2
    tf, args = _fft_fit(n, R)
    old, new = _both_preconditioners(tf, args)
    full = torch.ones(n, n, n, R, dtype=torch.bool)
    div = lambda Y: tf._divergence(Y, full).reshape(*Y.shape[:3], -1)
    dE_ = args[3]
    H = lambda xV, xE: (xV, 2 * tf.lam_E * xE + tf.smooth_E * tf._smooth_apply(xE)
                        + tf.eq_w * torch.func.vjp(div, xE)[1](div(xE))[0] + 1e-3 * dE_ * xE)
    g = torch.Generator().manual_seed(2)
    bV = torch.zeros(n, n, n, R, dtype=torch.float64)
    bE = torch.randn(n, n, n, R, 6, generator=g, dtype=torch.float64)
    for iters in (1, 5, 15):
        _, x0 = tf._cg_solve(H, old, bV, bE, iters); r0 = tf.last_cg_residual
        _, x1 = tf._cg_solve(H, new, bV, bE, iters); r1 = tf.last_cg_residual
        rel = float((x1 - x0).norm() / x0.norm())
        assert rel < 1e-7 and abs(r1 - r0) <= 1e-4 * r0 + 1e-9, (iters, rel, r0, r1)


@pytest.mark.parametrize("storage", ["chosen", "complex"])
def test_fft_compact_float32(storage, monkeypatch):
    """With single-precision volumes and rays, as in a fit, fft_compact gives the 'fft' preconditioner's output to
    single-precision rounding.  This holds with the storage that the symbol selects and with complex storage forced."""
    from mbirtorch.hsnt.strain import tomo
    if storage == 'complex':
        monkeypatch.setattr(tomo, 'SYMBOL_IMAG_TOL', -1.0)
    n, R = 16, 2
    tf, args = _fft_fit(n, R, dtype=torch.float32)
    old, new = _both_preconditioners(tf, args)
    assert storage == 'chosen' or tf.fft_compact_storage == 'complex'
    g = torch.Generator().manual_seed(3)
    rV, rE = torch.randn(n, n, n, R, generator=g), torch.randn(n, n, n, R, 6, generator=g)
    z0, z1 = old(rV, rE)[1], new(rV, rE)[1]
    assert z1.dtype == torch.float32
    rel = float((z1 - z0).norm() / z0.norm())
    assert rel < 1e-5, (rel, tf.fft_compact_storage, tf.symbol_imag_rel)


def test_fft_compact_symbol_storage(monkeypatch):
    """The half-spectrum symbol is the full symbol's half: its real part when it is stored as real numbers, and all of
    it when complex storage is forced.  With complex storage, fft_compact still gives the 'fft' preconditioner's output
    to rounding."""
    from mbirtorch.hsnt.strain import tomo
    n, R = 12, 2
    tf, args = _fft_fit(n, R)
    full = tf._prior_symbol()[:, :, :n // 2 + 1]
    tol = 1e-14 * float(full.abs().max())
    half = tf._prior_symbol_half()
    assert tf.fft_compact_storage == 'real' and torch.allclose(half, full.real, rtol=0, atol=tol)
    monkeypatch.setattr(tomo, 'SYMBOL_IMAG_TOL', -1.0)
    half = tf._prior_symbol_half()
    assert tf.fft_compact_storage == 'complex' and torch.allclose(half, full, rtol=0, atol=tol)
    old, new = _both_preconditioners(tf, args)
    g = torch.Generator().manual_seed(5)
    rV = torch.randn(n, n, n, R, generator=g, dtype=torch.float64)
    rE = torch.randn(n, n, n, R, 6, generator=g, dtype=torch.float64)
    z0, z1 = old(rV, rE)[1], new(rV, rE)[1]
    rel = float((z1 - z0).norm() / z0.norm())
    assert rel < FFT_COMPACT_REL_TOL, rel


def test_fft_compact_empty_material():
    """A material without voxels gets no data block and no damping from fft_compact, as from the default 'fft'
    preconditioner.  Here the second of two materials is empty, and the two outputs agree to rounding.  The test leaves
    out the equilibrium prior.  With it, the empty material's softest block would be only its ridge, and the rounding
    would come close to FFT_COMPACT_REL_TOL."""
    n, R = 16, 2
    tf, args = _fft_fit(n, R, elastic=False)
    tf.V[..., 1] = 0
    assert not bool(tf.material_voxels()[..., 1].any())
    old, new = _both_preconditioners(tf, args)
    g = torch.Generator().manual_seed(6)
    rV = torch.randn(n, n, n, R, generator=g, dtype=torch.float64)
    rE = torch.randn(n, n, n, R, 6, generator=g, dtype=torch.float64)
    z0, z1 = old(rV, rE)[1], new(rV, rE)[1]
    assert bool(torch.isfinite(z1).all())
    rel = float((z1 - z0).norm() / z0.norm())
    assert rel < FFT_COMPACT_REL_TOL, rel


def test_fft_compact_leaves_fft_ray():
    """fft_compact applies to precond='fft' only.  With precond='fft_ray' the default path runs, and its output is the
    same with the option on and off."""
    n, R = 12, 1
    tf, args = _fft_fit(n, R)
    tf.precond = 'fft_ray'
    g = torch.Generator().manual_seed(7)
    ray = torch.randn(n, n, n, R, 6, 6, generator=g, dtype=torch.complex128)
    tf._ray_symbol = lambda F: ray                       # a stand-in for the data's strain symbol

    def compact(*a, **k):
        raise AssertionError("the compact preconditioner ran with precond='fft_ray'")
    tf._fft_preconditioner_compact = compact
    old, new = _both_preconditioners(tf, args)
    rV = torch.randn(n, n, n, R, generator=g, dtype=torch.float64)
    rE = torch.randn(n, n, n, R, 6, generator=g, dtype=torch.float64)
    assert torch.equal(old(rV, rE)[1], new(rV, rE)[1])


def _old_data_block(tf, F, own):
    """The data strain block as _fft_preconditioner computes it, from C back-projections: (R, C, C) float64."""
    R, C, V, w = tf.R, tf.C, tf.V, tf.w_ray
    n, P_ = V.shape[0], F.shape[0]
    dUU = torch.diagonal(F[:, R:, R:], dim1=1, dim2=2)
    zero = torch.zeros(P_, R, device=V.device, dtype=V.dtype)
    bUU = torch.stack([tf.proj.back(torch.cat([zero, (dUU[:, :, None] * (w[:, k:k + 1] * w)[:, None, :]).reshape(P_, R * C)], 1))
                       [..., R:].reshape(n, n, n, R, C) for k in range(C)], -2)
    return torch.stack([(V[..., r, None, None] ** 2 * bUU[..., r, :, :])[own[..., r]].double().mean(0)
                        for r in range(R)])


@pytest.mark.parametrize("fine", [False, True])
def test_fft_compact_data_block_projectors(fine):
    """The data strain block from one forward projection (the adjoint identity) matches the mean of the
    back-projections to single-precision rounding.  The test uses a MultiProjector with mounts (fractional shifts and
    a rotation) and a FineProjector."""
    from mbirtorch.hsnt.strain.tomo import FineProjector
    R, C = 1, 6
    if fine:
        proj = FineProjector(N // 2, MOUNTS[:2], R * (1 + C), channel_offsets=[0.0, 0.3], ror_mask=False, factor=2)
    else:
        proj = MultiProjector(N, MOUNTS, R * (1 + C), channel_offsets=[0.0, 0.3, -0.2], ror_mask=False)
    tf, (F, *_) = _fft_fit(N, R, dtype=torch.float32, proj=proj)
    own = tf.material_voxels()
    new, old = tf._data_strain_block(F, own), _old_data_block(tf, F, own)
    rel = float((new - old).abs().max() / old.abs().max())
    assert rel < 1e-5, rel


def _volume_step_fit(seed=0, one_model_views=True):
    """A TomoFit for one volume step on a MultiProjector with mounts: one bcc material in a cube, flat data, random
    strain tensors, random view weights and the equilibrium prior. one_model_views is passed to the MultiProjector."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    from mbirtorch.hsnt.strain.tomo import TomoFit
    C = 6
    pr = MultiProjector(N, MOUNTS, 2, channel_offsets=[0.0, 0.3, -0.2], ror_mask=False,
                        one_model_views=one_model_views)
    dev = pr.fwd(torch.zeros(N, N, N, 2)).device                 # the device the projector runs on
    lam = np.linspace(2.0, 4.4, 60)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=3, n_contam=1,
                device=dev)
    g = torch.zeros(M.n_global, dtype=torch.float64, device=dev)
    sl = M.slices[0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3; g[sl['c'][0]] = 0.2
    P = sum(pr.counts)
    kw = dict(device=dev)
    fit = FitB(M, torch.ones(P, M.K, **kw), g, torch.ones(P, 1, **kw), torch.zeros(1, M.K, **kw),
               eps=torch.zeros(P, 1, **kw), chunk=512)
    gen = torch.Generator().manual_seed(seed)
    V = torch.zeros(N, N, N, 1); V[3:9, 3:9, 3:9] = 0.05
    E = 1e-4 * torch.randn(N, N, N, 1, C, generator=gen)
    w = 2 * torch.rand(pr.V, C, generator=gen) - 1
    return TomoFit(fit, pr, V.to(dev), E.to(dev), weights=w.to(dev), elastic=[(211e9, 0.29)])


def test_volume_step_fft_compact():
    """A volume step with fft_compact runs on the package projector, and it agrees with the default step: the Newton
    decrement and the objective's decrease match to the precision of single-precision conjugate gradients."""
    a, b = _volume_step_fit(), _volume_step_fit()
    a.fft_compact = False
    b.fft_compact = True
    ua, da = a.volume_step()
    ub, db = b.volume_step()
    assert b.fft_compact_storage in ('real', 'complex') and math.isfinite(ub) and db > 0, (ub, db)
    assert abs(ub - ua) <= 5e-2 * abs(ua) and abs(db - da) <= 5e-2 * abs(da), (ua, ub, da, db)


def _speedups_off(obj):
    """Sets every speedup option that is on by default to off on a FitB, or on a TomoFit and its fit, and returns obj.
    A reference arm that must take the old default path uses it."""
    from mbirtorch.hsnt.strain.model import FitB
    fit = obj if isinstance(obj, FitB) else obj.fit
    if fit is not obj:
        obj.fft_compact = False
        obj.spread_grouped = False
    fit.lean_syncs = False
    fit.closed_form_strain = False
    fit.lean_passes = False
    fit.fisher_rays = 0
    fit.drop_kept()
    return obj


def _small_fit(seed=0, shape_model=None):
    """A small FitB on the CPU: one bcc material, per-ray dilations, one contamination component, counts from the model.
    shape_model is passed to Hybrid2. The speedup options are off (_speedups_off)."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    lam = np.linspace(2.0, 4.4, 300)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=5, n_contam=1,
                device='cpu', dtype=torch.float64, shape_model=shape_model)
    g = torch.zeros(M.n_global, dtype=torch.float64)
    sl = M.slices[0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3; g[sl['c'][0]] = 0.2
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor([0.0, 0.001, 0.002, 0.0, 0.1, 0.02, 0.0, 0.0], dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    P = 40
    W = 0.5 + torch.rand(P, 1, generator=gen, dtype=torch.float64)
    eps = 1e-3 * torch.randn(P, 1, generator=gen, dtype=torch.float64)
    f = torch.zeros(1, M.K, dtype=torch.float64); f[0, 100:140] = 0.05
    fit = _speedups_off(FitB(M, torch.ones(P, M.K, dtype=torch.float64), g, W, f, eps=eps, chunk=16))
    with torch.no_grad():
        m = torch.cat([fit.mean(s)[0] for s in fit.chunks()])
    fit.T = torch.poisson(50.0 * m, generator=gen) / 50.0
    return fit


def test_global_grad_matches_global_system():
    """FitB.global_grad (automatic differentiation) equals global_system's gradient for the parameters other than the
    resolution kernel."""
    fit = _small_fit()
    M = fit.M
    idx = [i for i in range(M.n_global) if not (M.res_sl[0] <= i < M.res_sl[1]) and float(M.ub[i]) > float(M.lb[i])]
    g1, _ = fit.global_system(idx)
    g2 = fit.global_grad(idx)
    assert torch.allclose(g1, g2, rtol=1e-6, atol=1e-8 * float(g1.abs().max()))


def test_kernel_moves_on_fisher_steps():
    """With fisher_every > 1 and a free resolution kernel, the kernel moves on every global step while its moves are
    large, and afterwards only on the steps that recompute the Fisher information; the loss never rises."""
    from mbirtorch.hsnt.strain import model as mdl
    fit = _small_fit()
    M = fit.M
    fit.fisher_rays = 0                   # the kernel's lag applies to the Fisher information from all rays
    fit.fisher_every = 3
    res = list(range(M.res_sl[0], M.res_sl[1]))
    loss = [fit.loss()]
    moved, recomputed = [], []
    for k in range(9):
        g0 = fit.g.clone()
        # the first three steps as if the kernel were still moving, the rest as if it had settled
        fit.last_kernel_step_se = (2.0 if k < 3 else 0.5) * mdl.KERNEL_LAG_SE
        l, _ = fit.global_step()
        loss.append(float(l))
        moved.append(bool((fit.g[res] != g0[res]).any()))
        recomputed.append(fit._fisher[2] == 1)
    assert all(b <= a + 1e-9 * abs(a) for a, b in zip(loss, loss[1:]))
    for k in range(3, 9):
        if not recomputed[k]:
            assert not moved[k]                                          # the kernel is held between Fisher steps
    assert recomputed[:3] == [True, True, True]                         # a moving kernel recomputes every step
    assert not all(recomputed[3:])


GRAD_RTOL = 1e-6                 # global_grad against global_system's gradient, relative
CUBIC4_GRAD_RTOL = 1e-5          # the same with the cubic4 model, as in test_cubic4_global_grad
GRAD_ATOL_SHARE = 1e-8           # the absolute tolerance of those checks, as a share of the largest gradient entry
STRATA_RTOL = 1e-10              # test_fisher_rays_strata_counts: the subset against the full Fisher information
FISHER_SUBSET_SHARE = 4          # test_fisher_rays_subset_accuracy and the global steps sample one ray in this many.
# test_fisher_rays_subset_accuracy: the largest entry of the subset's minus the full Fisher information, in units of the
# square root of the product of the full one's two diagonal entries
FISHER_SUBSET_TOL = 0.1
FISHER_TEST_VIEWS = 8            # The fisher_rays tests' fits have this many views.
KERNEL_START_FACTOR = 1.2        # test_fisher_rays_global_steps starts the kernel's widths this many times too wide.
GLOBAL_TEST_STEPS = 6            # test_fisher_rays_global_steps takes this many global steps,
GLOBAL_TEST_EVERY = 2            # and it recomputes the Fisher information on every this many steps.
SETTLED_KERNEL_SE = 0.5          # A kernel's last move of this many times KERNEL_LAG_SE counts as settled.
STEP_LOSS_RTOL = 1e-9            # A global step may raise the loss by at most this share (rounding).
ACCURACY_TEST_RAYS = 1200        # test_fisher_rays_subset_accuracy's fit has this many rays.
AUTO_TEST_RAYS_PER_PARAMETER = 2  # test_fisher_rays_auto sets FISHER_RAYS_PER_PARAMETER to this value.


def _free_globals(fit):
    """Returns the indices of the fit's free global parameters, the resolution kernel's slots included."""
    M = fit.M
    return [i for i in range(M.n_global) if float(M.ub[i]) > float(M.lb[i])]


def _fisher_fit(groups, repeat=1, masked_groups=0, seed=5):
    """Returns a small FitB with the cubic4 model for the fisher_rays checks. It has one bcc material, FISHER_TEST_VIEWS
    views and one contamination component. Its rays are stored view by view, and within a view their maps and strains
    vary smoothly, as along a detector's rows. Each of the groups distinct rays is stored repeat times in a row. A ray
    mask leaves out the first masked_groups groups. The counts come from the model."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB, harmonics4
    lam = np.linspace(2.0, 4.4, 300)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=5, n_contam=1,
                device='cpu', dtype=torch.float64, shape_model='cubic4')
    g = torch.zeros(M.n_global, dtype=torch.float64)
    sl = M.slices[0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3; g[sl['c'][0]] = 0.2
    axis = torch.tensor([np.sin(0.9) * np.cos(0.4), np.sin(0.9) * np.sin(0.4), np.cos(0.9)], dtype=torch.float64)
    g[sl['x'][0]:sl['x'][1]] = 0.05 * harmonics4(axis)
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor([0.0005, 0.001, 0.002, 0.0005, 0.1, 0.02, 0.0, 0.0], dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    d = torch.randn(FISHER_TEST_VIEWS, 3, generator=gen, dtype=torch.float64)
    M.view_dirs = (d / d.norm(dim=1, keepdim=True)).numpy()
    per = groups // FISHER_TEST_VIEWS
    view = torch.arange(groups) // per
    phase = 2 * np.pi * torch.rand(FISHER_TEST_VIEWS, generator=gen, dtype=torch.float64)[view]
    u = 2 * np.pi * (torch.arange(groups) % per).double() / per + phase          # each ray's place along its view
    W = (1.0 + 0.5 * torch.sin(u))[:, None]
    eps = (1e-3 * torch.cos(2 * u))[:, None]
    f = torch.zeros(1, M.K, dtype=torch.float64); f[0, 100:140] = 0.05
    mask = torch.ones(groups * repeat, dtype=torch.bool); mask[:masked_groups * repeat] = False
    fit = _speedups_off(FitB(M, torch.ones(groups * repeat, M.K, dtype=torch.float64), g,
                             W.repeat_interleave(repeat, 0), f, eps=eps.repeat_interleave(repeat, 0), chunk=16,
                             ray_mask=mask if masked_groups else None))
    fit.ray_view = view.repeat_interleave(repeat)
    with torch.no_grad():
        m = torch.cat([fit.mean(s)[0] for s in fit.chunks()])
    fit.T = torch.poisson(50.0 * m, generator=gen) / 50.0
    return fit


@pytest.mark.parametrize('lean', [False, True])
def test_global_grad_with_kernel(lean):
    """FitB.global_grad with the resolution kernel's slots equals global_system's gradient: to GRAD_RTOL for a powder,
    and to CUBIC4_GRAD_RTOL with the cubic4 model. This also holds with lean_syncs."""
    for fit, rtol in ((_small_fit(), GRAD_RTOL), (_fisher_fit(40), CUBIC4_GRAD_RTOL)):
        fit.lean_syncs = lean
        M = fit.M
        idx = _free_globals(fit)
        assert any(M.res_sl[0] <= i < M.res_sl[1] for i in idx)
        g1, _ = fit.global_system(idx)
        g2 = fit.global_grad(idx)
        assert torch.allclose(g1, g2, rtol=rtol, atol=GRAD_ATOL_SHARE * float(g1.abs().max())), (g1, g2)


def test_fisher_rays_all_rays_exact():
    """With fisher_rays at the number of rays, subset_fisher gives global_system's Fisher information exactly."""
    fit = _fisher_fit(80)
    idx = _free_globals(fit)
    _, F = fit.global_system(idx)
    fit.fisher_rays = fit.P
    assert torch.equal(fit.subset_fisher(idx), F)
    assert fit.fisher_subset_info == dict(rays=fit.P, asked=fit.P, kept=fit.P)


@pytest.mark.parametrize('lean', [False, True])
def test_fisher_rays_strata_counts(lean):
    """When the rays come in runs of identical rays and each stratum is one run, the subset's Fisher information equals
    the full one to rounding, because each sampled ray counts for its stratum. This checks the sampled rays' Jacobians
    and their counts, also with lean_syncs and with a ray mask that leaves out whole runs."""
    groups, repeat, masked = 64, 4, 8
    fit = _fisher_fit(groups, repeat, masked)
    fit.lean_syncs = lean
    idx = _free_globals(fit)
    _, F = fit.global_system(idx)
    fit.fisher_rays = groups - masked
    Fs = fit.subset_fisher(idx)
    assert fit.fisher_subset_info == dict(rays=groups - masked, asked=groups - masked, kept=(groups - masked) * repeat)
    assert torch.allclose(Fs, F, rtol=STRATA_RTOL, atol=STRATA_RTOL * float(torch.diagonal(F).abs().max()))


def test_fisher_rays_subset_accuracy():
    """With one ray in FISHER_SUBSET_SHARE, every entry of the subset's Fisher information is within FISHER_SUBSET_TOL
    of the full one's, in units of the square root of the product of the full one's two diagonal entries. The rays'
    maps and strains vary smoothly within each view, as on a detector."""
    from mbirtorch.hsnt.strain.model import INFO_FLOOR
    fit = _fisher_fit(ACCURACY_TEST_RAYS)
    idx = _free_globals(fit)
    _, F = fit.global_system(idx)
    fit.fisher_rays = fit.P // FISHER_SUBSET_SHARE
    Fs = fit.subset_fisher(idx)
    assert fit.fisher_subset_info['rays'] == fit.P // FISHER_SUBSET_SHARE
    d = torch.diagonal(F)
    live = d > INFO_FLOOR * d.max()
    s = d[live].rsqrt()
    err = float((s[:, None] * (Fs - F)[live][:, live] * s[None, :]).abs().max())
    assert err < FISHER_SUBSET_TOL, err


def test_fisher_rays_global_steps():
    """With fisher_rays and fisher_every GLOBAL_TEST_EVERY, the global steps never raise the loss, and they recompute
    the Fisher information on every GLOBAL_TEST_EVERY-th step. The resolution kernel also moves on the steps in
    between, because the gradient covers it."""
    from mbirtorch.hsnt.strain import model as mdl
    fit = _fisher_fit(200)
    M = fit.M
    res = list(range(*M.res_sl))
    g = fit.g.clone()
    g[M.res_sl[0] + 1] *= KERNEL_START_FACTOR; g[M.res_sl[0] + 2] *= KERNEL_START_FACTOR
    fit.g = g
    fit.fisher_rays = fit.P // FISHER_SUBSET_SHARE
    fit.fisher_every = GLOBAL_TEST_EVERY
    loss = [fit.loss()]
    moved, recomputed = [], []
    for _ in range(GLOBAL_TEST_STEPS):
        fit.last_kernel_step_se = SETTLED_KERNEL_SE * mdl.KERNEL_LAG_SE
        g0 = fit.g.clone()
        l, _ = fit.global_step()
        loss.append(float(l))
        moved.append(bool((fit.g[res] != g0[res]).any()))
        recomputed.append(fit._fisher[2] == 1)
    assert all(b <= a + STEP_LOSS_RTOL * abs(a) for a, b in zip(loss, loss[1:]))
    assert recomputed == [k % GLOBAL_TEST_EVERY == 0 for k in range(GLOBAL_TEST_STEPS)]
    assert any(mv for mv, rc in zip(moved, recomputed) if not rc)
    assert fit._fisher_fallback is None                    # the uniform sample judges every parameter well enough


def test_fisher_rays_auto(monkeypatch):
    """fisher_rays FISHER_RAYS_AUTO asks for FISHER_RAYS_PER_PARAMETER rays per free global parameter."""
    from mbirtorch.hsnt.strain import model as mdl
    monkeypatch.setattr(mdl, 'FISHER_RAYS_PER_PARAMETER', AUTO_TEST_RAYS_PER_PARAMETER)
    fit = _fisher_fit(200)
    idx = _free_globals(fit)
    fit.fisher_rays = mdl.FISHER_RAYS_AUTO
    fit.subset_fisher(idx)
    n = AUTO_TEST_RAYS_PER_PARAMETER * len(idx)
    assert n < fit.P and fit.fisher_subset_info == dict(rays=n, asked=n, kept=fit.P)


# ---- the automatic fallback of fisher_rays
FISHER_EXACT_RTOL = 1e-12        # The fallback's Fisher information equals global_system's to this relative tolerance.
RARE_TOTAL_RAYS = 4096           # test_fisher_rays_rare_material_falls_back's fit has this many rays.
RARE_RAYS = 10                   # Its second material crosses only this many consecutive rays.
RARE_START = 1000                # The second material's rays start at this ray.
RARE_STRATUM = 64                # Its ray subset samples one ray in this many.
RARE_BINS = 100                  # Its data have this many wavelength bins.
RARE_DOSE = 50.0                 # The data have this many counts per bin at a transmission of 1.
RARE_SEED = 31                   # The fit's random maps and counts come from this seed.
RARE_WAVELENGTHS = (2.0, 4.4)    # The data's wavelength bins span this range (angstrom).
RARE_BCC_A0 = 2.8665             # The bcc material has this lattice parameter (angstrom).
RARE_FCC_A0 = 3.524              # The fcc material has this lattice parameter (angstrom).
RARE_A_RANGE = 0.01              # Each lattice parameter may move by this fraction.
RARE_HEIGHT = 0.3                # Each material's edge heights start at this value.
RARE_SMOOTH = 0.2                # Each material's first smooth coefficient starts at this value.
RARE_SMOOTH_DEGREE = 3           # The smooth part of each spectrum has this Legendre degree.
RARE_BCC_MIN_FRACTION = 0.5      # The bcc material's map on each ray is this value plus a uniform draw from [0, 1).
# The new tests' resolution kernel has these eight parameters, in the order of Hybrid2.res_sl.
TEST_KERNEL = (0.0, 0.001, 0.002, 0.0, 0.1, 0.02, 0.0, 0.0)
UNIFORM_TEST_STEPS = 3           # test_fisher_rays_uniform_sample_no_fallback takes this many global steps.
SMALL_TEST_STEPS = 2             # test_fisher_rays_small_problem_default_path takes this many global steps.
# _check_fit holds these resolution-kernel slots (s0, s1 and lam_R, as offsets from Hybrid2.res_sl[0]), because the
# model's derivative with respect to each of them is zero at its lower bound 0.
ZERO_SLOPE_KERNEL_SLOTS = (0, 1, 4)
CHECK_FISHER_EVERY = 1           # The tests of the fisher_rays check recompute the Fisher information on every step.
# On global step k, test_fisher_rays_check_bound sets to zero the Fisher information's rows and columns of the edge
# heights in entry k, each given by its offset from the first edge height of material 0 (M.slices[0]['h'][0]).
CHECK_BOUND_SCHEDULE = ((), (0,), (1,), (0, 1), (0,), (), (1,))
CHECK_BOUND_FULL_STEPS = (0, 1, 2)   # The global steps of that test that compute the full Fisher information are these.
# On global step k, test_fisher_rays_rechecks_new_information sets to zero the Fisher information's rows and columns of
# the edge heights in entry k, each given by its offset from the first edge height of material 0.
NEW_INFO_SCHEDULE = ((0,), (), (0,), ())
NEW_INFO_FULL_STEPS = (0, 1)     # The global steps of that test that compute the full Fisher information are these.
# From this global step on, test_fisher_rays_late_fallback sets to zero the subset's Fisher information of the first
# edge height of material 0.
LATE_FALLBACK_STEP = 2


def _rare_material_fit():
    """Returns a FitB on the CPU in float64 with a bcc and an fcc material. The bcc material crosses all
    RARE_TOTAL_RAYS rays, and the fcc material crosses only RARE_RAYS consecutive rays from RARE_START. The counts come
    from the model. The speedup options are off (_speedups_off)."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    lam = np.linspace(*RARE_WAVELENGTHS, RARE_BINS)
    a0s = (RARE_BCC_A0, RARE_FCC_A0)
    M = Hybrid2(lam, np.full_like(lam, RARE_DOSE),
                [(lt, (a * (1 - RARE_A_RANGE), a * (1 + RARE_A_RANGE))) for lt, a in zip(('bcc', 'fcc'), a0s)],
                smooth_degree=RARE_SMOOTH_DEGREE, n_contam=1, device='cpu', dtype=torch.float64)
    g = torch.zeros(M.n_global, dtype=torch.float64)
    for r, a0 in enumerate(a0s):
        sl = M.slices[r]
        g[sl['a']] = a0; g[sl['h'][0]:sl['h'][1]] = RARE_HEIGHT; g[sl['c'][0]] = RARE_SMOOTH
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor(TEST_KERNEL, dtype=torch.float64)
    gen = torch.Generator().manual_seed(RARE_SEED)
    W = torch.zeros(RARE_TOTAL_RAYS, M.R, dtype=torch.float64)
    W[:, 0] = RARE_BCC_MIN_FRACTION + torch.rand(RARE_TOTAL_RAYS, generator=gen, dtype=torch.float64)
    W[RARE_START:RARE_START + RARE_RAYS, 1] = 1.0
    fit = _speedups_off(FitB(M, torch.ones(RARE_TOTAL_RAYS, M.K, dtype=torch.float64), g, W,
                             torch.zeros(1, M.K, dtype=torch.float64)))
    with torch.no_grad():
        m = torch.cat([fit.mean(s)[0] for s in fit.chunks()])
    fit.T = torch.poisson(RARE_DOSE * m, generator=gen) / RARE_DOSE
    return fit


def test_fisher_rays_rare_material_falls_back():
    """A material that crosses fewer rays than one stratum of the fisher_rays subset gets no sampled ray, or one that
    stands for far more rays than cross it. The first global step finds that the subset misjudges it, warns once, and
    takes its step with global_system's Fisher information. Every later global step uses all rays, without the
    subset."""
    fit = _rare_material_fit()
    fit.fisher_rays = RARE_TOTAL_RAYS // RARE_STRATUM
    idx = _free_globals(fit)
    _, F_ref = fit.global_system(idx)
    with pytest.warns(FallbackWarning, match='all .* rays'):
        fit.global_step()
    assert fit._fisher_fallback is not None
    F = fit._fisher[1]
    assert torch.allclose(F, F_ref, rtol=FISHER_EXACT_RTOL,
                          atol=FISHER_EXACT_RTOL * float(torch.diagonal(F_ref).abs().max()))
    calls = []
    subset = fit.subset_fisher
    fit.subset_fisher = lambda idx_: calls.append(1) or subset(idx_)
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)          # The fit warns only once.
        fit.global_step()
    assert calls == []


def _check_fit():
    """Returns the fit of the tests of the fisher_rays check: _fisher_fit with ACCURACY_TEST_RAYS rays, a ray subset of
    one ray in FISHER_SUBSET_SHARE, and the Fisher information recomputed on every global step (CHECK_FISHER_EVERY).
    The kernel slots ZERO_SLOPE_KERNEL_SLOTS are held at their start values, so that no parameter loses its information
    during the steps."""
    fit = _fisher_fit(ACCURACY_TEST_RAYS)
    M = fit.M
    for k in ZERO_SLOPE_KERNEL_SLOTS:                 # A parameter whose bounds are equal is held (global_step).
        i = M.res_sl[0] + k
        M.lb[i] = M.ub[i] = fit.g[i]
    fit.fisher_rays = fit.P // FISHER_SUBSET_SHARE
    fit.fisher_every = CHECK_FISHER_EVERY
    return fit


class _FisherSchedule:
    """_FisherSchedule wraps a fit's subset_fisher and global_system for the tests of the fisher_rays check. Each
    wrapper sets to zero the rows and columns of its Fisher information that belong to the global parameters in sub
    (subset_fisher) or in full (global_system). A test sets step, sub and full before each global step. The wrappers
    record the step of each of their calls in sub_steps and full_steps."""

    def __init__(self, fit):
        self.step, self.sub, self.full = None, (), ()
        self.sub_steps, self.full_steps = [], []
        subset_fisher, global_system = fit.subset_fisher, fit.global_system

        def subset_fisher_(idx):
            self.sub_steps.append(self.step)
            return self.zeroed(subset_fisher(idx), idx, self.sub)

        def global_system_(idx):
            self.full_steps.append(self.step)
            grad, F = global_system(idx)
            return grad, self.zeroed(F, idx, self.full)
        fit.subset_fisher, fit.global_system = subset_fisher_, global_system_

    @staticmethod
    def zeroed(F, idx, params):
        """Returns F, the Fisher information of the global parameters idx, with the rows and columns of the parameters
        params set to zero. F itself is not changed."""
        if not params:
            return F
        k = [list(idx).index(i) for i in params]
        F = F.clone()
        F[k, :] = 0.0
        F[:, k] = 0.0
        return F


def _scheduled_full_steps(schedule):
    """Takes one global step of a new _check_fit per entry of schedule. Each entry names edge heights by their offset
    from the first edge height of material 0 (M.slices[0]['h'][0]). Their rows and columns of the Fisher information
    are zero on that step. The subset's and the full Fisher information are changed alike. Returns the steps that
    computed the full Fisher information. It asserts that each step used the subset, that nothing was warned and that
    the fit did not switch to all rays."""
    fit = _check_fit()
    h0 = fit.M.slices[0]['h'][0]
    wrap = _FisherSchedule(fit)
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        for step, edges in enumerate(schedule):
            wrap.step = step
            wrap.sub = wrap.full = tuple(h0 + e for e in edges)
            fit.global_step()
    assert fit._fisher_fallback is None
    assert wrap.sub_steps == list(range(len(schedule))), wrap.sub_steps
    return tuple(wrap.full_steps)


def test_fisher_rays_uniform_sample_no_fallback():
    """When every parameter's information is spread over many rays, the fisher_rays subset passes the check: the first
    global step computes the full Fisher information once, and the later steps use the subset alone. Nothing is
    warned. The fit holds the kernel slots that lose their information at their lower bound (_check_fit), because such
    a parameter would ask for a new check."""
    fit = _check_fit()
    full, sub = [], []
    global_system, subset_fisher = fit.global_system, fit.subset_fisher
    fit.global_system = lambda idx_: full.append(1) or global_system(idx_)
    fit.subset_fisher = lambda idx_: sub.append(1) or subset_fisher(idx_)
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        for _ in range(UNIFORM_TEST_STEPS):
            fit.global_step()
    assert fit._fisher_fallback is None
    assert len(full) == 1 and len(sub) == UNIFORM_TEST_STEPS, (full, sub)


def test_fisher_rays_check_bound():
    """A due global step checks the fisher_rays subset against all rays only when the subset informs or freezes a
    parameter in a way that no earlier check confirmed. Two edge heights lose and regain their information on the
    schedule CHECK_BOUND_SCHEDULE. Only the first step and the first step that freezes each of them compute the full
    Fisher information, so the checks stay within one plus the number of free parameters."""
    assert _scheduled_full_steps(CHECK_BOUND_SCHEDULE) == CHECK_BOUND_FULL_STEPS


def test_fisher_rays_rechecks_new_information():
    """An edge height that the fisher_rays subset first freezes and then informs gets a full check both times, because
    a check of the frozen parameter does not confirm the subset's information on it. The later steps of
    NEW_INFO_SCHEDULE repeat both states and use the subset alone."""
    assert _scheduled_full_steps(NEW_INFO_SCHEDULE) == NEW_INFO_FULL_STEPS


def test_fisher_rays_late_fallback():
    """The fit switches to all rays when the fisher_rays subset misjudges a parameter after an earlier check passed.
    From step LATE_FALLBACK_STEP on, the subset's Fisher information of an edge height is zero, and the full one's is
    not. That step warns once, and it takes its step with global_system's Fisher information. The next step uses all
    rays without the subset."""
    fit = _check_fit()
    global_system = fit.global_system           # This method is not wrapped. Its call does not count as a check.
    wrap = _FisherSchedule(fit)
    h0 = fit.M.slices[0]['h'][0]
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        for step in range(LATE_FALLBACK_STEP):
            wrap.step = step
            fit.global_step()
    assert fit._fisher_fallback is None
    _, F_ref = global_system(_free_globals(fit))
    wrap.step, wrap.sub = LATE_FALLBACK_STEP, (h0,)
    with pytest.warns(FallbackWarning, match='all .* rays'):
        fit.global_step()
    assert fit._fisher_fallback == fit._fisher_sel[0]
    assert LATE_FALLBACK_STEP in wrap.full_steps, wrap.full_steps
    F = fit._fisher[1]
    assert torch.allclose(F, F_ref, rtol=FISHER_EXACT_RTOL,
                          atol=FISHER_EXACT_RTOL * float(torch.diagonal(F_ref).abs().max()))
    wrap.step = LATE_FALLBACK_STEP + 1
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)          # The fit warns only once.
        fit.global_step()
    assert wrap.step not in wrap.sub_steps, wrap.sub_steps


def test_fisher_rays_small_problem_default_path():
    """When the fisher_rays subset would hold at least FISHER_RAYS_MAX_SHARE of the rays, the global step takes the
    default path with all rays: it never calls subset_fisher, it warns nothing, and its steps equal those with
    fisher_rays 0."""
    from mbirtorch.hsnt.strain.model import FISHER_RAYS_AUTO
    a, b = _small_fit(), _small_fit()
    a.fisher_rays, b.fisher_rays = FISHER_RAYS_AUTO, 0
    calls = []
    subset_fisher = a.subset_fisher
    a.subset_fisher = lambda idx_: calls.append(1) or subset_fisher(idx_)
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        for _ in range(SMALL_TEST_STEPS):
            a.global_step(); b.global_step()
    assert calls == [] and a._fisher_fallback is None
    assert torch.equal(a.g, b.g)


def test_contamination_newton_kkt():
    """The contamination Newton step's inner solution meets the optimality conditions of its model: zero gradient where
    a coefficient is positive, gradient >= 0 where it is held at 0."""
    fit = _small_fit()
    fit.build_atoms()
    M = fit.M; na = fit.A.shape[1]
    gen = torch.Generator().manual_seed(1)
    A = fit.A.double()
    Fx = (1.0 + torch.rand(M.K, 1, 1, generator=gen, dtype=torch.float64)) * 50.0
    gc = torch.randn(1, na, generator=gen, dtype=torch.float64) * 5.0
    fit.lam_c = torch.full((1, na), 1.0, dtype=torch.float64)
    c0 = torch.zeros(1, na, dtype=torch.float64)
    c = fit._contamination_newton(A, gc, Fx, c0)
    H = A.T @ (Fx[:, 0, 0, None] * A)
    grad = gc[0] + H @ (c[0] - c0[0]) + fit.lam_c[0]
    pos = c[0] > 1e-9
    scale = float(gc.abs().max())
    assert float(grad[pos].abs().max()) < 1e-2 * scale and float(grad[~pos].min()) > -1e-2 * scale


@pytest.mark.parametrize("method", ["median", "otsu"])
def test_material_mask_face_overshoot(method):
    """With density overshoot on a cube's faces, the 'median' and 'otsu' material masks hold the whole cube and give
    the equilibrium prior an interior, while the 'max' mask holds only the overshooting faces."""
    from mbirtorch.hsnt.strain.tomo import TomoFit
    n = 16
    V = torch.zeros(n, n, n, 1)
    V[3:13, 3:13, 3:13] = 1.0
    V[3, 3:13, 3:13] = V[12, 3:13, 3:13] = 2.5                                   # overshoot on two faces
    V += 0.02 * torch.rand(n, n, n, 1, generator=torch.Generator().manual_seed(0))
    cube = torch.zeros(n, n, n, 1, dtype=torch.bool); cube[3:13, 3:13, 3:13] = True
    tf = object.__new__(TomoFit)
    tf.V = V; tf.mask_freeze = False; tf._mask = None
    tf.mask_method = 'max'
    assert int(tf._eq_mask().sum()) == 0
    tf.mask_method = method
    m = tf.material_voxels()
    assert int((m & cube).sum()) >= 0.9 * int(cube.sum()) and int((m & ~cube).sum()) <= 0.1 * int(cube.sum())
    assert int(tf._eq_mask().sum()) >= 0.5 * 8 ** 3


def test_traction_aligned_faces():
    """traction_model='aligned' keeps the axis model's conditions on a cube's flat faces (all but its edges) and drops
    most of them on a sphere, whose staircase faces have oblique normals."""
    import torch
    from mbirtorch.hsnt.strain.tomo import TomoFit

    n = 24
    c = (n - 1) / 2
    y, x, z = torch.meshgrid(*[torch.arange(n, dtype=torch.float64) - c] * 3, indexing='ij')
    g = torch.Generator().manual_seed(5)
    E = 1e-3 * torch.randn(n, n, n, 1, 6, generator=g, dtype=torch.float64)
    kept = {}
    for name, occ in (('cube', (x.abs() < 7) & (y.abs() < 7) & (z.abs() < 7)), ('sphere', x ** 2 + y ** 2 + z ** 2 < 8.5 ** 2)):
        tf = object.__new__(TomoFit)
        tf.V = occ.to(torch.float64)[..., None]; tf._eq_inside = occ[..., None]; tf.elastic = [(211e9, 0.29)]
        counts = []
        for model in ('axis', 'aligned'):
            tf.traction_model = model
            counts.append(int((tf._traction(E) != 0).any(-1).sum()))
        kept[name] = counts[1] / counts[0]
    assert kept['cube'] > 0.7 and kept['sphere'] < 0.3, kept          # the cube loses its edge rows (diagonal normal)


def test_traction_faces_bonded_interface():
    """Two materials side by side: the faces they share are bonded and get no traction condition; their faces against
    empty space do."""
    from mbirtorch.hsnt.strain.tomo import TomoFit
    n = 8
    tf = object.__new__(TomoFit)
    tf.elastic = [(211e9, 0.29), (117e9, 0.34)]
    inside = torch.zeros(n, n, n, 2, dtype=torch.bool)
    inside[2:6, 2:4, 2:6, 0] = True                                                 # material 0: x in [2, 4)
    inside[2:6, 4:6, 2:6, 1] = True                                                 # material 1: x in [4, 6)
    tf._eq_inside = inside
    E = torch.zeros(n, n, n, 2, 6, dtype=torch.float64)
    E[..., 0, 0] = 1e-3 * inside[..., 0].double(); E[..., 1, 0] = 1e-3 * inside[..., 1].double()
    t = tf._traction(E)                                                             # (n, n, n, R, 3 normals, 3)
    assert float(t[3, 3, 3, 0, 0].abs().max()) == 0.0                               # material 0 at the interface (x = 3)
    assert float(t[3, 4, 3, 1, 0].abs().max()) == 0.0                               # material 1 at the interface (x = 4)
    assert float(t[3, 2, 3, 0, 0].abs().max()) > 0.0                                # material 0's free face (x = 2)
    assert float(t[3, 5, 3, 1, 0].abs().max()) > 0.0                                # material 1's free face (x = 5)


def test_equilibrium_corner_stencil():
    """The corner stencil sees a stress alternating along one axis (invisible to central differences), gives the exact
    derivative of a linear stress, and still misses a pattern alternating along two axes (an hourglass mode)."""
    from mbirtorch.hsnt.strain.tomo import TomoFit
    n = 8
    idx = torch.arange(n, dtype=torch.float64)
    full = torch.ones(n, n, n, 1, dtype=torch.bool)
    res = {}
    for stencil in ('central', 'corner'):
        tf = object.__new__(TomoFit); tf.elastic = [(1.0, 0.0)]; tf.eq_stencil = stencil   # nu = 0: sigma_xx = e_xx
        E = torch.zeros(n, n, n, 1, 6, dtype=torch.float64)
        E[..., 0, 0] = ((-1.0) ** idx)[None, :, None].expand(n, n, n)                      # e_xx alternating along x
        alt = tf._divergence(E, full)[1:-1, 1:-2, 1:-1, 0, 0].abs().max()
        E[..., 0, 0] = 0.01 * idx[None, :, None].expand(n, n, n)                            # linear along x
        lin = tf._divergence(E, full)[1:-1, 1:-2, 1:-1, 0, 0]
        E[..., 0, 0] = ((-1.0) ** (idx[:, None] + idx[None, :]))[..., None].expand(n, n, n)  # alternating along x and y
        hg = tf._divergence(E, full)[1:-1, 1:-2, 1:-1, 0, 0].abs().max()
        res[stencil] = (float(alt), lin, float(hg))
    assert res['central'][0] < 1e-12 and res['corner'][0] > 0.5                 # only the corner stencil sees it
    assert torch.allclose(res['corner'][1], torch.full_like(res['corner'][1], 0.01))
    assert res['corner'][2] < 1e-12                                              # the hourglass mode remains


def test_d0_step_recovers_a_dilatation_field():
    """A lattice strain that is a pure dilatation D I with no elastic strain: d0_step finds D (up to the constant the
    equilibrium prior cannot see without traction), and the elastic strain E - D I is then in equilibrium."""
    from mbirtorch.hsnt.strain.tomo import TomoFit, D0_SMOOTH_SD, D0_RIDGE_SD
    n = 14
    c = (n - 1) / 2; g = torch.arange(n, dtype=torch.float64) - c
    zz, yy, xx = torch.meshgrid(g, g, g, indexing='ij')
    D_true = 2e-4 * torch.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * 2.5 ** 2))[..., None]
    tf = object.__new__(TomoFit); tf.elastic = [(1.0, 0.3)]; tf.eq_stencil = 'corner'; tf.traction = False
    tf.eq_w = 1.0 / 1e-5 ** 2; tf.C = 6; tf.R = 1
    tf.V = torch.ones(n, n, n, 1, dtype=torch.float64)
    tf.E = torch.cat([D_true[..., None].expand(n, n, n, 1, 3), torch.zeros(n, n, n, 1, 3, dtype=torch.float64)], -1)
    inner = torch.zeros(n, n, n, 1, dtype=torch.bool); inner[1:-2, 1:-2, 1:-2] = True
    tf._eq_mask = lambda: inner
    tf.material_voxels = lambda: torch.ones(n, n, n, 1, dtype=torch.bool)
    tf.D = torch.zeros_like(tf.V); tf.d0_smooth_sd, tf.d0_ridge_sd = D0_SMOOTH_SD, D0_RIDGE_SD
    tf.d0_step(cg_iters=400)
    err = (tf.D - tf.D.mean()) - (D_true - D_true.mean())
    assert float(err.abs().max()) < 0.1 * float(D_true.max())
    res = tf._eq_residual(tf._elastic(tf.E), inner)
    assert float(res.abs().max()) < 0.05 * float(tf._eq_residual(tf.E, inner).abs().max())


def test_ray_spectrum_float32():
    """In single precision, with elastic anisotropy and the strain's spread (whose widths come in double precision), the
    no-gradient windowed spectrum runs and matches the dense sum."""
    from mbirtorch.hsnt.strain.model import Hybrid2
    lam = np.linspace(1.2, 4.4, 600)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=5, n_contam=1,
                device='cpu', dtype=torch.float32)
    gen = torch.Generator().manual_seed(6)
    g = torch.zeros(M.n_global, dtype=torch.float64); sl = M.slices[0]; ne = sl['h'][1] - sl['h'][0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3; g[sl['c'][0]] = 0.2
    M.set_dec(0, 0.8 + 0.4 * torch.rand(ne, generator=gen, dtype=torch.float64), 0.05 * torch.randn(ne, generator=gen, dtype=torch.float64))
    eps = (2e-3 * torch.randn(30, 2, generator=gen, dtype=torch.float64))
    sd = 3e-4 * torch.rand(30, 1, generator=gen, dtype=torch.float64)
    with torch.no_grad():
        a = M.spectra(g, eps, None, sd=sd)
        M.dense_spectra = True; b = M.spectra(g, eps, None, sd=sd); M.dense_spectra = False
    assert a.dtype == torch.float32
    assert torch.allclose(a, b, rtol=0, atol=1e-5 * float(b.abs().max()))


def test_fine_projector():
    """FineProjector: back is the adjoint of fwd, and a volume constant over each factor^3 block projects as the
    detector-resolution projector projects its block values (lengths in detector pixels)."""
    from mbirtorch.hsnt.strain.tomo import FineProjector
    n, f = 8, 2
    series = [(((0, 1, 2), (1, 1, 1), (0.0, 0.0, 0.0)), ANGLES), (((1, 2, 0), (1, 1, 1), (1.0, -2.0, 0.0)), ANGLES)]
    fp = FineProjector(n, series, 2, channel_offsets=[0.0, 0.0], ror_mask=False, factor=f)
    cp = MultiProjector(n, series, 2, channel_offsets=[0.0, 0.0], ror_mask=False)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(f * n, f * n, f * n, 2, generator=g); y = torch.rand(fp.V * n * n, 2, generator=g)
    a = float((fp.fwd(x).cpu() * y).sum()); b = float((x * fp.back(y).cpu()).sum())   # back returns on the projector's device
    assert abs(a - b) <= 1e-4 * abs(a)
    xc = torch.rand(n, n, n, 2, generator=g)
    xf = xc.repeat_interleave(f, 0).repeat_interleave(f, 1).repeat_interleave(f, 2)
    pf, pc = fp.fwd(xf), cp.fwd(xc)
    assert float((pf - pc).abs().max()) < 0.05 * float(pc.abs().max())


def test_measure_noise():
    """measure_noise finds the rays that miss an attenuating disk and recovers the dispersion and the inflation of
    noise with a known variance (4 x Poisson) and correlation (0.5 between neighbouring channels)."""
    from mbirtorch.hsnt.strain.noise import measure_noise, AIR_MARGIN
    images, side, K, phi, rho = 20, 24, 400, 4.0, 0.5
    g = torch.Generator().manual_seed(0)
    d = torch.full((K,), 1000.0, dtype=torch.float64)
    yy, xx = np.meshgrid(np.arange(side) - side / 2, np.arange(side) - side / 2, indexing='ij')
    disk = np.broadcast_to(yy ** 2 + xx ** 2 < 36, (images, side, side))
    m = torch.where(torch.as_tensor(disk.reshape(-1)), 0.7, 1.0).double()[:, None].expand(-1, K)
    w = torch.randn(images, side, side + 1, K, generator=g, dtype=torch.float64)
    x = (w[:, :, :-1] + w[:, :, 1:]) / np.sqrt(2.0)                              # correlation 0.5 along channels
    T = m + torch.sqrt(phi * m / d) * x.reshape(-1, K)
    out = measure_noise(T, d.numpy(), (images, side, side))
    near = np.broadcast_to(yy ** 2 + xx ** 2 < (6 + AIR_MARGIN) ** 2, (images, side, side))
    assert out['air_rays'] <= int((~near).sum())                                 # no ray near the disk counts as air
    assert abs(out['dispersion'] / phi - 1) < 0.02
    assert abs(out['inflation'] - (1 + 2 * rho)) < 0.05


def test_measure_noise_per_bin():
    """measure_noise's per-bin dispersion follows a dispersion that ramps from 3 to 6 across the bins, and its mean
    equals the overall dispersion; the bins of a dose gap keep the overall value."""
    from mbirtorch.hsnt.strain.noise import measure_noise, DISPERSION_BLOCK
    images, side, K = 16, 16, 640
    g = torch.Generator().manual_seed(2)
    phi = torch.linspace(3.0, 6.0, K, dtype=torch.float64)
    d = torch.full((K,), 900.0, dtype=torch.float64); d[300:310] = 0.0
    T = 1.0 + torch.sqrt(phi / d.clamp_min(1.0)) * torch.randn(images * side * side, K, generator=g, dtype=torch.float64)
    T[:, 300:310] = 0.0
    out = measure_noise(T, d.numpy(), (images, side, side))
    per = np.asarray(out['dispersion_per_bin'])
    assert per.shape == (K,) and np.isfinite(per).all()
    live = np.ones(K, bool); live[299:311] = False
    blocks = np.convolve(phi.numpy(), np.ones(DISPERSION_BLOCK) / DISPERSION_BLOCK, mode='same')   # the block average
    inner = live & (np.arange(K) > DISPERSION_BLOCK) & (np.arange(K) < K - DISPERSION_BLOCK)
    assert np.abs(per[inner] / blocks[inner] - 1).max() < 0.1
    assert np.all(per[302:308] == out['dispersion'])


def test_measure_noise_dose_gaps():
    """Bins with a dose of zero (shutter gaps) are left out of the noise measurement."""
    from mbirtorch.hsnt.strain.noise import measure_noise
    images, side, K, phi = 8, 16, 300, 3.0
    g = torch.Generator().manual_seed(1)
    d = torch.full((K,), 800.0, dtype=torch.float64); d[100:110] = 0.0
    T = 1.0 + torch.sqrt(phi / d.clamp_min(1.0)) * torch.randn(images * side * side, K, generator=g, dtype=torch.float64)
    T[:, 100:110] = 0.0
    out = measure_noise(T, d.numpy(), (images, side, side))
    assert np.isfinite(out['dispersion']) and abs(out['dispersion'] / phi - 1) < 0.03
    assert abs(out['inflation'] - 1) < 0.05


def test_data_loss_memo():
    """The remembered data loss equals a fresh computation, goes stale when the state changes, and the global and
    contamination steps take the same steps with it as without it."""
    import copy
    fit = _small_fit()
    calls = []
    mean0 = type(fit).mean
    fit.mean = lambda *a, **k: (calls.append(1), mean0(fit, *a, **k))[1]
    l1 = fit.loss(); n1 = len(calls); l2 = fit.loss()
    assert l1 == l2 and len(calls) == n1                                         # the second loss costs no pass
    with torch.no_grad():
        fit.g[0] += 1e-6                                                          # an in-place change: stale
    l3 = fit.loss()
    assert len(calls) > n1 and l3 != l1
    del fit.mean
    fit.build_atoms(); fit.contamination_newton = True
    a, b = copy.deepcopy(fit), copy.deepcopy(fit)
    b.use_memo = False
    for x in (a, b):
        x.contamination_atoms_step()
        x.global_step(1e-3)
        x.contamination_atoms_step()
    assert torch.equal(a.g, b.g) and torch.equal(a.c, b.c) and torch.equal(a.f, b.f)
    assert abs(a.loss() - b.loss()) <= 1e-9 * abs(b.loss())


def test_lattice_search():
    """lattice_search finds the loss minimum in the lattice parameter from a start two edge widths away, to within
    the tolerance of a fine grid around the truth, with fewer loss evaluations than the old two-scan grid."""
    fit = _small_fit()
    M = fit.M; ia = M.slices[0]['a']
    a_true = float(fit.g[ia])
    with torch.no_grad():
        fit.g[ia] = a_true * (1 + 0.0012)
    n_eval = []
    loss0 = type(fit).loss
    fit.loss = lambda *x, **k: (n_eval.append(1), loss0(fit, *x, **k))[1]
    l_best, a_best = fit.lattice_search(0)
    del fit.loss
    fine = np.linspace(a_true * (1 - 0.001), a_true * (1 + 0.001), 401)
    vals = []
    for a_ in fine:
        gg = fit.g.clone(); gg[ia] = a_
        vals.append(fit.loss(g=gg))
    a_grid = fine[int(np.argmin(vals))]
    assert abs(a_best - a_grid) <= 2 * (fine[1] - fine[0]) and l_best <= min(vals) + 1e-9 * abs(min(vals))
    assert len(n_eval) < 33 + 25 + 2


def test_spectra_float32():
    """The per-pixel spectra in float32 (the model's dtype on the GPU) agree with float64 to single precision, and so
    do their derivatives with respect to the dilation (what the strain is fitted from)."""
    from mbirtorch.hsnt.strain.model import Hybrid2
    lam = np.linspace(2.0, 4.4, 300)
    mk = lambda dt: Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=5,
                            n_contam=1, device='cpu', dtype=dt)
    M32, M64 = mk(torch.float32), mk(torch.float64)
    g = torch.zeros(M64.n_global, dtype=torch.float64)
    sl = M64.slices[0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3; g[sl['c'][0]] = 0.2
    eps = 1e-3 * torch.randn(64, 1, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    s32, s64 = M32.spectra(g, eps), M64.spectra(g, eps)
    assert s32.dtype == torch.float32 and s64.dtype == torch.float64
    assert float((s32.double() - s64).abs().max()) < 1e-6 * float(s64.abs().max())
    tg = torch.ones_like(eps)
    d32 = torch.func.jvp(lambda e_: M32.spectra(g, e_), (eps,), (tg,))[1]
    d64 = torch.func.jvp(lambda e_: M64.spectra(g, e_), (eps,), (tg,))[1]
    assert float((d32.double() - d64).abs().max()) < 1e-3 * float(d64.abs().max())


def test_normalize_views():
    """normalize_views recovers each view's exposure, and with tiles each sensor's gain, from the air rays, and flags
    the (view, tile) far off."""
    from mbirtorch.hsnt.strain.noise import normalize_views
    images, side, K = 12, 16, 200
    g = torch.Generator().manual_seed(3)
    yy, xx = np.meshgrid(np.arange(side) - side / 2, np.arange(side) - side / 2, indexing='ij')
    disk = torch.as_tensor(np.broadcast_to(yy ** 2 + xx ** 2 < 16, (images, side, side)).reshape(-1))
    m = torch.where(disk, 0.6, 1.0).double()[:, None].expand(-1, K)
    gain = 1.0 + 0.004 * torch.randn(images, 2, 2, generator=g, dtype=torch.float64); gain[7, 1, 0] = 1.05
    pix = gain.repeat_interleave(side // 2, 1).repeat_interleave(side // 2, 2).reshape(-1)
    T = m * pix[:, None] + 0.002 * torch.randn(images * side * side, K, generator=g, dtype=torch.float64)
    lv, flagged = normalize_views(T, (images, side, side), tiles=(2, 2))
    rel = gain.reshape(images, 4).numpy(); rel = rel / np.median(rel)
    assert np.abs(lv - rel).max() < 3e-4 and flagged == [(7, 2)]
    lv2, _ = normalize_views(T, (images, side, side), tiles=(2, 2))
    assert np.abs(lv2 - 1).max() < 3e-4                                         # every view and tile at the median level


def test_shape_models_checked():
    """Hybrid2 accepts only the shape models in SHAPE_MODELS; the removed March-Dollase model texture3d is refused."""
    import pytest
    from mbirtorch.hsnt.strain.model import Hybrid2
    lam = np.linspace(2.0, 4.4, 50)
    with pytest.raises(ValueError):
        Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.83, 2.90))], device='cpu', dtype=torch.float64, shape_model='texture3d')


def _small_fit_cubic4(seed=0, coeffs=None):
    """A small FitB on the CPU with the cubic4 model: one bcc material, eight views with their own beam directions,
    per-ray dilations, and counts from a degree-4 cubic texture with coefficients coeffs (default: a fiber about a
    tilted axis)."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB, harmonics4
    lam = np.linspace(2.0, 4.4, 300)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=5, n_contam=1,
                device='cpu', dtype=torch.float64, shape_model='cubic4')
    g = torch.zeros(M.n_global, dtype=torch.float64)
    sl = M.slices[0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3; g[sl['c'][0]] = 0.2
    if coeffs is None:
        n = torch.tensor([np.sin(0.9) * np.cos(0.4), np.sin(0.9) * np.sin(0.4), np.cos(0.9)], dtype=torch.float64)
        coeffs = 0.05 * harmonics4(n)
    g[sl['x'][0]:sl['x'][1]] = torch.as_tensor(coeffs, dtype=torch.float64)
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor([0.0, 0.001, 0.002, 0.0, 0.1, 0.02, 0.0, 0.0], dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    d = torch.randn(8, 3, generator=gen, dtype=torch.float64); M.view_dirs = (d / d.norm(dim=1, keepdim=True)).numpy()
    P = 40
    W = 0.5 + torch.rand(P, 1, generator=gen, dtype=torch.float64)
    eps = 1e-3 * torch.randn(P, 1, generator=gen, dtype=torch.float64)
    fit = _speedups_off(FitB(M, torch.ones(P, M.K, dtype=torch.float64), g, W, torch.zeros(1, M.K, dtype=torch.float64),
                             eps=eps, chunk=16))
    fit.ray_view = torch.arange(P) % 8
    with torch.no_grad():
        m = torch.cat([fit.mean(s)[0] for s in fit.chunks()])
    fit.T = torch.poisson(50.0 * m, generator=gen) / 50.0
    return fit


def test_cubic4_harmonics():
    """harmonics4 gives 9 orthogonal functions with a root mean square of 1 over the sphere. Each is orthogonal to the
    degree-2 monomials, so none has a degree-2 part. cubic_k4 is 0.4 for bcc 200, -0.1 for 110, 211 and 220, and -4/15
    for 222."""
    from mbirtorch.hsnt.strain.model import harmonics4, cubic_k4, _sphere_quadrature
    pts, w = _sphere_quadrature()
    Y = harmonics4(pts)
    G = (w[:, None, None] * Y[:, :, None] * Y[:, None, :]).sum(0)
    assert torch.allclose(G, torch.eye(9, dtype=torch.float64), atol=1e-10)
    for q in (pts[:, 0] ** 2, pts[:, 0] * pts[:, 1], pts[:, 2] ** 2):
        assert float((w[:, None] * Y * q[:, None]).sum(0).abs().max()) < 1e-12
    assert np.allclose(cubic_k4('bcc', [4, 2, 6, 8, 12]), [0.4, -0.1, -0.1, -0.1, -4 / 15])


def test_cubic4_factor():
    """The cubic4 factor is 1 for a random powder. Below each edge it equals the pole density averaged over the
    diffraction cone, checked against a direct average on the cone."""
    from mbirtorch.hsnt.strain.model import harmonics4
    fit = _small_fit_cubic4()
    M = fit.M; sl = M.slices[0]; x0 = sl['x'][0]; g = fit.g.clone()
    g0 = g.clone(); g0[x0:x0 + 9] = 0.0
    assert torch.allclose(M.texture_factor(g0, 0), torch.ones(()).double(), atol=1e-12)
    F = M.texture_factor(g, 0)                                            # (views, edges, Kf)
    c = g[x0:x0 + 9]; k4 = M.mats[0]['k4']
    b = torch.as_tensor(M.view_dirs[0], dtype=torch.float64)
    e = 2.0 * float(g[sl['a']]) / torch.sqrt(M.mats[0]['ns'])
    for ie in range(len(e)):
        for frac in (0.999, 0.97, 0.9, 0.8):
            k = int(torch.argmin((M.lamf - frac * e[ie]).abs()))
            st = float(M.lamf[k] / e[ie])
            if st >= 1.0:
                continue
            # normals at 90 deg - theta from the beam: a cone about b with cos(half angle) = sin(theta)
            u = torch.linalg.svd(torch.stack([b, torch.randn(3, dtype=torch.float64)]))[2][-1]; v = torch.linalg.cross(b, u)
            phi = torch.linspace(0, 2 * np.pi, 721, dtype=torch.float64)[:-1]
            sn = st * b[None] + np.sqrt(1 - st * st) * (torch.cos(phi)[:, None] * u[None] + torch.sin(phi)[:, None] * v[None])
            direct = (1 + k4[ie] * (harmonics4(sn) @ c)).mean()
            assert abs(float(F[0, ie, k]) - float(direct)) < 1e-9


def test_cubic4_global_grad():
    """With cubic4 the gradient by automatic differentiation (global_grad) equals global_system's for the texture
    coefficients and the heights."""
    fit = _small_fit_cubic4()
    M = fit.M; sl = M.slices[0]
    idx = list(range(sl['h'][0], sl['h'][1])) + list(range(sl['x'][0], sl['x'][1]))
    g1, F = fit.global_system(idx)
    g2 = fit.global_grad(idx)
    assert torch.allclose(g1, g2, rtol=1e-5, atol=1e-7 * float(g1.abs().max()))
    assert bool((torch.diagonal(F) > 0).all())


def test_texture_field_global_grad():
    """With a texture field (two basis functions, each ray its own weights) the gradient by automatic differentiation
    (global_grad) equals global_system's for the heights and the texture coefficients, whose derivatives
    global_system forms in closed form."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    lam = np.linspace(2.0, 4.4, 300)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=5, n_contam=1,
                device='cpu', dtype=torch.float64, shape_model='cubic4', texture_basis=2)
    g = torch.zeros(M.n_global, dtype=torch.float64)
    sl = M.slices[0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3; g[sl['c'][0]] = 0.2
    gen = torch.Generator().manual_seed(3)
    g[sl['x'][0]:sl['x'][1]] = 0.05 * torch.randn(sl['x'][1] - sl['x'][0], generator=gen, dtype=torch.float64)
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor([0.0, 0.001, 0.002, 0.0, 0.1, 0.02, 0.0, 0.0], dtype=torch.float64)
    d = torch.randn(8, 3, generator=gen, dtype=torch.float64); M.view_dirs = (d / d.norm(dim=1, keepdim=True)).numpy()
    P = 40
    W = 0.5 + torch.rand(P, 1, generator=gen, dtype=torch.float64)
    eps = 1e-3 * torch.randn(P, 1, generator=gen, dtype=torch.float64)
    fit = FitB(M, torch.ones(P, M.K, dtype=torch.float64), g, W, torch.zeros(1, M.K, dtype=torch.float64), eps=eps, chunk=16)
    fit.ray_view = torch.arange(P) % 8
    q = torch.rand(P, generator=gen, dtype=torch.float64)
    fit.tex_Q = [torch.stack([q, 1 - q], 1)]
    fit.eps_sd = 2e-4 * torch.rand(P, 1, generator=gen, dtype=torch.float64)
    with torch.no_grad():
        m = torch.cat([fit.mean(s)[0] for s in fit.chunks()])
    fit.T = torch.poisson(50.0 * m, generator=gen) / 50.0
    idx = list(range(sl['h'][0], sl['h'][1])) + list(range(sl['x'][0], sl['x'][1]))
    g1, F = fit.global_system(idx)
    g2 = fit.global_grad(idx)
    assert torch.allclose(g1, g2, rtol=1e-5, atol=1e-7 * float(g1.abs().max()))
    assert bool((torch.diagonal(F) > 0).all())


@pytest.mark.parametrize('case', ['powder', 'views', 'field', 'anisotropic'])
@pytest.mark.parametrize('spread', [False, True])
def test_ray_spectrum_matches_dense(case, spread):
    """The windowed per-ray spectra (Hybrid2._ray_spectrum) equal the sum of every edge over every fine bin, with and
    without the strain's spread along the rays, and so do their derivatives with respect to the global parameters and
    the strains."""
    from mbirtorch.hsnt.strain.model import Hybrid2, RayTexture, harmonics4
    lam = np.linspace(1.2, 4.4, 600)
    nb = 2 if case == 'field' else 1
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=5, n_contam=1,
                device='cpu', dtype=torch.float64, edge_slope=True,
                shape_model=None if case == 'powder' else 'cubic4', texture_basis=nb)
    gen = torch.Generator().manual_seed(5)
    g = torch.zeros(M.n_global, dtype=torch.float64)
    sl = M.slices[0]; ne = sl['h'][1] - sl['h'][0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3 + 0.1 * torch.rand(ne, generator=gen, dtype=torch.float64)
    g[sl['t'][0]:sl['t'][1]] = 0.2 * torch.randn(sl['t'][1] - sl['t'][0], generator=gen, dtype=torch.float64)
    g[sl['c'][0]] = 0.2
    if case != 'powder':
        g[sl['x'][0]:sl['x'][1]] = 0.05 * torch.randn(sl['x'][1] - sl['x'][0], generator=gen, dtype=torch.float64)
    d = torch.randn(8, 3, generator=gen, dtype=torch.float64); M.view_dirs = (d / d.norm(dim=1, keepdim=True)).numpy()
    P = 30
    if case == 'anisotropic':
        M.set_dec(0, 0.8 + 0.4 * torch.rand(ne, generator=gen, dtype=torch.float64), 0.05 * torch.randn(ne, generator=gen, dtype=torch.float64))
        eps = 2e-3 * torch.randn(P, 2, generator=gen, dtype=torch.float64)
    else:
        eps = 2e-3 * torch.randn(P, 1, generator=gen, dtype=torch.float64)
    views = None if case == 'powder' else torch.arange(P) % 8
    if case == 'field':
        q = torch.rand(P, generator=gen, dtype=torch.float64)
        views = RayTexture(views, [torch.stack([q, 1 - q], 1)])
    sd = 3e-4 * torch.rand(P, 1, generator=gen, dtype=torch.float64) if spread else None

    def run(dense, gg, ee):
        M.dense_spectra = dense
        return M.spectra(gg, ee, views, sd=sd)
    a, b = run(False, g, eps), run(True, g, eps)
    with torch.no_grad():                                 # the fused window kernel (no derivatives wanted)
        c_ = run(False, g, eps)
    assert torch.allclose(c_, b, rtol=0, atol=1e-10 * float(b.abs().max()))
    x = M.lamf / M.lam_ref
    assert M._ray_spectrum(g, 0, eps, views, sd, x, torch.zeros_like(x)) is not None    # the windowed path ran
    assert torch.allclose(a, b, rtol=0, atol=1e-10 * float(b.abs().max()))
    w = torch.randn_like(b)
    grads = []
    for dense in (False, True):
        gp = g.clone().requires_grad_(True)
        grads.append(torch.autograd.grad((run(dense, gp, eps) * w).sum(), gp)[0])
    assert torch.allclose(grads[0], grads[1], rtol=1e-8, atol=1e-10 * float(grads[1].abs().max()))
    tang = torch.randn_like(eps)
    jv = [torch.func.jvp(lambda ee: run(dense, g, ee), (eps,), (tang,))[1] for dense in (False, True)]
    assert torch.allclose(jv[0], jv[1], rtol=0, atol=1e-8 * float(jv[1].abs().max()))
    # the closed-form derivative with respect to the lattice parameter equals forward-mode differentiation of the dense sum
    tg = torch.zeros_like(g); tg[sl['a']] = 1.0
    M.dense_spectra = True
    _, d_ref = torch.func.jvp(lambda gg: M.spectra(gg, eps, views, sd=sd)[:, 0], (g,), (tg,))
    M.dense_spectra = False
    mu_da = M.spectra_da(g, eps, 0, views, sd)
    assert torch.allclose(mu_da[0], b[:, 0], rtol=0, atol=1e-10 * float(b.abs().max()))
    assert torch.allclose(mu_da[1], d_ref, rtol=0, atol=1e-8 * float(d_ref.abs().max()))
    if case == 'views':                      # given texture factors (a phantom's true textures) are used as given
        tex = [1 + 0.3 * torch.rand(8, ne, M.Kf, generator=gen, dtype=torch.float64)]
        M.dense_spectra = False; a = M.spectra(g, eps, views, tex=tex, sd=sd)
        M.dense_spectra = True; b = M.spectra(g, eps, views, tex=tex, sd=sd)
        assert torch.allclose(a, b, rtol=0, atol=1e-12 * float(b.abs().max()))
        assert not torch.allclose(a, M.spectra(g, eps, views, sd=sd))
    M.dense_spectra = False


@pytest.mark.parametrize('known', [None, 0, 1])
def test_warmup_searches_unknown_materials(known):
    """warmup finds the lattice type and parameter of every material nothing is given about (here fcc Ni and bcc Fe),
    and, when one material is given by its spectrum, finds the other one instead of the given one again."""
    from mbirtorch.hsnt.strain.tomo import Projector, warmup
    from mbirtorch.hsnt.strain.bragg import BraggSpectrum
    torch.manual_seed(0)
    n = 12; th = np.linspace(0, np.pi, 9, endpoint=False)
    proj = Projector(n, th, 2, 0.0, False)
    lam = np.linspace(1.5, 5.0, 500)
    mats = [('fcc', 3.524), ('bcc', 2.8665)]
    mu = []
    for lt, a in mats:
        m = BraggSpectrum(lam, lattice=lt, a_range=(a * 0.999, a * 1.001), resolution=(0.002, 0.0))
        p_ = torch.zeros(m.n_params, dtype=torch.float64); p_[0] = a; p_[1:1 + m.n_edges] = 1.0; p_[1 + m.n_edges] = 0.3
        mu.append(m(p_).detach().numpy())
    mu = 0.08 * np.stack(mu) / np.stack(mu).mean(1, keepdims=True)
    V = torch.zeros(n, n, n, 2); V[2:6, 3:9, 3:9, 0] = 1; V[6:10, 3:9, 3:9, 1] = 1
    # The projector returns its rays on its own device, which is a GPU when there is one. The counts are drawn by the
    # seeded CPU generator on every machine. The fit computes on the projector's device.
    rays = proj.fwd(V); dev = rays.device.type
    T = torch.poisson(1e5 * torch.exp(-(rays.cpu().double() @ torch.as_tensor(mu)))) / 1e5
    lat = [None, None]; a0 = [None, None]; init = None
    if known is not None:
        lat[known] = mats[known][0]; a0[known] = mats[known][1]
        init = [None, None]; init[known] = mu[known]
    _, info = warmup(T.float().to(dev), lam, np.full(lam.size, 1e5), lat, a0, proj, device=dev, init_spectra=init,
                     d0_from_data=True, noise=None, normalize=False, a_range=0.01)      # not the search's +-2%
    for (lt, a), (lt_f, a_f) in zip(mats, info['found']):
        assert lt_f == lt and abs(a_f / a - 1) < 1e-3


def test_warm_start_refuses_a_flat_search():
    """When every lattice parameter fits the components almost exactly (here smooth curves with no edges, which the
    search's smooth basis holds), warm_start stops with an error instead of returning an arbitrary lattice."""
    from mbirtorch.hsnt.strain.bragg import warm_start
    lam = np.linspace(1.5, 5.0, 300); x = (lam - lam.mean()) / np.ptp(lam)
    H = np.stack([1 + 0.5 * x, 1 - 0.3 * x + 0.2 * x ** 2])
    with pytest.raises(ValueError, match='cannot tell lattice parameters apart'):
        warm_start(H, lam, n_bragg=1, lattices=[('bcc', (2.80, 2.93))], device='cpu')


def test_fine_projector_set_mount():
    """A FineProjector whose mount is changed in place projects like one built with that mount, series by series, and
    reports the mount in voxels of the detector grid."""
    from mbirtorch.hsnt.strain.tomo import FineProjector
    n = 8; ang = np.linspace(0, np.pi, 5)
    series = [(((0, 1, 2), (1, 1, 1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), ang), (((1, 2, 0), (1, 1, 1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), ang)]
    vol = torch.rand(2 * n, 2 * n, 2 * n, 1, generator=torch.Generator().manual_seed(1))
    P = FineProjector(n, series, 1, channel_offsets=[0.0, 0.0], ror_mask=False)
    P.set_mount(1, shift=(0.3, -0.2, 0.1), rot=(0.5, 0.0, -0.4), offset=0.25)
    moved = [series[0], (((1, 2, 0), (1, 1, 1), (0.3, -0.2, 0.1), (0.5, 0.0, -0.4)), ang)]
    Q = FineProjector(n, moved, 1, channel_offsets=[0.0, 0.25], ror_mask=False)
    for k in range(2):
        assert torch.allclose(P.project_series(k, vol), Q.project_series(k, vol), atol=1e-6)
    assert np.allclose(P.parts[1][0][2], (0.3, -0.2, 0.1)) and abs(P.offsets[1] - 0.25) < 1e-12
    assert torch.allclose(P.fwd(vol), Q.fwd(vol), atol=1e-6)


def test_set_view_offsets():
    """MultiProjector and FineProjector: per-view channel offsets set in place project like a projector built with them,
    and they survive with_q."""
    from mbirtorch.hsnt.strain.tomo import FineProjector
    n = 8; ang = np.linspace(0, np.pi, 5)
    series = [(((0, 1, 2), (1, 1, 1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), ang), (((1, 2, 0), (1, 1, 1), (0.3, 0.0, 0.0), (0.0, 0.0, 0.0)), ang)]
    extra = [np.zeros(5), np.array([0.0, 0.4, 0.0, -0.3, 0.0])]
    for make, side in ((lambda **k: MultiProjector(n, series, 1, channel_offsets=[0.0, 0.0], ror_mask=False, **k), n),
                       (lambda **k: FineProjector(n, series, 1, channel_offsets=[0.0, 0.0], ror_mask=False, **k), 2 * n)):
        vol = torch.rand(side, side, side, 1, generator=torch.Generator().manual_seed(2))
        P = make(); P.set_view_offsets(1, extra[1])
        Q = make(view_extra=extra)
        for k in range(2):
            assert torch.allclose(P.project_series(k, vol).cpu(), Q.project_series(k, vol).cpu(), atol=1e-6)
        assert np.allclose(P.with_q(2).view_extra[1], extra[1])


def test_set_view_angles():
    """MultiProjector and FineProjector: per-view angle offsets set in place project like a projector whose series has
    those angles, and they survive with_q."""
    from mbirtorch.hsnt.strain.tomo import FineProjector
    n = 8; ang = np.linspace(0, np.pi, 5); extra = np.array([0.0, 0.01, 0.0, -0.02, 0.0])
    mk = lambda a_: [(((0, 1, 2), (1, 1, 1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), ang), (((1, 2, 0), (1, 1, 1), (0.3, 0.0, 0.0), (0.0, 0.0, 0.0)), a_)]
    for cls, side in ((MultiProjector, n), (FineProjector, 2 * n)):
        vol = torch.rand(side, side, side, 1, generator=torch.Generator().manual_seed(3))
        P = cls(n, mk(ang), 1, channel_offsets=[0.0, 0.0], ror_mask=False); P.set_view_angles(1, extra)
        Q = cls(n, mk(ang + extra), 1, channel_offsets=[0.0, 0.0], ror_mask=False)
        for k in range(2):
            assert torch.allclose(P.project_series(k, vol).cpu(), Q.project_series(k, vol).cpu(), atol=1e-6)
        assert np.allclose(P.with_q(2).view_angle_extra[1], extra)


def test_volume_step_retries_with_zero_mean_step():
    """A volume step that finds no decrease retries with more damping. With zero_mean_step, each retry's curvature
    product must still be finite (q363 and q368: a RecursionError on the first retry)."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    from mbirtorch.hsnt.strain.tomo import TomoFit

    pr = MultiProjector(N, MOUNTS, 2, channel_offsets=[0.0, 0.3, -0.2], ror_mask=False)
    dev = pr.fwd(torch.zeros(N, N, N, 2)).device                 # the device the projector runs on
    lam = np.linspace(2.0, 4.4, 60)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=3, n_contam=1,
                device=dev)
    g = torch.zeros(M.n_global, dtype=torch.float64, device=dev)
    sl = M.slices[0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3; g[sl['c'][0]] = 0.2
    P = sum(pr.counts)
    kw = dict(device=dev)
    fit = FitB(M, torch.ones(P, M.K, **kw), g, torch.ones(P, 1, **kw), torch.zeros(1, M.K, **kw),
               eps=torch.zeros(P, 1, **kw), chunk=512)
    V = torch.zeros(N, N, N, 1, device=dev); V[3:9, 3:9, 3:9] = 0.05
    tf = TomoFit(fit, pr, V, torch.zeros(N, N, N, 1, 1, device=dev))
    assert tf.zero_mean_step
    tf.loss = lambda mask=None: -math.inf                     # no step can lower this: every pass is a retry
    tf.volume_step()


def test_open_beam_likelihood():
    """With a measured open beam (open_beam_obs n), the likelihood is the negative binomial's: it tends to the Poisson
    deviance as n grows; FitB.residual is the deviance's derivative; FitB.weight is the inverse of the variance of
    transmissions divided by a noisy open beam (simulated), times the counts squared."""
    from mbirtorch.hsnt.strain.model import deviance_terms
    gen = torch.Generator().manual_seed(3)
    m = 0.2 + 0.8 * torch.rand(2000, dtype=torch.float64, generator=gen)
    T = m * (1 + 0.05 * torch.randn(2000, dtype=torch.float64, generator=gen))
    d = 400.0
    assert torch.allclose(deviance_terms(m, T, d, 1e9), deviance_terms(m, T, d), rtol=1e-5, atol=1e-6)
    fit = _small_fit_cubic4()
    fit.open_beam_obs = 1.1
    sl = slice(0, 16)
    with torch.no_grad():
        mm = fit.mean(sl)[0].clamp_min(1e-12)
    Tm = fit.T[sl]
    mv = mm.clone().requires_grad_(True)
    g_auto = torch.autograd.grad(fit.deviance_terms(sl, mv, Tm).sum(), mv)[0]
    assert torch.allclose(fit.residual(sl, Tm, mm), g_auto, rtol=1e-8, atol=1e-10)
    # a noisy open beam: T = S / O with S ~ Poisson(d m), O ~ Poisson(n d) / n
    n, dd, mt = 1.1, 400.0, torch.tensor([0.25, 0.6, 1.0], dtype=torch.float64)
    S = torch.poisson((dd * mt).expand(200000, 3).contiguous(), generator=gen)
    O = torch.poisson(torch.full((200000, 3), n * dd, dtype=torch.float64), generator=gen) / n
    var = (S / O).var(0)
    var_model = mt / dd * (1 + mt / n)                          # 1 / (weight / d^2)
    assert torch.allclose(var, var_model, rtol=0.03)


def test_shared_open_beam():
    """One noisy open beam shared by every view (FitB.set_shared_open_beam): at the current state the Poisson bound's
    residual equals the derivative of the exact profile likelihood (the open beam profiled out), and pixel and global
    steps on the bound never raise the exact deviance (majorize-minimize)."""
    fit = _small_fit_cubic4()
    P, K = fit.T.shape; pix = torch.arange(P) // 8                        # 5 detector pixels, each seen in 8 views
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        m_true = torch.cat([fit.mean(s)[0] for s in fit.chunks()])
    b_true = 50.0 * torch.ones(5, K, dtype=torch.float64); s_ = 0.9 + 0.2 * torch.rand(P, generator=gen, dtype=torch.float64)
    O = torch.poisson(b_true, generator=gen).clamp_min(1.0)
    S = torch.poisson(s_[:, None] * b_true[pix] * m_true, generator=gen)
    fit.T = (S / (s_[:, None] * O[pix])).to(fit.T.dtype)
    fit.set_shared_open_beam(O, pix, s_)
    # the bound's residual is the exact profile likelihood's derivative at the current state
    with torch.no_grad():
        mm = torch.cat([fit.mean(s)[0] for s in fit.chunks()]).clamp_min(1e-12)
    mv = mm.clone().requires_grad_(True)
    A = torch.zeros(5, K, dtype=torch.float64).index_add(0, pix, s_[:, None] * mv)
    nll = ((O + torch.zeros(5, K, dtype=torch.float64).index_add(0, pix, S)) * torch.log1p(A)).sum() - (S * torch.log(mv)).sum()
    g_exact = torch.autograd.grad(nll, mv)[0]
    g_bound = torch.cat([fit.residual(s, fit.T[s], mm[s]) for s in fit.chunks()])
    assert torch.allclose(g_bound, g_exact, rtol=1e-6, atol=1e-8 * float(g_exact.abs().max()))
    # majorize-minimize: the exact deviance does not rise
    d0 = fit.open_beam_deviance()
    devs = [d0]
    for _ in range(3):
        fit.pixel_step(); fit.global_step()
        devs.append(fit.open_beam_deviance())
    assert all(b <= a * (1 + 1e-9) for a, b in zip(devs, devs[1:])) and devs[-1] < d0


def test_edge_spread():
    """Summed edges: edge_step reduces to the sharp step at zero width and equals a Monte-Carlo average of shifted sharp
    steps; spectra with a strain spread equal a fine Gauss-Hermite average of spectra over the strain; and
    TomoFit.refresh_spread gives each ray's variance of the strain along it."""
    from mbirtorch.hsnt.strain.model import edge_step, Hybrid2, FitB
    from mbirtorch.hsnt.strain.tomo import TomoFit
    off = torch.linspace(-0.05, 0.06, 400, dtype=torch.float64); hf = 0.01
    assert torch.equal(edge_step(off, hf, 0.0), (off / hf).clamp(0, 1))
    w = 0.013
    z = torch.randn(200000, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    mc = ((off[:, None] + w * z[None]) / hf).clamp(0, 1).mean(1)
    assert float((edge_step(off, hf, w) - mc).abs().max()) < 4e-3
    # spectra with a spread against a Monte-Carlo average over the strain (a Gauss-Hermite rule converges slowly here:
    # each sharp edge is a ramp narrower than its nodes' spacing)
    lam = np.linspace(3.6, 4.5, 400)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=3, n_contam=1,
                device='cpu', dtype=torch.float64)
    g = torch.zeros(M.n_global, dtype=torch.float64); sl = M.slices[0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3
    eps = torch.tensor([[1e-3], [-5e-4]], dtype=torch.float64); sd = torch.tensor([[2e-3], [5e-4]], dtype=torch.float64)
    zs = torch.randn(20000, 1, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    for i in range(2):
        mc = M.spectra(g, eps[i:i + 1] + zs * sd[i:i + 1])[:, 0].mean(0)
        assert float((M.spectra(g, eps[i:i + 1], sd=sd[i:i + 1])[0, 0] - mc).abs().max()) < 1.5e-2 * float(mc.abs().max())
    # the rays' strain spread
    pr = MultiProjector(N, MOUNTS, 2, channel_offsets=[0.0, 0.3, -0.2], ror_mask=False)
    dev = pr.fwd(torch.zeros(N, N, N, 2)).device
    M2 = Hybrid2(np.linspace(2.0, 4.4, 60), np.full(60, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=3,
                 n_contam=1, device=dev)
    P = sum(pr.counts); kw = dict(device=dev)
    fit = FitB(M2, torch.ones(P, M2.K, **kw), torch.zeros(M2.n_global, dtype=torch.float64, **kw), torch.ones(P, 1, **kw),
               torch.zeros(1, M2.K, **kw), eps=torch.zeros(P, 1, **kw), chunk=512)
    V = torch.zeros(N, N, N, 1, device=dev); V[3:9, 3:9, 3:9] = 0.05
    E = 1e-3 * torch.randn(N, N, N, 1, 1, generator=torch.Generator().manual_seed(2)).to(dev)
    tf = TomoFit(fit, pr, V, E); tf.sync(); tf.edge_spread = True
    tf.spread_grouped = False                    # This test checks the reference path. test_spread_grouped compares.
    tf.refresh_spread()
    W = pr.with_q(1).fwd(V); m1 = pr.with_q(1).fwd(V * E[..., 0]); m2 = pr.with_q(1).fwd(V * E[..., 0] ** 2)
    keep = W[:, 0] > 1e-3
    var = (m2 / W.clamp_min(1e-12) - (m1 / W.clamp_min(1e-12)) ** 2)[keep, 0]
    assert torch.allclose(fit.eps_sd[keep, 0] ** 2, var.clamp_min(0), rtol=1e-3, atol=1e-10)


# SPREAD_MOUNTS is MOUNTS with whole shifts along each series' rotation axis. Its projectors therefore have no
# fractional detector row offset. With a fractional row offset, each stacked volume's detector row position carries a
# float32 rounding that grows with its place in the stack. The default refresh_spread stacks 21 products per material,
# and the grouped path stacks 7. Their spreads then differ by up to a few 1e-4 of a ray's mean square strain.
SPREAD_MOUNTS = [('x', ANGLES),
                 (((1, 2, 0), (1, 1, 1), (0.4, -2.3, 2.0), (-0.35, 1.93, -0.63)), ANGLES),
                 (((2, 0, 1), (-1, 1, -1), (3.0, -0.5, 0.0)), ANGLES)]
SPREAD_TOL = 3e-5               # the two paths' largest difference in a ray's mean square strain, relative to the sum
                                # of its terms' absolute values, without fractional row offsets
SPREAD_TOL_ROWS = 2e-3          # the same with fractional row offsets


def _spread_fit(kind, n, R, mounts, C=6):
    """A TomoFit with edge_spread on, R materials and C strain components (6 for the tensor, 1 for one strain). kind
    'plain' uses one Projector, 'multi' a MultiProjector with the given mounts and 'fine' a FineProjector (factor 2)
    with them. The strain has a mean and a spread inside the material."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    from mbirtorch.hsnt.strain.tomo import TomoFit, Projector, FineProjector
    from mbirtorch.hsnt.strain.geometry import series_weights
    offs = [0.0, 0.3, -0.2]
    if kind == 'plain':
        pr, m, wts = Projector(n, ANGLES, 2, ror_mask=False), n, series_weights([('z', ANGLES)])
    else:
        pr = MultiProjector(n, mounts, 2, channel_offsets=offs, ror_mask=False) if kind == 'multi' else \
            FineProjector(n, mounts, 2, channel_offsets=offs, ror_mask=False, factor=2)
        m = n if kind == 'multi' else 2 * n
        wts = series_weights(mounts)
    if C == 1:
        wts = None                                       # one strain, seen alike by every view
    dev = pr.fwd(torch.zeros(m, m, m, 2)).device
    mats = [('bcc', (2.8665 * 0.99, 2.8665 * 1.01)), ('fcc', (3.52 * 0.99, 3.52 * 1.01))][:R]
    lam = np.linspace(2.0, 4.4, 40)
    M = Hybrid2(lam, np.full_like(lam, 50.0), mats, smooth_degree=3, n_contam=1, device=dev)
    P = pr.V * n * n; kw = dict(device=dev)
    fit = FitB(M, torch.ones(P, M.K, **kw), torch.zeros(M.n_global, dtype=torch.float64, **kw), torch.ones(P, R, **kw),
               torch.zeros(1, M.K, **kw), eps=torch.zeros(P, R, **kw), chunk=512)
    fit.lean_passes = False                              # The test counts refresh_spread's own reuse of its result.
    gen = torch.Generator().manual_seed(7)
    V = torch.zeros(m, m, m, R)
    a, b = m // 4, 3 * m // 4
    V[a:b, a:b, a:b, 0] = 0.05
    if R > 1:
        V[a + 2:b + 2, a - 1:b - 1, a:b, 1] = 0.03
    E = (1e-3 + 3e-4 * torch.randn(m, m, m, R, C, generator=gen)) * (V[..., None] > 0)
    tf = TomoFit(fit, pr, V.to(dev), E.to(dev), weights=wts)
    tf.edge_spread = True
    return tf


@pytest.mark.parametrize("kind,n,R,rows,C", [("plain", 16, 1, False, 6), ("multi", 16, 2, False, 6),
                                             ("multi", 24, 1, False, 6), ("fine", 12, 2, False, 6),
                                             ("multi", 16, 2, True, 6), ("fine", 12, 2, True, 6), ("multi", 16, 2, False, 1)])
def test_spread_grouped(kind, n, R, rows, C):
    """TomoFit.spread_grouped gives the default refresh_spread's strain spread up to float rounding, with a plain,
    mounted or fine-grid projector, one or two materials and one or six strain components. It keeps its last result
    while nothing it reads changes, also across syncs of the same volumes. It makes a new result after a change of the
    strain or of the geometry. rows=True uses MOUNTS, whose fractional detector row offsets need the looser tolerance."""
    from mbirtorch.hsnt.strain.tomo import EPS_W_MIN
    tf = _spread_fit(kind, n, R, MOUNTS if rows else SPREAD_MOUNTS, C)
    fit = tf.fit
    tol = SPREAD_TOL_ROWS if rows else SPREAD_TOL
    iu = [(c, d) for c in range(C) for d in range(c, C)]
    npair = len(iu)

    def grouped():
        tf.spread_grouped = True
        tf.refresh_spread()
        return fit.eps_sd

    def compare(out):
        # The default path's spread at the current state is the reference. It replaces fit.eps_sd, and the next
        # grouped call therefore computes a new result. The tolerance scales with each ray's sum of the absolute
        # values of its terms.
        tf.spread_grouped = False
        tf.refresh_spread()
        ref = fit.eps_sd
        tf.spread_grouped = True
        m = tf.V.shape[0]
        prods = torch.stack([(tf.V * tf.E[..., c] * tf.E[..., d]).abs() for c, d in iu], -1).reshape(m, m, m, R * npair)
        ww = torch.stack([(1.0 if c == d else 2.0) * (tf.w_ray[:, c] * tf.w_ray[:, d]).abs() for c, d in iu], -1)
        terms = (tf.proj.with_q(R * npair).fwd(prods).reshape(-1, R, npair) * ww[:, None, :]).sum(-1)
        scale = terms / fit.W.clamp_min(EPS_W_MIN)
        diff = (out ** 2 - ref ** 2).abs()
        keep = fit.W > EPS_W_MIN
        assert float(ref[keep].max()) > 0
        assert bool((diff <= tol * scale)[keep].all()), float((diff / scale.clamp_min(1e-30))[keep].max())

    def move(k):
        # The k-th of two changes of the geometry: a channel offset, a mount's shift or a mount's channel offset.
        if kind == 'plain':
            tf.proj.set_geometry((0.3, -0.2)[k])
        elif k == 0:
            tf.proj.set_mount(1, shift=(0.6, -2.0, 1.4 if rows else 1.0))
        else:
            tf.proj.set_mount(2, offset=0.4)

    out = grouped()
    skips = getattr(tf, 'spread_skips', 0)
    assert grouped() is out and tf.spread_skips == skips + 1          # nothing changed: the result is kept
    compare(out)
    tf.sync(); out = grouped(); tf.sync()                              # a new sync of the same volumes ...
    assert grouped() is out and tf.spread_skips == skips + 2          # ... keeps the result
    compare(out)
    out = grouped(); tf.E.mul_(1.5)                                    # the strain changes in place
    new = grouped()
    assert new is not out and tf.spread_skips == skips + 2
    compare(new)
    tf.sync(); out = grouped(); move(0)                                # the geometry changes without a sync
    new = grouped()
    assert new is not out and tf.spread_skips == skips + 2
    compare(new)
    tf.sync(); out = grouped(); move(1); tf.sync()                     # the geometry changes, then a sync
    new = grouped()
    assert new is not out and tf.spread_skips == skips + 2
    compare(new)


def test_tabulated_spectrum():
    """A tabulated material (Hybrid2.set_tabulated): its smooth part is c0 x the table's non-Bragg shape + c1 lambda /
    lambda_ref with the spline fixed at zero; the height prior (FitB.height_prior) is zero at heights proportional to
    the table and grows with relative deviations."""
    fit = _small_fit_cubic4()
    M = fit.M; sl = M.slices[0]
    nb = 1.0 + 0.5 * torch.linspace(0, 1, M.Kf, dtype=torch.float64) ** 2
    M.set_tabulated(0, nonbragg=nb)
    assert float(M.ub[sl['s'][0]:sl['s'][1]].abs().max()) == 0.0
    g = fit.g.clone(); g[sl['h'][0]:sl['h'][1]] = 0.0; g[sl['s'][0]:sl['s'][1]] = 0.0
    g[sl['c'][0]] = 2.0; g[sl['c'][0] + 1] = 0.0
    assert torch.allclose(M.spectra(g)[0], 2.0 * nb / nb[M.Kf // 2], rtol=1e-12)
    ne = sl['h'][1] - sl['h'][0]; t = torch.linspace(1.0, 2.0, ne, dtype=torch.float64)
    fit.height_prior = dict(targets={0: t}, sd=0.05)
    g2 = fit.g.clone(); g2[sl['h'][0]:sl['h'][1]] = 3.0 * t
    base = float(fit.spline_penalty(g2)); fit.height_prior = None; base -= float(fit.spline_penalty(g2))
    assert abs(base) < 1e-9
    fit.height_prior = dict(targets={0: t}, sd=0.05)
    g3 = g2.clone(); g3[sl['h'][0]] *= 1.1
    assert float(fit.spline_penalty(g3)) - float(fit.spline_penalty(g2)) > 1.0


def test_elastic_anisotropy():
    """Each reflection's own lattice strain (Hybrid2.set_dec): the ray Jacobian's columns for the mean strain and the
    mean trace equal automatic derivatives of the model, and a volume step on data from an anisotropic model with
    three series lowers the loss."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    from mbirtorch.hsnt.strain.tomo import TomoFit
    from mbirtorch.hsnt.strain.geometry import series_weights
    fit = _small_fit_cubic4()
    M = fit.M; ne = len(M.mats[0]['ns'])
    al = torch.linspace(0.8, 1.3, ne, dtype=torch.float64); be = torch.linspace(-0.1, 0.06, ne, dtype=torch.float64)
    M.set_dec(0, al, be)
    P = fit.T.shape[0]; gen = torch.Generator().manual_seed(5)
    fit.eps = torch.cat([fit.eps, 2e-3 * torch.randn(P, 1, generator=gen, dtype=torch.float64)], 1)   # mean traces
    sl = slice(0, 16); cache = fit._state(fit.g)
    Jac, m = fit.ray_jacobian(sl, cache)
    W, eps = fit.W[sl], fit.eps[sl]
    U, Ut = W * eps[:, :1], W * eps[:, 1:]
    f = lambda W_, U_, Ut_: fit.mean(sl, W=W_, eps=torch.cat([U_ / W_, Ut_ / W_], 1), cache=cache)[0]
    for k, tang in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        t = [torch.full_like(W, float(v)) for v in tang]
        _, dm = torch.func.jvp(f, (W, U, Ut), tuple(t))
        assert torch.allclose(Jac[..., k], dm, rtol=1e-6, atol=1e-8 * float(dm.abs().max()))
    # a volume step with three series and an anisotropic model
    pr = MultiProjector(N, MOUNTS, 2, channel_offsets=[0.0, 0.3, -0.2], ror_mask=False)
    dev = pr.fwd(torch.zeros(N, N, N, 2)).device
    wts = series_weights([(m_[0] if isinstance(m_[0], str) else tuple(m_[0][:3]), a_) for m_, a_ in [(mm, mm[1]) for mm in MOUNTS]])
    lam = np.linspace(2.0, 4.4, 80)
    M2 = Hybrid2(lam, np.full_like(lam, 200.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=3, n_contam=1,
                 device=dev)
    ne2 = len(M2.mats[0]['ns']); M2.set_dec(0, torch.linspace(0.8, 1.3, ne2), torch.linspace(-0.1, 0.06, ne2))
    g = torch.zeros(M2.n_global, dtype=torch.float64, device=dev); s2 = M2.slices[0]
    g[s2['a']] = 2.8665; g[s2['h'][0]:s2['h'][1]] = 0.3; g[s2['c'][0]] = 0.2
    Pn = sum(pr.counts); kw = dict(device=dev)
    V = torch.zeros(N, N, N, 1, device=dev); V[3:9, 3:9, 3:9] = 0.05
    E_true = 1.5e-3 * torch.randn(N, N, N, 1, 6, generator=torch.Generator().manual_seed(6)).to(dev) * (V[..., None] > 0)
    fit2 = FitB(M2, torch.ones(Pn, M2.K, **kw), g, torch.ones(Pn, 1, **kw), torch.zeros(1, M2.K, **kw),
                eps=torch.zeros(Pn, 1, **kw), chunk=512)
    tf = TomoFit(fit2, pr, V, E_true.clone(), weights=wts)
    assert tf.anisotropic
    tf.sync()
    with torch.no_grad():
        fit2.T = torch.cat([fit2.mean(s_)[0] for s_ in fit2.chunks()])                # noiseless data at the truth
    tf.E = torch.zeros_like(E_true); tf.sync()
    l0 = tf.data_loss(); tf.volume_step(); l1 = tf.data_loss()
    assert l1 < l0


# Host storage matches device storage on a GPU to this relative tolerance. Sums by atomic additions vary run to run.
GPU_RTOL = 1e-8
# float32 data match to this relative tolerance. One unit in the last place of a float32 is about 1.2e-7.
FLOAT32_RTOL = 1e-6


def _shared_open_beam_data(ref):
    """Counts for the small fit ref (_small_fit_cubic4) with one noisy open beam shared by every view, drawn as in
    test_shared_open_beam. Returns the measured transmission T and (the open beam (5 pixels, bins), each ray's pixel,
    each ray's exposure), the arguments of FitB.set_shared_open_beam."""
    P, K = ref.T.shape
    pix = torch.arange(P) // 8; gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        m_true = torch.cat([ref.mean(s)[0] for s in ref.chunks()])
    b_true = 50.0 * torch.ones(5, K, dtype=torch.float64)
    s_ = 0.9 + 0.2 * torch.rand(P, generator=gen, dtype=torch.float64)
    O = torch.poisson(b_true, generator=gen).clamp_min(1.0)
    S = torch.poisson(s_[:, None] * b_true[pix] * m_true, generator=gen)
    return (S / (s_[:, None] * O[pix])).to(ref.T.dtype), (O, pix, s_)


def _storage_fit(ref, T, device, shared=None, **storage):
    """A copy of the small cubic4 fit ref (_small_fit_cubic4) that computes on device, with the transmission T and the
    FitB storage options in storage. shared: the arguments of FitB.set_shared_open_beam, or None for none."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    M = Hybrid2(np.linspace(2.0, 4.4, 300), np.full(300, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))],
                smooth_degree=5, n_contam=1, device=device, dtype=torch.float64, shape_model='cubic4')
    M.view_dirs = ref.M.view_dirs
    fit = FitB(M, T, ref.g.to(device), ref.W.to(device), ref.f.to(device), eps=ref.eps.to(device), chunk=16, **storage)
    fit.ray_view = ref.ray_view.to(device)
    if shared is not None:
        fit.set_shared_open_beam(*shared)
    return fit


def _storage_twins(device, shared):
    """Two copies of the small cubic4 fit (_small_fit_cubic4) that compute on device: one with the default storage of
    the transmission, and one with t_storage 'host' (FitB). With shared, both use one noisy open beam shared by every
    view (_shared_open_beam_data)."""
    ref = _small_fit_cubic4()
    T, ob = _shared_open_beam_data(ref) if shared else (ref.T, None)
    return (_speedups_off(_storage_fit(ref, T.clone().to(device), device, ob, t_storage='device')),
            _speedups_off(_storage_fit(ref, T.clone(), device, ob, t_storage='host', t_prefetch=False)))


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_host_storage_matches_device(device, shared):
    """FitB with t_storage 'host' keeps T in host memory (pinned for a GPU) and gives the numbers of the default storage:
    the data loss, every chunk of T, the ray and global systems, and pixel, contamination and global steps. They agree
    bitwise on the CPU, and to GPU_RTOL on a GPU. With a shared open beam, T_chunk gives the bound's transmission that
    refresh_open_beam builds in the default storage, and the exact deviance agrees too."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('no GPU')
    a, b = _storage_twins(device, shared)
    exact = device == 'cpu'

    def same(x, y):
        if not torch.is_tensor(x):
            return x == y if exact else abs(x - y) <= GPU_RTOL * abs(y)
        if exact:
            return torch.equal(x, y)
        return torch.allclose(x, y, rtol=GPU_RTOL, atol=GPU_RTOL * float(y.abs().max()))
    assert b.T.device.type == 'cpu' and b.device == a.device and b.P == a.P
    if device == 'cuda':
        assert b.t_pinned and b.T.is_pinned()
    assert same(b.data_loss(), a.data_loss())
    ca, cb = a._state(a.g), b._state(b.g)
    for sl in a.chunks():
        assert b.T_chunk(sl).device == a.T_chunk(sl).device and same(b.T_chunk(sl), a.T_chunk(sl))
        (Ga, Fa), (Gb, Fb) = a.ray_system(sl, ca), b.ray_system(sl, cb)
        assert same(Gb, Ga) and same(Fb, Fa)
    M = a.M
    idx = [i for i in range(M.n_global) if not (M.res_sl[0] <= i < M.res_sl[1]) and float(M.ub[i]) > float(M.lb[i])]
    (ga, Fga), (gb, Fgb) = a.global_system(idx), b.global_system(idx)
    assert same(gb, ga) and same(Fgb, Fga)
    for x in (a, b):
        x.build_atoms(); x.contamination_newton = True
        x.contamination_atoms_step(); x.pixel_step(); x.global_step(1e-3)
    for key in ('g', 'W', 'eps', 'c', 'f'):
        assert same(getattr(b, key), getattr(a, key)), key
    assert same(b.data_loss(), a.data_loss())
    if shared:
        assert same(b.open_beam_deviance(), a.open_beam_deviance())
        for sl in a.chunks():                                       # the bound moved with the state, in both storages
            assert same(b.T_chunk(sl), a.T_chunk(sl))


def test_noise_with_host_data():
    """measure_noise and normalize_views on a T in host memory, with device set to a GPU, give what they give on the
    same T kept on that GPU: the air rays, the dispersion, the inflation, the view levels and the normalized T."""
    if not torch.cuda.is_available():
        pytest.skip('no GPU')
    from mbirtorch.hsnt.strain.noise import measure_noise, normalize_views
    images, side, K = 12, 16, 200
    g = torch.Generator().manual_seed(3)
    yy, xx = np.meshgrid(np.arange(side) - side / 2, np.arange(side) - side / 2, indexing='ij')
    disk = torch.as_tensor(np.broadcast_to(yy ** 2 + xx ** 2 < 16, (images, side, side)).reshape(-1))
    m = torch.where(disk, 0.6, 1.0).double()[:, None].expand(-1, K)
    gain = 1.0 + 0.004 * torch.randn(images, 2, 2, generator=g, dtype=torch.float64)
    pix = gain.repeat_interleave(side // 2, 1).repeat_interleave(side // 2, 2).reshape(-1)
    T = (m * pix[:, None] + 0.02 * torch.randn(images * side * side, K, generator=g, dtype=torch.float64)).float()
    d = np.full(K, 2500.0)
    shape = (images, side, side)
    on_gpu = measure_noise(T.cuda(), d, shape, tiles=(2, 2))
    on_host = measure_noise(T.clone().pin_memory(), d, shape, tiles=(2, 2), device='cuda')
    assert on_host['air_rays'] == on_gpu['air_rays'] and on_host['air_rays_per_view'] == on_gpu['air_rays_per_view']
    for key in ('dispersion', 'inflation'):
        assert abs(on_host[key] - on_gpu[key]) <= GPU_RTOL * abs(on_gpu[key]), key
    Tg, Th = T.cuda(), T.clone().pin_memory()
    lv_g, flag_g = normalize_views(Tg, shape, tiles=(2, 2))
    lv_h, flag_h = normalize_views(Th, shape, tiles=(2, 2), device='cuda')
    assert Th.device.type == 'cpu' and flag_h == flag_g and np.allclose(lv_h, lv_g, rtol=GPU_RTOL, atol=0)
    assert torch.allclose(Th, Tg.cpu(), rtol=FLOAT32_RTOL, atol=0)


# The noise measured on host data matches the noise of the same data on the GPU to this relative tolerance.
NOISE_RTOL = 1e-12


@pytest.mark.parametrize('noise', [None, 'measure'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_warmup_host_storage(device, noise):
    """warmup with t_storage 'host' gives a TomoFit whose fit keeps T in host memory, and the same start as the
    default storage. The test uses a ray mask. On a GPU, the test gives T in pinned host memory (pinned_empty). The
    fit then keeps that tensor itself, and the noise measurement and the factorization read it from the host. The
    default storage keeps the given T itself too. With noise 'measure', both storages measure the same dispersion and
    inflation. The factorization and the fit's sums on a GPU vary in the last bits from run to run, so the check of
    the start allows for rounding. The phantom leaves air around the sample, which the noise measurement needs."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('no GPU')
    from mbirtorch.hsnt.strain.model import pinned_empty
    from mbirtorch.hsnt.strain.tomo import Projector, warmup
    from mbirtorch.hsnt.strain.bragg import BraggSpectrum
    n = 12; th = np.linspace(0, np.pi, 9, endpoint=False)
    proj = Projector(n, th, 2, 0.0, False)
    lam = np.linspace(1.5, 5.0, 500)
    mats = [('fcc', 3.524), ('bcc', 2.8665)]
    mu = []
    for lt, a in mats:
        m = BraggSpectrum(lam, lattice=lt, a_range=(a * 0.999, a * 1.001), resolution=(0.002, 0.0))
        p_ = torch.zeros(m.n_params, dtype=torch.float64); p_[0] = a; p_[1:1 + m.n_edges] = 1.0; p_[1 + m.n_edges] = 0.3
        mu.append(m(p_).detach().numpy())
    mu = 0.08 * np.stack(mu) / np.stack(mu).mean(1, keepdims=True)
    V = torch.zeros(n, n, n, 2); V[4:6, 4:8, 4:8, 0] = 1; V[6:8, 4:8, 4:8, 1] = 1       # air 4 voxels on every side
    # The projector returns its rays on its own device, which is a GPU when there is one. The fit must compute on that
    # device, so a case that asks for another device cannot run on this machine.
    rays = proj.fwd(V)
    if rays.device.type != device:
        pytest.skip(f'the projector computes on {rays.device.type}, not on {device}')
    torch.manual_seed(0)
    # The counts are drawn by the seeded CPU generator on every machine. Ta and Tb below put T where each case asks.
    T = (torch.poisson(1e5 * torch.exp(-(rays.cpu().double() @ torch.as_tensor(mu)))) / 1e5).float()
    ray_mask = torch.ones(T.shape[0], dtype=torch.bool); ray_mask[:n] = False         # one detector row of view 0
    Ta = T.clone().to(device)                                       # the default storage's T, on the fit's device
    if device == 'cuda':                                            # the host storage's T, pinned on a GPU
        Tb = pinned_empty(T.shape, T.dtype, 'cuda'); Tb.copy_(T)
        assert Tb.is_pinned()
    else:
        Tb = T.clone()
    out = []
    for storage, Tin in (('device', Ta), ('host', Tb)):
        torch.manual_seed(0)
        out.append(warmup(Tin, lam, np.full(lam.size, 1e5), [lt for lt, _ in mats], [a for _, a in mats], proj,
                          device=device, init_spectra=list(mu), d0_from_data=True, noise=noise, normalize=False,
                          ray_mask=ray_mask, t_storage=storage, t_prefetch=False))
    (a, ia), (b, ib) = out
    assert a.fit.T is Ta and b.fit.T is Tb                          # neither storage made a copy of T
    assert b.fit.t_storage == 'host' and b.fit.T.device.type == 'cpu' and b.fit.P == a.fit.P
    assert b.fit.device == a.fit.device == torch.device(Ta.device)
    if device == 'cuda':
        assert b.fit.t_pinned
    if noise == 'measure':
        assert ib['noise']['air_rays'] == ia['noise']['air_rays'] > 0
        for key in ('dispersion', 'inflation'):
            assert abs(ib['noise'][key] - ia['noise'][key]) <= NOISE_RTOL * abs(ia['noise'][key]), key
        assert torch.allclose(b.fit.M.d, a.fit.M.d, rtol=NOISE_RTOL, atol=0)
    assert torch.allclose(b.V, a.V, rtol=1e-4, atol=1e-6 * float(a.V.abs().max()))
    assert torch.allclose(b.fit.g, a.fit.g, rtol=1e-4, atol=1e-6 * float(a.fit.g.abs().max()))
    assert abs(b.fit.data_loss() - a.fit.data_loss()) <= 1e-5 * abs(a.fit.data_loss())


# The tensors in test_pinned_empty have this many rows, which is not a multiple of a page's worth of rows.
PINNED_TEST_ROWS = 100003


def test_pinned_empty():
    """pinned_empty gives a host tensor of the requested shape and dtype. On a GPU, the tensor and its views are pinned,
    and copies to the GPU and back keep the values. After the tensor is freed, new tensors can be pinned again, so the
    freed memory does not stay registered. Without CUDA, the tensor is pageable."""
    import gc
    from mbirtorch.hsnt.strain.model import pinned_empty
    if not torch.cuda.is_available():
        t = pinned_empty((3, 5), torch.float32)
        assert t.shape == (3, 5) and t.dtype == torch.float32 and not t.is_pinned()
        return
    for _ in range(2):                                              # the second round pins memory again after a free
        for dtype in (torch.float32, torch.float16, torch.float64):
            t = pinned_empty((PINNED_TEST_ROWS, 7), dtype, 'cuda')
            assert t.shape == (PINNED_TEST_ROWS, 7) and t.dtype == dtype and t.is_contiguous()
            assert t.is_pinned() and t[5:].is_pinned()
            t.copy_((torch.arange(t.numel(), dtype=torch.float64) % 1000).reshape(t.shape))
            g = t.to('cuda', non_blocking=True) * 2
            back = pinned_empty(t.shape, dtype, 'cuda')
            back.copy_(g, non_blocking=True)
            torch.cuda.synchronize()
            assert back.is_pinned() and torch.equal(back, t * 2)
            del t, g, back
            gc.collect()
    assert pinned_empty((0, 7), torch.float32, 'cuda').shape == (0, 7)


# float16 storage may move the small fit's deviance by at most this many noise units, sqrt(N / 2).
HALF_NOISE_UNITS = 0.1


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('storage', ['device', 'host'])
def test_half_storage(storage, shared):
    """FitB with t_dtype 'float16' stores T in float16 (on the model's device, or in host memory for t_storage 'host')
    and gives each chunk as the float16-rounded T in the model's dtype. With a shared open beam, each chunk is the
    rounded measured T times the bound's ratio, within float16's rounding of the default storage's. At the small fit's
    counts (50 per bin), the deviance moves by far less than one noise unit. The logged relative rounding is below
    float16's half unit, 2^-11. A T with a value above HALF_MAX or below -HALF_MAX is refused."""
    from mbirtorch.hsnt.strain.model import HALF_MAX
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ref = _small_fit_cubic4()
    T, ob = _shared_open_beam_data(ref) if shared else (ref.T, None)
    a = _storage_fit(ref, T.clone().to(device), device, ob, t_storage='device')
    b = _storage_fit(ref, T.clone(), device, ob, t_storage=storage, t_dtype='float16', t_prefetch=False)
    assert b.T.dtype == torch.float16 and b.T.device.type == ('cpu' if storage == 'host' else a.device.type)
    for sl in a.chunks():
        x = b.T_chunk(sl)
        assert x.dtype == a.M.dt and x.device == a.device
        if shared:
            assert torch.allclose(x, a.T_chunk(sl), rtol=2.0 ** -10, atol=0)
        else:
            assert torch.equal(x, a.T_chunk(sl).half().to(a.M.dt))
    assert abs(b.data_loss() - a.data_loss()) < HALF_NOISE_UNITS * math.sqrt(a.n_points / 2)
    info = b.t_half_info
    assert info['max'] == float(T.max()) and info['min'] == float(T.min())
    assert 0 < info['rms_relative_rounding'] < 2.0 ** -11
    assert info['median_counts_per_bin'] > 0 and info['tiny_share'] == 0.0
    for big in (2 * HALF_MAX, -2 * HALF_MAX):
        with pytest.raises(ValueError, match='float16 storage'):
            _storage_fit(ref, torch.full_like(ref.T, big), device, t_dtype='float16')


PREFETCH_TEST_RAYS = 20000   # The larger transmission in test_host_prefetch has this many rays.
PREFETCH_TEST_CHUNK = 1024   # Its chunks have this many rays.


def test_host_prefetch():
    """With t_prefetch, host storage gives every chunk of T exactly as stored: a chunk prefetched by the call before
    (chunks in order, the next pass's first chunk), a mispredicted one (smaller chunks, a repeated chunk, the reverse
    order), a chunk changed in place after its prefetch, and a chunk after T is replaced. A fit with prefetching takes
    the same steps as one without, to GPU_RTOL."""
    if not torch.cuda.is_available():
        pytest.skip('no GPU')
    from mbirtorch.hsnt.strain.model import FitB
    ref = _small_fit_cubic4()
    a = _storage_fit(ref, ref.T.clone(), 'cuda', t_storage='host', t_prefetch=False)
    b = _storage_fit(ref, ref.T.clone(), 'cuda', t_storage='host', t_prefetch=True)
    assert b.t_prefetch and not a.t_prefetch
    P, half = b.P, max(1, b.chunk // 2)
    order = (b.chunks() + b.chunks() + [slice(i, min(i + half, P)) for i in range(0, P, half)]
             + [b.chunks()[1], b.chunks()[1]] + b.chunks()[::-1])
    for sl in order:
        x = b.T_chunk(sl)
        assert x.device == b.device and torch.equal(x.cpu(), ref.T[sl]), sl
    s0, s1 = b.chunks()[0], b.chunks()[1]
    b.T_chunk(s0)                                               # this call prefetches the chunk s1
    torch.cuda.synchronize()
    b.T[s1] *= 3                                                # an in-place change after the prefetch
    assert torch.equal(b.T_chunk(s1).cpu(), 3 * ref.T[s1])
    b.T = ref.T.clone() * 2                                     # a new T: an earlier prefetch must not be used
    for sl in b.chunks():
        assert torch.equal(b.T_chunk(sl).cpu(), 2 * ref.T[sl]), sl
    b.T = ref.T.clone()
    for x in (a, b):
        x.build_atoms(); x.contamination_newton = True
        x.contamination_atoms_step(); x.pixel_step(); x.global_step(1e-3)
    for key in ('g', 'W', 'eps', 'c', 'f'):
        y, z = getattr(b, key), getattr(a, key)
        assert torch.allclose(y, z, rtol=GPU_RTOL, atol=GPU_RTOL * float(z.abs().max())), key
    # a larger T, with chunks read in order twice while the main stream is kept busy between them
    Tbig = torch.rand(PREFETCH_TEST_RAYS, b.M.K, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    c = FitB(b.M, Tbig, b.g, torch.ones(PREFETCH_TEST_RAYS, 1, dtype=torch.float64, device='cuda'), b.f,
             chunk=PREFETCH_TEST_CHUNK, t_storage='host', t_prefetch=True)
    busy = torch.randn(PREFETCH_TEST_CHUNK, PREFETCH_TEST_CHUNK, device='cuda')
    got = []
    for sl in c.chunks() * 2:
        got.append(c.T_chunk(sl) * 1.0)                         # a computation on the chunk, on the main stream
        busy = torch.tanh(busy @ busy)
    torch.cuda.synchronize()
    assert torch.equal(torch.cat(got).cpu(), torch.cat([Tbig, Tbig]))


def _lean_fit(case, dtype, seed=7, device='cpu'):
    """Returns a small FitB for the lean_syncs checks. It has one bcc material, three chunks of rays and eight views.
    Each view has five rays, so that chunks see different views. The rays have strains with a spread along the beam,
    and the counts come from the model. case is 'powder', 'views' (a global cubic4 texture), 'field' (a texture field),
    'anisotropic' (elastic anisotropy, a global texture and a free edge slope) or 'floor' (a global texture that brings
    view 0, and so the first chunk, to CUBIC4_FLOOR). The cases 'field' and 'anisotropic' have a nonzero edge slope t.
    device is where the fit's arrays live."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB, harmonics4, CUBIC4_FLOOR, P4_MIN
    lam = np.linspace(1.6, 4.4, 400)
    nb = 2 if case == 'field' else 1
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=5, n_contam=1,
                device=device, dtype=dtype, shape_model=None if case == 'powder' else 'cubic4', texture_basis=nb,
                edge_slope=case == 'anisotropic')
    gen = torch.Generator().manual_seed(seed)
    g = torch.zeros(M.n_global, dtype=torch.float64)
    sl = M.slices[0]; ne = sl['h'][1] - sl['h'][0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3 + 0.1 * torch.rand(ne, generator=gen, dtype=torch.float64)
    g[sl['c'][0]] = 0.2
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor([0.0, 0.001, 0.002, 0.0, 0.1, 0.02, 0.0, 0.0], dtype=torch.float64)
    if case in ('field', 'anisotropic'):
        g[sl['t'][0]:sl['t'][1]] = torch.linspace(-0.2, 0.3, ne, dtype=torch.float64)      # a nonzero edge slope
    d = torch.randn(8, 3, generator=gen, dtype=torch.float64); M.view_dirs = (d / d.norm(dim=1, keepdim=True)).numpy()
    if case != 'powder':
        g[sl['x'][0]:sl['x'][1]] = 0.05 * torch.randn(sl['x'][1] - sl['x'][0], generator=gen, dtype=torch.float64)
    if case == 'floor':
        # a texture whose term is -3 in view 0 and small in the other views: the 200 reflection (k4 = 0.4) of view 0
        # reaches the floor, so the first chunk takes the dense sum and the other chunks the windowed sum
        Y = harmonics4(torch.as_tensor(M.view_dirs))                                    # (8 views, 9)
        target = torch.tensor([-3.0, 0.1, 0.2, -0.1, 0.0, 0.1, -0.2, 0.3], dtype=torch.float64)
        c = torch.linalg.pinv(Y) @ target
        g[sl['x'][0]:sl['x'][1]] = c
        kA = M.mats[0]['k4'].cpu()[None, :] * (Y @ c)[:, None]
        hit = ((1 + torch.minimum(kA * P4_MIN, kA)) <= CUBIC4_FLOOR).any(1)
        assert hit.tolist() == [True] + [False] * 7
    P = 40
    W = 0.5 + torch.rand(P, 1, generator=gen, dtype=torch.float64)
    eps = 1e-3 * torch.randn(P, 2 if case == 'anisotropic' else 1, generator=gen, dtype=torch.float64)
    if case == 'anisotropic':
        M.set_dec(0, 0.8 + 0.4 * torch.rand(ne, generator=gen, dtype=torch.float64), 0.05 * torch.randn(ne, generator=gen, dtype=torch.float64))
    kw = dict(dtype=dtype, device=device)
    fit = _speedups_off(FitB(M, torch.ones(P, M.K, **kw), g.to(device), W.to(**kw), torch.zeros(1, M.K, **kw),
                             eps=eps.to(**kw), chunk=16))
    fit.ray_view = (torch.arange(P) // 5).to(device)
    fit.eps_sd = (3e-4 * torch.rand(P, 1, generator=gen, dtype=torch.float64)).to(**kw)
    if case == 'field':
        q = torch.rand(P, generator=gen, dtype=torch.float64)
        fit.tex_Q = [torch.stack([q, 1 - q], 1).to(**kw)]
    with torch.no_grad():
        m = torch.cat([fit.mean(s)[0] for s in fit.chunks()])
    fit.T = torch.poisson(50.0 * m.double().cpu(), generator=gen).to(**kw) / 50.0
    return fit


def _pixel_step_result(fit):
    """Runs one pixel_step and returns the new maps, the new strains and the number of window extents that the step
    computed (Hybrid2.window_extent). The fit's maps and strains are then set back to their values before the step."""
    M = fit.M
    W0, e0 = fit.W.clone(), fit.eps.clone()
    calls = []
    extent = M.window_extent
    M.window_extent = lambda *a, **k: calls.append(1) or extent(*a, **k)
    try:
        fit.pixel_step()
    finally:
        del M.window_extent
    out = (fit.W.clone(), fit.eps.clone(), len(calls))
    fit.W, fit.eps = W0, e0
    return out


def _lean_outputs(fit):
    """Returns the quantities that the lean_syncs option touches, at the fit's current state."""
    M = fit.M
    out = {}
    cache = fit._state(fit.g)
    out['mu'], out['band'], out['kappa'] = cache
    fit.use_memo = False
    loss = fit.data_loss()
    assert isinstance(loss, float)
    out['loss'] = torch.tensor(loss)
    eps2 = fit.eps * 1.3
    out['loss_eps'] = torch.tensor(fit.data_loss(eps=eps2))
    with torch.no_grad():
        out['mean'] = torch.cat([fit.mean(s, cache=cache)[0] for s in fit.chunks()])
        out['ray_loss'] = torch.cat([fit.ray_loss(s, fit.W[s], eps2[s], cache, eps_all=eps2) for s in fit.chunks()])
        G, F, dev = [], [], []
        for s in fit.chunks():
            g_, f_ = fit.ray_system(s, cache, deviance=dev)
            G.append(g_); F.append(f_)
    out['G'], out['F'], out['dev'] = torch.cat(G), torch.cat(F), torch.tensor(sum(float(v) for v in dev))
    idx = [i for i in range(M.n_global) if float(M.ub[i]) > float(M.lb[i]) and not (M.res_sl[0] <= i < M.res_sl[1])]
    out['gs_grad'], out['gs_F'] = fit.global_system(idx)
    out['gg'] = fit.global_grad(idx)
    full = [i for i in range(M.n_global) if float(M.ub[i]) > float(M.lb[i])]
    out['gs_grad_kernel'], out['gs_F_kernel'] = fit.global_system(full)
    # the contamination's passes: a trial loss, and the Pearson dispersion and thresholds of a step (state restored)
    fit.build_atoms()
    out['loss_c'] = torch.tensor(fit.loss_c_data(0.01 * torch.ones_like(fit.c)))
    f0, lam0, disp0 = fit.f.clone(), fit.lam1.clone(), fit.dispersion
    fit.contamination_step()
    assert isinstance(fit.dispersion, float)
    out['dispersion'], out['lam1'] = torch.tensor(fit.dispersion), fit.lam1.clone()
    fit.f, fit.lam1, fit.dispersion = f0, lam0, disp0
    g0 = fit.g.clone()
    with torch.no_grad():
        out['lattice'] = torch.tensor(fit.lattice_search(0, half_width=0.002))
    fit.g = g0
    return out


@pytest.mark.parametrize('case', ['powder', 'views', 'field', 'anisotropic', 'floor'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_lean_syncs_matches_default(case, dtype, monkeypatch):
    """FitB.lean_syncs (the band kept between passes, one window width and one cubic4 floor decision per pass) gives the
    default path's spectra, losses, ray systems, global systems, global gradient and lattice search, to rounding. A
    pass's window width holds every chunk's own window, in single and in double precision. With 'floor', some chunks
    take the dense sum and others the windowed sum, as on the default path. A pixel_step works out one window width for
    the whole step, and in double precision it gives the default path's maps and strains."""
    from mbirtorch.hsnt.strain import model
    from mbirtorch.hsnt.strain.model import WINDOW_BUCKET
    # The test checks the shared window plan, not the rule that limits its size.
    monkeypatch.setattr(model, 'WINDOW_PLAN_MAX_RATIO', math.inf)
    fit = _lean_fit(case, dtype)
    M = fit.M
    ref = _lean_outputs(fit)
    fit.lean_syncs = True
    got = _lean_outputs(fit)
    tol = 1e-12 if dtype == torch.float64 else 1e-6
    for k in ref:
        a, b = ref[k].double(), got[k].double()
        assert torch.allclose(a, b, rtol=tol, atol=tol * float(a.abs().max())), k
    # the pass's width against each chunk's own, with the strains in either precision
    widths = fit._window_widths(fit.g, fit.eps)
    assert all(w % WINDOW_BUCKET == 0 for w in widths)
    for s in fit.chunks():
        for e_ in (fit.eps[s], fit.eps[s].double()):
            lo, hi, wd = M.window_extent(fit.g, 0, e_, fit.eps_sd[s])
            assert int(M.window_width(lo, hi, wd)) <= widths[0]
    if case == 'floor':
        floors = [fit.window_plan(s, fit.g, fit.eps)[0].floor for s in fit.chunks()]
        assert floors == [True, False, False]
    # pixel_step writes each chunk's new strains into fit.eps. With lean_syncs, it still works out the window width once
    # for the whole step, which is one window extent per chunk and material. pixel_step fits the strain along the beam
    # alone, so the case with traces ('anisotropic') is left out.
    if case != 'anisotropic':
        fit.lean_syncs = False
        W_ref, e_ref, _ = _pixel_step_result(fit)
        fit.lean_syncs = True
        W_got, e_got, n_extents = _pixel_step_result(fit)
        assert n_extents == len(fit.chunks()) * M.R
        assert not torch.equal(e_ref, fit.eps)                                         # the step moved the strains
        if dtype == torch.float64:
            assert _close(W_got, W_ref, 1e-9) and _close(e_got, e_ref, 1e-9)
    # the band is built again only when the resolution kernel's values change
    built = []
    band = M.band
    M.band = lambda g_: built.append(1) or band(g_)
    fit.data_loss(); fit.data_loss()
    assert built == []
    fit.lean_syncs = False
    fit.data_loss()
    del M.band
    assert built == [1]


# test_lean_syncs_volume_step runs in float32, the model's default dtype. The variants' models agree to float32 rounding,
# and the forward projector's atomic adds also change them by rounding from one run to the next. Near the truth, these
# roundings leave the data loss uncertain by a few parts in a million. The deviance is computed in float64
# (model.deviance_terms), so its own rounding adds almost nothing to that.
VOLUME_STEP_LOSS_RTOL = 1e-5         # the trial data loss and the loss before the step, relative to the default path's
VOLUME_STEP_AFTER_RTOL = 1e-3        # the loss after one volume step, relative to the default path's loss before it


def test_lean_syncs_volume_step():
    """A tomographic fit with several chunks of rays and a strain spread along the beam, with FitB.lean_syncs, with
    FitB.closed_form_strain and with both: each gives the default path's data loss at a trial state (to rounding), and
    a volume step that lowers the objective as much."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    from mbirtorch.hsnt.strain.tomo import TomoFit
    pr = MultiProjector(N, MOUNTS, 2, channel_offsets=[0.0, 0.3, -0.2], ror_mask=False)
    dev = pr.fwd(torch.zeros(N, N, N, 2)).device
    lam = np.linspace(2.0, 4.4, 80)
    M = Hybrid2(lam, np.full_like(lam, 200.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=3, n_contam=1,
                device=dev)
    g = torch.zeros(M.n_global, dtype=torch.float64, device=dev); s = M.slices[0]
    g[s['a']] = 2.8665; g[s['h'][0]:s['h'][1]] = 0.3; g[s['c'][0]] = 0.2
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor([0.0, 0.001, 0.002, 0.0, 0.1, 0.02, 0.0, 0.0], dtype=torch.float64,
                                              device=dev)
    P = sum(pr.counts); kw = dict(device=dev)
    V = torch.zeros(N, N, N, 1, device=dev); V[3:9, 3:9, 3:9] = 0.05
    E_true = 1.5e-3 * torch.randn(N, N, N, 1, 1, generator=torch.Generator().manual_seed(6)).to(dev) * (V[..., None] > 0)

    def run(lean, closed=False, T=None):
        """Returns the trial data loss, the loss before one volume step and the loss after it, and the data. With T
        None, the data are noiseless data at the truth. A given T is used as the data."""
        fit = FitB(M, torch.ones(P, M.K, **kw), g, torch.ones(P, 1, **kw), torch.zeros(1, M.K, **kw),
                   eps=torch.zeros(P, 1, **kw), chunk=512)
        tf = _speedups_off(TomoFit(fit, pr, V.clone(), E_true.clone()))      # Each variant sets its options below.
        tf.sync(); tf.edge_spread = True; tf.refresh_spread()
        if T is None:
            with torch.no_grad():
                T = torch.cat([fit.mean(c)[0] for c in fit.chunks()])              # noiseless data at the truth
        fit.T = T.clone()
        fit.lean_syncs = lean; fit.closed_form_strain = closed
        tf.E = 0.5 * E_true; tf.sync(); tf.refresh_spread()                         # the spread at the start
        trial = tf.data_loss(V, 0.8 * E_true)
        l0 = tf.loss(); tf.volume_step(); l1 = tf.loss()                           # data + prior
        assert isinstance(trial, float)
        return (trial, l0, l1), T
    assert len(range(0, P, 512)) > 1                                                 # several chunks
    # Every variant fits the default run's data. The projector's atomic adds would otherwise give each run its own data,
    # and those data differ by rounding.
    ref, T = run(False)
    for name, lean, closed in (('lean_syncs', True, False), ('closed_form_strain', False, True),
                               ('lean_syncs and closed_form_strain', True, True)):
        got, _ = run(lean, closed, T)
        for k, what in ((0, 'trial data loss'), (1, 'loss before the volume step')):
            assert abs(got[k] - ref[k]) <= VOLUME_STEP_LOSS_RTOL * abs(ref[k]), \
                f'{name}: the {what} is {got[k]!r}, and the default path gives {ref[k]!r}'
        assert got[2] < got[1], f'{name}: the volume step raised the loss from {got[1]!r} to {got[2]!r}'
        assert abs(got[2] - ref[2]) <= VOLUME_STEP_AFTER_RTOL * abs(ref[1]), \
            f'{name}: the loss after the volume step is {got[2]!r}, and the default path gives {ref[2]!r}'


DEVIANCE_TEST_SEED = 12              # the seed of test_deviance_terms_float64's random draws
DEVIANCE_TEST_BINS = 4096            # the number of bins that the test draws
DEVIANCE_TEST_COUNTS = 200.0         # each bin's counts at a transmission of 1
DEVIANCE_TEST_T_RANGE = (0.3, 0.9)   # the range of the drawn transmissions
DEVIANCE_TEST_MISFIT = 1e-4          # the standard deviation of the model's relative difference from the data
DEVIANCE_TEST_RTOL = 1e-7            # the summed terms' tolerance, relative to their exact sum


def test_deviance_terms_float64():
    """deviance_terms computes its terms in float64 for a float32 model and float32 data. Here the model nearly equals
    the data, and each term is smaller than the float32 rounding of its parts. The summed terms equal a sum that has no
    such cancellation, which is d T (u - log(1 + u)) with u = (m - T) / T. Terms computed in float32 would carry
    rounding errors larger than the terms themselves."""
    from mbirtorch.hsnt.strain.model import deviance_terms
    gen = torch.Generator().manual_seed(DEVIANCE_TEST_SEED)
    lo, hi = DEVIANCE_TEST_T_RANGE
    T = (lo + (hi - lo) * torch.rand(DEVIANCE_TEST_BINS, generator=gen, dtype=torch.float64)).float()
    u = DEVIANCE_TEST_MISFIT * torch.randn(DEVIANCE_TEST_BINS, generator=gen, dtype=torch.float64)
    m = (T.double() * (1 + u)).float()
    d = torch.full((DEVIANCE_TEST_BINS,), DEVIANCE_TEST_COUNTS)
    got = deviance_terms(m, T, d)
    assert got.dtype == torch.float64
    # The reference starts from the difference of the float32 values, which is exact in float64.
    Td = T.double(); ud = (m.double() - Td) / Td
    ref = float((DEVIANCE_TEST_COUNTS * Td * (ud - torch.log1p(ud))).sum())
    assert abs(float(got.sum()) - ref) <= DEVIANCE_TEST_RTOL * ref, (float(got.sum()), ref)


def _close(got, ref, tol):
    """True when got equals ref to relative tolerance tol, with an absolute floor of tol times ref's largest value.
    It is a check for the closed-form strain derivatives."""
    a, b = got.double(), ref.double()
    return torch.allclose(a, b, rtol=tol, atol=tol * float(b.abs().max()))


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_closed_form_strain_matches_jvp(dtype):
    """Hybrid2.spectra_de gives the spectra and their forward-mode derivatives with respect to each ray's strain along
    the beam and its mean trace. The check has two materials (one with elastic anisotropy), a nonzero edge slope, a
    global cubic4 texture and a strain spread, and it runs with and without a window plan. ray_jacobian with
    FitB.closed_form_strain equals the default path, and a chunk with a ray at CUBIC4_FLOOR falls back to forward
    mode."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB, WindowPlan, harmonics4
    lam = np.linspace(1.6, 4.4, 400)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01)),
                                               ('fcc', (3.6149 * 0.99, 3.6149 * 1.01))],
                smooth_degree=5, n_contam=1, device='cpu', dtype=dtype, shape_model='cubic4')
    gen = torch.Generator().manual_seed(11)
    g = torch.zeros(M.n_global, dtype=torch.float64)
    for r, a0 in enumerate((2.8665, 3.6149)):
        sl = M.slices[r]; ne = sl['h'][1] - sl['h'][0]
        g[sl['a']] = a0; g[sl['h'][0]:sl['h'][1]] = 0.2 + 0.1 * torch.rand(ne, generator=gen, dtype=torch.float64)
        g[sl['c'][0]] = 0.1
        # a nonzero edge slope, of opposite signs in the two materials, so that the slope's terms are checked
        g[sl['t'][0]:sl['t'][1]] = (1 - 2 * r) * torch.linspace(-0.2, 0.3, ne, dtype=torch.float64)
        g[sl['x'][0]:sl['x'][1]] = 0.05 * torch.randn(9, generator=gen, dtype=torch.float64)
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor([0.0, 0.001, 0.002, 0.0, 0.1, 0.02, 0.0, 0.0], dtype=torch.float64)
    d = torch.randn(8, 3, generator=gen, dtype=torch.float64); M.view_dirs = (d / d.norm(dim=1, keepdim=True)).numpy()
    ne0 = M.slices[0]['h'][1] - M.slices[0]['h'][0]
    M.set_dec(0, 0.8 + 0.4 * torch.rand(ne0, generator=gen, dtype=torch.float64),
              0.05 * torch.randn(ne0, generator=gen, dtype=torch.float64))
    P = 40
    eps = (1e-3 * torch.randn(P, 4, generator=gen, dtype=torch.float64)).to(dtype)    # strains (R), mean traces (R)
    sd = (3e-4 * torch.rand(P, 2, generator=gen, dtype=torch.float64)).to(dtype)
    views = torch.arange(P) // 5
    tol = 1e-9 if dtype == torch.float64 else 1e-4
    for plan in (None, [WindowPlan(64, False), WindowPlan(64, False)]):
        for r in range(2):
            mu, d_eps, d_tau = M.spectra_de(g, eps, r, views, sd, plan=plan)
            ok = {'mu': _close(mu, M.spectra(g, eps, views, sd=sd)[:, r], tol)}
            for col, got in ((r, d_eps), (2 + r, d_tau)):
                tg = torch.zeros(eps.shape, dtype=torch.float64); tg[:, col] = 1
                _, dm = torch.func.jvp(lambda e_: M.spectra(g, e_, views, sd=sd)[:, r], (eps.double(),), (tg,))
                ok[f'column {col}'] = _close(got, dm, tol)
            assert all(ok.values()), (r, ok)
        assert M.spectra_de(g, eps[:, :2], 0, views, sd, plan=plan)[2] is None              # no trace columns
    # ray_jacobian: view 0's texture brings material 0 to the floor, so the first chunk falls back to forward mode
    Y = harmonics4(torch.as_tensor(M.view_dirs))
    x0, x1 = M.slices[0]['x']
    g[x0:x1] = torch.linalg.pinv(Y) @ torch.tensor([-3.0, 0.1, 0.2, -0.1, 0.0, 0.1, -0.2, 0.3], dtype=torch.float64)
    W = (0.5 + torch.rand(P, 2, generator=gen, dtype=torch.float64)).to(dtype)
    fit = FitB(M, torch.ones(P, M.K, dtype=dtype), g, W, torch.zeros(1, M.K, dtype=dtype), eps=eps, chunk=16)
    fit.ray_view = views; fit.eps_sd = sd
    with torch.no_grad():
        m = torch.cat([fit.mean(s)[0] for s in fit.chunks()])
    fit.T = torch.poisson(50.0 * m.double(), generator=gen).to(dtype) / 50.0
    for lean in (False, True):
        fit.lean_syncs = lean; fit.closed_form_strain = False
        fit._warned = set()                       # The fallback warns once per fit, so each round starts afresh.
        cache = fit._state(fit.g)
        with torch.no_grad():
            ref = [fit.ray_jacobian(s, cache) for s in fit.chunks()]
            fit.closed_form_strain = True
            with pytest.warns(FallbackWarning, match='texture'):
                got = [fit.ray_jacobian(s, cache) for s in fit.chunks()]
            assert [fit._ray_jacobian_closed(s, cache) is None for s in fit.chunks()] == [True, False, False]
        for (J0, m0), (J1, m1) in zip(ref, got):
            assert J1.shape == J0.shape == (J0.shape[0], M.K, 6)
            assert _close(J1, J0, tol) and _close(m1, m0, tol)


@pytest.mark.parametrize('case', ['powder', 'views', 'field', 'anisotropic'])
def test_closed_form_strain_ray_jacobian(case):
    """ray_jacobian with FitB.closed_form_strain equals the default path for a powder, a global texture, a texture
    field and elastic anisotropy, with and without lean_syncs. The texture field and the anisotropic case have a
    nonzero edge slope."""
    fit = _lean_fit(case, torch.float64)
    for lean in (False, True):
        fit.lean_syncs = lean; fit.closed_form_strain = False
        cache = fit._state(fit.g)
        with torch.no_grad():
            ref = [fit.ray_jacobian(s, cache) for s in fit.chunks()]
            fit.closed_form_strain = True
            got = [fit.ray_jacobian(s, cache) for s in fit.chunks()]
        for (J0, m0), (J1, m1) in zip(ref, got):
            assert _close(J1, J0, 1e-9) and _close(m1, m0, 1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='the check needs a GPU')
@pytest.mark.parametrize('case', ['powder', 'views', 'field', 'anisotropic'])
def test_lean_syncs_no_gpu_waits(case, monkeypatch):
    """With FitB.lean_syncs, a pass of the data loss over the chunks of rays does not wait for the GPU once the pass's
    window widths and cubic4 floor decisions are known. torch.cuda.set_sync_debug_mode makes any wait raise an error.
    The pass gives the same loss as data_loss."""
    from mbirtorch.hsnt.strain import model
    # The test checks the shared window plan, not the rule that limits its size.
    monkeypatch.setattr(model, 'WINDOW_PLAN_MAX_RATIO', math.inf)
    fit = _lean_fit(case, torch.float32, device='cuda')
    fit.lean_syncs = True; fit.use_memo = False
    cache = fit._state(fit.g)
    ref = fit.data_loss()                       # this pass works out the plan and compiles the windowed sum
    fit.data_loss()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        with torch.no_grad():
            tot = 0.0
            for s in fit.chunks():
                m = fit.mean(s, cache=cache)[0]
                tot += fit._chunk_total(fit.deviance_terms(s, m, fit.T[s]).sum(dtype=torch.float64))
    finally:
        torch.cuda.set_sync_debug_mode('default')
    assert abs(float(tot) - ref) <= 1e-5 * abs(ref)


def test_prolong():
    """prolong (coarse-to-fine): factor 1 returns its input unchanged, which is the default path. Mode 'constant'
    copies each coarse voxel into its block of fine voxels. Mode 'trilinear' reproduces a linear field at the fine
    voxel centres between the outermost coarse centres. The weighted form keeps a strain that is constant over a
    material's voxels constant up to the faces, where the plain form dilutes it. The weighted form also equals its
    definition, computed from whole-tensor interpolations, where some fine voxels hold no material."""
    from mbirtorch.hsnt.strain.tomo import prolong, PROLONG_WEIGHT_MIN
    n, f = 6, 2
    x = torch.rand(n, n, n, 2, 3, generator=torch.Generator().manual_seed(0))
    for mode in ('constant', 'trilinear'):
        assert torch.equal(prolong(x, 1, mode), x)
        assert torch.equal(prolong(x, 1, mode, weight=torch.rand(n, n, n, 2, 1)), x)
    xc = prolong(x, f, 'constant')
    assert xc.shape == (f * n, f * n, f * n, 2, 3)
    blocks = xc.reshape(n, f, n, f, n, f, 2, 3).permute(0, 2, 4, 1, 3, 5, 6, 7).reshape(n, n, n, f ** 3, 2, 3)
    assert torch.equal(blocks, x[:, :, :, None].expand(n, n, n, f ** 3, 2, 3))
    # a linear field at the coarse voxel centres (i + 0.5) / n, against its values at the fine centres (j + 0.5) / (f n)
    cc = (torch.arange(n, dtype=torch.float64) + 0.5) / n; cf = (torch.arange(f * n, dtype=torch.float64) + 0.5) / (f * n)
    lin = lambda c: 0.3 + 1.1 * c[:, None, None] - 0.7 * c[None, :, None] + 2.0 * c[None, None, :]
    xt = prolong(lin(cc)[..., None], f, 'trilinear')[..., 0]
    inner = slice(f // 2, f * n - f // 2)
    assert torch.allclose(xt[inner, inner, inner], lin(cf)[inner, inner, inner], rtol=0, atol=1e-12)
    # a constant strain in a cube of material, with zero strain and no material outside
    V = torch.zeros(n, n, n, 1); V[1:5, 1:5, 1:5] = 1.0
    E = torch.zeros(n, n, n, 1, 6); E[1:5, 1:5, 1:5] = 1e-3
    own = prolong(V, f, 'trilinear') > 0
    Ew = prolong(E, f, 'trilinear', weight=V[..., None])
    assert torch.allclose(Ew[own], torch.full_like(Ew[own], 1e-3), rtol=1e-5, atol=0)
    assert float(prolong(E, f, 'trilinear')[own].min()) < 0.9e-3
    # the weighted form against its definition, with an empty slab of coarse voxels where the plain form is used
    gen = torch.Generator().manual_seed(1)
    xd = torch.rand(n, n, n, 2, 3, generator=gen, dtype=torch.float64)
    wd = torch.rand(n, n, n, 2, 1, generator=gen, dtype=torch.float64); wd[:2] = 0
    for mode in ('constant', 'trilinear'):
        den = prolong(wd, f, mode)
        ref = torch.where(den > PROLONG_WEIGHT_MIN, prolong(wd * xd, f, mode) / den.clamp_min(PROLONG_WEIGHT_MIN),
                          prolong(xd, f, mode))
        assert bool((den <= PROLONG_WEIGHT_MIN).any())
        assert torch.allclose(prolong(xd, f, mode, weight=wd), ref, rtol=1e-12, atol=0)
    assert prolong(torch.zeros(0, n, n, 2), f).shape == (0, f * n, f * n, 2)
    for bad in (dict(factor=1.5), dict(factor=f, mode='cubic'), dict(factor=f, weight=V),
                dict(factor=f, weight=torch.ones(n, n, n, 2, 1))):
        with pytest.raises(ValueError):
            prolong(E, **bad)


def test_prolong_projections():
    """A coarse volume copied into blocks (prolong 'constant') and projected on a detector f times finer, then averaged
    over f x f detector blocks, gives f times the coarse projection, because each projector measures lengths in its own
    pixels. The match is exact at views along the grid's axes (0 and 90 degrees), where each voxel's shadow covers
    whole detector pixels. The coarse-to-fine continuation relies on this when it divides each spectrum by f."""
    from mbirtorch.hsnt.strain.tomo import prolong
    n, f = 6, 2
    ang = np.array([0.0, np.pi / 2])
    series = [(((0, 1, 2), (1, 1, 1), (0.0, 0.0, 0.0)), ang), (((1, 2, 0), (1, 1, 1), (0.0, 0.0, 0.0)), ang)]
    cp = MultiProjector(n, series, 1, channel_offsets=[0.0, 0.0], ror_mask=False)
    fp = MultiProjector(f * n, series, 1, channel_offsets=[0.0, 0.0], ror_mask=False)
    xc = torch.rand(n, n, n, 1, generator=torch.Generator().manual_seed(3))
    pc = cp.fwd(xc).cpu()
    pf = fp.fwd(prolong(xc, f, 'constant')).cpu()
    pb = pf.reshape(cp.V, n, f, n, f, 1).mean((2, 4)).reshape(-1, 1)
    assert torch.allclose(pb, f * pc, rtol=1e-5, atol=1e-5 * float(pc.abs().max()))


# ---- FitB.lean_passes (speedup round 2): the passes over the rays and the projections whose results the fit has
LEAN_N, LEAN_VIEWS, LEAN_RAYS_PER_VIEW = 8, 6, 64   # the lean_passes fit: volume side, views and rays per view
LEAN_CHUNKS = 3                                     # ... the chunks of rays
LEAN_BINS = 80                                      # ... the wavelength bins
LEAN_DOSE = 200.0                                   # ... the counts per bin at transmission 1
LEAN_CG_ITERS = 10                                  # ... the conjugate-gradient iterations of each volume step
LEAN_ITERS = 3                                      # the iterations that the comparison runs
# The comparison runs the default path twice. Where the two runs are bitwise equal (a deterministic CPU), the lean path
# must be bitwise equal too. Elsewhere its difference from the default may be at most this many times the difference
# between the two default runs, which is the default path's own rounding noise.
LEAN_NOISE_MARGIN = 10.0
# On a GPU, the mount and view-offset steps' loss may also differ by this fraction, and their mounts and offsets by this
# many voxels, degrees or detector channels: the projector's atomic adds change float32 rounding from run to run.
LEAN_GPU_TOL = 1e-4
MOUNT_TEST_SHIFT = 0.3          # _mount_fit: the data's shift of series 1's mount (voxels) ...
VIEW_TEST_OFFSET = 0.3          # ... and of one view's own channel offset in series 0 (detector channels)
VIEW_TEST_INDEX = 2             # ... that view's index
# _mount_fit gives series 2 this mount shift (voxels), whose components and their MOUNT_FD probes stay off the
# half-voxel rounding ties of MultiProjector._split.
MOUNT_TEST_SERIES2_SHIFT = (3.0, -0.4, 0.25)


class _LeanProjector:
    """A test projector on the CPU for the lean_passes checks. Its rays are view-major, views x rays_per_view of them,
    and each ray is a weighted sum of RAY_VOXELS random voxels in any number of channels. back is the exact adjoint of
    fwd. The projector has the attributes that tomo._geometry_key reads."""
    RAY_VOXELS = 24

    def __init__(self, n, views, rays_per_view, gen):
        self.n, self.V, self.rays_per_view = n, views, rays_per_view
        P = views * rays_per_view
        self.idx = torch.randint(0, n ** 3, (P, self.RAY_VOXELS), generator=gen)
        self.wt = torch.rand(P, self.RAY_VOXELS, generator=gen)
        self.kind, self.angles, self.channel_offset = 'plain', np.zeros(views), 0.0
        self.view_offsets, self.row_offset, self.ror_mask = None, 0.0, False
        self.Q = None              # This is the channel count of TomoFit's projector. _lean_tomo_fit sets it.

    def with_q(self, Q):
        return self

    def fwd(self, vols):
        return (vols.reshape(-1, vols.shape[-1])[self.idx] * self.wt.to(vols.dtype)[..., None]).sum(1)

    def back(self, rays):
        out = torch.zeros(self.n ** 3, rays.shape[-1], dtype=rays.dtype)
        out.index_add_(0, self.idx.reshape(-1),
                       (self.wt.to(rays.dtype)[..., None] * rays[:, None, :]).reshape(-1, rays.shape[-1]))
        return out.reshape(self.n, self.n, self.n, -1)


def _lean_tomo_fit(texture, normalize, t_storage='device', speedups=False):
    """A small tomographic fit on the CPU for the lean_passes checks: one bcc material in a cube, the strain tensor with
    the equilibrium prior, the strain's spread along the rays, two contamination components and, with texture, the
    per-voxel cubic4 texture. Its data are Poisson counts from a true state, and the fit starts from another state.
    t_storage is passed to FitB. With speedups False, the speedup options are off from the start (_speedups_off), and
    with True they keep their defaults."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB, harmonics4
    from mbirtorch.hsnt.strain.tomo import TomoFit, STRAIN_SMOOTH2_SD
    n, nv, rpv = LEAN_N, LEAN_VIEWS, LEAN_RAYS_PER_VIEW
    gen = torch.Generator().manual_seed(23)
    proj = _LeanProjector(n, nv, rpv, gen)
    lam = np.linspace(2.0, 4.4, LEAN_BINS)
    M = Hybrid2(lam, np.full_like(lam, LEAN_DOSE), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=3,
                n_contam=2, device='cpu', shape_model='cubic4' if texture else None, texture_basis=9 if texture else 1)
    g = torch.zeros(M.n_global, dtype=torch.float64)
    sl = M.slices[0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3; g[sl['c'][0]] = 0.2
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor([0.0, 0.001, 0.002, 0.0, 0.1, 0.02, 0.0, 0.0], dtype=torch.float64)
    g[M.kap_sl[0]:M.kap_sl[1]] = torch.tensor([0.05, 0.2], dtype=torch.float64)
    d = torch.randn(nv, 3, generator=gen, dtype=torch.float64)
    d = d / d.norm(dim=1, keepdim=True)
    w = torch.stack([d[:, 0] ** 2, d[:, 1] ** 2, d[:, 2] ** 2, 2 * d[:, 0] * d[:, 1], 2 * d[:, 1] * d[:, 2],
                     2 * d[:, 0] * d[:, 2]], 1).float()
    P = nv * rpv
    proj.Q = M.R * (1 + w.shape[1])              # TomoFit's projector carries the volume fractions and the strain.
    f = torch.zeros(M.J, M.K); f[0, 20:40] = 0.03; f[1, 50:70] = 0.02
    fit = FitB(M, torch.ones(P, M.K), g, torch.ones(P, 1), f, eps=torch.zeros(P, 1), chunk=P // LEAN_CHUNKS,
               t_storage=t_storage)
    if not speedups:
        _speedups_off(fit)
    if texture:
        M.view_dirs = d.numpy()
        fit.ray_view = torch.arange(nv).repeat_interleave(rpv)
    V = torch.zeros(n, n, n, 1); V[2:6, 2:6, 2:6] = 1.0
    x = torch.linspace(-1.0, 1.0, n)[:, None, None].expand(n, n, n)
    E = 1e-3 * torch.stack([c * x for c in (1.0, -0.5, 0.3, 0.2, -0.1, 0.05)], -1)[..., None, :] * V[..., None]
    tf = TomoFit(fit, proj, V, E, weights=w, elastic=[(211e9, 0.29)], density_smooth=True, normalize=normalize,
                 smooth_order=2, smooth_sd=STRAIN_SMOOTH2_SD, cg_iters=LEAN_CG_ITERS)
    if not speedups:
        _speedups_off(tf)
    if texture:
        tf.set_texture_voxels(1.0, 1.0)
        u = torch.tensor([0.3, 0.5, 0.81], dtype=torch.float64)
        tf.Tx = (0.05 * harmonics4(u / u.norm())).float().reshape(1, 1, 1, 1, 9) * V[..., None]
        tf.sync()
    tf.edge_spread = True
    tf.refresh_spread()
    with torch.no_grad():
        m = torch.cat([fit.mean(s)[0] for s in fit.chunks()])
    fit.T = (torch.poisson(LEAN_DOSE * m.double(), generator=gen) / LEAN_DOSE).float()
    # the starting state
    tf.V = (0.7 * V + 0.05 * torch.rand(n, n, n, 1, generator=gen)).contiguous()
    tf.E = torch.zeros_like(tf.E)
    if texture:
        tf.Tx = torch.zeros_like(tf.Tx)
    g0 = fit.g.clone(); g0[sl['h'][0]:sl['h'][1]] *= 0.9
    fit.g = g0; fit.f = 0.5 * f; fit.eps_sd = None
    tf.sync()
    return tf


def _lean_run(texture, normalize, lean, keep=None, t_storage='device', options=None):
    """Runs LEAN_ITERS iterations of the driver's steps (fe_cube_multi.py: volume, texture, recentring, global and
    contamination steps) on a new _lean_tomo_fit, with lean_passes on or off. keep None leaves keep_gb at its default,
    'none' keeps no model parts, and 'one' keeps one chunk's. t_storage is passed to FitB. The speedup options start off
    (_speedups_off). options maps names of FitB or TomoFit options to the values that the fit is given, or is None for
    none. Returns the objective after each iteration, the passes over the rays in each iteration, the final state and,
    per iteration, whether the fit holds no kept model parts after it."""
    tf = _speedups_off(_lean_tomo_fit(texture, normalize, t_storage))
    fit = tf.fit; M = fit.M
    for name, value in (options or {}).items():
        setattr(fit if hasattr(fit, name) else tf, name, value)
    fit.lean_passes = lean
    if keep == 'none':
        fit.keep_gb = 0.0
    elif keep == 'one':                                      # room for one chunk's n and beta, not for two
        from mbirtorch.hsnt.strain.model import GB
        fit.keep_gb = 1.5 * fit.chunk * (M.K + M.J) * torch.finfo(M.dt).bits / 8 / GB
    fit.fisher_every = 2; fit.contamination_newton = True
    damp = 1e-3
    objectives, passes, freed = [], [], []
    for _ in range(LEAN_ITERS):
        p0 = fit.pass_count()
        tf.volume_step()
        if texture:
            with torch.no_grad():
                tf.texture_step()
        tf.recenter_strain(full=True)
        _, damp = fit.global_step(damp)
        fit.contamination_atoms_step()
        passes.append(fit.pass_count() - p0)
        freed.append(fit._kept is None)
        objectives.append(tf.loss() + float(fit.spline_penalty(fit.g)) + float((fit.lam_c * fit.c).sum()))
    state = dict(V=tf.V, E=tf.E, g=fit.g, f=fit.f, c=fit.c, W=fit.W, eps=fit.eps)
    if texture:
        state['Tx'] = tf.Tx
    return objectives, passes, {k: v.detach().clone() for k, v in state.items()}, freed


def _largest_rel(a, b):
    """The largest difference between a and b relative to b's largest magnitude."""
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return float((a - b).abs().max() / b.abs().max().clamp_min(torch.finfo(torch.float64).tiny))


def _lean_differences(run, ref):
    """The objectives' and each state's largest relative difference between two _lean_run results."""
    out = {'objective': _largest_rel(run[0], ref[0])}
    out.update({k: _largest_rel(run[2][k], ref[2][k]) for k in ref[2]})
    return out


@pytest.mark.parametrize('texture,normalize,keep,saved', [(True, True, None, 2.0), (True, False, 'none', 1.0),
                                                          (False, True, 'one', 1.0 / LEAN_CHUNKS)])
def test_lean_passes_matches_default(texture, normalize, keep, saved):
    """FitB.lean_passes gives the default path's objectives and states (volumes, strain, texture, spectrum and
    contamination parameters, ray maps) over LEAN_ITERS iterations of the driver's steps, bitwise when the default path
    is deterministic (see LEAN_NOISE_MARGIN). It makes at least `saved` fewer passes over the rays per iteration: the
    texture step's starting loss, and the contamination step's passes from the kept model parts. The cases cover the
    per-voxel texture with and without normalize_fractions, no texture, and the memory budget at no chunk and one chunk.
    The kept model parts are freed after each contamination step."""
    ref = _lean_run(texture, normalize, False)
    again = _lean_run(texture, normalize, False)
    lean = _lean_run(texture, normalize, True, keep)
    noise, got = _lean_differences(again, ref), _lean_differences(lean, ref)
    for k in got:
        assert got[k] <= LEAN_NOISE_MARGIN * noise[k], (k, got[k], noise[k])
    assert all(lean[3]) and all(ref[3])
    if all(v == 0 for v in noise.values()):              # the same steps were taken, so the passes compare per iteration
        assert ref[1] == again[1]
        for p_ref, p_lean in zip(ref[1], lean[1]):
            assert p_lean <= p_ref - saved + 1e-9, (ref[1], lean[1])
    else:
        assert sum(lean[1]) < sum(ref[1]), (ref[1], lean[1])


def _mount_fit(seed=0, speedups=False, one_model_views=True):
    """_volume_step_fit with noiseless data made with series 1's mount shifted by MOUNT_TEST_SHIFT voxels and one view
    of series 0 shifted by VIEW_TEST_OFFSET channels, so that a mount step and a view-offset step each have a clear way
    down. The fit starts at the unshifted geometry. Series 2's mount shift is MOUNT_TEST_SERIES2_SHIFT, in the data and
    in the fit. With speedups False, the speedup options are off (_speedups_off), and with True they keep their
    defaults. one_model_views is passed to the MultiProjector."""
    tf = _volume_step_fit(seed, one_model_views)
    if not speedups:
        _speedups_off(tf)
    fit, pr = tf.fit, tf.proj
    pr.set_mount(2, shift=MOUNT_TEST_SERIES2_SHIFT)           # The rotation stays as it is.
    (_, _, shift0, _), _ = pr.parts[1]
    views0 = len(pr.series[0][1])
    offs = np.zeros(views0); offs[VIEW_TEST_INDEX] = VIEW_TEST_OFFSET
    pr.set_mount(1, shift=[v + MOUNT_TEST_SHIFT for v in shift0])
    pr.set_view_offsets(0, offs)
    tf.sync()
    with torch.no_grad():
        fit.T = torch.cat([fit.mean(s)[0] for s in fit.chunks()]).clone()
    pr.set_mount(1, shift=list(shift0))
    pr.set_view_offsets(0, np.zeros(views0))
    tf.sync()
    return tf


def _mount_geometry(out_m, out_v):
    """Returns, as one float64 vector, the mounts that TomoFit.mount_step reported in out_m (each series' shift,
    rotation and offset) and the view offsets that view_offset_step reported in out_v."""
    geometry = [v for m in out_m['mounts'] for v in m['shift'] + m['rot'] + [m['offset']]] + \
        [v for offs in out_v['offsets'] for v in offs]
    return torch.tensor(geometry, dtype=torch.float64)


def _mount_run(lean):
    """A mount step and a view-offset step on a new _mount_fit, with lean_passes on or off, from a synced state whose
    data loss the fit holds (as after a contamination step). Returns each step's passes over the rays, each step's
    length, the loss after them, the mounts and view offsets after them, and the fit's device. The projectors keep one
    model per view (one_model_views off), because the one-model projector is not bitwise reproducible on the CPU.
    test_mount_steps_with_defaults covers one_model_views with lean_passes."""
    tf = _mount_fit(one_model_views=False)
    fit = tf.fit
    fit.lean_passes = lean
    tf.sync(); fit.data_loss()
    p0 = fit.pass_count()
    out_m = tf.mount_step()
    p1 = fit.pass_count()
    out_v = tf.view_offset_step()
    p2 = fit.pass_count()
    return ((p1 - p0, p2 - p1), (out_m['alpha'], out_v['alpha']), tf.loss(), _mount_geometry(out_m, out_v),
            fit.device)


def test_lean_passes_mount_and_view_steps():
    """With lean_passes, TomoFit.mount_step and view_offset_step skip the projection of unchanged volumes and take
    their starting loss from the fit's remembered data loss, so each makes one pass over the rays fewer than the
    default path. They take the same steps: bitwise on a deterministic device, and on a GPU within the default path's
    own run-to-run difference or LEAN_GPU_TOL (relative for the loss, in voxels, degrees and channels for the
    geometry)."""
    ref, again, lean = _mount_run(False), _mount_run(False), _mount_run(True)
    assert lean[1] == ref[1] == again[1], (ref[1], again[1], lean[1])
    assert ref[1][0] > 0, ref[1]                                       # the mount step was taken
    floor = 0.0 if ref[4].type == 'cpu' else LEAN_GPU_TOL
    noise = (abs(again[2] - ref[2]) / abs(ref[2]), float((again[3] - ref[3]).abs().max()))
    got = (abs(lean[2] - ref[2]) / abs(ref[2]), float((lean[3] - ref[3]).abs().max()))
    for g_, n_ in zip(got, noise):
        assert g_ <= max(LEAN_NOISE_MARGIN * n_, floor), (got, noise)
    assert ref[0] == again[0]
    assert all(abs(p_lean - (p_ref - 1.0)) < 1e-9 for p_ref, p_lean in zip(ref[0], lean[0])), (ref[0], lean[0])


# ---- the speedup options together (speedup round 2)
COMBINED_ROUNDS = 2             # test_fisher_rays_with_host_storage takes this many rounds of the fit's steps.
# On the CPU, a data loss that the fit remembers matches the data loss of a new pass to this relative tolerance. A
# remembered value from an earlier state differs by the loss change of a step, which is far larger.
MEMO_RTOL = 1e-12


@pytest.mark.parametrize('texture,normalize', [(True, False), (False, True)])
def test_lean_passes_with_other_options(texture, normalize, monkeypatch):
    """lean_passes gives the objectives and states of the same fit without it when the fit also has t_storage 'host',
    lean_syncs, closed_form_strain and fisher_rays at one ray in FISHER_SUBSET_SHARE. The comparison is the one of
    test_lean_passes_matches_default. The lean path makes fewer passes over the rays, and it frees its kept model parts
    after each contamination step. The TomoFit options fft_compact and spread_grouped are set on both arms too."""
    from mbirtorch.hsnt.strain import model
    # This test checks the subset path, not the subset's check. So the check never switches the fit to all rays.
    monkeypatch.setattr(model, 'FISHER_CHECK_TOL', math.inf)
    rays = LEAN_VIEWS * LEAN_RAYS_PER_VIEW
    options = dict(lean_syncs=True, closed_form_strain=True, fisher_rays=rays // FISHER_SUBSET_SHARE,
                   fft_compact=True, spread_grouped=True)
    ref, again, lean = [_lean_run(texture, normalize, on, t_storage='host', options=options)
                        for on in (False, False, True)]
    noise, got = _lean_differences(again, ref), _lean_differences(lean, ref)
    for k in got:
        assert got[k] <= LEAN_NOISE_MARGIN * noise[k], (k, got[k], noise[k])
    assert all(lean[3])
    assert sum(lean[1]) < sum(ref[1]), (ref[1], lean[1])


@pytest.mark.parametrize('check', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_fisher_rays_with_host_storage(device, check, monkeypatch):
    """With fisher_rays and lean_syncs, a fit with t_storage 'host', t_prefetch and lean_passes takes the same pixel,
    global and contamination steps as a fit with the default storage. The two fits agree bitwise on the CPU, and to
    GPU_RTOL on a GPU. After each global step, the data loss that each fit remembers is the data loss of a new pass.
    After each contamination step, a remembered data loss is that of a new pass too. The fit with lean_passes makes
    fewer passes over the rays, and it holds no kept model parts at the end. With check False, FISHER_CHECK_TOL is
    infinite, so the subset's check never switches a fit to all rays. With check True, the check runs as in a real fit,
    and the two fits switch to all rays in the same round or not at all."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('no GPU')
    from mbirtorch.hsnt.strain import model
    if not check:
        # This variant checks the subset path, not the subset's check. So the check never switches the fit to all rays.
        monkeypatch.setattr(model, 'FISHER_CHECK_TOL', math.inf)
    ref = _small_fit_cubic4()
    a = _storage_fit(ref, ref.T.clone().to(device), device, t_storage='device')
    b = _storage_fit(ref, ref.T.clone(), device, t_storage='host', t_prefetch=True)
    a.lean_passes = False
    exact = device == 'cpu'

    def same(x, y):
        if not torch.is_tensor(x):
            return x == y if exact else abs(x - y) <= GPU_RTOL * abs(y)
        if exact:
            return torch.equal(x, y)
        return torch.allclose(x, y, rtol=GPU_RTOL, atol=GPU_RTOL * float(y.abs().max()))

    def memo_is_current(fit, required):
        # The remembered value is compared with a new pass over all rays. use_memo False makes data_loss compute it.
        remembered = fit.recall_data_loss()
        if remembered is None:
            return not required
        fit.use_memo = False
        try:
            fresh = fit.data_loss()
        finally:
            fit.use_memo = True
        return abs(remembered - fresh) <= (MEMO_RTOL if exact else GPU_RTOL) * abs(fresh)

    for x in (a, b):
        x.fisher_rays = x.P // FISHER_SUBSET_SHARE
        x.lean_syncs = True
        x.closed_form_strain = False
        x.build_atoms(); x.contamination_newton = True
    b.lean_passes = True
    for _ in range(COMBINED_ROUNDS):
        for x in (a, b):
            x.pixel_step()
            x.global_step()
            assert memo_is_current(x, True)
            x.contamination_atoms_step()
            assert memo_is_current(x, False)
        for key in ('g', 'W', 'eps', 'c', 'f'):
            assert same(getattr(b, key), getattr(a, key)), key
        assert b._fisher_fallback == a._fisher_fallback, (a._fisher_fallback, b._fisher_fallback)
    assert b.fisher_subset_info['rays'] == a.P // FISHER_SUBSET_SHARE
    if not check:
        assert a._fisher_fallback is None
    assert b.pass_count() < a.pass_count(), (a.pass_count(), b.pass_count())
    assert b._kept is None


# test_mount_steps_with_defaults compares the mount and view-offset steps of a fit with every speedup option at its
# default against those of a fit with every option off and a model per view. The two fits' models agree only to
# rounding. lean_syncs and closed_form_strain agree to rounding in the model's dtype, float32, and one_model_views
# agrees to float32 rounding of the projections. The steps' finite differences (MOUNT_FD, VIEW_OFFSET_FD) divide that
# rounding by their small steps, so the two fits' steps differ by more than the rounding itself. That difference is
# estimated at about 1e-5 in the mounts and view offsets. The geometry tolerance below is 100 times that estimate. It
# stays below the 0.0057 change in a view offset that one flip of a rounding tie in MultiProjector._split caused.
# The loss tolerance has the same value and the same reference loss as VOLUME_STEP_AFTER_RTOL, which bounds the same
# kind of difference after a volume step.
# The two fits' data losses before and after the steps differ by at most this share of the loss before them.
DEFAULTS_MOUNT_LOSS_RTOL = 1e-3
# The two fits' mounts and view offsets after the steps differ by at most this many voxels, degrees or detector
# channels.
DEFAULTS_MOUNT_GEOMETRY_TOL = 1e-3


def test_mount_steps_with_defaults():
    """With every speedup option at its default, one_model_views, lean_passes, lean_syncs and closed_form_strain among
    them, TomoFit.mount_step and view_offset_step take the steps of a fit with every option off and a model per view.
    The step lengths are equal. The data losses before and after the steps agree to DEFAULTS_MOUNT_LOSS_RTOL, and the
    mounts and view offsets agree to DEFAULTS_MOUNT_GEOMETRY_TOL. Both fits use the data that the fit with every option
    off makes."""

    def run(on, T=None):
        """Runs a mount step and a view-offset step on a new _mount_fit with every option at its default (on True) or
        off (on False), from a synced state. A given T replaces the fit's data. Returns each step's length, the data
        loss before and after the steps, the mounts and view offsets after them, and the data."""
        tf = _mount_fit(speedups=on, one_model_views=on)
        fit = tf.fit
        if T is not None:
            fit.T = T.clone()
        tf.sync()
        before = fit.data_loss()
        out_m = tf.mount_step()
        out_v = tf.view_offset_step()
        return (out_m['alpha'], out_v['alpha']), (before, fit.data_loss()), _mount_geometry(out_m, out_v), fit.T

    off = run(False)
    on = run(True, off[3])
    assert on[0] == off[0], (on[0], off[0])
    assert off[0][0] > 0, off[0]                                       # the mount step was taken
    for got, ref in zip(on[1], off[1]):
        assert abs(got - ref) <= DEFAULTS_MOUNT_LOSS_RTOL * abs(off[1][0]), (on[1], off[1])
    diff = float((on[2] - off[2]).abs().max())
    assert diff <= DEFAULTS_MOUNT_GEOMETRY_TOL, (diff, on[2], off[2])


ONE_MODEL_RTOL = 1e-5           # one_model_views against a model per view: the largest difference relative to the largest
                                 # value, and the adjoint identity's relative error
ONE_MODEL_SEED = 20261123        # the random volumes, rays and per-view offsets of the one_model_views tests
ONE_MODEL_OFFSET = 0.5           # each view's own channel offset is uniform between -ONE_MODEL_OFFSET and ONE_MODEL_OFFSET
ONE_MODEL_ROW_OFFSET = 0.3       # the detector row offset of the one_model_views tests (a fractional shift along the axis)
ONE_MODEL_CHANNEL_OFFSET = 0.25  # the projector's own channel offset in test_one_model_views
ONE_MODEL_MOUNT_SHIFT = (2.6, -0.45, 0.35)   # a new mount shift (voxels) with fractional parts, set in place


def _max_rel(a, b):
    """The largest difference between a and b, relative to the largest magnitude of b."""
    a, b = torch.as_tensor(a).double().cpu(), torch.as_tensor(b).double().cpu()
    return float((a - b).abs().max() / b.abs().max())


def test_view_offset_model(any_device):
    """MultiAxisModelWithViewOffsets, one model whose views each have their own channel offset, projects and back
    projects like one MultiAxisParallelModel per view, to float32 rounding. On the CPU it runs the package's torch code,
    and on a GPU the Triton kernels. Its back projection is the adjoint of its projection."""
    import mbirtorch
    from mbirtorch.hsnt.strain.tomo import MultiAxisModelWithViewOffsets
    rng = np.random.default_rng(ONE_MODEL_SEED)
    views, rows, channels = 6, 10, 9
    az = np.linspace(0, np.pi, views, endpoint=False)
    offsets = rng.uniform(-ONE_MODEL_OFFSET, ONE_MODEL_OFFSET, views)
    one = MultiAxisModelWithViewOffsets((views, rows, channels), np.stack([az, np.zeros(views), offsets], 1))
    per = [mbirtorch.MultiAxisParallelModel((1, rows, channels), np.array([[a, 0.0]]), compile_mode='off') for a in az]
    for m, o in zip([one] + per, [0.0] + list(offsets)):
        m.configure_devices(devices=[any_device])
        m.set_params(no_warning=True, verbose=0, det_channel_offset=float(o), det_row_offset=ONE_MODEL_ROW_OFFSET,
                     use_ror_mask=False)
    vol = rng.random(tuple(one.get_params('recon_shape'))).astype(np.float32)
    sino = rng.random((views, rows, channels)).astype(np.float32)
    fwd, back = np.asarray(one.forward_project(vol)), np.asarray(one.back_project(sino))
    assert _max_rel(fwd, np.concatenate([np.asarray(m.forward_project(vol)) for m in per])) <= ONE_MODEL_RTOL
    assert _max_rel(back, sum(np.asarray(m.back_project(sino[v:v + 1])) for v, m in enumerate(per))) <= ONE_MODEL_RTOL
    a, b = float((fwd.astype(np.float64) * sino).sum()), float((vol.astype(np.float64) * back).sum())
    assert abs(a - b) <= ONE_MODEL_RTOL * abs(a)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='the Triton kernels need a GPU')
def test_view_offset_model_uses_kernels():
    """On a GPU where the package's multiaxis Triton kernels are usable, MultiAxisModelWithViewOffsets projects with
    them, and it keeps the attributes that the package reads from them."""
    from mbirtorch.kernel_availability import multiaxis_back_kernel_usable, multiaxis_forward_kernel_usable
    from mbirtorch.triton_multiaxis import _multiaxis_back_view_batch_triton, _multiaxis_forward_view_batch_triton
    from mbirtorch.hsnt.strain.tomo import MultiAxisModelWithViewOffsets
    views, side = 4, 6
    m = MultiAxisModelWithViewOffsets((views, side, side), np.zeros((views, 3)))
    m.configure_devices(devices=['cuda'])
    if not (multiaxis_forward_kernel_usable(m)[0] and multiaxis_back_kernel_usable(m)[0]):
        pytest.skip('the multiaxis Triton kernels are not usable on this GPU')
    fwd, back = m._view_batch_bodies()
    assert fwd.body is _multiaxis_forward_view_batch_triton and back.body is _multiaxis_back_view_batch_triton
    assert fwd._mbirtorch_no_compile and back._mbirtorch_no_compile and back._adds_into_output


def test_one_model_views():
    """Projector, MultiProjector and FineProjector with one_model_views=True project and back project like the default
    path (one model per view) to float32 rounding. The views have random fractional channel offsets of their own, and
    the series have fractional mount shifts. The match holds after the offsets and mounts change in place, and the
    back projection is the adjoint of the projection. The option survives with_q."""
    from mbirtorch.hsnt.strain.tomo import Projector, FineProjector
    rng = np.random.default_rng(ONE_MODEL_SEED)
    gen = torch.Generator().manual_seed(ONE_MODEL_SEED)
    n, Q = 10, 3
    th = np.linspace(0, np.pi, 7, endpoint=False)
    vo = rng.uniform(-ONE_MODEL_OFFSET, ONE_MODEL_OFFSET, th.size)
    x = torch.rand(n, n, n, Q, generator=gen)
    y = torch.rand(th.size * n * n, Q, generator=gen)
    for row_offset in (0.0, ONE_MODEL_ROW_OFFSET):
        old = Projector(n, th, Q, ONE_MODEL_CHANNEL_OFFSET, False, vo, row_offset, one_model_views=False)
        new = Projector(n, th, Q, ONE_MODEL_CHANNEL_OFFSET, False, vo, row_offset, one_model_views=True)
        assert len(old.models) == th.size and len(new.models) == 1
        fwd, back = new.fwd(x).cpu(), new.back(y).cpu()
        assert _max_rel(fwd, old.fwd(x)) <= ONE_MODEL_RTOL and _max_rel(back, old.back(y)) <= ONE_MODEL_RTOL
        a, b = float((fwd.double() * y.double()).sum()), float((x.double() * back.double()).sum())
        assert abs(a - b) <= ONE_MODEL_RTOL * abs(a)
        wider = new.with_q(Q + 1)
        assert wider.one_model_views and len(wider.models) == 1
    extra = [rng.uniform(-ONE_MODEL_OFFSET, ONE_MODEL_OFFSET, len(ANGLES)) for _ in MOUNTS]
    offsets = [0.0, 0.3, -0.2]
    for make, side in ((lambda **k: MultiProjector(N, MOUNTS, 2, channel_offsets=offsets, ror_mask=False, **k), N),
                       (lambda **k: FineProjector(N // 2, MOUNTS, 2, channel_offsets=offsets, ror_mask=False, **k), N)):
        old, new = make(view_extra=extra, one_model_views=False), make(view_extra=extra, one_model_views=True)
        assert all(len(p.models) == len(ANGLES) for _, p in old.parts) and all(len(p.models) == 1 for _, p in new.parts)
        vol = torch.rand(side, side, side, 2, generator=gen)
        rays = torch.rand(sum(old.counts), 2, generator=gen)
        for _ in range(2):
            assert _max_rel(new.fwd(vol), old.fwd(vol)) <= ONE_MODEL_RTOL
            assert _max_rel(new.back(rays), old.back(rays)) <= ONE_MODEL_RTOL
            changed = rng.uniform(-ONE_MODEL_OFFSET, ONE_MODEL_OFFSET, len(ANGLES))
            for p in (old, new):
                p.set_view_offsets(1, changed)
                p.set_mount(2, shift=ONE_MODEL_MOUNT_SHIFT)
        assert new.with_q(3).one_model_views


DEFAULTS_TEST_RAYS = 8           # test_speedup_defaults' fit has this many rays.
DEFAULTS_TEST_Q = 2              # test_speedup_defaults' projectors carry this many channels.
DEFAULTS_TEST_OFFSETS = [0.0, 0.3, -0.2]   # test_speedup_defaults' series have these channel offsets.
DEFAULTS_FINE_FACTOR = 2         # test_speedup_defaults' FineProjector has a grid this many times finer.
# The window tests set one chunk's strains to this value. With _lean_fit's grid, the shared window is then more than
# WINDOW_PLAN_MAX_RATIO times any chunk's own width.
OUTLIER_STRAIN = 8e-2
WINDOW_FALLBACK_RTOL = 1e-12     # The window fallback's losses and ray systems equal lean_syncs off to this tolerance.
WARN_ONCE_PASSES = 3             # test_fallback_warns_once makes this many passes.
CLOSED_FALLBACK_RTOL = 1e-12     # A chunk that falls back to forward mode equals closed_form_strain off to this.
CLOSED_FORM_RTOL = 1e-9          # A chunk in closed form equals forward mode to this, as in test_closed_form_strain_*.
CGROUP_TEST_AVAILABLE = 10 ** 10                 # test_host_free_bytes_cgroup's system has this much available memory.
CGROUP_TEST_TOTAL = 2 * 10 ** 10                 # Its system has this much memory in total.
# These are the memory.max, memory.current and inactive_file of test_host_free_bytes_cgroup's group.
CGROUP_TEST_CHILD = (10_000_000, 4_000_000, 1_000_000)
CGROUP_TEST_PARENT = (6_000_000, 3_000_000)      # These are the parent group's memory.max and memory.current.
CGROUP_TEST_V1 = (5_000_000, 1_000_000)          # These are a version 1 group's limit and usage.
CGROUP_TEST_V1_HIGH = 2 * CGROUP_TEST_TOTAL      # A version 1 limit above the total memory counts as no limit.
BUFFER_TEST_SHAPE = (64, 12)                     # test_transmission_buffer_cpu's transmission has this shape.
MESSAGE_TEST_BYTES = 3.2e9                       # test_storage_messages' transmission has this many bytes.
MESSAGE_TEST_HOST_FREE = 4.0e9                   # test_storage_messages' host has this many bytes free.
MESSAGE_TEST_GPU_FREE = 2.0e9                    # test_storage_messages' GPU has this many bytes free.
# In test_storage_choice, a memory with room has this many times the transmission's bytes free, and a tight memory
# has CHOICE_TEST_TIGHT times them free.
CHOICE_TEST_ROOM = 4.0
CHOICE_TEST_TIGHT = 1.0
FRACTION_TEST_TOTAL = 4 * 10 ** 9                # test_device_free_bytes_fraction's GPU has this many bytes.
FRACTION_TEST_FREE = 3 * 10 ** 9                 # The driver reports this many bytes of it free.
FRACTION_TEST_RESERVED = 2 * 10 ** 8             # PyTorch holds this many bytes of it.
FRACTION_TEST_ALLOCATED = 10 ** 8                # PyTorch's tensors use this many of the bytes that it holds.
FRACTION_TEST_SHARE = 0.25                       # PyTorch may use this share of the GPU's memory.
KEEP_TEST_SHARE_OF_NEED = 0.5    # test_lean_keep_remedy sets keep_gb to this share of the memory that all parts need.


def test_speedup_defaults():
    """Every speedup option is on by default: lean_syncs, closed_form_strain, lean_passes and fisher_rays
    FISHER_RAYS_AUTO on a FitB, fft_compact and spread_grouped on a TomoFit, and one_model_views on each projector. A
    fit on the CPU keeps T as given, without prefetch. Nothing is warned."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB, FISHER_RAYS_AUTO
    from mbirtorch.hsnt.strain.tomo import Projector, FineProjector
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        lam = np.linspace(*RARE_WAVELENGTHS, RARE_BINS)
        M = Hybrid2(lam, np.full_like(lam, RARE_DOSE),
                    [('bcc', (RARE_BCC_A0 * (1 - RARE_A_RANGE), RARE_BCC_A0 * (1 + RARE_A_RANGE)))],
                    smooth_degree=RARE_SMOOTH_DEGREE, n_contam=1, device='cpu', dtype=torch.float64)
        T = torch.ones(DEFAULTS_TEST_RAYS, M.K, dtype=torch.float64)
        fit = FitB(M, T, torch.zeros(M.n_global, dtype=torch.float64),
                   torch.ones(DEFAULTS_TEST_RAYS, 1, dtype=torch.float64), torch.zeros(1, M.K, dtype=torch.float64))
        assert fit.lean_syncs and fit.closed_form_strain and fit.lean_passes
        assert fit.fisher_rays == FISHER_RAYS_AUTO
        assert fit.t_storage == 'device' and fit.T is T and not fit.t_prefetch
        tf = _volume_step_fit()
        assert tf.fft_compact and tf.spread_grouped
        assert Projector(N, ANGLES, DEFAULTS_TEST_Q).one_model_views
        assert MultiProjector(N, MOUNTS, DEFAULTS_TEST_Q, channel_offsets=DEFAULTS_TEST_OFFSETS,
                              ror_mask=False).one_model_views
        assert FineProjector(N // DEFAULTS_FINE_FACTOR, MOUNTS[:2], DEFAULTS_TEST_Q,
                             channel_offsets=DEFAULTS_TEST_OFFSETS[:2],
                             ror_mask=False, factor=DEFAULTS_FINE_FACTOR).one_model_views


def test_warmup_fit_options():
    """warmup's fit_options sets each named FitB attribute on a fit, and a name that FitB does not have raises
    ValueError."""
    from mbirtorch.hsnt.strain.tomo import _apply_fit_options
    fit = _small_fit()
    _apply_fit_options(fit, dict(lean_syncs=True, fisher_rays=0))
    assert fit.lean_syncs and fit.fisher_rays == 0
    _apply_fit_options(fit, None)
    with pytest.raises(ValueError):
        _apply_fit_options(fit, dict(lean_sync=False))


def test_no_fallback_on_normal_path():
    """A small tomographic fit with every default on takes the volume, texture, global, contamination and pixel steps
    without any FallbackWarning."""
    from mbirtorch.hsnt.strain.model import T_STORAGE_AUTO
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        tf = _lean_tomo_fit(True, True, t_storage=T_STORAGE_AUTO, speedups=True)
        fit = tf.fit
        fit.contamination_newton = True
        tf.volume_step()
        with torch.no_grad():
            tf.texture_step()
        tf.recenter_strain(full=True)
        fit.global_step()
        fit.contamination_atoms_step()
        fit.pixel_step()


def test_host_free_bytes_cgroup(tmp_path, monkeypatch):
    """host_free_bytes is the system's available memory, or less when a control group leaves less: a version 2 group
    and its parents (inactive file pages count as free), or a version 1 memory group. A limit of 'max', a version 1
    limit above the total memory and a missing group file set no limit."""
    from mbirtorch.hsnt.strain import model
    monkeypatch.setattr(model.psutil, 'virtual_memory',
                        lambda: types.SimpleNamespace(available=CGROUP_TEST_AVAILABLE, total=CGROUP_TEST_TOTAL))
    root = tmp_path / 'cgroup'
    child = root / 'a' / 'b'
    child.mkdir(parents=True)
    (root / 'memory.max').write_text('max\n')
    lim, cur, inactive = CGROUP_TEST_CHILD
    (child / 'memory.max').write_text(f'{lim}\n')
    (child / 'memory.current').write_text(f'{cur}\n')
    (child / 'memory.stat').write_text(f'anon {cur}\ninactive_file {inactive}\n')
    plim, pcur = CGROUP_TEST_PARENT
    (root / 'a' / 'memory.max').write_text(f'{plim}\n')
    (root / 'a' / 'memory.current').write_text(f'{pcur}\n')
    proc = tmp_path / 'self_cgroup'
    proc.write_text('0::/a/b\n')
    monkeypatch.setattr(model, 'CGROUP_ROOT', str(root))
    monkeypatch.setattr(model, 'PROC_SELF_CGROUP', str(proc))
    assert model.host_free_bytes() == min(CGROUP_TEST_AVAILABLE, lim - cur + inactive, plim - pcur)
    (root / 'a' / 'memory.max').write_text('max\n')                     # the parent sets no limit now
    assert model.host_free_bytes() == min(CGROUP_TEST_AVAILABLE, lim - cur + inactive)
    v1 = root / 'memory' / 'job'
    v1.mkdir(parents=True)
    vlim, vuse = CGROUP_TEST_V1
    (v1 / 'memory.limit_in_bytes').write_text(f'{vlim}\n')
    (v1 / 'memory.usage_in_bytes').write_text(f'{vuse}\n')
    proc.write_text('5:cpu,cpuacct:/job\n4:memory:/job\n')
    assert model.host_free_bytes() == min(CGROUP_TEST_AVAILABLE, vlim - vuse)
    (v1 / 'memory.limit_in_bytes').write_text(f'{CGROUP_TEST_V1_HIGH}\n')
    assert model.host_free_bytes() == CGROUP_TEST_AVAILABLE
    monkeypatch.setattr(model, 'PROC_SELF_CGROUP', str(tmp_path / 'missing'))
    assert model.host_free_bytes() == CGROUP_TEST_AVAILABLE


def test_transmission_buffer_cpu():
    """For a fit on the CPU, transmission_buffer gives a CPU tensor with storage 'device', and resolve_t_storage keeps
    the given T. Nothing is warned."""
    from mbirtorch.hsnt.strain.model import transmission_buffer, resolve_t_storage
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        buf, storage = transmission_buffer(BUFFER_TEST_SHAPE, torch.float32, 'cpu')
        assert storage == 'device' and buf.device.type == 'cpu' and tuple(buf.shape) == BUFFER_TEST_SHAPE
        assert buf.dtype == torch.float32
        T = torch.ones(BUFFER_TEST_SHAPE)
        place, kept = resolve_t_storage(T, 'cpu')
        assert place == 'device' and kept is T


def test_storage_messages(monkeypatch):
    """Each warning about the transmission's storage gives its size, says that the results are the same, and names
    what the user can do. Under WSL, the pinning warning does not suggest a locked-memory limit. A warning about a
    transmission on the GPU names t_storage='device', which keeps it there without the warning."""
    from mbirtorch.hsnt.strain import model
    from mbirtorch.hsnt.strain.model import _storage_message, GB
    for wsl in (False, True):
        monkeypatch.setattr(model, '_under_wsl', lambda wsl=wsl: wsl)
        remedies = {'pinning': 'without WSL' if wsl else 'ulimit -l unlimited',
                    'host memory': 'give the job more memory'}
        for reason in ('pinning', 'host memory'):
            for kept_on in ('device', 'host'):
                text = _storage_message(reason, kept_on, MESSAGE_TEST_BYTES, MESSAGE_TEST_HOST_FREE,
                                        MESSAGE_TEST_GPU_FREE)
                assert f'{MESSAGE_TEST_BYTES / GB:.1f} GB' in text and 'results are the same' in text, text
                assert remedies[reason] in text, text
                assert not (wsl and 'ulimit' in text), text
                if kept_on == 'host':
                    assert 'GPU with at least' in text and f'{MESSAGE_TEST_GPU_FREE / GB:.1f} GB free' in text, text
                else:
                    assert 'more GPU memory' in text and "t_storage='device'" in text, text
    assert f'{MESSAGE_TEST_HOST_FREE / GB:.1f} GB of host memory' in _storage_message(
        'host memory', 'device', MESSAGE_TEST_BYTES, MESSAGE_TEST_HOST_FREE, MESSAGE_TEST_GPU_FREE)
    assert "t_storage='device'" in _storage_message('pinning', 'host', MESSAGE_TEST_BYTES)


def test_storage_choice():
    """_storage_choice chooses pinned host memory when the host has room and pinning is not known to fail. Otherwise
    it chooses the GPU when the GPU has room, and pageable host memory when it has not. It names the cause: a known
    pinning failure first, and the host memory otherwise."""
    from mbirtorch.hsnt.strain.model import _storage_choice
    b = MESSAGE_TEST_BYTES
    room, tight = CHOICE_TEST_ROOM * b, CHOICE_TEST_TIGHT * b
    assert _storage_choice(b, room, room, False) == ('pinned', None)
    assert _storage_choice(b, room, room, True) == ('device', 'pinning')
    assert _storage_choice(b, room, tight, True) == ('host', 'pinning')
    assert _storage_choice(b, tight, room, True) == ('device', 'pinning')
    assert _storage_choice(b, tight, room, False) == ('device', 'host memory')
    assert _storage_choice(b, tight, tight, False) == ('host', 'host memory')


def test_storage_warnings_once(monkeypatch):
    """Each storage fallback warns once per process for its place, cause and size of the transmission. A
    transmission of another size warns again."""
    from mbirtorch.hsnt.strain import model
    monkeypatch.setattr(model, '_PROCESS_WARNED', set())
    monkeypatch.setattr(model, '_under_wsl', lambda: False)
    sizes = (MESSAGE_TEST_BYTES, MESSAGE_TEST_HOST_FREE, MESSAGE_TEST_GPU_FREE)
    for place, reason, match in (('device', 'pinning', 'cannot pin'), ('host', 'pinning', 'without pinning'),
                                 ('device', 'host memory', 'of host memory is free'),
                                 ('host', 'host memory', 'without pinning')):
        with pytest.warns(FallbackWarning, match=match):
            model._storage_warning(place, reason, *sizes)
        with warnings.catch_warnings():
            warnings.simplefilter('error', FallbackWarning)
            model._storage_warning(place, reason, *sizes)
    with pytest.warns(FallbackWarning, match='cannot pin'):
        model._storage_warning('device', 'pinning', CHOICE_TEST_ROOM * MESSAGE_TEST_BYTES, *sizes[1:])


def test_device_free_bytes_fraction(monkeypatch):
    """On a GPU, device_free_bytes is the driver's free memory plus PyTorch's unused memory, or less when PyTorch's
    per-process memory fraction leaves less. The GPU is simulated, so the test runs on the CPU."""
    from mbirtorch.hsnt.strain import model
    cuda = torch.cuda
    monkeypatch.setattr(model, '_compute_device', lambda device: torch.device('cuda', 0))
    monkeypatch.setattr(cuda, 'mem_get_info', lambda device=None: (FRACTION_TEST_FREE, FRACTION_TEST_TOTAL))
    monkeypatch.setattr(cuda, 'memory_reserved', lambda device=None: FRACTION_TEST_RESERVED)
    monkeypatch.setattr(cuda, 'memory_allocated', lambda device=None: FRACTION_TEST_ALLOCATED)
    monkeypatch.setattr(cuda, 'get_device_properties',
                        lambda device=None: types.SimpleNamespace(total_memory=FRACTION_TEST_TOTAL))
    monkeypatch.setattr(cuda, 'get_per_process_memory_fraction', lambda device=None: FRACTION_TEST_SHARE,
                        raising=False)
    budget = FRACTION_TEST_FREE + FRACTION_TEST_RESERVED - FRACTION_TEST_ALLOCATED
    allowed = int(FRACTION_TEST_SHARE * FRACTION_TEST_TOTAL) - FRACTION_TEST_ALLOCATED
    assert allowed < budget
    assert model.device_free_bytes('cuda') == allowed
    monkeypatch.setattr(cuda, 'get_per_process_memory_fraction', lambda device=None: 1.0)
    assert model.device_free_bytes('cuda') == budget


def test_transmission_buffer_fallbacks_cpu(monkeypatch):
    """transmission_buffer keeps the given host T in pageable host memory, with a warning, when pinning fails or the
    host memory is short and the GPU has no room either. After one failure, it does not try to pin again. The GPU and
    the memory are simulated, so the test runs on the CPU."""
    from mbirtorch.hsnt.strain import model
    monkeypatch.setattr(model, '_PROCESS_WARNED', set())
    monkeypatch.setattr(model, '_PIN_FAILED', set())
    monkeypatch.setattr(model, '_compute_device', lambda device: torch.device('cuda', 0))
    monkeypatch.setattr(model, 'device_free_bytes', lambda device: 0)
    tries = []

    def no_pinning(shape, dtype, device=None):
        tries.append(shape)
        return _no_pinning(shape, dtype, device)
    monkeypatch.setattr(model, 'pinned_empty', no_pinning)
    T = torch.ones(BUFFER_TEST_SHAPE)
    with pytest.warns(FallbackWarning, match='without pinning'):
        buf, storage = model.transmission_buffer(T.shape, T.dtype, 'cuda', given=T)
    assert buf is T and storage == 'host' and len(tries) == 1 and 0 in model._PIN_FAILED
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)            # The same size and cause are reported once.
        buf, storage = model.transmission_buffer(T.shape, T.dtype, 'cuda', given=T)
    assert buf is T and storage == 'host' and len(tries) == 1
    monkeypatch.setattr(model, '_PIN_FAILED', set())
    monkeypatch.setattr(model, 'host_free_bytes', lambda: T.numel() * T.element_size())
    with pytest.warns(FallbackWarning, match='of host memory is free'):
        buf, storage = model.transmission_buffer(T.shape, T.dtype, 'cuda', given=T)
    assert buf is T and storage == 'host' and len(tries) == 1


def _no_pinning(shape, dtype, device=None):
    """A stand-in for model.pinned_empty on a computer where pinning fails: it returns pageable host memory."""
    return torch.empty(tuple(int(s) for s in shape), dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='host storage needs a GPU')
def test_t_storage_auto_pinning_fails(monkeypatch):
    """When pinning fails, t_storage 'auto' keeps T on the GPU, without prefetch, and warns once per process. The fit
    gives the data loss of an explicit 'device' fit. When the GPU has no room either, the fit keeps the given host T
    itself, without a copy."""
    from mbirtorch.hsnt.strain import model
    monkeypatch.setattr(model, '_PROCESS_WARNED', set())
    monkeypatch.setattr(model, '_PIN_FAILED', set())
    monkeypatch.setattr(model, 'pinned_empty', _no_pinning)
    ref = _small_fit_cubic4()
    with pytest.warns(FallbackWarning, match='cannot (be )?pin'):
        a = _storage_fit(ref, ref.T.clone(), 'cuda')
    assert a.t_storage == 'device' and a.T.is_cuda and not a.t_prefetch
    b = _storage_fit(ref, ref.T.clone().cuda(), 'cuda', t_storage='device')
    assert abs(a.data_loss() - b.data_loss()) <= GPU_RTOL * abs(b.data_loss())
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)                # The same computer is reported once.
        _storage_fit(ref, ref.T.clone(), 'cuda')
    monkeypatch.setattr(model, '_PROCESS_WARNED', set())
    monkeypatch.setattr(model, 'device_free_bytes', lambda device: 0)
    T = ref.T.clone()
    with pytest.warns(FallbackWarning, match='without pinning'):
        c = _storage_fit(ref, T, 'cuda')
    assert c.T is T and c.t_storage == 'host' and not c.t_prefetch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='host storage needs a GPU')
def test_t_storage_explicit_host_pin_fails(monkeypatch):
    """When pinning fails, t_storage 'host' keeps the given host T itself instead of a second pageable copy, and it
    warns once per process."""
    from mbirtorch.hsnt.strain import model
    monkeypatch.setattr(model, '_PROCESS_WARNED', set())
    monkeypatch.setattr(model, '_PIN_FAILED', set())
    monkeypatch.setattr(model, 'pinned_empty', _no_pinning)
    ref = _small_fit_cubic4()
    T = ref.T.clone()
    with pytest.warns(FallbackWarning, match='cannot (be )?pin'):
        fit = _storage_fit(ref, T, 'cuda', t_storage='host')
    assert fit.T is T and fit.t_storage == 'host' and not fit.t_pinned and not fit.t_prefetch
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        _storage_fit(ref, ref.T.clone(), 'cuda', t_storage='host')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='host storage needs a GPU')
def test_t_storage_auto_low_host_memory(monkeypatch):
    """When a pinned copy of T would take more than T_HOST_MAX_SHARE of the free host memory, t_storage 'auto' keeps T
    on the GPU and warns about the host memory. When the GPU has no room either, the fit keeps the given host T
    itself, without pinning and without prefetch, and warns about that."""
    from mbirtorch.hsnt.strain import model
    monkeypatch.setattr(model, '_PROCESS_WARNED', set())
    monkeypatch.setattr(model, '_PIN_FAILED', set())
    ref = _small_fit_cubic4()
    T = ref.T.clone()
    monkeypatch.setattr(model, 'host_free_bytes', lambda: T.numel() * T.element_size())
    with pytest.warns(FallbackWarning, match='host memory'):
        fit = _storage_fit(ref, T, 'cuda')
    assert fit.t_storage == 'device' and fit.T.is_cuda
    monkeypatch.setattr(model, 'device_free_bytes', lambda device: 0)
    with pytest.warns(FallbackWarning, match='without pinning'):
        fit = _storage_fit(ref, T, 'cuda')
    assert fit.t_storage == 'host' and fit.T is T and not fit.t_prefetch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='host storage needs a GPU')
def test_t_storage_auto_host(monkeypatch):
    """Where pinning works, t_storage 'auto' keeps a host T in pinned host memory with prefetch, and it keeps a T that
    is already pinned as the same tensor. Nothing is warned."""
    from mbirtorch.hsnt.strain import model
    monkeypatch.setattr(model, '_PROCESS_WARNED', set())
    monkeypatch.setattr(model, '_PIN_FAILED', set())
    if not model.pinned_empty((1,), torch.float32, 'cuda').is_pinned():
        pytest.skip('this computer cannot pin host memory (as under WSL)')
    ref = _small_fit_cubic4()
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        fit = _storage_fit(ref, ref.T.clone(), 'cuda')
        assert fit.t_storage == 'host' and fit.T.is_pinned() and fit.t_pinned and fit.t_prefetch
        Tp = model.pinned_empty(ref.T.shape, ref.T.dtype, 'cuda')
        Tp.copy_(ref.T)
        kept = _storage_fit(ref, Tp, 'cuda')
        assert kept.T is Tp and kept.t_storage == 'host' and kept.t_prefetch


def test_lean_keep_small_free_memory(monkeypatch):
    """With lean_passes and no free memory on the fit's device, the fit keeps no model parts, warns once over all its
    iterations, and gives the objectives and states of the fit without lean_passes. keep_gb 0, the user's own setting,
    is not warned about."""
    from mbirtorch.hsnt.strain import model
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        _lean_run(False, True, True, keep='none')
    ref, again = _lean_run(False, True, False), _lean_run(False, True, False)
    records = []
    keep_start = model.FitB._keep_start

    def spy(self, g, cache=None):
        rec = keep_start(self, g, cache)
        records.append(rec)
        return rec
    monkeypatch.setattr(model.FitB, '_keep_start', spy)
    monkeypatch.setattr(model, 'device_free_bytes', lambda device: 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        lean = _lean_run(False, True, True)
    hits = [w for w in caught
            if issubclass(w.category, FallbackWarning) and 'keeps the model values of 0' in str(w.message)]
    assert len(hits) == 1, [str(w.message) for w in caught]
    made = [r for r in records if r is not None]
    assert len(made) > 1 and all(not r['parts'] for r in made)
    noise, got = _lean_differences(again, ref), _lean_differences(lean, ref)
    for k in got:
        assert got[k] <= LEAN_NOISE_MARGIN * noise[k], (k, got[k], noise[k])


def test_lean_keep_remedy(monkeypatch):
    """The lean_passes warning names a remedy that works. When keep_gb holds all the model parts, it names the free
    memory that keeps them all. When keep_gb holds only part of them, it names the free memory that reaches keep_gb,
    and it says to raise keep_gb to keep them all."""
    from mbirtorch.hsnt.strain import model
    from mbirtorch.hsnt.strain.model import GB
    monkeypatch.setattr(model, 'device_free_bytes', lambda device: 0)
    fits = [_small_fit(), _small_fit()]
    M = fits[0].M
    need = fits[0].P * (M.K + M.J) * torch.finfo(M.dt).bits // 8
    fits[1].keep_gb = KEEP_TEST_SHARE_OF_NEED * need / GB
    with pytest.warns(FallbackWarning, match='To keep them all, free') as rec:
        fits[0]._keep_start(fits[0].g)
    assert all('keep_gb' not in str(w.message) for w in rec)
    with pytest.warns(FallbackWarning, match="raise the fit's keep_gb to"):
        fits[1]._keep_start(fits[1].g)


def _outlier_fit():
    """_lean_fit('powder') in float64 with lean_syncs on and the last chunk's strains at OUTLIER_STRAIN."""
    fit = _lean_fit('powder', torch.float64)
    eps = fit.eps.clone()
    eps[fit.chunks()[-1]] = OUTLIER_STRAIN
    fit.eps = eps
    fit.lean_syncs = True
    fit.use_memo = False
    return fit


def test_window_plan_oversized_falls_back():
    """With lean_syncs, a material whose shared window would be more than WINDOW_PLAN_MAX_RATIO times any chunk's own
    width gets per-chunk windows and a warning. The data loss and the ray systems equal those with lean_syncs off.
    Without the outlying chunk, the shared width stays and nothing is warned."""
    fit = _outlier_fit()
    with pytest.warns(FallbackWarning, match='chunk by chunk'):
        on = fit.data_loss()
    assert fit._window_widths(fit.g, fit.eps)[0] is None
    cache = fit._state(fit.g)
    sys_on = [fit.ray_system(s, cache) for s in fit.chunks()]
    fit.lean_syncs = False
    off = fit.data_loss()
    cache = fit._state(fit.g)
    sys_off = [fit.ray_system(s, cache) for s in fit.chunks()]
    assert abs(on - off) <= WINDOW_FALLBACK_RTOL * abs(off), (on, off)
    for (G1, F1), (G0, F0) in zip(sys_on, sys_off):
        assert _close(G1, G0, WINDOW_FALLBACK_RTOL) and _close(F1, F0, WINDOW_FALLBACK_RTOL)
    plain = _lean_fit('powder', torch.float64)
    plain.lean_syncs = True
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        plain.data_loss()
        assert isinstance(plain._window_widths(plain.g, plain.eps)[0], int)


def test_fallback_warns_once():
    """A fallback warns once per fit, also when its cause is found again on later passes."""
    fit = _outlier_fit()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for _ in range(WARN_ONCE_PASSES):
            fit.eps = fit.eps.clone()                 # new strains, so that each pass works out its windows again
            fit.data_loss()
    assert sum(issubclass(w.category, FallbackWarning) for w in caught) == 1


def test_closed_form_strain_fallbacks():
    """closed_form_strain falls back to forward mode for a chunk with a ray at CUBIC4_FLOOR, warns once, and gives the
    result with the option off. A powder and the extinction shape model need no warning: the powder takes the closed
    form, and the closed form does not apply to the extinction model."""
    fit = _lean_fit('floor', torch.float64)
    cache = fit._state(fit.g)
    with torch.no_grad():
        ref = [fit.ray_jacobian(s, cache) for s in fit.chunks()]
        fit.closed_form_strain = True
        with pytest.warns(FallbackWarning, match='texture'):
            got = [fit.ray_jacobian(s, cache) for s in fit.chunks()]
    for k, ((J0, m0), (J1, m1)) in enumerate(zip(ref, got)):
        tol = CLOSED_FALLBACK_RTOL if k == 0 else CLOSED_FORM_RTOL          # chunk 0 reaches the floor
        assert _close(J1, J0, tol) and _close(m1, m0, tol), k
    for other in (_lean_fit('powder', torch.float64), _small_fit(shape_model='extinction')):
        other.closed_form_strain = True
        cache = other._state(other.g)
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter('error', FallbackWarning)
            for s in other.chunks():
                other.ray_jacobian(s, cache)

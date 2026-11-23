"""Tests of the experimental strain-tomography projectors in ``mbirtorch.hsnt.strain``."""
import math
import numpy as np
import pytest
import torch

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
            assert torch.allclose(s_, c_, rtol=1e-2)


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


def _fft_fit(n, R, stencil='corner', order=2, elastic=True, dtype=torch.float64, seed=0, proj=None):
    """A TomoFit with only what the 'fft' preconditioner reads: a cube of material inside an empty volume, the package's
    prior strengths, and random ray information and view weights.  proj None gives a _RayProjector.  Returns the fit and
    the preconditioner's arguments (F, lam_, pV, dE_, fV, fE), on the projector's device."""
    from mbirtorch.hsnt.strain.tomo import TomoFit, EQUILIBRIUM_SD, STRAIN_SMOOTH2_SD, STRAIN_PRIOR_SD
    gen = torch.Generator().manual_seed(seed)
    C = 6
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


def _volume_step_fit(seed=0):
    """A TomoFit for one volume step on a MultiProjector with mounts: one bcc material in a cube, flat data, random
    strain tensors, random view weights and the equilibrium prior."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    from mbirtorch.hsnt.strain.tomo import TomoFit
    C = 6
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
    gen = torch.Generator().manual_seed(seed)
    V = torch.zeros(N, N, N, 1); V[3:9, 3:9, 3:9] = 0.05
    E = 1e-4 * torch.randn(N, N, N, 1, C, generator=gen)
    w = 2 * torch.rand(pr.V, C, generator=gen) - 1
    return TomoFit(fit, pr, V.to(dev), E.to(dev), weights=w.to(dev), elastic=[(211e9, 0.29)])


def test_volume_step_fft_compact():
    """A volume step with fft_compact runs on the package projector, and it agrees with the default step: the Newton
    decrement and the objective's decrease match to the precision of single-precision conjugate gradients."""
    a, b = _volume_step_fit(), _volume_step_fit()
    b.fft_compact = True
    ua, da = a.volume_step()
    ub, db = b.volume_step()
    assert b.fft_compact_storage in ('real', 'complex') and math.isfinite(ub) and db > 0, (ub, db)
    assert abs(ub - ua) <= 5e-2 * abs(ua) and abs(db - da) <= 5e-2 * abs(da), (ua, ub, da, db)


def _small_fit(seed=0):
    """A small FitB on the CPU: one bcc material, per-ray dilations, one contamination component, counts from the model."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    lam = np.linspace(2.0, 4.4, 300)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=5, n_contam=1,
                device='cpu', dtype=torch.float64)
    g = torch.zeros(M.n_global, dtype=torch.float64)
    sl = M.slices[0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3; g[sl['c'][0]] = 0.2
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor([0.0, 0.001, 0.002, 0.0, 0.1, 0.02, 0.0, 0.0], dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    P = 40
    W = 0.5 + torch.rand(P, 1, generator=gen, dtype=torch.float64)
    eps = 1e-3 * torch.randn(P, 1, generator=gen, dtype=torch.float64)
    f = torch.zeros(1, M.K, dtype=torch.float64); f[0, 100:140] = 0.05
    fit = FitB(M, torch.ones(P, M.K, dtype=torch.float64), g, W, f, eps=eps, chunk=16)
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
    fit = FitB(M, torch.ones(P, M.K, dtype=torch.float64), g, W, torch.zeros(1, M.K, dtype=torch.float64), eps=eps, chunk=16)
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
    T = torch.poisson(1e5 * torch.exp(-(proj.fwd(V).double() @ torch.as_tensor(mu)))) / 1e5
    lat = [None, None]; a0 = [None, None]; init = None
    if known is not None:
        lat[known] = mats[known][0]; a0[known] = mats[known][1]
        init = [None, None]; init[known] = mu[known]
    _, info = warmup(T.float(), lam, np.full(lam.size, 1e5), lat, a0, proj, device='cpu', init_spectra=init,
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
    tf = TomoFit(fit, pr, V, E); tf.sync(); tf.edge_spread = True; tf.refresh_spread()
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
    return (_storage_fit(ref, T.clone().to(device), device, ob),
            _storage_fit(ref, T.clone(), device, ob, t_storage='host'))


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
    torch.manual_seed(0)
    T = (torch.poisson(1e5 * torch.exp(-(proj.fwd(V).double() @ torch.as_tensor(mu)))) / 1e5).float()
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
                          ray_mask=ray_mask, t_storage=storage))
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
    a = _storage_fit(ref, T.clone().to(device), device, ob)
    b = _storage_fit(ref, T.clone(), device, ob, t_storage=storage, t_dtype='float16')
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
    a = _storage_fit(ref, ref.T.clone(), 'cuda', t_storage='host')
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

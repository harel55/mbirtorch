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
    M = Hybrid2(lam, np.ones_like(lam), [('fcc', (3.52 * 0.99, 3.52 * 1.01))], smooth_degree=5, n_contam=1, device='cpu',
                two_scale=True, kernel='ikeda_carpenter')
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


def _small_fit(seed=0):
    """A small FitB on the CPU: one bcc material, per-ray dilations, one contamination component, counts from the model."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    lam = np.linspace(2.0, 4.4, 300)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=5, n_contam=1,
                device='cpu', two_scale=True, dtype=torch.float64)
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
                            n_contam=1, device='cpu', two_scale=True, dtype=dt)
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


def _small_fit_texture(seed=0, strength=0.7, noise=True):
    """A small FitB on the CPU with a March-Dollase texture (texture3d) of the given strength: one bcc material, eight
    views with their own beam directions, per-ray dilations, counts from the model (their means when noise is False)."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    lam = np.linspace(2.0, 4.4, 300)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=5, n_contam=1,
                device='cpu', two_scale=True, dtype=torch.float64, shape_model='texture3d')
    g = torch.zeros(M.n_global, dtype=torch.float64)
    sl = M.slices[0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3; g[sl['c'][0]] = 0.2
    g[sl['x'][0]] = strength; g[sl['x'][0] + 1] = 0.9; g[sl['x'][0] + 2] = 0.4
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
    fit.T = torch.poisson(50.0 * m, generator=gen) / 50.0 if noise else m.clone()
    return fit


def test_texture_factor():
    """The texture factor is 1 for a random powder, averages to about 1 over beam directions, does not depend on the
    axis's sign, and differs between views for a textured sample."""
    fit = _small_fit_texture()
    M = fit.M; x0 = M.slices[0]['x'][0]; g = fit.g.clone()
    g[x0] = 1.0
    assert torch.allclose(M.texture_factor(g, 0), torch.ones(()).double(), atol=1e-9)
    g[x0] = 0.7
    d = torch.randn(4000, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    keep = M.view_dirs; M.view_dirs = (d / d.norm(dim=1, keepdim=True)).numpy()
    avg = M.texture_factor(g, 0).mean(0)                                     # over beam directions
    e = 2.0 * float(g[M.slices[0]['a']]) / np.sqrt(M.mats[0]['ns'].numpy())
    below = (M.lamf.numpy()[None, :] < 0.95 * e[:, None])                     # bins below each edge
    assert abs(float(avg.numpy()[below].mean()) - 1.0) < 0.03
    M.view_dirs = keep
    f1 = M.texture_factor(g, 0)
    g2 = g.clone(); g2[x0 + 1] = math.pi - g[x0 + 1]; g2[x0 + 2] = g[x0 + 2] + math.pi    # the opposite axis
    assert torch.allclose(f1, M.texture_factor(g2, 0), atol=1e-9)
    assert float((f1.max(0).values - f1.min(0).values).max()) > 0.1


def test_texture_global_grad():
    """With texture3d the gradient by automatic differentiation (global_grad) equals global_system's for the texture,
    height and lattice parameters (the per-view factor enters both)."""
    fit = _small_fit_texture()
    M = fit.M; sl = M.slices[0]
    idx = list(range(sl['h'][0], sl['h'][1])) + list(range(sl['x'][0], sl['x'][1]))
    g1, F = fit.global_system(idx)
    g2 = fit.global_grad(idx)
    assert torch.allclose(g1, g2, rtol=1e-5, atol=1e-7 * float(g1.abs().max()))
    assert bool((torch.diagonal(F) > 0).all())


def test_texture_search():
    """texture_search, started from a random powder, finds an axis whose loss is below the powder's and close to the
    truth's (the small fit's data come from a texture of strength 0.7)."""
    fit = _small_fit_texture()
    M = fit.M; x0 = M.slices[0]['x'][0]
    l_true = fit.loss()
    with torch.no_grad():
        fit.g[x0] = 1.0; fit.g[x0 + 1] = 0.3; fit.g[x0 + 2] = 2.0
    l_powder = fit.loss()
    l_best, (st, th, ph) = fit.texture_search(0, strength=0.7, n_dirs=48)
    assert l_best < l_powder and (l_best - l_true) < 0.5 * (l_powder - l_true)


def test_texture_search_weak():
    """texture_search finds a weak texture (strength 0.92, noiseless data): from a random powder, the axis it keeps is
    within 30 degrees of the truth, its strength is plate-like (below 1), and its loss is below the powder's."""
    fit = _small_fit_texture(strength=0.92, noise=False)
    M = fit.M; x0 = M.slices[0]['x'][0]
    with torch.no_grad():
        fit.g[x0] = 1.0; fit.g[x0 + 1] = 0.3; fit.g[x0 + 2] = 2.0
    l_powder = fit.loss()
    l_best, (st, th, ph) = fit.texture_search(0)
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    n_true = np.array([np.sin(0.9) * np.cos(0.4), np.sin(0.9) * np.sin(0.4), np.cos(0.9)])
    assert l_best < l_powder and st < 1.0 and abs(float(n @ n_true)) > np.cos(np.radians(30))


def _small_fit_cubic4(seed=0, coeffs=None):
    """Builds the small fit of _small_fit_texture with the cubic4 model. The data come from a degree-4 cubic texture
    with coefficients coeffs (default: a fiber about a tilted axis) on eight views."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB, harmonics4
    lam = np.linspace(2.0, 4.4, 300)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], smooth_degree=5, n_contam=1,
                device='cpu', two_scale=True, dtype=torch.float64, shape_model='cubic4')
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

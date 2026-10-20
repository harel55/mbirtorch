"""Tests of the experimental strain-tomography projectors in ``mbirtorch.hsnt.strain``."""
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
    a = float((proj.fwd(x) * y).sum())
    b = float((x * proj.back(y)).sum())
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
    M = Hybrid2(lam, np.ones_like(lam), [('fcc', (3.52 * 0.99, 3.52 * 1.01))], n_spline=4, n_contam=1, device='cpu',
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


def test_fft_preconditioner_symbol():
    """The strain priors' Fourier symbol (TomoFit._prior_symbol) reproduces their Hessian on a field away from the faces."""
    from mbirtorch.hsnt.strain.tomo import TomoFit
    n = 12
    tf = object.__new__(TomoFit)
    tf.R, tf.C = 1, 6
    tf.V = torch.ones(n, n, n, 1, dtype=torch.float64)
    tf.elastic = [(211e9, 0.29)]
    tf.lam_E, tf.smooth_E, tf.eq_w = 0.3, 2.0, 5.0
    g = torch.Generator().manual_seed(0)
    X = torch.zeros(n, n, n, 1, 6, dtype=torch.float64)
    X[4:8, 4:8, 4:8] = torch.randn(4, 4, 4, 1, 6, generator=g, dtype=torch.float64)
    full = torch.ones(n, n, n, 1, dtype=torch.bool)
    div = lambda Y: tf._divergence(Y, full).reshape(*Y.shape[:3], -1)
    direct = 2 * tf.lam_E * X + tf.smooth_E * tf.laplacian(X) + tf.eq_w * torch.func.vjp(div, X)[1](div(X))[0]
    sym = tf._prior_symbol()
    via = torch.fft.ifftn(torch.einsum('xyzrab,xyzrb->xyzra', sym, torch.fft.fftn(X, dim=(0, 1, 2))), dim=(0, 1, 2)).real
    assert torch.allclose(via, direct, atol=1e-10 * float(direct.abs().max()))


def _small_fit(seed=0):
    """A small FitB on the CPU: one bcc material, per-ray dilations, one contamination component, counts from the model."""
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    lam = np.linspace(2.0, 4.4, 300)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], n_spline=4, n_contam=1,
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

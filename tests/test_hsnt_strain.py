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


def test_spike_phantom_equilibrium():
    """The spike phantom's press-fit strain gives a stress with zero divergence inside each part (finite differences,
    away from the interfaces); the parts' volume fractions add to at most 1; the parts' largest strain is SPIKE_PEAK."""
    from scipy import ndimage as nd
    from mbirtorch.hsnt.strain.phantom import spike_tensor, spike_fields, SPIKE_ELASTIC, SPIKE_MATERIALS, SPIKE_PEAK
    N = 96                                       # fine enough that finite differences of the 1/r^2 field near the hole
    #                                              stay well under the tolerance (1.25% of a voxel's jump at 64)
    mat, eps = spike_tensor(N)
    assert all(int((mat == r).sum()) > 0 for r in range(3))
    frac, eps_parts = spike_fields(32)
    assert abs(float(np.abs(eps_parts).max()) - SPIKE_PEAK) < 1e-12                # the peak, on the parts' own fields
    assert float(frac.sum(-1).max()) <= 1 + 1e-12 and 0 < float(frac[..., 1].sum()) < float(frac[..., 2].sum())
    sig = np.zeros_like(eps)
    for r, m in enumerate(SPIKE_MATERIALS):
        E, nu = SPIKE_ELASTIC[m['name']]; lam_, mu_ = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
        sel = mat == r; tr = eps[..., :3].sum(-1)
        for k in range(3):
            sig[..., k] = np.where(sel, lam_ * tr + 2 * mu_ * eps[..., k], sig[..., k])
        for k in (3, 4, 5):
            sig[..., k] = np.where(sel, 2 * mu_ * eps[..., k], sig[..., k])
    D = lambda a, ax: np.gradient(a, 1.0 / N, axis=ax)                        # volume axes (y, x, z)
    sxx, syy, szz, sxy, syz, sxz = (sig[..., k] for k in range(6))
    f = np.sqrt((D(sxx, 1) + D(sxy, 0) + D(sxz, 2)) ** 2 + (D(sxy, 1) + D(syy, 0) + D(syz, 2)) ** 2 + (D(sxz, 1) + D(syz, 0) + D(szz, 2)) ** 2)
    jump = float(np.abs(sig).max()) * N                                        # a full stress jump over one voxel
    for r in range(3):
        interior = nd.binary_erosion(mat == r, iterations=2)
        assert float(np.percentile(f[interior], 99)) < 1e-2 * jump


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


def test_fe_cube_offset():
    """fe_cube_tensor's offset moves the cube and its strain by whole voxels when it is a multiple of 1 / N."""
    from mbirtorch.hsnt.strain.phantom import fe_cube_tensor
    N = 32
    c0, e0 = fe_cube_tensor(N)
    c1, e1 = fe_cube_tensor(N, 1.0 / N)
    roll = lambda a: np.roll(np.roll(np.roll(a, 1, 0), 1, 1), 1, 2)
    assert np.array_equal(roll(c0), c1)
    assert np.allclose(roll(e0), e1)


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
    mk = lambda dt: Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], n_spline=4,
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
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], n_spline=4, n_contam=1,
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
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], n_spline=4, n_contam=1,
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


def test_fiber_texture_degree4():
    """A cubic fiber texture (phantom.fiber_texture_factor) is 1 for a random powder and averages to 1 over beam
    directions. Its edge factors are mostly of degree 4, and each reflection's degree-4 part is k4(hkl) times one
    common function of the beam direction, which is the form of the cubic4 model."""
    from mbirtorch.hsnt.strain.phantom import fe_spectrum_model, fiber_texture_factor
    from mbirtorch.hsnt.strain.model import harmonics4, cubic_k4, _sphere_quadrature
    lam = np.linspace(1.9, 4.4, 600)
    M, g = fe_spectrum_model(lam, np.full_like(lam, 1000.0), 16, 'cpu', texture={'fiber': (0.75, 0.9, 0.4)})
    pts, w = _sphere_quadrature(); M.view_dirs = pts.numpy(); w = w.numpy()      # beam directions: an exact quadrature
    assert float((fiber_texture_factor(M, 0, 1.0, 0.9, 0.4) - 1).abs().max()) < 1e-12
    F = fiber_texture_factor(M, 0, 0.75, 0.9, 0.4).double().numpy()
    ns = M.mats[0]['ns'].numpy(); e = 2 * 0.5 * sum(M.mats[0]['a_range']) / np.sqrt(ns)
    at = np.array([F[:, i, int(np.argmin(np.abs(M.lamf.numpy() - 0.999 * ee)))] for i, ee in enumerate(e)])
    mean = at @ w
    assert np.allclose(mean, 1.0, atol=0.01)
    Y = harmonics4(pts).numpy()
    c = (at - mean[:, None]) @ (w[:, None] * Y)                            # each edge's degree-4 coefficients
    share = (c ** 2).sum(1) / (((at - mean[:, None]) ** 2) @ w)
    assert share.min() > 0.8
    ratio = c / np.array(cubic_k4('bcc', ns))[:, None]                    # the common degree-4 function, per edge
    assert np.allclose(ratio, ratio.mean(0), atol=0.03 * np.abs(ratio).max())


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


def test_skin_stress_field():
    """The skin phantom's stress (phantom.fe_cube_skin_stress) is in equilibrium inside the cube; on the faces the
    traction is near zero while the in-plane stresses are not, and they exceed the mid-plane's."""
    from mbirtorch.hsnt.strain.phantom import fe_cube_skin_stress
    from scipy import ndimage as nd
    cube, sig = fe_cube_skin_stress(32)
    g = lambda a, ax: np.gradient(a, axis=ax)                               # axes: rows y, cols x, slices z
    xx, yy, zz, xy, yz, xz = [sig[..., k] for k in range(6)]
    div = [g(xx, 1) + g(xy, 0) + g(xz, 2), g(xy, 1) + g(yy, 0) + g(yz, 2), g(xz, 1) + g(yz, 0) + g(zz, 2)]
    deep = nd.binary_erosion(cube, iterations=2)
    assert max(np.abs(d)[deep].max() for d in div) < 0.05 * np.abs(g(xx, 1))[deep].max()
    c0 = np.where(cube.any((0, 2)))[0][0]; face = cube[:, c0, :]; peak = np.abs(sig[cube]).max()
    traction = max(np.abs(sig[:, c0, :, k][face]).mean() for k in (0, 3, 5)) / peak
    inplane = max(np.abs(sig[:, c0, :, k][face]).mean() for k in (1, 2)) / peak
    mid = cube[:, 16, :]; inplane_mid = max(np.abs(sig[:, 16, :, k][mid]).mean() for k in (1, 2)) / peak
    assert traction < 0.05 and inplane > 3 * traction and inplane > 1.5 * inplane_mid


def test_diffraction_constants():
    """For an isotropic crystal every reflection's diffraction constants are -nu/E and (1 + nu)/E. For Fe the Hill
    moduli match the package's Fe values, and 200 responds about 35% more than 110 to a uniaxial stress."""
    from mbirtorch.hsnt.strain.phantom import diffraction_constants, hill_moduli, FE_SINGLE_CRYSTAL, FE_ELASTIC
    E, nu = FE_ELASTIC
    iso = (E * (1 - nu) / ((1 + nu) * (1 - 2 * nu)), E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu)))
    for gm in (0.0, 0.25, 1 / 3):
        s1, hs2 = diffraction_constants(iso, gm)
        assert np.isclose(s1, -nu / E) and np.isclose(hs2, (1 + nu) / E)
    Eh, nuh = hill_moduli(FE_SINGLE_CRYSTAL)
    assert abs(Eh / E - 1) < 0.01 and abs(nuh - nu) < 0.01
    r200, r110 = [sum(diffraction_constants(FE_SINGLE_CRYSTAL, gm)) for gm in (0.0, 0.25)]
    assert 1.25 < r200 / r110 < 1.45


def test_anisotropic_phantom_isotropic_limit(monkeypatch):
    """With isotropic single-crystal constants, the anisotropic phantom (one component per reflection group, the smooth
    part added once) reproduces the standard phantom's data. The cube is shifted off the grid's symmetry, as in the
    phantom runs: a symmetric cube puts many voxels' strains exactly between two levels, where rounding picks either."""
    import mbirtorch.hsnt.strain.phantom as P
    E, nu = P.FE_ELASTIC
    monkeypatch.setattr(P, 'FE_SINGLE_CRYSTAL', (E * (1 - nu) / ((1 + nu) * (1 - 2 * nu)), E * nu / ((1 + nu) * (1 - 2 * nu)),
                                                 E / (2 * (1 + nu))))
    kw = dict(n=8, angles=np.linspace(0, np.pi, 4), fine=2, device='cpu', levels=7, kernel='ikeda_carpenter',
              kernel_values=[0, 0.00118, 0.00039, 0.00059, 0, 0.0002, 0, 0], contamination=False, cube_shift=0.25)
    a = P.make_fe_cube(**kw); b = P.make_fe_cube(**kw, elastic_anisotropy=True)
    assert float((a['T'] == b['T']).double().mean()) > 0.9999


def _small_fit_debye(monkeypatch, seed=0):
    """_small_fit with the smooth part as the Debye phonon term (a coarse table, for speed)."""
    import mbirtorch.hsnt.strain.phonons as phonons
    from mbirtorch.hsnt.strain.model import Hybrid2, FitB
    monkeypatch.setattr(phonons, 'N_THETA', 12); monkeypatch.setattr(phonons, 'N_WL', 40)
    monkeypatch.setattr(phonons, 'BETA_POINTS', 60); monkeypatch.setattr(phonons, '_TABLES', {})
    lam = np.linspace(2.0, 4.4, 300)
    M = Hybrid2(lam, np.full_like(lam, 50.0), [('bcc', (2.8665 * 0.99, 2.8665 * 1.01))], n_contam=1, device='cpu',
                two_scale=True, dtype=torch.float64, smooth_model='debye', masses=[55.845])
    g = torch.zeros(M.n_global, dtype=torch.float64)
    sl = M.slices[0]
    g[sl['a']] = 2.8665; g[sl['h'][0]:sl['h'][1]] = 0.3; g[sl['c'][0]] = 0.2; g[sl['c'][0] + 1] = 0.1
    g[sl['s'][0]] = 1.0; g[sl['s'][0] + 1] = math.log(430.0)
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor([0.0, 0.001, 0.002, 0.0, 0.1, 0.02, 0.0, 0.0], dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    P = 40
    W = 0.5 + torch.rand(P, 1, generator=gen, dtype=torch.float64)
    fit = FitB(M, torch.ones(P, M.K, dtype=torch.float64), g, W, torch.zeros(1, M.K, dtype=torch.float64), chunk=16)
    with torch.no_grad():
        m = torch.cat([fit.mean(s)[0] for s in fit.chunks()])
    fit.T = torch.poisson(50.0 * m, generator=gen) / 50.0
    return fit


def test_debye_smooth_model(monkeypatch):
    """smooth_model 'debye': the phonon term enters the spectrum, global_grad equals global_system's gradient for its
    parameters, the starting fit runs, and rescaling a material leaves its Debye temperature alone."""
    from mbirtorch.hsnt.strain.model import ls_init_material
    fit = _small_fit_debye(monkeypatch)
    M = fit.M; sl = M.slices[0]
    g0 = fit.g.clone(); g0[sl['s'][0]] = 0.0
    assert float((M.spectra(fit.g) - M.spectra(g0)).abs().max()) > 1e-3                 # the term is not zero
    idx = [i for i in range(M.n_global) if not (M.res_sl[0] <= i < M.res_sl[1]) and float(M.ub[i]) > float(M.lb[i])]
    assert sl['s'][0] + 1 in idx                                                         # the Debye temperature is free
    g1, _ = fit.global_system(idx)
    g2 = fit.global_grad(idx)
    assert torch.allclose(g1, g2, rtol=1e-6, atol=1e-8 * float(g1.abs().max()))
    assert float(fit.spline_penalty(fit.g)) == 0.0
    theta = float(fit.g[sl['s'][0] + 1])
    fit.rescale_material(0, 2.0)
    assert abs(float(fit.g[sl['s'][0] + 1]) - theta) < 1e-12
    g = g0.clone(); g[sl['s'][0]] = 1.0; g[sl['s'][0] + 1] = math.log(430.0)
    tgt = np.interp(M.lam.cpu().numpy(), M.lamf.cpu().numpy(), M.spectra(g)[0].cpu().numpy())
    g_init = g.clone(); g_init[sl['s'][0]] = 0.0
    rms = ls_init_material(M, g_init, 0, tgt, 2.8665)
    assert np.isfinite(rms) and float(g_init[sl['s'][0]]) > 0.0                        # the phonon term is used

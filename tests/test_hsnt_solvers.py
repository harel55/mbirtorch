"""Tests for the NNAL solvers in mbirtorch.hsnt: the maximum-likelihood fit, its stop and compiled kernels, the rank
estimate, the spectra estimators and streaming.

Self-contained: small random nonnegative factorizations with mixed pixels stand in for a phantom. Each test runs on
every device of the repository's ``device`` fixture except MPS, which lacks the float64 the solvers accumulate in.
"""
import itertools

import numpy as np
import pytest
import torch

import mbirtorch.hsnt as hsnt
from mbirtorch.hsnt import _newton
from mbirtorch.hsnt._loss import _nnal_prep, stable_nnal, stable_nnal_derivatives
from mbirtorch.hsnt._streaming import _stream_factorization
from mbirtorch.hsnt.factorization import _initial_factors, _nnal_factorization
from mbirtorch.hsnt.spectra import (_auto_penalty, _empty_fit_loss, _guard_components, _select_supports,
                                    _support_selected_spectra, _unconstrained_spectra)
from hsnt_basis import material_basis


@pytest.fixture(autouse=True, scope="module")
def _one_torch_thread():
    """One intra-op thread, so parallel test workers do not each start one per core."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture
def dev(device):
    if device == "mps":
        pytest.skip("the hsnt solvers accumulate in float64, which MPS does not support")
    return device


def _problem(device, P=2048, K=200, R=3, dose=10.0, seed=0, noisy=True, dtype=torch.float32, alpha=0.5, absent=0.4):
    """T = counts / dose for X = W_true @ H_true: background pixels, pure pixels, and pixels mixing two or three
    materials, as at the boundaries of a real sample. alpha (the Dirichlet concentration of the mixtures) and absent
    (the chance a material is missing from a pixel) set how dense the mixtures are."""
    rng = np.random.default_rng(seed)
    H = rng.uniform(0.05, 1.0, size=(R, K))
    H[:, K // 3:] *= 0.5                                                            # rough edge structure
    W = rng.dirichlet(np.full(R, alpha), P) * rng.uniform(0.2, 2.0, (P, 1))
    W[rng.uniform(size=(P, R)) < absent] = 0.0                                      # absent materials
    W[: P // 8] = 0                                                                 # background pixels
    X = W @ H
    T = rng.poisson(dose * np.exp(-X)) / dose if noisy else np.exp(-X)
    return (torch.tensor(T, dtype=dtype, device=device), torch.tensor(W, dtype=torch.float64, device=device),
            torch.tensor(H, dtype=torch.float64, device=device))


def _sphere_problem(device, n=48, K=150, dose=100.0, seed=7):
    """Three overlapping spheres seen along one axis, with the reference spectra: most pixels mix two materials."""
    basis = material_basis()
    yy, xx = np.mgrid[:n, :n] + 0.5
    maps = []
    for (cy, cx), density in zip(((0.38, 0.38), (0.38, 0.62), (0.6, 0.5)), (0.25, 0.25, 0.75)):
        d2 = ((yy - cy * n) ** 2 + (xx - cx * n) ** 2) / (0.22 * n) ** 2
        maps.append(10.0 * density * np.sqrt(np.clip(1.0 - d2, 0.0, None)))       # a chord of a diameter-10 sphere
    X = torch.tensor(np.stack(maps, -1).reshape(-1, 3), dtype=torch.float32, device=device) @ torch.tensor(
        basis[:, ::basis.shape[1] // K].copy(), device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.poisson(dose * torch.exp(-X), generator=g) / dose


def _loss(W, H, T):
    return stable_nnal(W.double() @ H.double(), T.double()).item()


def _mle(T, max_steps=200, rel_tol=1e-8):
    return _nnal_factorization(T, 3, max_steps=max_steps, rel_tol=rel_tol, compile_mode="off")


def _skip_without_triton():
    from mbirtorch.kernel_availability import triton_available
    usable, reason = triton_available()
    if not usable:
        pytest.skip(f"torch.compile needs Triton on CUDA: {reason}")


def test_mle_fits_noisy_data_and_reaches_machine_precision_on_exact_data(dev):
    T, _, _ = _problem(dev)
    W, H, steps = _mle(T, max_steps=300, rel_tol=1e-6)
    W0, H0 = _initial_factors(T, 3)
    assert steps > 0 and W.min() >= 0 and H.min() >= 0 and _loss(W, H, T) < _loss(W0, H0, T)
    T, _, _ = _problem(dev, noisy=False, dtype=torch.float64)
    W, H, _ = _mle(T, max_steps=300, rel_tol=1e-6)
    assert _loss(W, H, T) < 1e-8 * T.numel()
    with pytest.raises(ValueError, match="compile_mode"):
        _nnal_factorization(T, 3, compile_mode="default")


def test_a_dead_component_is_revived(dev):
    """A component whose spectrum or map is zero gets no gradient and would stay out of the fit; the solve re-seeds
    it and reaches the loss of an ordinary start."""
    T, _, _ = _problem(dev)
    W0, H0 = _initial_factors(T, 3)
    W_ref, H_ref, _ = _mle(T)
    W0[:, 2] = 0
    H0[2] = 0                                                                       # dead in both factors
    W, H, _ = _nnal_factorization(T, 3, max_steps=200, rel_tol=1e-8, compile_mode="off", W_init=W0, H_init=H0)
    assert H[2].norm() > 0 and abs(_loss(W, H, T) - _loss(W_ref, H_ref, T)) <= 1e-6 * _loss(W_ref, H_ref, T)


def test_a_start_in_a_skewed_gauge_stops_where_the_balanced_start_does(dev):
    """The KKT stop measures each component's gradient against its factor, so a start with one component rescaled by
    1e6 (the same X) stops at the same loss in about as many steps; a plain gradient norm, dominated by the rescaled
    component, stopped it 0.036 nats high after 527 steps."""
    T, _, _ = _problem(dev)
    W0, H0 = _initial_factors(T, 3)
    W1, H1, s1 = _nnal_factorization(T, 3, compile_mode="off", W_init=W0, H_init=H0)
    D = torch.tensor([1e6, 1.0, 1.0], dtype=W0.dtype, device=dev)
    W2, H2, s2 = _nnal_factorization(T, 3, compile_mode="off", W_init=W0 / D, H_init=H0 * D[:, None])
    assert abs(_loss(W2, H2, T) - _loss(W1, H1, T)) <= 1e-8 * _loss(W1, H1, T) and s2 <= 2 * s1


def test_the_w_solve_reaches_stationarity_next_to_an_inward_entry(dev):
    """A bound entry with an inward gradient takes a scaled-gradient step outside the Newton system, so its partners'
    Newton moves do not assume a step it does not take: the W solve reaches a small projected gradient on data with
    nearly collinear spectra, where such pixels used to stall."""
    basis = material_basis()
    rng = np.random.default_rng(1)
    Hn = basis[:3, ::4][:, :300].astype(np.float64)
    W = np.zeros((4096, 3))
    W[np.arange(4096), rng.integers(0, 3, 4096)] = rng.uniform(0.5, 3.0, 4096) * 3.0
    W[:512] = 0
    T = torch.tensor(rng.poisson(3.0 * np.exp(-(W @ Hn))) / 3.0, dtype=torch.float32, device=dev)
    H = torch.tensor(Hn, dtype=torch.float32, device=dev)
    Ws = _newton.solve_W(T, H, max_steps=1000, rel_tol=1e-8)
    G, _ = stable_nnal_derivatives(Ws @ H, T, _nnal_prep(T))
    g = G @ H.T
    pg = torch.where(Ws > 0, g, g.clamp(max=0))
    assert (pg.double().norm() / (T.double() @ H.T.double()).norm()).item() < 1e-6


def test_a_streamed_fit_at_one_count_per_bin_stays_finite_and_descends():
    """At one count per bin the streamed polish neither raises its loss (its snap of frozen spectrum entries is kept
    only where it does not raise a bin's loss) nor follows the zero-count direction past the attenuation bound."""
    from mbirtorch.hsnt._fit import _fit
    T = _sphere_problem("cpu", n=32, K=100, dose=1.0).numpy()
    W, H, rep = _fit(T, 3, device="cpu", mode="stream", chunk_pixels=512, max_passes=10, compile_mode="off")
    losses = rep["loss_per_pass"]
    assert np.isfinite(W).all() and np.isfinite(H).all() and len(losses) >= 2
    assert float((W.max(0) * H.max(1)).max()) <= 27.7                  # each component's peak attenuation
    assert all(b <= a * (1 + 1e-9) for a, b in zip(losses, losses[1:]))


def test_a_fit_reports_the_zero_count_divergence(caplog):
    """The fit behind dehydrate records the largest attenuation it puts on a zero count, and warns when entries sit
    at the attenuation bound, full and streamed; at one count per bin it also names the components that model the zero counts. At
    dose 100 it records neither.  Each component's peak attenuation stays at the attenuation bound."""
    from mbirtorch.hsnt._fit import _fit
    for dose, mode in ((1.0, "full"), (1.0, "stream"), (100.0, "full")):
        T = _sphere_problem("cpu", n=32, K=100, dose=dose).numpy()
        caplog.clear()
        with caplog.at_level("WARNING", logger="mbirtorch.hsnt"):
            W, H, rep = _fit(T, 3, device="cpu", mode=mode, chunk_pixels=512, max_steps=150, max_passes=3,
                             compile_mode="off")
        assert float((W.max(0) * H.max(1)).max()) <= 27.7      # each component's peak, at most -log(1e-12)
        diverges = rep["zero_count_entries_at_bound"] > 0
        assert diverges == (dose == 1.0) and diverges == ("zero-count entries" in caplog.text)
        captured = bool(rep["zero_count_components"])
        assert captured == (dose == 1.0) and captured == ("the zero counts rather than a material" in caplog.text)
        if mode == "full":                            # bounded, the fit stops on its own, not at max_steps
            assert not rep["mle_hit_max_steps"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_and_eager_solves_agree():
    """The compiled solve ends at the eager loss. Fused rounding in the compiled block step can leave ulp-sized
    residues where the eager step lands on zero; the snap to zero makes both steps take the same zero decisions."""
    _skip_without_triton()
    T, _, _ = _problem("cuda")
    W1, H1, _ = _nnal_factorization(T, 3, max_steps=200, rel_tol=1e-8, compile_mode="off")
    W2, H2, _ = _nnal_factorization(T, 3, max_steps=200, rel_tol=1e-8, compile_mode="on")
    assert abs(_loss(W1, H1, T) - _loss(W2, H2, T)) <= 1e-6 * _loss(W1, H1, T)
    T, _, _ = _problem("cuda", dose=3.0)
    prep = _nnal_prep(T)
    W, H = _initial_factors(T, 3)
    eager, compiled = _newton._kernels("off")[3], _newton._kernels("on")[3]
    for _ in range(3):
        X = W @ H
        We, _, _ = eager(W, H, X, T, prep, 0)
        Wc, _, _ = compiled(W, H, X, T, prep, 0)
        assert torch.equal(We == 0, Wc == 0)
        W = We
        H, _, _ = eager(H, W, W @ H, T, prep, 1)


def test_rank_estimate_finds_the_rank_with_pooling(dev):
    """The sphere phantom has rank 3 at full resolution and pooled.  (A rank estimated from attenuation data is run
    by the dehydrate call in test_hsnt_cli.py.)"""
    T = _sphere_problem(dev).reshape(48, 48, -1)
    n, _, detail = hsnt.estimate_rank(T, "transmission", max_rank=5, device=dev, pool=2)
    assert n == 3 and detail["rank_full"] == 3 and detail["pool_block"] == 2 and detail["pooled"]["pixels"] == 24 * 24


def test_spectra_estimators(dev):
    """The unconstrained estimate keeps W >= 0. Support selection holds the coefficients off its supports at zero,
    and with no penalty keeps at least as many. A component selected almost nowhere reverts to the maximum-likelihood
    treatment."""
    T, _, _ = _problem(dev)
    W, H, _ = _mle(T)
    Wu, Hu, _ = _unconstrained_spectra(T, W, H)
    Ws, Hs, support, _ = _support_selected_spectra(T, W, H, dose=10.0)
    assert Wu.min() >= 0 and Hu.shape == H.shape and Ws.min() >= 0 and Hs.min() >= 0
    assert bool((Ws[~support] == 0).all())
    Wz, Hz, support_z, _ = _support_selected_spectra(T, W, H, dose=10.0, penalty=0.0)
    assert support_z.sum() >= support.sum() and Wz.min() >= 0 and bool((Wz[~support_z] == 0).all())
    P = 5000
    guard = torch.rand(P, 3, device=dev) > 0.5
    guard[:, 1] = False
    guard[:3, 1] = True
    W_mle, W0 = torch.rand(P, 3, device=dev), torch.zeros(P, 3, device=dev)
    with pytest.warns(UserWarning):
        weak = _guard_components(guard, W_mle, W0)
    assert weak.tolist() == [False, True, False] and bool(guard[:, 1].all()) and torch.equal(W0[:, 1], W_mle[:, 1])


def _enumerated_supports(T, W, H, dose, lam):
    """The reference search: every subset of the R materials, fitted by a constrained W solve."""
    R = H.shape[0]
    prep = _nnal_prep(T)
    rowwise = _newton._kernels("off")[2]
    subsets = [list(c) for r in range(1, R + 1) for c in itertools.combinations(range(R), r)]
    f0 = _empty_fit_loss(T, prep)
    crit, fits, W_sub = [f0 * dose], [f0], []
    for S in subsets:
        Ws = _newton.solve_W(T, H[S].contiguous(), W[:, S].contiguous(), 100, 1e-12)
        f = rowwise(Ws @ H[S], T, prep, 1, dtype=torch.float64)
        crit.append(f * dose + lam * len(S))
        fits.append(f)
        W_sub.append(Ws)
    best = torch.stack(crit, 1).argmin(1)
    W0 = torch.zeros_like(W)
    for j, S in enumerate(subsets):
        m = (best == j + 1).nonzero().squeeze(1)
        W0[m[:, None], torch.tensor(S, device=T.device)[None, :]] = W_sub[j][m]
    return W0 > 0, torch.stack(fits, 1).gather(1, best[:, None]).squeeze(1)


def _dense_rank5(dev, dose=1000.0):
    """Rank 5 at high dose, every material in every sample pixel; the enumeration's criterion and supports."""
    T, _, _ = _problem(dev, P=512, K=100, R=5, dose=dose, alpha=1.0, absent=0.0)
    W, H, _ = _nnal_factorization(T, 5, max_steps=200, rel_tol=1e-8, compile_mode="off")
    lam = 2 * np.log(T.shape[1])
    s_enum, f_enum = _enumerated_supports(T, W, H, dose, lam)
    return T, W, H, (dose * f_enum + lam * s_enum.sum(1)).sum().item(), s_enum


def test_branch_and_bound_matches_the_enumeration(dev):
    """Branch and bound reproduces the exhaustive search's supports on nearly every pixel of the test problem at the
    same criterion, at rank 3 and on dense rank-5 mixtures. 'auto' is the penalty it names."""
    T, _, _ = _problem(dev, dose=10.0)
    W, H, _ = _mle(T)
    lam = 2 * np.log(T.shape[1])
    s_enum, f_enum = _enumerated_supports(T, W, H, 10.0, lam)
    s_bb, W_bb, f_bb = _select_supports(T, W, H, dose=10.0, penalty=2.0)

    def criterion(s, f):
        return (10.0 * f + lam * s.sum(1)).sum().item()

    assert (s_bb == s_enum).all(1).double().mean() > 0.95
    assert criterion(s_bb, f_bb) <= criterion(s_enum, f_enum) * (1 + 2e-3)
    assert W_bb.min() >= 0 and bool((W_bb[~s_bb] == 0).all()) and s_bb.dtype == torch.bool
    # At rank 5 on dense mixtures, where a third of the pixels keep all five components, it matches as well.
    T5, W5, H5, crit_enum5, s_enum5 = _dense_rank5(dev)
    s5, _, f5 = _select_supports(T5, W5, H5, dose=1000.0, penalty=2.0)
    assert (s5.sum(1) == 5).double().mean() > 0.2 and (s5 == s_enum5).all(1).double().mean() > 0.95
    assert (1000.0 * f5 + 2 * np.log(T5.shape[1]) * s5.sum(1)).sum().item() <= crit_enum5 * (1 + 1e-4)
    s_auto = _select_supports(T, W, H, dose=10.0, penalty="auto")[0]
    assert torch.equal(s_auto, _select_supports(T, W, H, dose=10.0, penalty=_auto_penalty(T.mean(1), 10.0))[0])


def test_streaming_matches_the_full_solve(dev):
    """Streamed by chunks of pixels, the MLE lands within 1% of the loss of the solve held whole, and keeps W >= 0."""
    T, _, _ = _problem(dev, P=4096)
    tiles = [T[i:i + 1024].cpu() for i in range(0, 4096, 1024)]
    Wm, Hm, _ = _mle(T)
    W_chunks, H, passes = _stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024, device=dev)
    W = torch.cat([w.to(dev) for w in W_chunks])
    assert passes >= 1 and W.min() >= 0 and _loss(W, H, T) <= 1.01 * _loss(Wm, Hm, T)


def test_open_beam_weights(dev):
    """Weights of 1 change nothing.  Per-bin weights give the derivatives of the weighted loss, and a weighted fit of
    counts drawn with a varying open beam has a lower Poisson negative log-likelihood than the unweighted fit."""
    T, _, _ = _problem(dev, P=1024, K=120, dose=20.0, seed=3, dtype=torch.float64)
    ones = torch.ones(1, T.shape[1], dtype=T.dtype, device=dev)
    W, H, _ = _nnal_factorization(T, 3, compile_mode="off")
    W1, H1, _ = _nnal_factorization(T, 3, compile_mode="off", weights=ones)
    assert torch.equal(W, W1) and torch.equal(H, H1)
    support, _, _ = _select_supports(T, W, H, 20.0)
    support1, _, _ = _select_supports(T, W, H, 20.0, weights=ones)
    assert torch.equal(support, support1)

    # The derivatives of the weighted loss match autograd.
    w = torch.linspace(0.3, 1.7, T.shape[1], dtype=T.dtype, device=dev)[None, :]
    X = (W @ H).detach().requires_grad_(True)
    stable_nnal(X, T, _nnal_prep(T, w)).backward()
    G, _ = stable_nnal_derivatives(X.detach(), T, _nnal_prep(T, w))
    assert torch.allclose(X.grad, G, rtol=1e-10, atol=1e-12)

    # Counts drawn with the open beam 20 * w: the weighted fit is the better fit of their likelihood.
    _, Wt, Ht = _problem(dev, P=1024, K=120, dose=20.0, seed=3, dtype=torch.float64, noisy=False)
    g = torch.Generator(device=dev).manual_seed(5)
    I0 = 20.0 * w
    Tw = torch.poisson(I0 * torch.exp(-(Wt @ Ht)), generator=g) / I0
    nll = lambda W_, H_: float((I0 * (torch.exp(-(W_ @ H_)) + Tw * (W_ @ H_))).sum())
    Wu, Hu, _ = _nnal_factorization(Tw, 3, compile_mode="off")
    Ww, Hw, _ = _nnal_factorization(Tw, 3, compile_mode="off", weights=w)
    assert nll(Ww, Hw) < nll(Wu, Hu)
    Ws, Hs, _, _ = _support_selected_spectra(Tw, Ww, Hw, 20.0, compile_mode="off", weights=w)
    assert torch.isfinite(Hs).all()

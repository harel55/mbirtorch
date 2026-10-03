"""Tests for the NNAL solvers in mbirtorch.hsnt: the maximum-likelihood fit, its stop and compiled kernels, the rank
estimate, the spectra estimators and streaming.

Self-contained: small random nonnegative factorizations with mixed pixels stand in for a phantom. Each test runs on
every device of the repository's ``device`` fixture except MPS, which lacks the float64 the solvers accumulate in.
"""
import itertools
import warnings

import numpy as np
import pytest
import torch

import mbirtorch.hsnt as hsnt
from mbirtorch.hsnt import _linalg, _newton, spectra
from mbirtorch.hsnt._loss import _nnal_prep, stable_nnal, stable_nnal_derivatives
from mbirtorch.hsnt._streaming import _stream_factorization
from mbirtorch.hsnt.factorization import _initial_factors, _nnal_factorization, _zero_count_divergence
from mbirtorch.hsnt.spectra import (_auto_penalty, _empty_fit_loss, _fit_free_sets, _guard_components, _select_supports,
                                    _support_selected_spectra, _unconstrained_spectra)


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
    """Three overlapping spheres seen along one axis, with the packaged spectra: most pixels mix two materials."""
    basis, _ = hsnt.load_material_basis()
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
    Wd, Hd = W_ref.clone(), H_ref.clone()
    Hd[1] = 0                                                                       # only the spectrum dead
    Wn, Hn, n = _linalg._reseed_dead(Wd, Hd)
    assert n == 1 and Hn[1].min() > 0 and Wn[:, 1].min() > 0 and torch.equal(Hn[0], Hd[0])
    c = torch.tensor([1e8, 1.0, 1.0], dtype=W_ref.dtype, device=dev)            # the same fit, with component 0's
    Wg, Hg = W_ref * c, H_ref / c[:, None]                                       # scale moved into its map
    assert _linalg._reseed_dead(Wg, Hg)[2] == 0                                  # small next to the others: alive
    Hg[1] = 0
    Wn, Hn, n = _linalg._reseed_dead(Wg, Hg)
    seed = Wn[:, 1].double().mean() * Hn[1].double().mean()
    assert n == 1 and 1e-6 * seed_ref(W_ref, H_ref) < seed < 1e-2 * seed_ref(W_ref, H_ref)
    Wb, Hb = W_ref * 1e3, H_ref * 1e3                                              # one live component, far too large
    Wb[:, 1:], Hb[1:] = 0, 0
    Wn, Hn, n = _linalg._reseed_dead(Wb, Hb, T)                                   # the seed's scale from the data
    scale = _linalg._attenuation_scale(T, 3)
    assert n == 2 and 0 < Wn[:, 1:].min() and Wn[:, 1:].max() <= 1e-2 * scale and Hn[1:].max() <= 1e-2 * scale
    T1, _, _ = _problem(dev, dose=1.0)                                            # one count per bin: T >= 1 where
    W1, H1, _ = _mle(T1)                                                          # there are counts
    W0, H0 = _initial_factors(T1, 3)
    W0[:, 2], H0[2] = 0, 0
    W, H, _ = _nnal_factorization(T1, 3, max_steps=200, rel_tol=1e-8, compile_mode="off", W_init=W0, H_init=H0)
    assert H[2].norm() > 0 and abs(_loss(W, H, T1) - _loss(W1, H1, T1)) <= 1e-6 * _loss(W1, H1, T1)


def seed_ref(W, H):
    """The product of the mean map and mean spectrum entries of a component, the median over components."""
    return (W.double().mean(0) * H.double().mean(1)).median()


def test_a_reseed_that_ends_no_lower_is_undone(dev, monkeypatch):
    """A re-seed after the joint solve whose continued solve ends above the state before it leaves that state."""
    T, _, _ = _problem(dev)
    W_ref, H_ref, _ = _mle(T)
    W0, H0 = _initial_factors(T, 3)
    calls = []

    def harmful(W, H, T=None, seed=0):          # after the warm-up, re-seed component 0 at a scale no solve recovers
        calls.append(1)
        if len(calls) <= 5:
            return W, H, 0
        W, H = W.clone(), H.clone()
        W[:, 0], H[0] = 1e3, 1e3
        return W, H, 1

    monkeypatch.setattr(_newton, "_reseed_dead", harmful)
    W, H, _ = _nnal_factorization(T, 3, max_steps=200, rel_tol=1e-8, compile_mode="off", W_init=W0, H_init=H0)
    assert len(calls) > 5 and torch.equal(W, W_ref) and torch.equal(H, H_ref)       # the state before the re-seed


def test_the_block_step_is_gauge_equivariant(dev):
    """Rescaling one component (map / c, spectrum x c) leaves W H and the loss unchanged, and the block solves with
    it: the fixed-spectra W solve ends at the same loss, and a block H step from the rescaled MLE snaps no entry of
    the other components and does not raise the loss."""
    T, _, _ = _problem(dev)
    W, H, _ = _mle(T)
    L = _loss(_newton.solve_W(T, H), H, T)
    prep = _nnal_prep(T)
    _, _, _, step = _newton._kernels("off")
    for c in (1e-6, 1e6):
        D = torch.tensor([c, 1.0, 1.0], dtype=H.dtype, device=dev)
        assert abs(_loss(_newton.solve_W(T, H * D[:, None]), H * D[:, None], T) - L) <= 1e-9 * L
        Wc, Hc = W / D, H * D[:, None]
        Hn, _, _ = step(Hc, Wc, Wc @ Hc, T, prep, 1)
        assert torch.equal(Hn[1:] > 0, Hc[1:] > 0) and _loss(Wc, Hn, T) <= _loss(Wc, Hc, T) * (1 + 1e-9)
        Wn, _, _ = step(Wc, Hc, Wc @ Hc, T, prep, 0)
        assert _loss(Wn, Hc, T) <= _loss(Wc, Hc, T) * (1 + 1e-9)
        for X in (Wc @ Hn, Wn @ Hc):                                                 # the MLE stays a fixed point
            assert ((X - W @ H).norm() / (W @ H).norm()).item() < 1e-4


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
    basis, _ = hsnt.load_material_basis()
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


def test_the_zero_count_divergence_is_reported(dev):
    """At about one count per bin the loss keeps falling along a component that grows on the zero counts, which
    stops at the attenuation bound, and _zero_count_divergence reports the entries there; at dose 100 it reports
    nothing."""
    for dose, diverges in ((1.0, True), (100.0, False)):
        T = _sphere_problem(dev, n=32, K=100, dose=dose)
        W, H, _ = _nnal_factorization(T, 3, max_steps=150, compile_mode="off")
        x_max, n_above = _zero_count_divergence(W, H, T)
        assert (n_above > 0) == diverges and (x_max > 20) == diverges
        assert bool(((W.amax(0) * H.amax(1)) <= _newton._X_MAX * (1 + 1e-5)).all())    # each component's peak
        # Next to the component grown on the zero counts, the block steps' snap of frozen entries is checked against
        # the row loss, so a step from the fit does not raise the loss.
        prep = _nnal_prep(T)
        _, _, _, step = _newton._kernels("off")
        Wn, _, _ = step(W, H, W @ H, T, prep, 0)
        Hn, _, _ = step(H, W, W @ H, T, prep, 1)
        assert _loss(Wn, H, T) <= _loss(W, H, T) * (1 + 1e-9) and _loss(W, Hn, T) <= _loss(W, H, T) * (1 + 1e-9)


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
    dose 100 it records neither."""
    from mbirtorch.hsnt._fit import _fit
    for dose, mode in ((1.0, "full"), (1.0, "stream"), (100.0, "full")):
        T = _sphere_problem("cpu", n=32, K=100, dose=dose).numpy()
        caplog.clear()
        with caplog.at_level("WARNING", logger="mbirtorch.hsnt"):
            _, _, rep = _fit(T, 3, device="cpu", mode=mode, chunk_pixels=512, max_steps=150, max_passes=3,
                             compile_mode="off")
        diverges = rep["zero_count_entries_at_bound"] > 0
        assert diverges == (dose == 1.0) and diverges == ("zero-count entries" in caplog.text)
        captured = bool(rep["zero_count_components"])
        assert captured == (dose == 1.0) and captured == ("the zero counts rather than a material" in caplog.text)
        if mode == "full":                            # bounded, the fit stops on its own, not at max_steps
            assert not rep["mle_hit_max_steps"]


def test_a_component_that_models_the_zero_counts_is_named(caplog):
    """A component whose attenuation lies on a block of zero counts is named, with its share, in the report and the
    warning, which counts the components left for the materials; without it no component is named."""
    from mbirtorch.hsnt._fit import _report_zero_counts
    T = torch.full((40, 30), 0.5)
    T[:4, :5] = 0
    W, H = torch.full((40, 2), 0.3), torch.ones(2, 30)
    W[:, 1], H[1] = 0, 0
    W[:4, 1], H[1, :5] = 20.0, 3.0                          # component 1 lives on the zero block
    rep = {}
    with caplog.at_level("WARNING", logger="mbirtorch.hsnt"):
        _report_zero_counts(rep, W, H, T, "cpu", 16)
    share = rep["zero_count_share"]
    assert rep["zero_count_components"] == [1] and share[1] == 1.0 and share[0] < 0.1
    assert "components on zero counts: 1 (100.00%, 20 at the bound)" in caplog.text
    assert "the materials have 1 of the 2 components" in caplog.text
    W[:4, 1], H[1, :5], W[10:20, 1], H[1, 10:20] = 0.0, 0.0, 1.0, 1.0      # the same component on counted entries
    rep = {}
    caplog.clear()
    with caplog.at_level("WARNING", logger="mbirtorch.hsnt"):
        _report_zero_counts(rep, W, H, T, "cpu", 16)
    assert rep["zero_count_components"] == [] and "rather than a material" not in caplog.text


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


def test_solves_from_nearby_starts_stop_at_the_same_point(dev):
    """The joint solve stops after several quiet steps in a row: from starts 1e-7 apart it ends at the same loss
    (stopping at the first quiet step spreads these by 2e-5 on the CPU and 5e-7 on CUDA on this problem)."""
    T = _sphere_problem(dev)
    W0, H0 = _initial_factors(T, 3)
    rng = torch.Generator(device=dev).manual_seed(0)
    losses = []
    for k in range(3):
        eps = 0.0 if k == 0 else 1e-7
        W, H, _ = _nnal_factorization(T, 3, compile_mode="off",
                                      W_init=W0 * (1 + eps * torch.randn(W0.shape, generator=rng, device=dev)),
                                      H_init=H0 * (1 + eps * torch.randn(H0.shape, generator=rng, device=dev)))
        losses.append(_loss(W, H, T))
    assert (max(losses) - min(losses)) / min(losses) < 1e-8


def test_rank_estimate_finds_the_rank_with_pooling_and_near_max_rank(dev):
    """The sphere phantom has rank 3 at full resolution and pooled, given as transmission or as attenuation; with the
    true rank one below max_rank a real component is among the last three gains, and it must not raise the noise floor
    and collapse the estimate."""
    T = _sphere_problem(dev).reshape(48, 48, -1)
    n, _, detail = hsnt.estimate_rank(T, "transmission", max_rank=5, device=dev, pool=2)
    assert n == 3 and detail["rank_full"] == 3 and detail["pool_block"] == 2 and detail["pooled"]["pixels"] == 24 * 24
    assert hsnt.estimate_rank(-torch.log(T), max_rank=5, device=dev, pool=2)[0] == 3             # attenuation
    rng = np.random.default_rng(0)
    P, K, R = 2000, 300, 5
    x = np.linspace(0, 1, K)
    H = np.stack([0.1 + 0.9 * np.exp(-((x - (r + 0.5) / R) / (0.6 / R)) ** 2) for r in range(R)])
    W = rng.dirichlet(np.full(R, 0.5), P) * rng.uniform(0.3, 2.0, (P, 1))
    T = (rng.poisson(50.0 * np.exp(-W @ H)) / 50.0).astype(np.float32)
    rank, _, detail = hsnt.estimate_rank(T, "transmission", max_rank=6, device=dev)
    assert rank == R and detail["full"]["noise_tail"]


def test_a_rank_estimated_without_image_axes_at_low_counts_warns(dev):
    """Pixels given without image axes cannot be pooled: below 64 counts per bin, where pooling would run, the estimate
    warns that the full-resolution test alone can miss components, and its note says so. With the image axes, or at
    1000 counts per bin, it does not warn."""
    with pytest.warns(UserWarning, match="full resolution only"):
        _, note, detail = hsnt.estimate_rank(_sphere_problem(dev, dose=3.0), "transmission", max_rank=4, device=dev)
    assert "no pooling" in note and detail["pool_block"] == 0
    for data in (_sphere_problem(dev, dose=3.0).reshape(48, 48, -1), _sphere_problem(dev, dose=1000.0)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hsnt.estimate_rank(data, "transmission", max_rank=4, device=dev)
        assert not any("full resolution only" in str(w.message) for w in caught)


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


def test_branch_and_bound_matches_the_enumeration_and_scales(dev):
    """Branch and bound reproduces the exhaustive search's supports on nearly every pixel of the test problem at the
    same criterion. At a rank the enumeration cannot reach (12), from the maximum-likelihood maps for 12 spectra and
    with no charge, it warns that the search is not exhaustive, selects at most _M_MAX components or all 12 (half the
    pixels, at the maps and loss of the full fit), and never does worse than the full set or the empty one. 'auto' is
    the penalty it names."""
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
    rng = np.random.default_rng(5)
    H12 = torch.tensor(rng.uniform(0.05, 1.0, (12, T.shape[1])), dtype=torch.float32, device=dev)
    W12 = _newton.solve_W(T, H12, torch.full((T.shape[0], 12), 0.05, device=dev), 100, 1e-12)
    prep = _nnal_prep(T)
    f_full = _newton._kernels("off")[2](W12 @ H12, T, prep, 1, dtype=torch.float64)
    with pytest.warns(UserWarning, match="support selection at rank 12 is not exhaustive"):
        s12, W0, f12 = _select_supports(T, W12, H12, dose=10.0, penalty=0.0)
    n = s12.sum(1)
    full = n == 12
    assert s12.shape == (T.shape[0], 12) and bool(((n <= spectra._M_MAX) | full).all()) and torch.isfinite(f12).all()
    assert full.double().mean() > 0.3 and torch.equal(W0[full], W12[full]) and torch.equal(f12[full], f_full[full])
    assert bool((f12 <= f_full + 1e-9).all()) and bool((f12 <= _empty_fit_loss(T, prep) + 1e-9).all())
    s_auto = _select_supports(T, W, H, dose=10.0, penalty="auto")[0]
    assert torch.equal(s_auto, _select_supports(T, W, H, dose=10.0, penalty=_auto_penalty(T.mean(1), 10.0))[0])


def _dense_rank5(dev, dose=1000.0):
    """Rank 5 at high dose, every material in every sample pixel; the enumeration's criterion and supports."""
    T, _, _ = _problem(dev, P=512, K=100, R=5, dose=dose, alpha=1.0, absent=0.0)
    W, H, _ = _nnal_factorization(T, 5, max_steps=200, rel_tol=1e-8, compile_mode="off")
    lam = 2 * np.log(T.shape[1])
    s_enum, f_enum = _enumerated_supports(T, W, H, dose, lam)
    return T, W, H, (dose * f_enum + lam * s_enum.sum(1)).sum().item(), s_enum


def test_branch_and_bound_selects_the_full_set_at_rank_5(dev):
    """At rank 5 the search is exhaustive: on dense mixtures it matches the enumeration, a third of the pixels keep all
    five components, and the refit on the supports ends within 1 nat per pixel of the maximum-likelihood loss (supports
    of at most four components end it about 2.3 nats per pixel higher)."""
    T, W, H, crit_enum, s_enum = _dense_rank5(dev)
    s, _, f = _select_supports(T, W, H, dose=1000.0, penalty=2.0)
    assert (s.sum(1) == 5).double().mean() > 0.2 and (s == s_enum).all(1).double().mean() > 0.95
    assert (1000.0 * f + 2 * np.log(T.shape[1]) * s.sum(1)).sum().item() <= crit_enum * (1 + 1e-4)
    Ws, Hs, _, _ = _support_selected_spectra(T, W, H, dose=1000.0)
    assert 1000.0 * (_loss(Ws, Hs, T) - _loss(W, H, T)) / T.shape[0] < 1.0


def test_branch_and_bound_short_of_the_full_set(dev, monkeypatch):
    """A search whose subsets stop short of the full set still compares it, at the loss of the full fit: one size
    short, the search stays exhaustive and silent; two short, the pixels at its largest subset are reported."""
    T, W, H, crit_enum, _ = _dense_rank5(dev)
    lam = 2 * np.log(T.shape[1])
    monkeypatch.setattr(spectra, "_M_MAX", 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s, W0, f = _select_supports(T, W, H, dose=1000.0, penalty=2.0)
    full = s.sum(1) == 5
    assert full.double().mean() > 0.2 and torch.equal(W0[full], W[full])
    assert (1000.0 * f + lam * s.sum(1)).sum().item() <= crit_enum * (1 + 1e-4)
    monkeypatch.setattr(spectra, "_M_MAX", 3)
    with pytest.warns(UserWarning, match=r"rank 5 is not exhaustive \(it is up to rank 4\).* select 3"):
        s3, _, _ = _select_supports(T, W, H, dose=1000.0, penalty=2.0)
    assert (s3.sum(1) == 5).double().mean() > 0.2 and not bool((s3.sum(1) == 4).any())


def test_pixel_fits_meet_the_kkt_conditions(dev):
    """From a uniform start, where the coupled Newton step pushes entries at zero outward, every pixel's w >= 0 fit
    reaches its KKT point."""
    T, _, Ht = _problem(dev, P=512)
    H = Ht.float()
    idx = torch.arange(3, device=dev).expand(512, 3).contiguous()
    valid = torch.ones_like(idx, dtype=torch.bool)
    w, _ = _fit_free_sets(T, H, idx, valid, torch.full((512, 3), 0.5, device=dev), steps=8)
    g = stable_nnal_derivatives(w @ H, T, _nnal_prep(T))[0] @ H.T / T.shape[1]
    assert w.min() >= 0 and torch.where(w > 0, g.abs(), (-g).clamp(min=0)).max() < 1e-6


def test_streaming_matches_the_full_solve(dev):
    """Streamed by chunks of pixels, the MLE, the unconstrained estimate and support selection land within 1% of the
    loss of the solve held whole, and keep W >= 0. The MLE starts from pixels drawn across all the chunks, so a
    leading chunk free of the sample does not keep it from the full solve's loss."""
    T, _, _ = _problem(dev, P=4096)
    tiles = [T[i:i + 1024].cpu() for i in range(0, 4096, 1024)]
    Wm, Hm, _ = _mle(T)
    W_chunks, H, passes = _stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024, device=dev)
    W = torch.cat([w.to(dev) for w in W_chunks])
    assert passes >= 1 and W.min() >= 0 and _loss(W, H, T) <= 1.01 * _loss(Wm, Hm, T)
    open_beam = torch.tensor(np.random.default_rng(5).poisson(10.0, (1024, T.shape[1])) / 10.0, dtype=T.dtype)
    T2 = torch.cat([open_beam.to(dev), T])
    W_chunks, H, _ = _stream_factorization([open_beam] + tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024,
                                           device=dev)
    Wm2, Hm2, _ = _mle(T2)
    assert _loss(torch.cat([w.to(dev) for w in W_chunks]), H, T2) <= (1 + 1e-4) * _loss(Wm2, Hm2, T2)
    W_chunks, H, _ = _stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024, device=dev,
                                           nonneg_W=False)
    W = torch.cat([w.to(dev) for w in W_chunks])
    Wu, Hu, _ = _unconstrained_spectra(T, Wm, Hm)
    assert W.min() >= 0 and abs(_loss(W, H, T) - _loss(Wu, Hu, T)) <= 1e-2 * _loss(Wu, Hu, T)
    stats = {}
    W_chunks, H, _ = _stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024, device=dev,
                                           stats=stats, support_selection=dict(dose=10.0))
    W = torch.cat([w.to(dev) for w in W_chunks])
    S = torch.cat(stats["support_chunks"]).to(dev)
    assert S.shape == W.shape and bool((W[~S] == 0).all()) and W.min() >= 0
    Ws, Hs, Sm, _ = _support_selected_spectra(T, Wm, Hm, dose=10.0)
    assert abs(S.sum(1).double().mean().item() - Sm.sum(1).double().mean().item()) < 0.1
    assert _loss(W, H, T) <= 1.01 * _loss(Ws, Hs, T)


def test_weights_of_one_change_nothing(dev):
    """Per-entry weights of 1 give the unweighted solve exactly; a constant factor on the weights does not move the
    minimizer (exp/overlap-weights)."""
    T, _, _ = _problem(dev, P=1024, K=120)[:3]
    W0, H0, _ = _nnal_factorization(T, 3, compile_mode='off')
    W1, H1, _ = _nnal_factorization(T, 3, compile_mode='off', weights=torch.ones_like(T))
    assert torch.equal(W0 @ H0, W1 @ H1)
    A = 0.7 + 0.3 * torch.rand(T.shape, generator=torch.Generator().manual_seed(1)).to(T)
    Wa, Ha, _ = _nnal_factorization(T, 3, compile_mode='off', weights=A)
    Wb, Hb, _ = _nnal_factorization(T, 3, compile_mode='off', weights=2.5 * A)
    assert torch.allclose(Wa @ Ha, Wb @ Hb, rtol=0, atol=1e-4)


def test_weighted_loss_and_derivatives(dev):
    """The weighted loss and its derivatives are the unweighted ones times the weights, entry by entry."""
    T, _, _ = _problem(dev, P=256, K=40)[:3]
    X = torch.rand(T.shape, generator=torch.Generator().manual_seed(2)).to(T)
    A = torch.rand(T.shape, generator=torch.Generator().manual_seed(3)).to(T)
    G, Z = stable_nnal_derivatives(X, T)
    Gw, Zw = stable_nnal_derivatives(X, T, _nnal_prep(T, A))
    assert torch.allclose(Gw, G * A) and torch.allclose(Zw, Z * A)
    from mbirtorch.hsnt._loss import _nnal_elementwise
    assert torch.allclose(_nnal_elementwise(X, T, _nnal_prep(T, A)), _nnal_elementwise(X, T, _nnal_prep(T)) * A)


def test_weighted_stream_matches_full(dev):
    """A streamed weighted fit reaches the full weighted fit's loss."""
    from mbirtorch.hsnt._fit import _fit
    T, _, _ = _problem(dev, P=2048, K=120)[:3]
    Tn = T.cpu().numpy()
    A = (0.7 + 0.3 * np.random.default_rng(4).random(Tn.shape)).astype(np.float32)
    _, _, rf = _fit(Tn, 3, device=dev, weights=A, mode='full')
    _, _, rs = _fit(Tn, 3, device=dev, weights=A, mode='stream', chunk_pixels=600, max_passes=10)
    assert abs(rs['loss_mle'] - rf['loss_mle']) <= 1e-5 * abs(rf['loss_mle'])

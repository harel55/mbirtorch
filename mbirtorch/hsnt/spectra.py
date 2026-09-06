import math

import torch

from ._loss import _nnal_prep
from ._newton import _joint_newton_pcg, _kernels, solve_W


def unconstrained_spectra(T, W, H, max_steps=300, cg_max=10, rel_tol=1e-10, w_max_steps=100, compile_mode=None):
    """Re-estimate the spectra with the bound on the pixel coefficients dropped, then re-solve W >= 0.

    The maximum-likelihood spectra are biased by the truncation of pixel
    coefficients at zero: a coefficient whose true value is zero is estimated
    positive half the time and clipped the other half, an O(1/sqrt m) effect per
    pixel (m = information per pixel) that does not average out over pixels.
    Dropping the bound removes it -- the pixel problem stays strictly convex for
    any real w; estimating H with W free is a semi-NMF step (Ding, Li & Jordan
    2010) and the relax-where-it-truncates logic of NEG-ML in PET -- at the price
    of the variance reduction the constraint provides (the implicit regularisation
    of a sign constraint, Slawski & Hein 2013),
    so it wins once pixels are plentiful (above ~10^5 at dose 3; the crossover
    moves to larger P at lower dose) and loses a little below that. Cost: a
    continuation of the joint solve from the ML point, cheaper than the ML solve.
    Not for coefficients that are physically nonnegative and mostly zero, such as
    fractions over a dictionary of many similar atoms: there the unconstrained fit
    is ill-conditioned and the bound carries real information; see
    support_selected_spectra. Measurements: docs/hsnt_solver_notes.md, section 5.

    Returns (W, H, steps) with W >= 0 re-solved for the returned H.
    """
    nnal_fn, deriv, _, _ = _kernels(compile_mode)
    prep = _nnal_prep(T)
    _, Hu, steps, _ = _joint_newton_pcg(T, W, H, max_steps=max_steps, cg_max=cg_max, rel_tol=rel_tol,
                                        prep=prep, nnal=nnal_fn, deriv=deriv, nonneg_W=False)
    Wc = solve_W(T, Hu, W, w_max_steps, 1e-12, compile_mode=compile_mode)
    return Wc, Hu, steps


def support_selected_spectra(T, W, H, dose, penalty=None, max_steps=300, cg_max=10, rel_tol=1e-10,
                             w_max_steps=100, compile_mode=None, verbose=False):
    """Choose each pixel's material subset by penalised likelihood, then refit with the supports fixed.

    The truncation bias of the ML spectra (see unconstrained_spectra) comes from
    coefficients whose true value is zero. If those are identified and held at
    zero, the remaining coefficients sit in the interior and only the much
    smaller curvature bias is left, while W >= 0 -- and the variance reduction it
    brings -- is kept. Every nonempty subset of the R materials is solved for
    every pixel (2^R - 1 grouped convex solves, so R <= 6), each pixel takes the
    subset minimising  dose * loss + penalty * |subset|  (the empty subset is
    allowed: a pixel with no material), and (W on the selected supports, H) is
    then refit jointly. The selection is a model-selection step and carries its
    own errors; a stronger penalty helped monotonically up to the default 2 log K,
    and a single select/refit round is the optimum (iterating degrades). It lifts
    the maps a little as well; the gauge fix, pure_pixel_gauge, lifts them far
    more. Measurements: docs/hsnt_solver_notes.md, section 5.

    Args:
        dose: open-beam counts per pixel and bin, which converts the loss to
            log-likelihood units for the penalty.
        penalty: per selected coefficient, in log-likelihood units. Default 2 log K.

    Returns (W, H, support, steps) with support a bool mask of W's shape.
    """
    R, K = H.shape
    if R > 6:
        raise ValueError("support_selected_spectra enumerates all 2^R - 1 subsets; for R > 6 use "
                         "unconstrained_spectra (signed coefficients do not truncate).")
    import itertools
    penalty = 2.0 * math.log(K) if penalty is None else penalty
    nnal_fn, deriv, rowwise, _ = _kernels(compile_mode)
    prep = _nnal_prep(T)
    subsets = [list(c) for r in range(1, R + 1) for c in itertools.combinations(range(R), r)]
    crit = [rowwise(torch.zeros_like(T), T, prep, 1, dtype=torch.float64) * dose]      # the empty subset
    W_sub = []
    for S in subsets:
        idx = torch.tensor(S, device=T.device)
        Ws = solve_W(T, H[idx].contiguous(), W[:, idx].contiguous(), w_max_steps, 1e-12, compile_mode=compile_mode)
        crit.append(rowwise(Ws @ H[idx], T, prep, 1, dtype=torch.float64) * dose + penalty * len(S))
        W_sub.append(Ws)
    best = torch.stack(crit, 1).argmin(1)                    # 0 = empty, j + 1 = subsets[j]
    W0 = torch.zeros_like(W)
    for j, S in enumerate(subsets):
        m = best == j + 1
        if m.any():
            W0[m.nonzero().squeeze(1)[:, None], torch.tensor(S, device=T.device)[None, :]] = W_sub[j][m]
    support = W0 > 0
    Wn, Hn, steps, _ = _joint_newton_pcg(T, W0, H, max_steps=max_steps, cg_max=cg_max, rel_tol=rel_tol,
                                         prep=prep, nnal=nnal_fn, deriv=deriv, w_mask=support)
    if verbose:
        print(f'  supports: mean size {support.sum(1).double().mean().item():.2f}; joint refit {steps} steps', flush=True)
    return Wn, Hn, support, steps


def pure_pixel_gauge(T, W, H, dose, penalty=None, seeds=8, kmeans_steps=100, w_max_steps=200, random_state=0,
                     compile_mode=None, verbose=False):
    """Re-mix the factorization so that each spectrum is the mean fitted spectrum of one cluster of pure pixels.

    The likelihood does not identify the gauge. For any invertible A with
    W A^-1 >= 0 and A H >= 0, (W A^-1, A H) fits the data exactly as well as
    (W, H); nonnegativity only confines A to a polytope, and the joint solver
    stops wherever its path ends inside it. On the phantom the fitted rows were
    combinations of the true spectra with condition numbers 49 (dose 3) and 333
    (dose 30), and the maps inherited the mixing: 8.4 dB against 11.5 for the
    same subspace in the true gauge. The gauge is fixed with a prior the
    likelihood lacks: most pixels contain a single material. The coefficient
    rows of W, taken as directions, then form R clusters whose centres are the
    rows of A, and A @ H are the pure pixels' mean spectra. The remedy follows
    Chowdhury et al. (ICIP 2023; IEEE Trans. Comput. Imaging 2025), who cluster
    the coefficient vectors of an NMF fit of the same kind of data and take the
    cluster means as the mixing; the same construction is K-P-Means (Xu et al.
    2014) and the vertex hunting of Topic-SCORE. This function applies it to the
    exact-rank Poisson fit in the pixel domain. Pixels without
    material (likelihood ratio against X = 0 below `penalty`, in log-likelihood
    units) are left out, the clusters are found by intensity-weighted k-means
    from several k-means++ seeds, and W >= 0 is re-solved with the re-mixed H
    held fixed. H is deliberately not refit: a joint refit returns to the MLE
    gauge. Recovers the oracle rotation's maps (11.48 vs 11.47 dB at dose 3,
    22.14 vs 22.18 at dose 30, 65k pixels; +3.1 and +5.0 dB over the MLE) for
    the cost of one fixed-H W solve; the result was the same with or without
    the intensity weights, at penalty 2 to 8 log K, and with constrained or
    unconstrained per-pixel means. The estimate of A is noise-limited on small
    images: at 4k pixels the maps stop 0.3 dB short of the oracle (7.9 -> 10.7
    vs 11.0 at dose 3), at 16k 0.1 dB short, with the axis error falling as
    1/sqrt(pixels). Requires every material to have pure pixels:
    a material present only in mixtures pulls its axis into the data cone, and
    a mixture-dominated dataset yields clusters that are not materials. The
    order of the returned spectra is the cluster order, which is arbitrary.
    Measurements: docs/hsnt_solver_notes.md, section 5.

    Args:
        dose: open-beam counts per pixel and bin, converting the loss to
            log-likelihood units for the material test.
        penalty: likelihood-ratio threshold for a pixel to count as carrying
            material. Default 2 log K, the same as support_selected_spectra's
            penalty for one coefficient.
        seeds: k-means++ restarts; the lowest within-cluster dispersion wins.
        random_state: seed of the k-means++ draws, so two calls agree.

    Returns (W, H, A, labels): W >= 0 re-solved for the new H; H = A @ H_in
    with the rows of A normalised to unit sum; labels the cluster of each
    pixel, -1 for pixels left out of the clustering.
    """
    R, K = H.shape
    penalty = 2.0 * math.log(K) if penalty is None else penalty
    _, _, rowwise, _ = _kernels(compile_mode)
    prep = _nnal_prep(T)
    llr = dose * (rowwise(torch.zeros_like(T), T, prep, 1, dtype=torch.float64)
                  - rowwise(W @ H, T, prep, 1, dtype=torch.float64))
    keep = llr > penalty
    n_keep = int(keep.sum())
    if n_keep < R:
        raise ValueError(f"pure_pixel_gauge: {n_keep} pixels carry material at penalty {penalty:.1f}; need at least {R}")
    Wk = W[keep].double()
    s = Wk.sum(1)                                                        # intensity: the weight of a pixel's direction
    U = Wk / s[:, None].clamp_min(torch.finfo(torch.float64).tiny)       # direction on the simplex
    gen = torch.Generator(device=W.device)
    best = None
    for seed in range(seeds):
        gen.manual_seed(random_state + seed)
        C = U[torch.multinomial(s, 1, generator=gen)]
        for _ in range(1, R):                                            # k-means++: seed far from the seeds so far
            p = s * torch.cdist(U, C).pow(2).amin(1)
            p = p if p.sum() > 0 else torch.ones_like(p)
            C = torch.cat([C, U[torch.multinomial(p / p.sum(), 1, generator=gen)]])
        for _ in range(kmeans_steps):
            labels = torch.cdist(U, C).argmin(1)
            Cn = C.clone()
            for k in range(R):
                m = labels == k
                if m.any():
                    Cn[k] = (s[m, None] * U[m]).sum(0) / s[m].sum()
            converged = torch.allclose(Cn, C, atol=1e-9, rtol=0.0)
            C = Cn
            if converged:
                break
        objective = (s * torch.cdist(U, C).pow(2).amin(1)).sum().item()
        if best is None or objective < best[0]:
            best = (objective, labels)
    _, labels = best
    A = torch.eye(R, dtype=torch.float64, device=W.device)               # an empty cluster keeps the fitted axis
    for k in range(R):
        m = labels == k
        if m.any():
            A[k] = Wk[m].sum(0)                                          # intensity-weighted mean direction
    A = A / A.sum(1, keepdim=True)
    H_new = (A @ H.double()).to(H.dtype).clamp_(min=0)
    W0 = torch.linalg.lstsq(A.T, W.double().T)[0].T.clamp_(min=0).to(W.dtype).contiguous()   # W A^-1, the same maps in the new gauge
    W_new = solve_W(T, H_new, W0, w_max_steps, 1e-12, compile_mode=compile_mode)
    full = torch.full((T.shape[0],), -1, dtype=torch.long, device=T.device)
    full[keep] = labels
    if verbose:
        sizes = [int((labels == k).sum()) for k in range(R)]
        print(f'  pure_pixel_gauge: {n_keep} of {T.shape[0]} pixels carry material; clusters {sizes}; '
              f'cond(A) {torch.linalg.cond(A).item():.1f}', flush=True)
    return W_new, H_new, A, full

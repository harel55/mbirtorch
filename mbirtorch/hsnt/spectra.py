"""Re-estimates of the spectra that remove the truncation bias of the maximum-likelihood fit."""
import itertools
import math
import warnings

import torch

from ._loss import _nnal_prep, _nnal_rowwise, stable_nnal_derivatives
from ._newton import _ARMIJO_FLOOR, _joint_newton_pcg, _kernels, _resolve_compile, solve_W


# The unconstrained re-estimate stops at a relative loss change of 1e-8 within 100 steps: with W free the loss is flat
# along the mixing gauge, and a longer solve only wanders along it.
_UNC_MAX_STEPS, _UNC_REL_TOL = 100, 1e-8
# The joint refit on the supports, and the W >= 0 re-solves.
_REFIT_MAX_STEPS, _REFIT_REL_TOL, _W_MAX_STEPS, _CG_MAX = 300, 1e-10, 100, 10
# Branch and bound: candidate materials per pixel, and the largest subset of them searched. Up to rank 6 the search is
# exhaustive; above, it covers every subset of a pixel's six best single materials, and the full set, and warns.
_K_TOP, _M_MAX = 6, 6


def _unconstrained_spectra(T, W, H, compile_mode='auto'):
    """Re-estimate the spectra with the bound on the pixel coefficients dropped, then re-solve W >= 0.

    The maximum-likelihood spectra are biased by the truncation of pixel coefficients at zero: a coefficient whose
    true value is zero is estimated positive half the time and clipped the other half, and the spectra, shared by
    every pixel, absorb that excess. Dropping the bound while H is estimated removes the bias, at the price of the
    variance the bound suppresses, so it gains when the pixels are many: on a sphere phantom at 3 counts per bin, 7 to
    12 dB on the spectra at 10^6 pixels (0.11 nats per pixel above the maximum-likelihood loss), and about nothing at
    4 x 10^4 to 6.5 x 10^4 pixels, scored after the best linear mixing of the fitted spectra onto the true ones.
    Returns (W, H, steps): W >= 0 re-solved for the returned H, and the steps of the free-W solve.
    """
    compile_mode = _resolve_compile(compile_mode, T)
    nnal_fn, deriv, _, _ = _kernels(compile_mode)
    prep = _nnal_prep(T)
    _, Hu, steps, _ = _joint_newton_pcg(T, W, H, max_steps=_UNC_MAX_STEPS, cg_max=_CG_MAX, rel_tol=_UNC_REL_TOL,
                                        prep=prep, nnal=nnal_fn, deriv=deriv, nonneg_W=False)
    Wc = solve_W(T, Hu, W, _W_MAX_STEPS, 1e-12, compile_mode=compile_mode)
    return Wc, Hu, steps


_FREE_SET_ELEMS = 2 ** 27       # elements of one (pixels, m, bins) block: 0.5 GB in float32, about eight live


def _fit_free_sets(T, H, idx, valid, w0, steps=8, rows=None):
    """Newton fit of every pixel on its own small set of materials, batched over pixels, with w >= 0.

    idx (P, m) holds material indices (anything where valid is False is padding), w0 (P, m) the start. The per-pixel
    Hessian is the m x m block of the free set, so one Newton step costs a (P, m, K) gather-product whatever R is;
    entries at zero with an outward gradient are frozen (two-metric projection), the step is projected onto w >= 0,
    and a per-pixel Armijo test on the float64 row loss checks the projected point. Pixels are processed in chunks
    sized by _FREE_SET_ELEMS (they are independent), and `rows` names the pixels of T to fit when T is the whole data,
    so that only a chunk of rows is ever gathered. Returns (w, f) with f the per-pixel loss."""
    P, m = idx.shape
    chunk = max(1, _FREE_SET_ELEMS // (m * H.shape[1]))
    if P <= chunk and rows is None:
        return _fit_free_sets_block(T, H, idx, valid, w0, steps)
    ws, fs = [], []
    for s in range(0, P, chunk):
        e = min(P, s + chunk)
        Tc = T[s:e] if rows is None else T[rows[s:e]]
        w, f = _fit_free_sets_block(Tc, H, idx[s:e], valid[s:e], w0[s:e], steps)
        ws.append(w)
        fs.append(f)
        del Tc
    return torch.cat(ws), torch.cat(fs)


def _fit_free_sets_block(T, H, idx, valid, w0, steps):
    m = idx.shape[1]
    dev = T.device
    prep = _nnal_prep(T)
    Hf = H[idx.clamp(min=0)] * valid[:, :, None]                                       # (P, m, K)
    w = (w0 * valid).clone()
    X = torch.einsum('pm,pmk->pk', w, Hf)
    f = _nnal_rowwise(X, T, prep, 1, dtype=torch.float64)
    eye = torch.eye(m, device=dev, dtype=T.dtype)
    for _ in range(steps):
        G, Z = stable_nnal_derivatives(X, T, prep)
        g = torch.einsum('pk,pmk->pm', G, Hf)
        Hs = torch.einsum('pk,pmk,pnk->pmn', Z, Hf, Hf) + 1e-6 * eye
        free = valid & ~((w <= 0) & (g > 0))
        Hs = torch.where(free[:, :, None] & free[:, None, :], Hs, eye.expand_as(Hs))
        rhs = torch.where(free, g, torch.zeros_like(g))
        # pinv: identical or zero spectra make Hs singular
        d = (torch.linalg.pinv(Hs) @ rhs.unsqueeze(-1)).squeeze(-1) * free
        slope = (g * d).sum(1)
        d = torch.where((slope <= 0)[:, None], rhs / Hs.diagonal(dim1=1, dim2=2).clamp(min=1e-12), d)
        # No cap at the largest feasible step: a free entry at zero that the coupled step pushes outward would make
        # it zero for the whole pixel. The projection clips it instead.
        alpha = torch.ones_like(slope)
        accepted = torch.zeros_like(alpha)
        done = torch.zeros_like(alpha, dtype=torch.bool)
        for _ in range(8):
            trial = torch.where(done, torch.zeros_like(alpha), alpha)
            wt = (w - trial[:, None] * d)
            wt = wt.clamp(min=0) * valid
            decrease = (g * (w - wt)).sum(1).clamp(min=0)              # first-order decrease along the projected step
            ft = _nnal_rowwise(torch.einsum('pm,pmk->pk', wt, Hf), T, prep, 1, dtype=torch.float64)
            ok = (ft <= f - 1e-4 * decrease + _ARMIJO_FLOOR * torch.finfo(T.dtype).eps * f.abs()) | (trial == 0)
            accepted = torch.where(ok & ~done, trial, accepted)
            done |= ok
            if bool(done.all()):
                break
            alpha = alpha * 0.5
        w_new = (w - accepted[:, None] * d)
        w_new = w_new.clamp(min=0) * valid
        X = torch.einsum('pm,pmk->pk', w_new, Hf)
        f_new = _nnal_rowwise(X, T, prep, 1, dtype=torch.float64)
        moved = (f - f_new).abs() > 1e-9 * f.abs()
        w, f = w_new, f_new
        if not bool(moved.any()):
            break
    return w, f


def _scatter_support(idx, valid, w, R):
    """(P, R) support mask and coefficients from padded per-pixel sets; only valid slots are written (padding
    indices must not collide with a real material)."""
    P = idx.shape[0]
    pp, slot = valid.nonzero(as_tuple=True)
    support = torch.zeros(P, R, dtype=torch.bool, device=idx.device)
    support[pp, idx[pp, slot]] = True
    W = torch.zeros(P, R, dtype=w.dtype, device=idx.device)
    W[pp, idx[pp, slot]] = w[pp, slot]
    return support, W


def _empty_fit_loss(T, prep):
    """Per-pixel loss of the empty subset, f_p(0), in pixel chunks, so a zero X the size of T is never built whole."""
    log_T, positive, all_positive = prep[:3]
    weights = prep[3] if len(prep) > 3 else None
    chunk = max(1, 2 ** 23 // T.shape[1])
    out = []
    for s in range(0, T.shape[0], chunk):
        Tc = T[s:s + chunk]
        prep_c = (log_T[s:s + chunk], positive[s:s + chunk], all_positive,
                  None if weights is None else weights[s:s + chunk])
        out.append(_nnal_rowwise(torch.zeros_like(Tc), Tc, prep_c, 1, dtype=torch.float64))
    return torch.cat(out)


def _support_sets(support, W):
    """Padded per-pixel index sets (idx, valid, w0) of a (P, R) bool support, for _fit_free_sets."""
    valid, idx = torch.sort(support, dim=1, descending=True, stable=True)
    return idx, valid, W.gather(1, idx)


def _solve_W_on_support(T, H, W, support, steps=30):
    """W >= 0 given H, restricted to each pixel's support: exact per-pixel Newton."""
    idx, valid, w0 = _support_sets(support, W.clamp(min=0))
    w, _ = _fit_free_sets(T, H, idx, valid, w0, steps=steps)
    return _scatter_support(idx, valid, w, W.shape[1])[1]


def _weak_components(counts, P, min_support=None):
    """Components selected in fewer than min_support pixels (default max(2R, P / 1000)), with a warning.

    A component kept in only a handful of pixels would be refit from those alone, and its spectrum, and the gauge
    of the others with it, would be unusable; such a component reverts to the maximum-likelihood treatment (free in
    every pixel, W >= 0 deciding its zeros). counts is the number of pixels selecting each component.
    """
    R = counts.numel()
    if min_support is None:
        min_support = max(2 * R, P // 1000)
    weak = counts < min_support
    if bool(weak.any()):
        warnings.warn(f"support selection: component(s) {weak.nonzero().flatten().tolist()} selected in fewer than "
                      f"{min_support} pixels; kept free in every pixel")
    return weak


def _guard_components(support, W_mle, W0, min_support=None):
    """Revert the weak components (see _weak_components) of a (P, R) support in place. Returns the weak mask."""
    weak = _weak_components(support.sum(0), support.shape[0], min_support)
    if bool(weak.any()):
        support[:, weak] = True
        W0[:, weak] = W_mle[:, weak]
    return weak


def _select_branch_bound(T, W, H, dose, lam, f_full, k_top=_K_TOP, m_max=_M_MAX, plausible=None, prep=None):
    """Per-pixel subset search: exact single-material fits for every material (R batched one-dimensional solves,
    restricted to `plausible` pixels when given), then subsets of 2 to m_max materials among each pixel's k_top best
    singletons for the pixels the lower bound leaves open, then the full set. W is the full-model fit and f_full its
    per-pixel loss. The pixel loss is monotone in the subset, so f_full bounds every subset from below: a set of `size`
    materials can beat the current best only if dose * (f_best - f_full) > lam * (size - size(best)). When the subsets
    stop short of the full set (m_max < R), the full set enters as the fit W, with no further fit. The search is
    exhaustive when every subset short of the full set is searched (k_top = R and m_max >= R - 1); otherwise a warning
    says so, with the shares of the pixels at the full set and at m_max. Returns (idx, valid, w, f) as padded per-pixel
    sets of width R."""
    P = T.shape[0]
    R = H.shape[0]
    dev = T.device
    k_top = min(k_top, R)
    m_max = min(m_max, k_top)
    prep = _nnal_prep(T) if prep is None else prep
    f0 = _empty_fit_loss(T, prep)
    F1 = torch.full((P, R), float('inf'), device=dev, dtype=torch.float64)
    W1 = torch.zeros(P, R, device=dev, dtype=T.dtype)
    for r in range(R):
        rows = torch.arange(P, device=dev) if plausible is None else plausible[:, r].nonzero().squeeze(1)
        if rows.numel() == 0:
            continue
        idx1 = torch.full((rows.numel(), 1), r, dtype=torch.long, device=dev)
        valid1 = torch.ones_like(idx1, dtype=torch.bool)
        w_start = torch.full((rows.numel(), 1), 0.5, device=dev, dtype=T.dtype)
        w1, f1 = _fit_free_sets(T, H, idx1, valid1, w_start, rows=rows)
        F1[rows, r] = f1
        W1[rows, r] = w1[:, 0]
    best_f1, r1 = F1.min(1)
    one = best_f1 * dose + lam < f0 * dose                          # best singleton beats the empty set
    best_idx = torch.full((P, R), -1, dtype=torch.long, device=dev)
    best_valid = torch.zeros(P, R, dtype=torch.bool, device=dev)
    best_w = torch.zeros(P, R, device=dev, dtype=T.dtype)
    best_idx[one, 0] = r1[one]
    best_valid[one, 0] = True
    best_w[one, 0] = W1[one, r1[one]]
    best_f = torch.where(one, best_f1, f0)
    order = F1.argsort(1)[:, :k_top]                                # candidate materials per pixel
    n_cand = torch.isfinite(F1).sum(1)
    for size in range(2, m_max + 1):
        can = ((best_f - f_full) * dose > lam * (size - best_valid.sum(1).double())) & (n_cand >= size)
        cp = can.nonzero().squeeze(1)
        if cp.numel() == 0:
            break
        for combo in itertools.combinations(range(k_top), size):
            idx_c = order[cp][:, list(combo)]
            ok = torch.isfinite(F1[cp[:, None], idx_c]).all(1)
            if not bool(ok.any()):
                continue
            cq = cp[ok]
            idx_c = idx_c[ok]
            valid_c = torch.ones(cq.numel(), size, dtype=torch.bool, device=dev)
            w_c, f_c = _fit_free_sets(T, H, idx_c, valid_c, (W1[cq[:, None], idx_c] / size).clamp(min=1e-3), rows=cq)
            better = f_c * dose + lam * size < best_f[cq] * dose + lam * best_valid[cq].sum(1).double()
            bp = cq[better]
            best_idx[bp] = -1
            best_valid[bp] = False
            best_w[bp] = 0
            best_idx[bp, :size] = idx_c[better]
            best_valid[bp, :size] = True
            best_w[bp, :size] = w_c[better]
            best_f[bp] = f_c[better]
    if m_max < R:                                                   # the full set: the fit W, its loss f_full
        full = f_full * dose + lam * R < best_f * dose + lam * best_valid.sum(1).double()
        best_idx[full] = torch.arange(R, device=dev)
        best_valid[full] = True
        best_w[full] = W[full].to(best_w.dtype)
        best_f = torch.where(full, f_full, best_f)
    if k_top < R or m_max < R - 1:                                  # some subsets short of the full set go unsearched
        n = best_valid.sum(1)
        at_full, at_cap = ((n == R).double().mean().item(), (n == m_max).double().mean().item())
        warnings.warn(f"support selection at rank {R} is not exhaustive (it is up to rank {min(k_top, m_max + 1)}): "
                      f"it tries the subsets of up to {m_max} of each pixel's {k_top} best single components, and the "
                      f"full set; {100 * at_full:.3g}% of the pixels select all {R} and {100 * at_cap:.3g}% select "
                      f"{m_max}, and a better support may be among those not tried")
    return best_idx, best_valid, best_w, best_f


def _auto_penalty(pixel_means, dose):
    """The automatic charge per selected material, as a multiple of log K, from the counts of the median pixel (dose
    times its mean transmission): 0.5 below 10 counts per bin, 2 above 100, log-linear in between.

    The rule is tuned on one phantom (three spheres, 10^6 pixels, the median pixel background, so its counts are the
    dose), where it is the minimax choice against a constant 2. At 1 to 2.4 counts, 2 falls up to 8.6 dB short of 0.5
    on one spectrum, the component guard reverting a column in most fits, and 0.5 up to 3.4 dB short of 2 on another
    (losses above the maximum-likelihood fit's: 0.75 to 0.93 nats per pixel for 0.5, 0.48 to 0.55 for 2). Between 3
    and 10 counts, where the rule still charges 0.5, 2 is as good or up to 1.5 dB better on the spectra (losses 0.47
    to 0.58 for 2, 0.41 to 0.46 for 0.5).
    """
    counts = float(dose) * torch.median(pixel_means.float()).item()
    frac = min(1.0, max(0.0, (math.log10(max(counts, 1e-12)) - 1.0)))          # 0 at 10 counts, 1 at 100
    return 0.5 * 4 ** frac


def _penalty_nats(penalty, T, dose):
    """The charge per selected material in nats: penalty is 'auto' or a multiple of log K."""
    K = T.shape[1]
    if isinstance(penalty, str):
        if penalty != "auto":
            raise ValueError(f"penalty must be a multiple of log K or 'auto', got {penalty!r}")
        return _auto_penalty(T.mean(1), dose) * math.log(K)
    if not float(penalty) >= 0:
        raise ValueError(f"penalty must be nonnegative, got {penalty!r}")
    return float(penalty) * math.log(K)


def _select_supports(T, W, H, dose, penalty='auto', wald_screen=0.0):
    """Each pixel's material subset S by penalized likelihood: minimize dose * f_p(S) + lam * size(S), lam the charge
    in nats (see _penalty_nats), by branch and bound (_select_branch_bound).

    wald_screen > 0 skips the single-material fit of material r in the pixels whose full-fit Wald statistic is below
    wald_screen times the charge: faster for sparse supports, but a faint material the full fit truncated to zero is
    then never considered. Returns (support, W0, f): the (pixels, R) bool support, the coefficients on it, and the
    per-pixel loss.
    """
    if not dose > 0:
        raise ValueError(f"support selection needs a positive dose, got {dose!r}")
    R = H.shape[0]
    lam = _penalty_nats(penalty, T, dose)
    prep = _nnal_prep(T)
    X = W @ H
    f_full = _nnal_rowwise(X, T, prep, 1, dtype=torch.float64)
    plausible = None
    if wald_screen > 0:
        _, Z = stable_nnal_derivatives(X, T, prep)
        wald = 0.5 * dose * (W.double() ** 2) * (Z @ (H * H).T).double()
        plausible = wald > wald_screen * lam
    idx, valid, w, f = _select_branch_bound(T, W, H, dose, lam, f_full, k_top=_K_TOP, m_max=_M_MAX,
                                            plausible=plausible, prep=prep)
    support, W0 = _scatter_support(idx, valid, w, R)
    return support, W0, f


def _warn_collinear_rows(H, cosine=0.999):
    """Two spectra that came out (nearly) proportional split their pixels' coefficients arbitrarily: say so."""
    Hn = H.double() / H.double().norm(dim=1, keepdim=True).clamp_min(1e-300)
    C = (Hn @ Hn.T).fill_diagonal_(0)
    if bool((C > cosine).any()):
        i, j = (C > cosine).nonzero()[0].tolist()
        warnings.warn(f"spectra {i} and {j} are proportional (cosine {C[i, j].item():.4f}): their pixel coefficients "
                      "are not separately identified; consider a smaller rank")


def _support_selected_spectra(T, W, H, dose, penalty='auto', wald_screen=0.0, compile_mode='auto'):
    """Choose each pixel's material subset by penalized likelihood, then refit with the other coefficients held at 0.

    The truncation bias of the maximum-likelihood spectra comes from coefficients whose true value is zero. Once those
    are identified and held at zero, the remaining ones sit in the interior and W >= 0, with the variance reduction it
    brings, is kept. Each pixel takes the subset minimizing dose * loss + charge * (subset size), the empty subset
    included (_select_supports), and W on the supports and H are then refit jointly. One selection round is used:
    re-selecting from refit spectra compounds the selection errors. A component selected in almost no pixel reverts to the maximum-likelihood
    treatment (_guard_components). The selection works in the basis W and H are in: where the components are strongly
    mixed combinations of the materials, a pixel of one material needs several of them, and the supports mostly
    separate the sample from the background.

    Returns (W, H, support, steps): the refit factors, the (pixels, R) bool support (all True in a column that
    reverted), and the refit's steps.
    """
    compile_mode = _resolve_compile(compile_mode, T)
    nnal_fn, deriv, _, _ = _kernels(compile_mode)
    prep = _nnal_prep(T)
    support, W0, _ = _select_supports(T, W, H, dose, penalty=penalty, wald_screen=wald_screen)
    _guard_components(support, W, W0)
    Wn, Hn, steps, _ = _joint_newton_pcg(T, W0, H, max_steps=_REFIT_MAX_STEPS, cg_max=_CG_MAX, rel_tol=_REFIT_REL_TOL,
                                         prep=prep, nnal=nnal_fn, deriv=deriv, w_mask=support)
    _warn_collinear_rows(Hn)
    return Wn, Hn, support, steps

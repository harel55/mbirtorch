"""This module re-estimates the spectra H after the maximum-likelihood fit.  The bound W >= 0 in
that fit biases the spectra at low dose.  This module provides two ways to remove the bias.  The
first way drops the bound and refits.  The second way chooses, for each pixel, the subset of
materials present in that pixel.  This subset is called the support of the pixel.  The second way
then refits with all other coefficients held at zero."""
import itertools
import math
import warnings

import torch

from ._loss import _nnal_prep, _nnal_rowwise, stable_nnal_derivatives
from ._newton import _ARMIJO_FLOOR, _joint_newton_pcg, _kernels, _resolve_compile, solve_W


# These are the maximum steps and the relative tolerance of the unconstrained re-estimate.
_UNC_MAX_STEPS, _UNC_REL_TOL = 100, 1e-8
# These are the step limits and tolerances of the refit on the supports and of the W >= 0 solves.
_REFIT_MAX_STEPS, _REFIT_REL_TOL, _W_MAX_STEPS, _CG_MAX = 300, 1e-10, 100, 10
# These are the number of candidate materials per pixel and the largest subset size that the
# branch and bound search tries.  The search is exhaustive up to rank 6.
_K_TOP, _M_MAX = 6, 6


def _unconstrained_spectra(T, W, H, compile_mode='auto', weights=None):
    """This function re-estimates the spectra without the bound W >= 0.  It then solves for
    W >= 0 with the new spectra held fixed.  It returns (W, H, steps).  ``weights`` multiplies
    each entry's loss (see ``_nnal_prep``).

    Removing the bound removes the bias of the spectra, but it adds variance.  For this reason,
    the method helps only when there are many pixels.
    """
    compile_mode = _resolve_compile(compile_mode, T)
    nnal_fn, deriv, _, _ = _kernels(compile_mode)
    prep = _nnal_prep(T, weights)
    _, Hu, steps, _ = _joint_newton_pcg(T, W, H, max_steps=_UNC_MAX_STEPS, cg_max=_CG_MAX, rel_tol=_UNC_REL_TOL,
                                        prep=prep, nnal=nnal_fn, deriv=deriv, nonneg_W=False)
    Wc = solve_W(T, Hu, W, _W_MAX_STEPS, 1e-12, compile_mode=compile_mode, weights=weights)
    return Wc, Hu, steps


_FREE_SET_ELEMS = 2 ** 27       # This is the number of elements in one (pixels, m, bins) block.


def _weight_rows(weights, rows):
    """This function returns the weights of the pixels ``rows`` (a slice or an index tensor).  Weights
    with one row, such as one weight per bin, apply to every pixel and are returned unchanged."""
    return weights if weights is None or weights.dim() < 2 or weights.shape[0] == 1 else weights[rows]


def _fit_free_sets(T, H, idx, valid, w0, steps=8, rows=None, weights=None):
    """This function fits the coefficients w >= 0 of each pixel on its own small set of materials.
    It uses projected Newton steps on chunks of pixels.  The array ``idx`` (P, m) holds material
    indices, ``valid`` marks the entries of ``idx`` that are real, and ``w0`` is the starting value.
    ``weights`` multiplies each entry's loss (see ``_nnal_prep``).  The function returns (w, f),
    where f is the loss of each pixel."""
    P, m = idx.shape
    chunk = max(1, _FREE_SET_ELEMS // (m * H.shape[1]))
    if P <= chunk and rows is None:
        return _fit_free_sets_block(T, H, idx, valid, w0, steps, weights)
    ws, fs = [], []
    for s in range(0, P, chunk):
        e = min(P, s + chunk)
        Tc = T[s:e] if rows is None else T[rows[s:e]]
        wc = _weight_rows(weights, slice(s, e) if rows is None else rows[s:e])
        w, f = _fit_free_sets_block(Tc, H, idx[s:e], valid[s:e], w0[s:e], steps, wc)
        ws.append(w)
        fs.append(f)
        del Tc
    return torch.cat(ws), torch.cat(fs)


def _fit_free_sets_block(T, H, idx, valid, w0, steps, weights=None):
    m = idx.shape[1]
    dev = T.device
    prep = _nnal_prep(T, weights)
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
        # The code uses pinv because identical or zero spectra make Hs singular.
        d = (torch.linalg.pinv(Hs) @ rhs.unsqueeze(-1)).squeeze(-1) * free
        slope = (g * d).sum(1)
        d = torch.where((slope <= 0)[:, None], rhs / Hs.diagonal(dim1=1, dim2=2).clamp(min=1e-12), d)
        # The step is not limited to the largest feasible step.  Instead, the projection clips w at zero.
        alpha = torch.ones_like(slope)
        accepted = torch.zeros_like(alpha)
        done = torch.zeros_like(alpha, dtype=torch.bool)
        for _ in range(8):
            trial = torch.where(done, torch.zeros_like(alpha), alpha)
            wt = (w - trial[:, None] * d)
            wt = wt.clamp(min=0) * valid
            decrease = (g * (w - wt)).sum(1).clamp(min=0)              # This is the first-order decrease along the projected step.
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
    """This function builds the (P, R) support mask and the (P, R) coefficients from padded sets
    of material indices, one set per pixel.  It writes only the entries that ``valid`` marks."""
    P = idx.shape[0]
    pp, slot = valid.nonzero(as_tuple=True)
    support = torch.zeros(P, R, dtype=torch.bool, device=idx.device)
    support[pp, idx[pp, slot]] = True
    W = torch.zeros(P, R, dtype=w.dtype, device=idx.device)
    W[pp, idx[pp, slot]] = w[pp, slot]
    return support, W


def _empty_fit_loss(T, prep):
    """This function computes the loss f_p(0) of each pixel when the pixel has no materials.  It
    processes the pixels in chunks."""
    log_T, positive, all_positive, weights = prep
    chunk = max(1, 2 ** 23 // T.shape[1])
    out = []
    for s in range(0, T.shape[0], chunk):
        Tc = T[s:s + chunk]
        prep_c = (log_T[s:s + chunk], positive[s:s + chunk], all_positive, _weight_rows(weights, slice(s, s + chunk)))
        out.append(_nnal_rowwise(torch.zeros_like(Tc), Tc, prep_c, 1, dtype=torch.float64))
    return torch.cat(out)


def _support_sets(support, W):
    """This function converts a (P, R) support mask to padded sets of material indices, one set
    per pixel.  It returns (idx, valid, w0)."""
    valid, idx = torch.sort(support, dim=1, descending=True, stable=True)
    return idx, valid, W.gather(1, idx)


def _solve_W_on_support(T, H, W, support, steps=30, weights=None):
    """This function solves for W >= 0 with H fixed.  It solves each pixel by Newton steps on the
    support of that pixel.  ``weights`` multiplies each entry's loss (see ``_nnal_prep``)."""
    idx, valid, w0 = _support_sets(support, W.clamp(min=0))
    w, _ = _fit_free_sets(T, H, idx, valid, w0, steps=steps, weights=weights)
    return _scatter_support(idx, valid, w, W.shape[1])[1]


def _weak_components(counts, P, min_support=None):
    """This function returns a mask of the components that are selected in fewer than
    ``min_support`` pixels, and it warns if there are any.  The default ``min_support`` is
    max(2R, P / 1000).  The argument ``counts`` holds the number of pixels that select each
    component."""
    R = counts.numel()
    if min_support is None:
        min_support = max(2 * R, P // 1000)
    weak = counts < min_support
    if bool(weak.any()):
        warnings.warn(f"support selection: component(s) {weak.nonzero().flatten().tolist()} selected in fewer than "
                      f"{min_support} pixels; kept free in every pixel")
    return weak


def _guard_components(support, W_mle, W0, min_support=None):
    """This function finds the components that are selected in too few pixels.  For these
    components, it sets ``support`` to True in every pixel and copies the maximum-likelihood
    coefficients into ``W0``.  It changes both arrays in place, and it returns the mask of these
    components."""
    weak = _weak_components(support.sum(0), support.shape[0], min_support)
    if bool(weak.any()):
        support[:, weak] = True
        W0[:, weak] = W_mle[:, weak]
    return weak


def _select_branch_bound(T, W, H, dose, lam, f_full, k_top=_K_TOP, m_max=_M_MAX, plausible=None, prep=None):
    """This function finds the best subset of materials for each pixel by branch and bound.  It
    first fits each material alone.  It then fits subsets of 2 to ``m_max`` materials, chosen from
    the ``k_top`` materials that fit the pixel best alone.  Finally, it tries the full set.  The
    function warns when the search is not exhaustive.  It returns (idx, valid, w, f) as padded
    sets of width R, one set per pixel."""
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
        w1, f1 = _fit_free_sets(T, H, idx1, valid1, w_start, rows=rows, weights=prep[3])
        F1[rows, r] = f1
        W1[rows, r] = w1[:, 0]
    best_f1, r1 = F1.min(1)
    one = best_f1 * dose + lam < f0 * dose                          # The best single material beats the empty set.
    best_idx = torch.full((P, R), -1, dtype=torch.long, device=dev)
    best_valid = torch.zeros(P, R, dtype=torch.bool, device=dev)
    best_w = torch.zeros(P, R, device=dev, dtype=T.dtype)
    best_idx[one, 0] = r1[one]
    best_valid[one, 0] = True
    best_w[one, 0] = W1[one, r1[one]]
    best_f = torch.where(one, best_f1, f0)
    order = F1.argsort(1)[:, :k_top]                                # These are the candidate materials of each pixel.
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
            w_c, f_c = _fit_free_sets(T, H, idx_c, valid_c, (W1[cq[:, None], idx_c] / size).clamp(min=1e-3), rows=cq,
                                      weights=prep[3])
            better = f_c * dose + lam * size < best_f[cq] * dose + lam * best_valid[cq].sum(1).double()
            bp = cq[better]
            best_idx[bp] = -1
            best_valid[bp] = False
            best_w[bp] = 0
            best_idx[bp, :size] = idx_c[better]
            best_valid[bp, :size] = True
            best_w[bp, :size] = w_c[better]
            best_f[bp] = f_c[better]
    if m_max < R:                                                   # The full set uses the fitted W and its loss f_full.
        full = f_full * dose + lam * R < best_f * dose + lam * best_valid.sum(1).double()
        best_idx[full] = torch.arange(R, device=dev)
        best_valid[full] = True
        best_w[full] = W[full].to(best_w.dtype)
        best_f = torch.where(full, f_full, best_f)
    if k_top < R or m_max < R - 1:                                  # Some subsets smaller than the full set are not tried.
        n = best_valid.sum(1)
        at_full, at_cap = ((n == R).double().mean().item(), (n == m_max).double().mean().item())
        warnings.warn(f"support selection at rank {R} is not exhaustive (it is up to rank {min(k_top, m_max + 1)}): "
                      f"it tries the subsets of up to {m_max} of each pixel's {k_top} best single components, and the "
                      f"full set; {100 * at_full:.3g}% of the pixels select all {R} and {100 * at_cap:.3g}% select "
                      f"{m_max}, and a better support may be among those not tried")
    return best_idx, best_valid, best_w, best_f


def _auto_penalty(pixel_means, dose):
    """This function returns the automatic penalty per selected material as a multiple of log K.
    It computes the penalty from the counts per bin of the median pixel.  The penalty is 0.5 below
    10 counts, 2 above 100 counts, and linear in log(counts) in between."""
    counts = float(dose) * torch.median(pixel_means.float()).item()
    frac = min(1.0, max(0.0, (math.log10(max(counts, 1e-12)) - 1.0)))          # frac is 0 at 10 counts and 1 at 100.
    return 0.5 * 4 ** frac


def _penalty_nats(penalty, T, dose):
    """This function converts the penalty to nats.  The penalty is either 'auto' or a multiple of
    log K."""
    K = T.shape[1]
    if isinstance(penalty, str):
        if penalty != "auto":
            raise ValueError(f"penalty must be a multiple of log K or 'auto', got {penalty!r}")
        return _auto_penalty(T.mean(1), dose) * math.log(K)
    if not float(penalty) >= 0:
        raise ValueError(f"penalty must be nonnegative, got {penalty!r}")
    return float(penalty) * math.log(K)


def _select_supports(T, W, H, dose, penalty='auto', wald_screen=0.0, weights=None):
    """This function selects the subset S of materials for each pixel.  It minimizes
    dose * f_p(S) + lam * size(S) by branch and bound, where lam is the penalty in nats.  If
    ``wald_screen`` > 0, the function uses a Wald statistic to skip unlikely single-material fits.
    With ``weights`` (see ``_nnal_prep``), dose times the weights must be the open beam count of
    each bin, so that dose * f_p is the negative log-likelihood.  It returns (support, W0, f).
    These are the (pixels, R) bool support, the coefficients on the support, and the loss of each
    pixel."""
    if not dose > 0:
        raise ValueError(f"support selection needs a positive dose, got {dose!r}")
    R = H.shape[0]
    lam = _penalty_nats(penalty, T, dose)
    prep = _nnal_prep(T, weights)
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
    """This function warns if two spectra are nearly proportional."""
    Hn = H.double() / H.double().norm(dim=1, keepdim=True).clamp_min(1e-300)
    C = (Hn @ Hn.T).fill_diagonal_(0)
    if bool((C > cosine).any()):
        i, j = (C > cosine).nonzero()[0].tolist()
        warnings.warn(f"spectra {i} and {j} are proportional (cosine {C[i, j].item():.4f}): their pixel coefficients "
                      "are not separately identified; consider a smaller rank")


def _support_selected_spectra(T, W, H, dose, penalty='auto', wald_screen=0.0, compile_mode='auto', weights=None):
    """This function chooses the subset of materials for each pixel by penalized likelihood.  It
    then refits W and H with all other coefficients held at zero.

    Holding the coefficients that are truly zero at zero removes the bias of the spectra.  If a
    component is selected in almost no pixels, the function keeps that component free in every
    pixel, as in the maximum-likelihood fit.  The function returns (W, H, support, steps).  These
    are the refitted factors, the (pixels, R) bool support, and the number of refit steps.  With
    ``weights``, dose times the weights must be the open beam count of each bin (see
    ``_select_supports``).
    """
    compile_mode = _resolve_compile(compile_mode, T)
    nnal_fn, deriv, _, _ = _kernels(compile_mode)
    prep = _nnal_prep(T, weights)
    support, W0, _ = _select_supports(T, W, H, dose, penalty=penalty, wald_screen=wald_screen, weights=weights)
    _guard_components(support, W, W0)
    Wn, Hn, steps, _ = _joint_newton_pcg(T, W0, H, max_steps=_REFIT_MAX_STEPS, cg_max=_CG_MAX, rel_tol=_REFIT_REL_TOL,
                                         prep=prep, nnal=nnal_fn, deriv=deriv, w_mask=support)
    _warn_collinear_rows(Hn)
    return Wn, Hn, support, steps

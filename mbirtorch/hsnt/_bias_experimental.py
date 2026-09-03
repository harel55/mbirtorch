import math

import torch

from . import _newton
from ._loss import _nnal_prep, stable_nnal, stable_nnal_derivatives
from ._newton import _joint_newton_pcg, _kernels, solve_W


# -----------------------------------------------------------------------
# Bias-corrected spectra: the modified profile likelihood
# -----------------------------------------------------------------------
_CORRECTIONS = ('cox_reid', 'barndorff_nielsen', 'orthant', 'bootstrap')


def _kr_matrices(Z, A, B):
    """Per-pixel R x R matrices  M_p = sum_k Z_pk a_k b_k^T  for A, B of shape (R, K).

    One GEMM of Z (P x K) against the K x R^2 Khatri-Rao product of A and B.
    """
    R = A.shape[0]
    KR = (A[:, None, :] * B[None, :, :]).reshape(R * R, -1)      # row r*R+s holds a_r * b_s
    return (Z @ KR.T).reshape(Z.shape[0], R, R)


def _adjustment_value_and_grad(T, W, H, dose, correction, H_ref, chunk=8192):
    """Value of the modified profile objective and the total H-gradient of its adjustment.

    At W = w_hat(H), the ML solution for the pixels given H,

        F(H) = L(W, H) + a(H),

    with the Cox-Reid adjustment

        a = 1/(2 dose) sum_p [ log det M_p - log det H H^T ],     M_p = sum_k Z_pk h_k h_k^T,

    M_p being the observed information of pixel p's coefficients per unit dose
    (the same R x R matrix block_newton_step forms) and the Gram term fixing the
    gauge (see the comment in the code), or the Barndorff-Nielsen
    adjustment in Severini's (1998) form,

        a = 1/dose sum_p [ log|det N_p| - 1/2 log det M_p ],   N_p = sum_k Z_pk h_k hhat_k^T,

    hhat being the reference ML estimate of H. Each is restricted to the pixel's
    free coordinates: a coefficient at the bound is not estimated, so there is no
    estimation noise to correct for. A pixel with no material contributes nothing.

    The adjustment depends on H directly and through w_hat(H), and its gradient
    needs both. One Newton step of the W problem from its converged point,
    W1 = W - M^-1 g(W, H), is a differentiable function of H whose derivative at
    the fixed point equals that of w_hat(H) (implicit function theorem), so
    autograd through a(H, W1(H)) returns the exact total gradient.

    Returns (F, da/dH) with F in the units of stable_nnal and the gradient in
    H's dtype. Pixels are processed in chunks to bound autograd memory.
    """
    R = H.shape[0]
    Hg = H.detach().clone().requires_grad_(True)
    Href = (H if H_ref is None else H_ref).detach()
    eye = torch.eye(R, dtype=W.dtype, device=W.device)
    scale = W.abs().amax(-1, keepdim=True).mean()
    loss_total = 0.0
    adj_total = 0.0
    for s in range(0, T.shape[0], chunk):
        Tc = T[s:s + chunk]
        W0 = W[s:s + chunk].detach()
        prep = _nnal_prep(Tc)
        X0 = W0 @ Hg
        G0, Z0 = stable_nnal_derivatives(X0, Tc, prep)
        g = G0 @ Hg.T                                                   # gradient of the pixel loss in w
        free = ~((W0 <= _newton._ACTIVE_TOL * scale) & (g.detach() > 0))        # block_newton_step's active set
        fm = free[:, :, None] & free[:, None, :]
        M0 = torch.where(fm, _kr_matrices(Z0, Hg, Hg), eye.expand_as(fm).to(W.dtype))
        d = torch.linalg.solve(M0, torch.where(free, g, torch.zeros_like(g))[..., None])[..., 0]
        W1 = W0 - torch.where(free, d, torch.zeros_like(d))
        X1 = W1 @ Hg
        with torch.no_grad():
            loss_total += stable_nnal(X1, Tc, prep, dtype=torch.float64).item()
        Z1 = torch.exp(-X1)
        M1 = torch.where(fm, _kr_matrices(Z1, Hg, Hg), eye.expand_as(fm).to(W.dtype))
        logdet_M = torch.logdet(M1)
        if correction == 'cox_reid':
            # Gauge. The likelihood is invariant under (W, H) -> (W A, A^-1 H), but
            # log det M_p shifts by -2 log|det A|, so along the exactly flat scaling
            # direction the adjusted objective would be linear and unbounded -- the
            # outer loop drifted 97% along it. Measuring the information relative to
            # the subspace's Gram matrix H H^T (a flat prior on the pixel's
            # attenuation profile in R^K rather than on its coordinates) makes the
            # adjustment invariant; it equals Cox-Reid in the gauge H H^T = I.
            gram = torch.where(fm, (Hg @ Hg.T)[None].expand_as(fm), eye.expand_as(fm).to(W.dtype))
            adj = (logdet_M - torch.logdet(gram)).sum() / (2.0 * dose)
        elif correction == 'barndorff_nielsen':
            N1 = torch.where(fm, _kr_matrices(Z1, Hg, Href), eye.expand_as(fm).to(W.dtype))
            adj = (torch.linalg.slogdet(N1)[1].sum() - 0.5 * logdet_M.sum()) / dose
        elif correction == 'orthant':
            # EXPERIMENTAL. Integrated likelihood over w >= 0: the invariant Laplace
            # term on the free coordinates plus, for each coordinate at the bound
            # with outward gradient g and curvature m, the one-dimensional integral
            #   int_0^inf exp(-dose (g w + m w^2 / 2)) dw
            #     = (dose m)^-1/2 sqrt(2 pi) exp(gamma^2/2) Phi(-gamma),  gamma = g sqrt(dose/m),
            # along the profile direction (|h_r| dw) so that it is gauge invariant.
            # Incomplete: the bound coordinates are treated as independent 1-D
            # integrals with marginal curvature, where the correlated spectra call
            # for a joint orthant probability. See docs/hsnt_solver_notes.md, section 5.
            gram = torch.where(fm, (Hg @ Hg.T)[None].expand_as(fm), eye.expand_as(fm).to(W.dtype))
            adj_free = (logdet_M - torch.logdet(gram)).sum() / (2.0 * dose)
            G1 = Tc - Z1
            bound = ~free
            g1 = torch.where(bound, (G1 @ Hg.T).clamp(min=0), torch.zeros_like(W0))     # outward gradient at the bound
            m_rr = Z1 @ (Hg * Hg).T                                                      # marginal curvature
            hn2 = (Hg * Hg).sum(1)[None, :]
            gamma = g1 * torch.sqrt(dose / m_rr)
            log_int = 0.5 * math.log(2 * math.pi) + 0.5 * gamma * gamma + torch.special.log_ndtr(-gamma)
            bound_term = 0.5 * torch.log(m_rr / hn2) - log_int
            adj = adj_free + torch.where(bound, bound_term, torch.zeros_like(bound_term)).sum() / dose
        else:
            raise ValueError(f"correction must be one of {_CORRECTIONS}")
        adj.backward()
        adj_total += adj.item()
    return loss_total + adj_total, Hg.grad.to(H.dtype)


def _bootstrap_score_bias(T, W, H, dose, n_sim, seed, chunk, w_steps, compile_mode, deriv):
    """Parametric-bootstrap estimate of the profile score's bias at (W, H).

    Counts are simulated from the fitted model, T_b ~ Poisson(dose exp(-W H)) / dose,
    the pixel coefficients are re-solved under the same nonnegativity constraints
    with H fixed, and the profile score W_b^T (T_b - Z_b) is averaged. Were (W, H)
    the truth, that average would be the score's expectation -- the quantity a
    bias-corrected estimating equation subtracts (Kuk 1995; Hahn & Newey 2004).
    It captures whatever biases the per-pixel solve, including the truncation of
    coefficients at zero, which no curvature-based adjustment sees. The noise of
    the estimate falls as 1/sqrt(P n_sim), so it is most precise where the bias
    matters. The seed is fixed across outer iterations (common random numbers)
    so the fixed-point map is smooth.
    """
    R = H.shape[0]
    gen = torch.Generator(device=T.device).manual_seed(seed)
    b = torch.zeros(H.shape, dtype=torch.float64, device=H.device)
    for s in range(0, T.shape[0], chunk):
        Wc = W[s:s + chunk]
        lam = dose * torch.exp(-(Wc @ H))
        for _ in range(n_sim):
            Tb = torch.poisson(lam, generator=gen) / dose
            Wb = solve_W(Tb, H, Wc, w_steps, 1e-12, compile_mode=compile_mode)
            Gb, _ = deriv(Wb @ H, Tb, _nnal_prep(Tb))
            b += (Wb.T @ Gb).to(torch.float64)
    return b / n_sim


def bias_corrected_spectra(T, W, H, dose, correction='bootstrap', max_outer=20, rel_tol=1e-8, h_tol=1e-3,
                           joint_steps=None, cg_max=10, chunk=8192, n_sim=8, seed=0, w_steps=30,
                           relax=1.0, callback=None, compile_mode=None, verbose=False):
    """Move a converged ML factorization to the root of a bias-corrected estimating equation.

    Kept for reference: none of these corrections helped on the phantom. The
    dominant bias of the ML spectra is the truncation of pixel coefficients at
    zero, which the curvature adjustments do not see and the bootstrap
    underestimates; unconstrained_spectra and support_selected_spectra are the
    estimators that work. See docs/hsnt_solver_notes.md, section 5.

    H is shared by every pixel and estimated jointly with R nuisance coefficients
    per pixel, each pixel carrying a fixed amount of information (K bins at the
    given dose). That is the Neyman-Scott setting: the ML estimate of H is not
    consistent -- its bias is set by the information per pixel, not the pixel
    count -- and on this problem converging the ML solve further lowered the
    spectral SNR while the loss kept falling. The modified profile likelihood
    (Barndorff-Nielsen 1983; Cox & Reid 1987) subtracts 1/2 log det j_ww(H, w_hat(H))
    per pixel, the Laplace approximation to integrating the nuisance parameters
    out; it removes the leading term of the bias, and is exact (REML) in the
    Neyman-Scott normal example.

    Solved as a fixed point. At the current (W, H) the adjustment's total
    gradient c = da/dH is computed, including its dependence through w_hat(H);
    the ML problem with the linear tilt <c, H> is then re-solved jointly from the
    current point, whose stationarity conditions are grad_W L = 0 and
    grad_H L + c = 0 -- those of the modified profile likelihood once c has
    stopped changing. The tilt is a perturbation of relative size ~1/(dose K), so
    a handful of outer iterations suffice.

    Args:
        T, W, H: data and a converged ML factorization (e.g. joint_newton at 1e-8).
        dose: open-beam counts per pixel and bin, so that T = counts / dose. It sets
            the weight of the adjustment relative to the loss; a scalar.
        correction: 'bootstrap' (default) subtracts a parametric-bootstrap estimate of
            the profile score's bias (n_sim simulated data sets per outer iteration,
            common random numbers from `seed`); it is the only one of the three that
            sees the truncation of coefficients at zero, which dominates on data with
            pure-material pixels. 'cox_reid' and 'barndorff_nielsen' (Severini's
            approximation, with the input H as reference) are the curvature-based
            adjustments; they assume interior per-pixel maxima and an
            information-orthogonal parametrization, neither of which holds here.
        rel_tol: stop when an outer iteration changes F by less than this, relatively
            (also the tolerance of each inner joint solve).
        h_tol: or when it changes H by less than this, relative to |H|. The default
            1e-3 is below the correction's own noise (about 1/sqrt(n_sim) of the ML
            estimate's sampling error for the bootstrap).
        joint_steps: Newton steps of the tilted joint solve per outer iteration. None
            means 1 for the bootstrap and 100 (a full solve) for the gradient-based
            adjustments. A full solve with the bootstrap tilt is unsafe: a linear tilt
            on H with W >= 0 is unbounded below -- shrink a component's W column while
            growing its H row and the loss stands still while <c, H> falls -- and one
            run moved H by 176% at the step cap. One damped, CG-truncated Newton step
            per outer iteration is bounded by construction; the outer loop supplies
            the iteration.
        relax: under-relaxation of the outer update, H <- H_prev + relax (H_new - H_prev)
            (W re-solved), for fixed-point maps that are not contractions at relax=1.
        callback: optional f(iteration, W, H, F, tilt) called once per outer iteration.

    Returns:
        (W, H, outer_iterations).
    """
    if correction not in _CORRECTIONS:
        raise ValueError(f"correction must be one of {_CORRECTIONS}")
    if joint_steps is None:
        joint_steps = 1 if correction == 'bootstrap' else 100
    nnal_fn, deriv, _, _ = _kernels(compile_mode)
    prep = _nnal_prep(T)
    R = H.shape[0]
    H_ref = H.detach().clone()
    W = solve_W(T, H, W, 50, 1e-12, compile_mode=compile_mode)
    F_prev, H_prev, rises = None, None, 0
    for it in range(max_outer):
        if correction == 'bootstrap':
            tilt = -_bootstrap_score_bias(T, W, H, dose, n_sim, seed, chunk, w_steps, compile_mode, deriv).to(H.dtype)
            F = nnal_fn(W @ H, T, prep, dtype=torch.float64).item()      # no adjusted objective: stop on dH
        else:
            F, tilt = _adjustment_value_and_grad(T, W, H, dose, correction, H_ref, chunk)
        # Along the gauge orbit {A H} the loss is exactly flat, so any component
        # of the tilt there makes the tilted objective unbounded: with the
        # bootstrap tilt |H| grew 200x in five iterations while the loss stood
        # still. Complementary slackness would make the score orthogonal to that
        # orbit exactly, but the per-pixel solves are converged only to a
        # tolerance, so the orthogonality is enforced: each row of the tilt is
        # projected onto the complement of H's row space. The removed part would
        # only have moved H within its own row space, which changes no fit.
        Hd = H.to(torch.float64)
        gram = Hd @ Hd.T
        tilt = (tilt.to(torch.float64) - torch.linalg.solve(gram, Hd @ tilt.to(torch.float64).T).T @ Hd).to(H.dtype)
        dH = float('nan') if H_prev is None else ((H - H_prev).norm() / H.norm()).item()
        if verbose:
            print(f'  outer {it}: F {F:.8e}  |dH|/|H| {dH:.2e}  |c| {tilt.norm().item():.3e}', flush=True)
        if callback is not None:
            callback(it, W, H, F, tilt)
        if F_prev is not None:
            if dH <= h_tol or (correction != 'bootstrap' and abs(F - F_prev) <= rel_tol * abs(F)):
                break
            rises = rises + 1 if (F > F_prev and correction != 'bootstrap') else 0
            if rises >= 3:
                if verbose:
                    print('  F rose three times in a row; stopping at the last iterate', flush=True)
                break
        F_prev, H_prev = F, H.clone()
        W, H, _, _ = _joint_newton_pcg(T, W, H, max_steps=joint_steps, cg_max=cg_max, rel_tol=rel_tol,
                                       prep=prep, nnal=nnal_fn, deriv=deriv, tilt_H=tilt)
        if relax != 1.0:
            H = (H_prev + relax * (H - H_prev)).clamp_(min=0)
            W = solve_W(T, H, W, 50, 1e-12, compile_mode=compile_mode)
    return W, H, it + 1

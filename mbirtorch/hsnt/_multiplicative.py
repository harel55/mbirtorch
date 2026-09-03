import torch

from ._loss import _nnal_prep, stable_nnal, stable_nnal_derivatives


def _shifted(V, ratio, shift, mode, mean_dim):
    """V <- max((V + d) * ratio - d, 0), the shifted multiplicative step.

    A plain multiplicative update cannot revive an entry that reaches exactly
    zero, since 0 * anything = 0. On the demo problem 5.5% of W starts at zero
    and that fraction never moves over 6000 iterations, even though 1.77% of all
    entries are zero with a NEGATIVE gradient -- not KKT points, just frozen.

    The offset does not bias the answer. Expanding gives V*ratio + d*(ratio - 1),
    so an interior fixed point needs (ratio - 1)(V + d) = 0, and V + d > 0 forces
    ratio = 1; a zero entry stays zero only while d*(ratio - 1) <= 0, i.e.
    ratio <= 1. Both updates are built so ratio > 1 exactly when the gradient is
    negative, so the fixed-point set is {V >= 0, grad >= 0, V*grad = 0} -- the KKT
    set -- for every d > 0. No decay schedule is needed for correctness.

    mode='boundary' offsets only entries currently at zero, leaving the interior
    step bit-for-bit the original update. mode='const' offsets everything, which
    converges faster but rides on the raw ratio, so an undamped ratio needs a cap.
    """
    if shift <= 0:
        return V * ratio
    # The offset is sized from the per-component mean, but a component that has
    # collapsed has a mean of zero -- so the offset that exists to revive it would
    # vanish exactly when it is needed. Floor each component's scale at a fraction
    # of the whole factor's mean so a dead row or column still gets a live offset.
    per_component = V.mean(dim=mean_dim, keepdim=True)
    floor = 1e-2 * V.mean()
    scale = shift * torch.maximum(per_component, floor).clamp_min(torch.finfo(V.dtype).tiny)
    if mode == 'boundary':
        scale = torch.where(V <= 0, scale.expand_as(V), torch.zeros_like(V))
    return (V * ratio + scale * (ratio - 1.0)).clamp_(min=0)


def quadratic_update(W, H, T, unused=None, update_H=True, prep=None,
                     shift=1e-1, shift_mode='boundary', ratio_max=2.0):
    """Iteratively reweighted multiplicative update for the NNAL.

    The second-order model of the NNAL at the current iterate X is

        L(X') ~ L(X) + G (X' - X) + (1/2) Z (X' - X)^2 = (1/2) Z (X' - V)^2 + c

    with G = T - exp(-X), Z = exp(-X) and target V = X - G/Z. Reweighted least
    squares on that model, relinearized every step, has the NNAL stationary point
    as its fixed point.

    Z V = Z X - G, which needs no exp(+X) and so stays bounded however large the
    attenuation gets. Z V is signed, so the multiplicative update splits it into
    positive and negative parts and puts each on the side of the ratio where it
    keeps W and H nonnegative; at the fixed point the ratio is one, which gives
    (exp(-X) - T) H^T = 0, exactly the NNAL stationarity condition.

    This replaces an earlier update that minimized (1/2) sum T (X + log T)^2 --
    the second-order model expanded about the noiseless solution rather than the
    current iterate, and never relinearized. That objective weights every
    zero-count measurement by exactly zero, so at low dose it discards a large
    fraction of the data (17.7% at dosage_rate=3) and converges somewhere other
    than the NNAL minimum. Here the weight is exp(-X) > 0 everywhere, so no
    measurement is dropped.

    Note this is a Gauss-Newton style surrogate, not a majorizer, so individual
    steps are not guaranteed to decrease the loss.

    Args:
        W: Feature matrix (spatial pixels × num_materials), PyTorch tensor
        H: Spectral basis matrix (num_materials × spectral channels), PyTorch tensor
        T: Data term matrix (spatial pixels × spectral channels), PyTorch tensor
        unused: Placeholder for auxiliary state (not used by this method).
        update_H: If False, keep H fixed and only update W.
        prep: Optional tuple from _nnal_prep(T).

    Returns:
        Updated (W, H) pair as PyTorch tensors
    """
    prep = _nnal_prep(T) if prep is None else prep
    tiny = torch.finfo(T.dtype).tiny

    X = W @ H
    G, Z = stable_nnal_derivatives(X, T, prep)
    ZV = Z * X - G
    # This ratio is undamped and can be very large. shift_mode='const' rides on it
    # for every entry and diverges to NaN within a few hundred iterations without a
    # cap; 'boundary' only offsets entries already at zero and is safe uncapped, so
    # the cap is applied only where it is needed and never alters the plain update.
    cap = ratio_max if (shift > 0 and shift_mode == 'const') else float('inf')
    ratio = ((ZV.clamp(min=0) @ H.T)
             / ((Z * X) @ H.T + (-ZV).clamp(min=0) @ H.T).clamp(min=tiny))
    W = _shifted(W, ratio.clamp(max=cap), shift, shift_mode, 0)

    if update_H:
        # Relinearize before the H step: the model is only second order accurate
        # at the iterate it was expanded about, and W has just moved.
        X = W @ H
        G, Z = stable_nnal_derivatives(X, T, prep)
        ZV = Z * X - G
        ratio = ((W.T @ ZV.clamp(min=0))
                 / (W.T @ (Z * X) + W.T @ (-ZV).clamp(min=0)).clamp(min=tiny))
        H = _shifted(H, ratio.clamp(max=cap), shift, shift_mode, 1)

    return W, H, 0.0


def newton_update(W, H, T, lr_init, update_H=True, prep=None):
    """PyTorch-optimized Newton update with automatic differentiation and line search.

    Args:
        W: Feature matrix (spatial pixels × num_materials), PyTorch tensor
        H: Spectral basis matrix (num_materials × spectral channels), PyTorch tensor
        T: Data term matrix (spatial pixels × spectral channels), PyTorch tensor
        lr_init: Initial learning rate for line search
        update_H: If False, keep H fixed and only update W.

    Returns:
        Updated (W, H) pair as PyTorch tensors
    """
    X = W @ H
    G, Z = stable_nnal_derivatives(X, T, prep)
    init_loss = stable_nnal(X, T, prep)

    # Compute gradients
    grad_W = G @ H.T
    grad_H = W.T @ G

    # Compute Hessian diagonal approximation manually (kept explicit for numerical stability)
    d2L_dW2 = Z @ H.T.pow(2)
    d2L_dH2 = W.T.pow(2) @ Z

    dW = grad_W / (d2L_dW2 + 1e-30)
    dH = grad_H / (d2L_dH2 + 1e-30) if update_H else torch.zeros_like(H)

    learning_rates = lr_init * torch.logspace(
        -10, 1, steps=13, base=2, dtype=T.dtype, device=T.device
    )
    W_candidates = torch.clip(W[None] - learning_rates[:, None, None] * dW[None], min=1e-30)
    if update_H:
        H_candidates = torch.clip(H[None] - learning_rates[:, None, None] * dH[None], min=1e-30)
    else:
        H_candidates = H.expand(learning_rates.shape[0], -1, -1)

    X_candidates = torch.bmm(W_candidates, H_candidates)
    candidate_losses = stable_nnal(X_candidates, T, prep)
    directional_derivatives = (
        torch.sum(grad_W[None] * (W_candidates - W[None]), dim=(1, 2))
        + torch.sum(grad_H[None] * (H_candidates - H[None]), dim=(1, 2))
    )
    armijo = candidate_losses <= init_loss + 1e-4 * directional_derivatives
    valid_losses = torch.where(armijo, candidate_losses, torch.full_like(candidate_losses, torch.inf))
    best_index = torch.argmin(valid_losses)
    best_index = best_index.reshape(1)
    return (
        W_candidates.index_select(0, best_index).squeeze(0),
        H_candidates.index_select(0, best_index).squeeze(0),
        learning_rates.index_select(0, best_index).squeeze(0),
    )

def _rebalance(W, H):
    """Equalize the scale of each component between W and H, leaving W @ H fixed.

    The factorization is invariant to W -> W D, H -> D^-1 H for any positive
    diagonal D, and nothing in a multiplicative update pins that gauge down. At
    low dosage it drifts to extremes -- H rows near 1e6 against W columns near
    1e-3 -- which starves the shifted step's offset (it scales with the mean of
    the factor being updated) and pushes float32 toward its limits. Setting
    ||W[:, r]|| = ||H[r, :]|| each step costs two norms and changes nothing else.
    """
    w = W.norm(dim=0); h = H.norm(dim=1)
    tiny = torch.finfo(W.dtype).tiny
    d = torch.sqrt((h.clamp_min(tiny) / w.clamp_min(tiny)))
    d = torch.where((w > tiny) & (h > tiny), d, torch.ones_like(d))
    return W * d, H / d[:, None]


def _reseed_dead(W, H, rel_tol=1e-6, mode='random'):
    """Re-seed any component that has died in BOTH factors at once.

    The shifted step revives an entry at zero whenever its ratio exceeds one. But a
    component whose W column and H row are both zero has a ratio of exactly zero in
    either update -- the numerator is a product with the dead factor -- so the
    offset term d*(ratio - 1) is negative and the clamp holds it at zero forever.
    No multiplicative step can reach it.

    What it is re-seeded WITH matters as much as whether it is. A constant seed
    keeps the loss honest but leaves the component flat, and a flat spectrum makes
    the gauge fit against the true spectra ill-conditioned: at dosage_rate=1 the
    constant seed reaches the same loss as joint_newton to 0.3% yet scores -84 dB
    on the recovered spectra against joint_newton's 21. A random positive seed
    breaks that symmetry and hands the dynamics something to differentiate; it
    scores 21.4 dB on the same case. Seeded, so runs are reproducible.
    """
    w = W.norm(dim=0); h = H.norm(dim=1)
    dead = (w <= rel_tol * w.max()) & (h <= rel_tol * h.max())
    n_dead = int(dead.sum())
    if n_dead == 0:
        return W, H
    live = ~dead
    W = W.clone(); H = H.clone()
    w_ref = W[:, live].mean() if bool(live.any()) else W.new_tensor(1.0)
    h_ref = H[live].mean() if bool(live.any()) else H.new_tensor(1.0)
    if mode == 'const':
        W[:, dead] = 1e-2 * w_ref
        H[dead] = 1e-2 * h_ref
        return W, H
    g = torch.Generator(device=W.device).manual_seed(0)
    W[:, dead] = 1e-2 * w_ref * torch.rand(W.shape[0], n_dead, generator=g, dtype=W.dtype, device=W.device)
    H[dead] = 1e-2 * h_ref * torch.rand(n_dead, H.shape[1], generator=g, dtype=H.dtype, device=H.device)
    return W, H


def multiplicative_update(W: torch.Tensor, H: torch.Tensor, T: torch.Tensor, unused: torch.Tensor = None,
                          update_H: bool = True, prep=None, shift=1e-2, shift_mode='const',
                          ratio_max=2.0, rebalance=True, reseed='random'):
    """PyTorch-optimized multiplicative update for non-negative factorization.

    Args:
        W: Feature matrix (spatial pixels × num_materials), PyTorch tensor
        H: Spectral basis matrix (num_materials × spectral channels), PyTorch tensor
        T: Data term matrix (spatial pixels × spectral channels), PyTorch tensor
        unused: Placeholder for auxiliary variables (not used in this implementation)
        update_H: If False, keep H fixed and only update W.

    Returns:
        Updated (W, H) pair as PyTorch tensors
    """
    damping_factor = 0.5
    Z = torch.exp(-W @ H)

    # Damping alone does not bound this ratio. At very low dosage T is almost all
    # zeros, so T @ H.T reaches the 1e-30 clip and the ratio reaches ~1e15; the
    # shifted step's offset term d*(ratio - 1) then overflows to NaN within one
    # iteration. Cap it wherever the shifted step rides on the raw ratio. shift=0
    # is left uncapped so it stays bit-for-bit the original update.
    cap = ratio_max if (shift > 0 and shift_mode == 'const') else float('inf')
    W = _shifted(W, (((Z @ H.T) / torch.clip(T @ H.T, min=1e-30)) ** damping_factor).clamp(max=cap),
                 shift, shift_mode, 0)
    if update_H:
        H = _shifted(H, (((W.T @ Z) / torch.clip(W.T @ T, min=1e-30)) ** damping_factor).clamp(max=cap),
                     shift, shift_mode, 1)
        # Only when the shift is on, so shift=0 stays bit-for-bit the original.
        if shift > 0 and reseed:
            W, H = _reseed_dead(W, H, mode=reseed)
        if shift > 0 and rebalance:
            W, H = _rebalance(W, H)

    return W, H, 0.0

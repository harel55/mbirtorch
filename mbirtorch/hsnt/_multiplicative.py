import torch



def _shifted(V, ratio, shift, mode, mean_dim):
    """V <- max((V + d) * ratio - d, 0), the shifted multiplicative step.

    A plain multiplicative update cannot revive an entry that reaches exactly
    zero, since 0 * anything = 0, so entries can freeze at zero with a negative
    gradient -- not KKT points. The offset does not bias the answer: expanding
    gives V*ratio + d*(ratio - 1), so an interior fixed point needs
    (ratio - 1)(V + d) = 0, and V + d > 0 forces ratio = 1; a zero entry stays
    zero only while d*(ratio - 1) <= 0, i.e. ratio <= 1. Both updates are built so
    ratio > 1 exactly when the gradient is negative, so the fixed-point set is
    {V >= 0, grad >= 0, V*grad = 0} -- the KKT set -- for every d > 0. No decay
    schedule is needed for correctness. See docs/hsnt_solver_notes.md, section 3.

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

    The default seed is small random positive values from a fixed generator, so
    runs are reproducible. A constant seed leaves the component flat, and a flat
    spectrum makes the gauge fit against the true spectra ill-conditioned; the
    random seed breaks that symmetry. See docs/hsnt_solver_notes.md, section 3.
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

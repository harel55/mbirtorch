import torch


def _randomized_svd(X, n_components, n_oversamples=10, n_iter=4, seed=0):
    """Truncated SVD via a randomized range finder.

    A full SVD of an (n_samples, n_features) matrix is wasted work when only a
    handful of components are wanted, and here n_components is the material count
    -- rarely above 20 against a thousand or more wavelength bins. Seeded so the
    result is reproducible.
    """
    n_rows, n_cols = X.shape
    rank = min(n_components + n_oversamples, n_rows, n_cols)
    generator = torch.Generator(device=X.device).manual_seed(seed)
    Q, _ = torch.linalg.qr(X @ torch.randn(n_cols, rank, generator=generator,
                                           dtype=X.dtype, device=X.device))
    for _ in range(n_iter):                      # power iterations sharpen the range
        Q, _ = torch.linalg.qr(X.T @ Q)
        Q, _ = torch.linalg.qr(X @ Q)
    U, singular_values, Vh = torch.linalg.svd(Q.T @ X, full_matrices=False)
    return Q @ U, singular_values, Vh


def nndsvda(X, n_components, fill='sqrt', fill_scale=1.0):
    """NNDSVD initialization for X ~= W @ H, with zeros filled.

    Every component after the first is one sign-half of a singular vector pair,
    so roughly half its entries are zero. Zeros are poison for multiplicative
    updates, which can never move an entry off zero, so they are filled.

    The fill value has to be chosen at the scale of a FACTOR entry, not of X.
    Each factor carries sqrt(s_k), so a typical W or H entry is of order
    sqrt(mean X) and their product is of order mean X. A fill of f in both
    factors therefore contributes f^2 to the product where both were filled.

      fill = mean X       (classic NNDSVDA)  -> f^2 = (mean X)^2, which only
                          matches the data when mean X ~ 1. At mean X = 34.5
                          it put 1190 into every noise component.
      fill = mean X / 100                    -> f^2 = (mean X)^2 / 1e4. Safe
                          against overshoot but so small that block_newton's
                          two-metric projection froze the filled entries at
                          zero and converged in a reduced subspace, 0.6% worse.
      fill = c * sqrt(mean X)  (default)     -> f^2 = c^2 * mean X: a fixed
                          fraction c^2 of a typical entry, whatever the scale of
                          X. Dimensionally right and scale-free.

    c = 1 was chosen by measurement, not by taste. block_newton's two-metric
    projection freezes any entry it drives to zero, and a fill it can push there
    in one step costs a component: at dosage_rate=3 the loss it reaches improves
    monotonically with the fill, 904,026 at mean/100, 902,547 at c=0.1, 900,323
    at c=0.3, 898,850 at classic mean, 898,514 at c=1 -- the last within 0.001% of
    the joint solver. The multiplicative and joint solvers are indifferent to c
    across that whole range. The price is an initial X that can exceed the data
    by up to 2x where fills coincide; that is bounded and scale-free, unlike the
    35x of classic NNDSVDA on a badly floored matrix, and no solver minds it.

    Args:
        X: Nonnegative array of shape (n_samples, n_features).
        n_components: Factorization rank.
        fill: 'sqrt' (default) fills zeros with fill_scale * sqrt(mean X);
            'mean' is classic NNDSVDA; 'small' is mean X / 100.
        fill_scale: multiplier for the 'sqrt' fill. Defaults to 1.0.

    Returns:
        W: Shape (n_samples, n_components).
        H: Shape (n_components, n_features).
    """
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    if n_components + 10 < min(X.shape):
        U, singular_values, Vh = _randomized_svd(X, n_components)
    else:
        U, singular_values, Vh = torch.linalg.svd(X, full_matrices=False)

    n_components = min(
        n_components,
        U.shape[1],
        Vh.shape[0],
    )

    W = torch.zeros((X.shape[0], n_components), dtype=X.dtype, device=X.device)
    H = torch.zeros((n_components, X.shape[1]), dtype=X.dtype, device=X.device)

    # First singular triplet.
    scale = torch.sqrt(singular_values[0])
    W[:, 0] = scale * torch.abs(U[:, 0])
    H[0, :] = scale * torch.abs(Vh[0, :])

    # Remaining components.
    for component in range(1, n_components):
        u = U[:, component]
        v = Vh[component, :]

        u_pos = torch.clip(u, min=0)
        u_neg = torch.clip(-u, min=0)
        v_pos = torch.clip(v, min=0)
        v_neg = torch.clip(-v, min=0)

        positive_strength = (
            torch.linalg.norm(u_pos) * torch.linalg.norm(v_pos)
        )
        negative_strength = (
            torch.linalg.norm(u_neg) * torch.linalg.norm(v_neg)
        )

        use_positive = positive_strength > negative_strength

        selected_u = torch.where(use_positive, u_pos, u_neg)
        selected_v = torch.where(use_positive, v_pos, v_neg)

        selected_u /= torch.linalg.norm(selected_u) + torch.finfo(X.dtype).eps
        selected_v /= torch.linalg.norm(selected_v) + torch.finfo(X.dtype).eps

        scale = torch.sqrt(singular_values[component])
        W[:, component] = scale * selected_u
        H[component, :] = scale * selected_v

    mean = torch.mean(X)
    if fill == 'mean':
        fill_value = mean
    elif fill == 'small':
        fill_value = mean * 1e-2
    else:
        fill_value = fill_scale * torch.sqrt(mean.clamp_min(0))
    W = torch.where(W == 0, fill_value, W)
    H = torch.where(H == 0, fill_value, H)

    return W, H


def _batched_spd_solve(M, g, jitter_rel=1e-9):
    """Solve M[b] d[b] = g[b] for a batch of small SPD matrices.

    Tikhonov damping scaled to each matrix keeps the factorization well posed when
    Z is near zero (heavily attenuated pixels), and the fallback is branch-free so
    no host synchronization is introduced inside the iteration.
    """
    rank = M.shape[-1]
    eye = torch.eye(rank, dtype=M.dtype, device=M.device)
    diag = torch.diagonal(M, dim1=-2, dim2=-1)
    lam = jitter_rel * diag.amax(-1).clamp_min(torch.finfo(M.dtype).tiny)
    A = M + lam[:, None, None] * eye
    L, info = torch.linalg.cholesky_ex(A)
    failed = (info > 0)[:, None, None]
    L = torch.where(failed, eye.expand_as(L), L)
    d = torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
    diag_A = torch.diagonal(A, dim1=-2, dim2=-1).clamp_min(torch.finfo(M.dtype).tiny)
    d = torch.where(failed[:, :, 0], g / diag_A, d)
    # A row carrying no curvature at all -- Z underflowed to zero across the whole
    # row, which happens in float32 once the attenuation exceeds ~88 -- leaves the
    # damping underflowed too, so the fallback divides by `tiny` and overflows.
    # There is no second-order information to act on there, so take no step and
    # let the gradient-driven steps of later iterations bring the iterate back
    # into range.
    return torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def _joint_dot(a, b, c, d):
    return (a * b).sum() + (c * d).sum()


def _joint_blocks(flat, rows, cols, rank, free, jitter):
    """(B,Q) upper triangle -> damped SPD (B,R,R) with frozen vars set to identity."""
    M = flat.new_zeros(flat.shape[0], rank, rank)
    M[:, rows, cols] = flat
    M[:, cols, rows] = flat
    eye = torch.eye(rank, dtype=M.dtype, device=M.device)
    M = torch.where(free[:, :, None] & free[:, None, :], M, eye.expand_as(M))
    scale = torch.diagonal(M, dim1=-2, dim2=-1).amax(-1).clamp_min(torch.finfo(M.dtype).tiny)
    M = M + (jitter * scale)[:, None, None] * eye
    L, info = torch.linalg.cholesky_ex(M)
    return torch.where((info > 0)[:, None, None], eye.expand_as(L), L)

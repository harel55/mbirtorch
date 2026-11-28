"""Tests of mbirtorch.hsnt.strain.eigenstrain: the elastic strain of an eigenstrain on the voxel grid."""


def _box(n=12):
    import torch
    w = torch.zeros(n, n, n, dtype=torch.float64)
    w[2:10, 3:9, 2:11] = 1.0
    return w


def test_uniform_eigenstrain_is_stress_free():
    """A uniform eigenstrain over a free body only expands it: the elastic strain is zero."""
    import torch
    from mbirtorch.hsnt.strain.eigenstrain import EigenstrainMap

    w = _box()
    M = EigenstrainMap(w, 0.29)
    es = torch.zeros(*w.shape, 6, dtype=torch.float64)
    es[..., 0], es[..., 3] = 1e-3, -4e-4
    assert float(M.apply(es * (w > 0)[..., None]).abs().max()) < 1e-9


def test_adjoint_matches_apply():
    """adjoint is the transpose of apply (to the solver's tolerance), and CoarseBasis.adjoint of CoarseBasis.apply."""
    import torch
    from mbirtorch.hsnt.strain.eigenstrain import EigenstrainMap, CoarseBasis

    w = _box()
    M = EigenstrainMap(w, 0.3)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(*w.shape, 6, generator=g, dtype=torch.float64) * (w > 0)[..., None]
    y = torch.randn(*w.shape, 6, generator=g, dtype=torch.float64) * (w > 0)[..., None]
    a, b = float((M.apply(x) * y).sum()), float((x * M.adjoint(y)).sum())
    assert abs(a - b) < 1e-6 * abs(a)
    B = CoarseBasis(w.shape[0], 3)
    c = torch.randn(B.shape(), generator=g, dtype=torch.float64)
    z = torch.randn(*w.shape, 6, generator=g, dtype=torch.float64)
    assert abs(float((B.apply(c) * z).sum()) - float((c * B.adjoint(z)).sum())) < 1e-9


def test_inclusion_stress_balances():
    """A dilatational eigenstrain in a central inclusion: the inclusion is in compression and the surrounding shell
    carries the balancing tension, so the volume-weighted mean stress over the free body is near zero."""
    import torch
    from mbirtorch.hsnt.strain.eigenstrain import EigenstrainMap
    from mbirtorch.hsnt.strain.tomo import TomoFit

    n = 14
    w = torch.zeros(n, n, n, dtype=torch.float64); w[2:12, 2:12, 2:12] = 1.0
    es = torch.zeros(n, n, n, 6, dtype=torch.float64); es[5:9, 5:9, 5:9, :3] = 1e-3
    M = EigenstrainMap(w, 0.3)
    e = M.apply(es)
    s = torch.stack(TomoFit._stress(e, 0.3), -1)
    tr_in = float(s[5:9, 5:9, 5:9, :3].sum(-1).mean())
    mean = s[w > 0].mean(0)
    assert tr_in < -1e-4
    assert float(mean.abs().max()) < 0.02 * abs(tr_in)

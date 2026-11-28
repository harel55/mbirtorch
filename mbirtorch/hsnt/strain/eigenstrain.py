"""Elastic strain from an eigenstrain, for strain tomography with the physics built in.

A residual-stress field is the elastic response of a body to an eigenstrain (an incompatible, stress-free strain such
as thermal or plastic strain): the body deforms by a displacement u, its total strain is sym(grad u), and its elastic
strain is the total strain less the eigenstrain. The elastic strain is in equilibrium and free of traction on the free
surfaces by construction, and every such field is the response to some eigenstrain. Fitting an eigenstrain on a coarse
basis therefore fits only physically admissible strain fields, with far fewer unknowns than a strain per voxel.

EigenstrainMap solves the elastic problem on the voxel grid with finite elements: one trilinear hexahedral element per
voxel, 2 x 2 x 2 Gauss points, the element's stiffness weighted by the voxel's volume fraction (an empty voxel carries
none, so the sample's surface is free), displacements at the voxel corners. It gives the linear map S from an eigenstrain
per voxel to the elastic strain per voxel (the element's mean) and its adjoint, both by conjugate gradients on the
stiffness matrix, which is never formed. Coarse bases (CoarseBasis) map fewer coefficients to the per-voxel eigenstrain.

Arrays are on the tomography's volume axes (y, x, z), and strains are in COMPONENTS order xx, yy, zz, xy, yz, xz with
tensor shears.
"""
import math

import torch

CG_ITERS = 2000            # conjugate-gradient iterations of each elastic solve, at most
CG_TOL = 1e-8              # ... until the residual norm falls below this fraction of the right-hand side's
STIFFNESS_RIDGE = 1e-9     # a ridge on every displacement, relative to the typical diagonal stiffness: it fixes the
                           # displacement of corners that touch no material, and it removes the free body's rigid motions
GAUSS = 0.5 / math.sqrt(3.0)   # the 2-point Gauss rule's offsets from the element centre, in voxel units
AXIS_OF = (1, 0, 2)        # the volume axis along x, y, z


def _corners(u):
    """The eight corner values of every element from corner-node values u (n+1, n+1, n+1, ...): a dict keyed by the
    corner's offsets (a, b, c) along the volume axes."""
    n = u.shape[0] - 1
    return {(a, b, c): u[a:a + n, b:b + n, c:c + n] for a in (0, 1) for b in (0, 1) for c in (0, 1)}


def _gauss_points():
    """The 8 Gauss points of an element as offsets (p, q, r) in [0, 1] along the volume axes."""
    g = (0.5 - GAUSS, 0.5 + GAUSS)
    return [(p, q, r) for p in g for q in g for r in g]


def element_strains(u):
    """Total strain sym(grad u) at each element's 8 Gauss points, (8, n, n, n, 6), from corner displacements u
    (n+1, n+1, n+1, 3) in (x, y, z) components."""
    cn = _corners(u)
    out = []
    for p, q, r in _gauss_points():
        w = {(a, b, c): (p if a else 1 - p) * (q if b else 1 - q) * (r if c else 1 - r) for (a, b, c) in cn}
        grad = []                                     # du / d(volume axis 0, 1, 2), each (n, n, n, 3)
        for ax in range(3):
            g = 0.0
            for key, val in cn.items():
                if key[ax] == 0:
                    continue
                low = list(key); low[ax] = 0; low = tuple(low)
                wt = w[key] / (p, q, r)[ax] if (p, q, r)[ax] else 0.0       # the weight without the axis's own factor
                g = g + wt * (val - cn[low])
            grad.append(g)
        dx, dy, dz = grad[AXIS_OF[0]], grad[AXIS_OF[1]], grad[AXIS_OF[2]]   # d/dx, d/dy, d/dz of (ux, uy, uz)
        out.append(torch.stack([dx[..., 0], dy[..., 1], dz[..., 2], 0.5 * (dy[..., 0] + dx[..., 1]),
                                0.5 * (dz[..., 1] + dy[..., 2]), 0.5 * (dz[..., 0] + dx[..., 2])], -1))
    return torch.stack(out)


class EigenstrainMap:
    """The linear map from an eigenstrain per voxel (n, n, n, 6) to the elastic strain per voxel (n, n, n, 6) of a
    body with stiffness weights w (n, n, n) (the volume fraction, in [0, 1]) and Lame constants over Young's modulus
    lam, mu (scalars or (n, n, n)), free of traction where it meets empty voxels. Use apply for the map and adjoint
    for its transpose. Each call solves the elastic problem once by conjugate gradients, from the last solution of the
    same kind (warm start)."""

    def __init__(self, w, nu, dtype=torch.float64):
        self.w = w.to(dtype).clamp(0, 1)
        self.lam = nu / ((1 + nu) * (1 - 2 * nu)); self.mu = 1 / (2 * (1 + nu))
        n = w.shape[0]; self.n = n; self.dtype = dtype; self.device = w.device
        ones = torch.zeros(n + 1, n + 1, n + 1, device=w.device, dtype=dtype)   # the material around each corner
        for (a, b, c), _ in _corners(ones).items():
            ones[a:a + n, b:b + n, c:c + n] += self.w
        self.diag = ((self.lam + 2 * self.mu) * ones / 8 + STIFFNESS_RIDGE * float(self.lam + 2 * self.mu))[..., None]
        self._warm = {}
        self.last_iters = {}

    def _energy(self, u, es):
        """The strain energy (1/2) sum w (eps - es) : C : (eps - es) over the Gauss points (each 1/8 of the voxel)."""
        e = element_strains(u) - es[None]
        tr = e[..., 0] + e[..., 1] + e[..., 2]
        dens = self.mu * ((e[..., :3] ** 2).sum(-1) + 2 * (e[..., 3:] ** 2).sum(-1)) + 0.5 * self.lam * tr * tr
        return 0.5 * (self.w[None] * dens).sum() / 8

    def _K(self, u):
        """The stiffness matrix times u (plus the ridge)."""
        zero = torch.zeros(self.n, self.n, self.n, 6, device=u.device, dtype=u.dtype)
        with torch.enable_grad():
            x = u.detach().requires_grad_(True)
            g = torch.autograd.grad(self._energy(x, zero), x)[0]
        return g + STIFFNESS_RIDGE * float(self.lam + 2 * self.mu) * u

    def _force(self, es):
        """The nodal forces of eigenstrain es: minus the energy's gradient at u = 0 (linear in es)."""
        with torch.enable_grad():
            x = torch.zeros(self.n + 1, self.n + 1, self.n + 1, 3, device=es.device, dtype=self.dtype, requires_grad=True)
            return -torch.autograd.grad(self._energy(x, es.to(self.dtype)), x, create_graph=True)[0]

    def _linear_T(self, fn, v, shape):
        """The transpose of the linear function fn applied to v, by a vector-Jacobian product."""
        with torch.enable_grad():
            x0 = torch.zeros(shape, device=v.device, dtype=self.dtype, requires_grad=True)
            return torch.autograd.grad(fn(x0), x0, grad_outputs=v.to(self.dtype))[0]

    def _mean_strain(self, u):
        """Each element's mean total strain, (n, n, n, 6)."""
        return element_strains(u).mean(0)

    def solve(self, f, key):
        """u with K u = f, by Jacobi-preconditioned conjugate gradients from the last solution under key."""
        u = self._warm.get(key)
        u = torch.zeros_like(f) if u is None else u.clone()
        r = f - self._K(u); z = r / self.diag; d = z.clone()
        rz = float((r * z).sum()); f2 = float((f * f).sum())
        it = 0
        for it in range(CG_ITERS):
            if not rz > 0 or float((r * r).sum()) <= CG_TOL ** 2 * f2:
                break
            q = self._K(d); a = rz / float((d * q).sum())
            u = u + a * d; r = r - a * q; z = r / self.diag
            rz_new = float((r * z).sum()); d = z + (rz_new / rz) * d; rz = rz_new
        self._warm[key] = u.detach()
        self.last_iters[key] = it
        return u

    def apply(self, es):
        """The elastic strain per voxel (n, n, n, 6) of the eigenstrain es (n, n, n, 6)."""
        es = es.to(self.dtype)
        u = self.solve(self._force(es).detach(), "apply")
        return (self._mean_strain(u) - es) * (self.w > 0)[..., None]

    def adjoint(self, g):
        """The transpose of apply at g (n, n, n, 6)."""
        g = (g.to(self.dtype) * (self.w > 0)[..., None])
        a = self._linear_T(self._mean_strain, g, (self.n + 1, self.n + 1, self.n + 1, 3))   # A^T g (nodal)
        v = self.solve(a, 'adjoint')                                                         # K^-1 A^T g (K symmetric)
        return self._linear_T(self._force, v, (self.n, self.n, self.n, 6)) - g


class CoarseBasis:
    """An eigenstrain on a grid `spacing` voxels apart, interpolated trilinearly to every voxel: coefficients
    (m, m, m, 6) with m = ceil((n - 1) / spacing) + 1. spacing 1 is the identity."""

    def __init__(self, n, spacing, device='cpu', dtype=torch.float64):
        self.n, self.s = n, int(spacing)
        self.m = n if self.s == 1 else int(math.ceil((n - 1) / self.s)) + 1
        self.device, self.dtype = device, dtype

    def shape(self):
        return (self.m, self.m, self.m, 6)

    def apply(self, c):
        if self.s == 1:
            return c
        span = (self.m - 1) * self.s                                # the coarse grid covers voxels 0 .. span
        x = c.permute(3, 0, 1, 2)[None]
        up = torch.nn.functional.interpolate(x, size=(span + 1,) * 3, mode='trilinear', align_corners=True)
        return up[0, :, :self.n, :self.n, :self.n].permute(1, 2, 3, 0)

    def adjoint(self, g):
        if self.s == 1:
            return g
        x0 = torch.zeros(self.shape(), device=g.device, dtype=g.dtype)
        _, vjp = torch.func.vjp(self.apply, x0)
        return vjp(g)[0]

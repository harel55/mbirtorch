"""Tomographic hybrid fit: the hybrid model (model.py) with the per-ray maps replaced by
projections of voxel volumes, so the views are coupled.

Unknowns: per material r, a density volume V_r >= 0 and a strain volume E_r; the global spectra, kernel and
contamination parameters of hybrid2. A ray's projected density is W = A V and its projected strain U = A (V E), with
A the package's parallel-beam projector; the ray's attenuation is sum_r W_r mu_r(lam; U_r / W_r), the spectrum dilated
by the density-weighted mean strain along the ray (exact to first order in the strain spread along the ray).

Steps: the volume step is a Gauss-Newton step on (V, E) through the projector, solved by preconditioned conjugate
gradients with V's active bound held, then a projected backtracking line search on the deviance. The global and
contamination steps are those of hybrid2.FitB on the rays. A weak Gaussian prior on the strain (standard deviation
STRAIN_PRIOR_SD) fixes E in voxels without material, and a quadratic smoothness prior on the strain differences between
the six neighbours of each voxel (standard deviation STRAIN_SMOOTH_SD) pools the strain over nearby voxels, as MBIR
does for densities.

Warm-up (the pipeline's, with the lattice types given): the package's bilinear factorization of the transmission
(hsnt.dehydrate), warm_start on its spectra for the given lattice types, least-squares initial spectra per material,
the dehydrated maps mapped onto them and refitted per ray to those spectra, and a least-squares tomographic reconstruction of those maps (conjugate gradients,
clipped to >= 0). Lattice parameters are fixed at their stress-free values (the d0 reference of strain imaging)."""
import math
import os
import sys
import time

import numpy as np
import torch

from .model import Hybrid2, FitB, ls_init_material
from .bragg import warm_start, BraggSpectrum
from ._dist import all_sum, all_sum_float, broadcast_objects, rank

STRAIN_PRIOR_SD = 1e-2          # standard deviation of the weak strain prior
STRAIN_SMOOTH_SD = 3e-4         # standard deviation of the strain difference between neighbouring voxels
JOINT_CG_ITERS = 60             # conjugate-gradient iterations of the joint (volumes + global) step
MASK_FLOOR_REL = 0.1            # the material mask ('median'): the typical density is the median over the voxels whose
                                # density exceeds this fraction of the maximum
MASK_MEDIAN_REL = 0.5           # ... and the mask is the voxels whose density exceeds this fraction of it
OTSU_BINS = 256                 # histogram bins of the 'otsu' material mask
VOLUME_FRACTION_MAX = 1.0       # with fraction=True, a voxel holds at most all of a material (a physical bound)
SUPPORT_REL = 0.1               # the support: voxels whose 3 x 3 x 3 mean density exceeds this fraction of the maximum
EQUILIBRIUM_SD = 1e-5           # the equilibrium prior's standard deviation of div(stress) / Young's modulus per voxel
QGGMRF_P, QGGMRF_Q, QGGMRF_T = 2.0, 1.2, 1.0   # the edge-preserving density prior's exponents and threshold (the
#                                              package's q-GGMRF defaults)
DENSITY_SMOOTH_REL = 0.2        # the density smoothness prior's standard deviation per neighbour difference, as a
                                # fraction of the material's typical density inside the sample (warm-up volume)
CG_ITERS = 30                   # conjugate-gradient iterations per volume step
INIT_CG_ITERS = 40              # conjugate-gradient iterations of the warm-up reconstruction
EPS_W_MIN = 1e-8                # rays with less projected density carry no strain
INIT_PIXEL_STEPS = 3            # per-ray map fits in the warm-up
MOUNT_FD = (0.05, 0.05, 0.1)    # finite-difference steps for the mount derivatives: degrees, voxels, detector channels
MOUNT_MAX_STEP = (0.5, 0.5, 0.5) # largest change per mount step: degrees, voxels, detector channels
ANCHOR_STIFFNESS = 1e3          # the spectrum scale pin's weight, relative to the largest global Fisher diagonal entry
TEXTURE_AXIS_START = (0.6, 0.4)  # texture3d: the axis's starting polar and azimuthal angles (generic, off the axes)


STACK_GAP = 2                    # empty slices between stacked volumes, so a fractional row offset cannot mix them


def detector_directions(angles, n=16):
    """Per view, the detector channel displacement of a unit shift of the volume along its rows and along its columns:
    (d_rows, d_cols), each (views,). Measured from the package projector with probe voxels, as beam_directions."""
    from .geometry import beam_directions
    nd = beam_directions(np.asarray(angles), n)                 # (views, 2): n = (-t_y, t_x), t the detector direction
    return -nd[:, 0], nd[:, 1]


class Projector:
    """Parallel-beam projection of Q stacked volumes (n, n, n, Q) to rays (P, Q), P = views * n * n ordered (view,
    detector row, channel), and its adjoint. One package model projects all Q volumes at once (stacked slices).
    view_offsets (views,) adds a channel offset per view and row_offset a detector row offset: together they shift the
    object by a fraction of a voxel without resampling it (MultiAxisParallelModel at zero elevation, one model per view
    when the view offsets differ)."""

    def __init__(self, n, angles, Q, channel_offset=0.0, ror_mask=True, view_offsets=None, row_offset=0.0):
        """channel_offset: the projector's det_channel_offset (the rotation axis's offset from the detector centre);
        ror_mask=False lets the volume fill its square, for objects that reach past the inscribed circle."""
        import mbirtorch
        self.n, self.V, self.Q = n, len(angles), Q
        self.angles, self.channel_offset, self.ror_mask = np.asarray(angles), float(channel_offset), ror_mask
        self.view_offsets = None if view_offsets is None or not np.any(view_offsets) else np.asarray(view_offsets, float)
        self.row_offset = float(row_offset)
        self.kind = self._kind(self.view_offsets, self.row_offset)
        self.gap = 0 if self.kind == 'plain' else STACK_GAP
        h = n + self.gap
        if self.kind == 'plain':
            self.models = [mbirtorch.ParallelBeamModel((self.V, Q * n, n), angles)]
        elif self.kind == 'rows':
            self.models = [mbirtorch.MultiAxisParallelModel((self.V, Q * h, n), np.stack([angles, np.zeros(self.V)], 1))]
        else:
            self.models = [mbirtorch.MultiAxisParallelModel((1, Q * h, n), np.array([[a, 0.0]])) for a in angles]
        self._set_offsets()

    @staticmethod
    def _kind(view_offsets, row_offset):
        """'plain' (one parallel-beam model), 'rows' (one multi-axis model: a row offset) or 'views' (a model per view)."""
        return 'views' if view_offsets is not None else ('rows' if row_offset else 'plain')

    def _set_offsets(self):
        if self.kind == 'plain':
            self.models[0].set_params(det_channel_offset=self.channel_offset, use_ror_mask=bool(self.ror_mask))
            return
        offs = self.channel_offset + (self.view_offsets if self.view_offsets is not None else np.zeros(self.V))
        if self.kind == 'rows':
            self.models[0].set_params(det_channel_offset=self.channel_offset, det_row_offset=self.row_offset,
                                      use_ror_mask=bool(self.ror_mask))
        else:
            for m, o in zip(self.models, offs):
                m.set_params(det_channel_offset=float(o), det_row_offset=self.row_offset, use_ror_mask=bool(self.ror_mask))

    def with_q(self, Q):
        """The same geometry for Q stacked volumes."""
        return self if Q == self.Q else Projector(self.n, self.angles, Q, self.channel_offset, self.ror_mask,
                                                  self.view_offsets, self.row_offset)

    def set_geometry(self, channel_offset, view_offsets=None, row_offset=0.0):
        """Change the offsets: in place when the kind of model stays the same, else by rebuilding. Returns self."""
        vo = None if view_offsets is None or not np.any(view_offsets) else np.asarray(view_offsets, float)
        if self._kind(vo, row_offset) != self.kind:
            self.__init__(self.n, self.angles, self.Q, channel_offset, self.ror_mask, vo, row_offset)
            return self
        self.channel_offset, self.view_offsets, self.row_offset = float(channel_offset), vo, float(row_offset)
        self._set_offsets()
        return self

    def _stack(self, vols):
        n, Q, g = self.n, self.Q, self.gap
        parts = []
        for q in range(Q):
            parts.append(vols[..., q])
            if g:
                parts.append(torch.zeros(n, n, g, device=vols.device, dtype=vols.dtype))
        return torch.cat(parts, 2).contiguous()

    def fwd(self, vols):
        n, Q, h = self.n, self.Q, self.n + self.gap
        stack = self._stack(vols)
        if len(self.models) == 1:
            s = self.models[0].forward_project(stack, output_sharded=True)          # (views, Q h, n)
        else:
            s = torch.cat([m.forward_project(stack, output_sharded=True) for m in self.models], 0)
        s = s.reshape(self.V, Q, h, n)[:, :, :n]
        return s.permute(1, 0, 2, 3).reshape(Q, -1).T.contiguous()

    def back(self, rays):
        n, Q, h = self.n, self.Q, self.n + self.gap
        s = rays.T.reshape(Q, self.V, n, n).permute(1, 0, 2, 3)
        if self.gap:
            s = torch.cat([s, torch.zeros(self.V, Q, self.gap, n, device=s.device, dtype=s.dtype)], 2)
        s = s.reshape(self.V, Q * h, n).contiguous()
        if len(self.models) == 1:
            b = self.models[0].back_project(s, output_sharded=True)                  # (n, n, Q h)
        else:
            b = sum(m.back_project(s[v:v + 1].contiguous(), output_sharded=True) for v, m in enumerate(self.models))
        b = b.reshape(n, n, Q, h)[..., :n]
        # with the rays split over processes, each holds some views: the back-projection is the sum over processes
        return all_sum(b.permute(0, 1, 3, 2).contiguous())


class MultiProjector:
    """Several rotation series of one sample: each series projects the sample volume permuted by its axis permutation
    (beam_geometry.PERMS) with its own angles. fwd stacks the series' rays (series-major); back sums the series'
    back-projections, permuted back to the sample frame. One series with the identity permutation is a Projector."""

    def __init__(self, n, series, Q, channel_offsets=None, ror_mask=True):
        """series: a list of (axis, angles), axis a key of beam_geometry.PERMS, or (transform, angles) with transform =
        (perm, signs, shift[, rot]): the series' volume is the sample volume permuted (frame axis i = sample axis
        perm[i]), flipped where signs < 0, rotated by rot (degrees about the frame's axes 0, 1, 2, as a rotation vector,
        about the grid's centre, by trilinear interpolation) and shifted by shift voxels (a measured mount). The whole
        part of the shift rolls the volume (wrapping around); the fractional part moves the detector instead (a
        channel offset per view and a row offset), so the volume is not resampled. The beam directions
        (beam_geometry.series_weights) do not include rot."""
        from .geometry import PERMS
        self.n, self.Q, self.series = n, Q, list(series)
        offs = channel_offsets or [0.0] * len(self.series)
        self.offsets, self.ror_mask = [float(o) for o in offs], ror_mask
        self.parts, self._dirs = [], []
        for (ax, ang), off in zip(self.series, self.offsets):
            tr = (PERMS[ax], (1, 1, 1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) if isinstance(ax, str) else \
                (tuple(int(v) for v in ax[0]), tuple(int(v) for v in ax[1]), tuple(float(v) for v in ax[2]),
                 tuple(float(v) for v in ax[3]) if len(ax) > 3 else (0.0, 0.0, 0.0))
            self._dirs.append(None)
            vo, ro = self._detector_shift(len(self.parts), ang, tr[2])
            self.parts.append((tr, Projector(n, ang, Q, off, ror_mask, vo, ro)))
        self.counts = [p.V * n * n for _, p in self.parts]
        self.V = sum(p.V for _, p in self.parts)

    def with_q(self, Q):
        return self if Q == self.Q else MultiProjector(self.n, self.series, Q, self.offsets, self.ror_mask)

    @staticmethod
    def _split(shift):
        """The whole part (rolled) and the fractional part (moved to the detector) of a shift."""
        whole = tuple(int(round(v)) for v in shift)
        return whole, tuple(v - w for v, w in zip(shift, whole))

    def _detector_shift(self, s, angles, shift):
        """Per-view channel offsets and the row offset that move series s's projections as a fractional shift of its
        volume (rows, cols, slices) would."""
        _, frac = self._split(shift)
        if not any(frac[:2]) and not frac[2]:
            return None, 0.0
        if self._dirs[s] is None:
            self._dirs[s] = detector_directions(angles)
        d_rows, d_cols = self._dirs[s]
        return frac[0] * d_rows + frac[1] * d_cols, frac[2]

    @staticmethod
    def _warp(v, shift, rot):
        """v (n, n, n, Q) -> its rotation by rot (degrees) about the grid's centre, then shift, by trilinear interpolation."""
        n = v.shape[0]; c = (n - 1) / 2
        w = torch.tensor(np.radians(rot), dtype=torch.float64)
        th = float(w.norm()); k = w / max(th, 1e-30)
        K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=torch.float64)
        R = (torch.eye(3, dtype=torch.float64) + math.sin(th) * K + (1 - math.cos(th)) * K @ K).to(v.device, torch.float32)
        ax = torch.arange(n, device=v.device, dtype=torch.float32) - c
        P = torch.stack(torch.meshgrid(ax, ax, ax, indexing='ij'), -1)
        q = P @ R.T + c - torch.tensor(shift, device=v.device, dtype=torch.float32)
        g = (2 * q / (n - 1) - 1).flip(-1)
        return torch.nn.functional.grid_sample(v.permute(3, 0, 1, 2)[None], g[None], mode='bilinear', padding_mode='zeros',
                                               align_corners=True)[0].permute(1, 2, 3, 0)

    @staticmethod
    def _to_series(vols, tr):
        """The series' volume: permuted, flipped, rotated, and rolled by the whole part of the shift (the fractional
        part is applied by the series' projector)."""
        perm, signs, shift, rot = tr
        v = vols.permute(*perm, 3)
        flips = [i for i, sg in enumerate(signs) if sg < 0]
        if flips:
            v = v.flip(flips)
        if any(rot):
            v = MultiProjector._warp(v.contiguous(), (0.0, 0.0, 0.0), rot)
        whole, _ = MultiProjector._split(shift)
        if any(whole):
            v = torch.roll(v, whole, (0, 1, 2))
        return v.contiguous()

    @staticmethod
    def _from_series(b, tr):
        perm, signs, shift, rot = tr
        whole, _ = MultiProjector._split(shift)
        if any(whole):
            b = torch.roll(b, tuple(-w for w in whole), (0, 1, 2))
        if any(rot):                                    # the adjoint of the interpolation, by automatic differentiation
            z = torch.zeros_like(b).requires_grad_(True)
            with torch.enable_grad():
                b = torch.autograd.grad(MultiProjector._warp(z, (0.0, 0.0, 0.0), rot), z, b)[0]
        flips = [i for i, sg in enumerate(signs) if sg < 0]
        if flips:
            b = b.flip(flips)
        inv = tuple(int(i) for i in np.argsort(perm))
        return b.permute(*inv, 3)

    def set_mount(self, s, shift=None, rot=None, offset=None):
        """Change series s's mount (shift in voxels, rot in degrees, as in __init__) and its centre-of-rotation offset."""
        (perm, signs, sh, ro), p = self.parts[s]
        tr = (perm, signs, tuple(float(v) for v in (sh if shift is None else shift)),
              tuple(float(v) for v in (ro if rot is None else rot)))
        if offset is not None:
            self.offsets[s] = float(offset)
        vo, row = self._detector_shift(s, p.angles, tr[2])
        p.set_geometry(self.offsets[s], vo, row)
        self.parts[s] = (tr, p)
        ax, ang = self.series[s]
        self.series[s] = ((perm, signs, tr[2], tr[3]), ang)

    def project_series(self, s, vols):
        """Series s's rays for the sample volumes."""
        tr, p = self.parts[s]
        return p.fwd(self._to_series(vols, tr))

    def fwd(self, vols):
        return torch.cat([p.fwd(self._to_series(vols, tr)) for tr, p in self.parts], 0)

    def back(self, rays):
        out = None
        for (tr, p), r in zip(self.parts, torch.split(rays, self.counts, 0)):
            b = self._from_series(p.back(r.contiguous()), tr)                # in the sample frame
            out = b if out is None else out + b
        return out.contiguous()


class FineProjector:
    """A MultiProjector on a volume grid `factor` times finer than the detector: the volume has side factor * n, it is
    projected onto a detector factor times finer, and each detector pixel averages its factor^2 fine pixels, as a
    detector integrates over its pixel area (the phantom generator makes its data the same way). Use it so faces and
    interfaces can sit between the detector's pixel positions without forcing ringing into the densities; it costs
    factor^3 the voxels. Lengths are in units of the detector pixel, as for a MultiProjector of side n. Mount shifts
    (voxels of n) and channel offsets (detector pixels) are given as for a MultiProjector of side n, also to
    set_mount and in parts, so TomoFit.mount_step works on it."""

    def __init__(self, n, series, Q, channel_offsets=None, ror_mask=True, factor=2):
        self.n_det, self.factor, self.Q = n, int(factor), Q
        self.series, self.ror_mask = list(series), ror_mask
        self.offsets = [float(o) for o in (channel_offsets or [0.0] * len(self.series))]
        f = self.factor
        fine_series = []
        for ax, ang in self.series:
            if isinstance(ax, str):
                fine_series.append((ax, ang))
            else:
                tr = (ax[0], ax[1], tuple(float(v) * f for v in ax[2])) + ((tuple(ax[3]),) if len(ax) > 3 else ())
                fine_series.append((tr, ang))
        self.inner = MultiProjector(f * n, fine_series, Q, [o * f for o in self.offsets], ror_mask)
        self.n = f * n                                       # the volume's side
        self.V = self.inner.V
        self.rays_per_view = n * n
        self.counts = [c // f ** 2 for c in self.inner.counts]

    def with_q(self, Q):
        return self if Q == self.Q else FineProjector(self.n_det, self.series, Q, self.offsets, self.ror_mask, self.factor)

    @property
    def parts(self):
        """Each series' mount and projector, with the shifts in voxels of n, as a MultiProjector of side n gives them."""
        f = self.factor
        return [((tr[0], tr[1], tuple(v / f for v in tr[2]), tr[3]), p) for tr, p in self.inner.parts]

    def set_mount(self, s, shift=None, rot=None, offset=None):
        """Changes series s's mount (shift in voxels of n, rot in degrees) and its centre-of-rotation offset (detector
        pixels), as MultiProjector.set_mount does."""
        f = self.factor
        self.inner.set_mount(s, shift=None if shift is None else [float(v) * f for v in shift], rot=rot,
                             offset=None if offset is None else float(offset) * f)
        if offset is not None:
            self.offsets[s] = float(offset)
        (perm, signs, sh, ro), _ = self.parts[s]
        self.series[s] = ((perm, signs, sh, ro), self.series[s][1])

    def project_series(self, s, vols):
        """Returns series s's rays (detector pixels) for the sample volumes on the fine grid."""
        n, f = self.n_det, self.factor
        rays = self.inner.project_series(s, vols)
        return rays.reshape(-1, n, f, n, f, rays.shape[-1]).mean((2, 4)).reshape(-1, rays.shape[-1]) / f

    def fwd(self, vols):
        n, f = self.n_det, self.factor
        rays = self.inner.fwd(vols)                                                   # (views N N, Q)
        return rays.reshape(self.V, n, f, n, f, -1).mean((2, 4)).reshape(self.V * n * n, -1) / f

    def back(self, rays):
        n, f = self.n_det, self.factor
        r = rays.reshape(self.V, n, 1, n, 1, -1).expand(self.V, n, f, n, f, rays.shape[-1]) / f ** 3
        return self.inner.back(r.reshape(self.V * f * n * f * n, -1).contiguous())


def _otsu_threshold(x):
    """Otsu's threshold of the values x: the cut of their histogram (OTSU_BINS bins) that maximizes the variance
    between the two classes."""
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return lo
    h = torch.histc(x.double(), bins=OTSU_BINS, min=lo, max=hi)
    c = lo + (torch.arange(OTSU_BINS, dtype=torch.float64, device=x.device) + 0.5) * (hi - lo) / OTSU_BINS
    w0 = torch.cumsum(h, 0); w1 = w0[-1] - w0
    m0 = torch.cumsum(h * c, 0) / w0.clamp_min(1e-30); m1 = ((h * c).sum() - torch.cumsum(h * c, 0)) / w1.clamp_min(1e-30)
    k = int(torch.argmax(w0 * w1 * (m0 - m1) ** 2))
    return lo + (k + 1) * (hi - lo) / OTSU_BINS


class TomoFit:
    """The volume part of the tomographic fit. Per material r: a density volume V (n, n, n, R) and strain volumes E
    (n, n, n, R, C), C = 1 (a scalar dilation, the same in every view) or 6 (the strain tensor, COMPONENTS order xx,
    yy, zz, xy, yz, xz with tensor shears). A view sees the weighted sum w_v . E: weights 1 for C = 1; for C = 6 the
    longitudinal ray transform n . eps . n of its beam direction n (beam_geometry.view_weights). A ray's strain is
    U / W with W = A V and U = sum_k w_k A(V E_k). Priors: a weak Gaussian on E (STRAIN_PRIOR_SD), a smoothness prior
    on E (smooth_sd per neighbour difference), optionally on V (density_smooth), and for C = 6 optionally 3D stress
    equilibrium (elastic: (Young's modulus, Poisson ratio) per material) on the voxels whose six neighbours hold the
    material, with standard deviation EQUILIBRIUM_SD of the divergence of stress / Young's modulus per voxel."""

    def __init__(self, fit, proj, V, E, weights=None, smooth_sd=STRAIN_SMOOTH_SD, density_smooth=False, elastic=None,
                 eq_sd=EQUILIBRIUM_SD, cg_iters=CG_ITERS, support=False, precond='jacobi', density_prior='quadratic',
                 voxel_scale=1.0, traction=False, mask_method='max', mask_freeze=False, fraction=False,
                 eq_stencil='corner'):
        """traction=True adds traction-free faces to the equilibrium prior: on the voxels at the sample's faces, the
        traction sigma . n / Young's modulus (n the outward normal) has the equilibrium prior's standard deviation.
        density_prior: 'quadratic' (a Gaussian prior on neighbour differences) or 'qggmrf' (edge-preserving: the
        same for small differences, much weaker for large jumps, so sharp interfaces are kept). voxel_scale: the voxel
        size relative to the grid the prior strengths are set for; the smoothness and equilibrium weights scale with it
        and the weak strain ridge with its cube, as discretized integrals over the volume, so the priors mean the same
        at any binning.
        mask_method: how the material's own voxels are found (for the equilibrium prior's interior and faces, the strain
        recentring and the density prior's scale): 'max' (density above half the maximum; a few overshooting voxels at
        the faces can leave it nearly empty), 'median' (density above MASK_MEDIAN_REL of the typical density), or 'otsu'
        (density above Otsu's threshold of its histogram). mask_freeze=True computes the mask once,
        from the starting volume, so the objective stays the same from one iteration to the next.
        fraction=True: V is each material's volume fraction, bounded by 0 and VOLUME_FRACTION_MAX (the material's full
        attenuation is carried by its spectrum; see warmup), so a voxel cannot hold more than all of a material; call
        normalize_fractions after each volume step. It removes the density's ringing at faces off the voxel grid, but
        the model error behind the ringing then goes into the strain at the faces, so it is off by default.
        eq_stencil: where the equilibrium prior evaluates div(stress): 'central' (at the voxels, central differences,
        which cannot see a strain that alternates from voxel to voxel) or 'corner' (the default; at the voxel corners, from the
        differences between the voxels on either side, averaged over the four pairs around the corner: it sees
        patterns that alternate along one axis; patterns alternating along two or three axes stay invisible, as the
        hourglass modes of one-point finite elements)."""
        self.fit = fit
        assert eq_stencil in ('central', 'corner'), eq_stencil
        self.eq_stencil = eq_stencil
        self.v_max = VOLUME_FRACTION_MAX if fraction else float('inf')
        assert mask_method in ('max', 'median', 'otsu'), mask_method
        self.mask_method, self.mask_freeze, self._mask = mask_method, bool(mask_freeze), None
        assert density_prior in ('quadratic', 'qggmrf'), density_prior
        self.density_prior, self.voxel_scale = density_prior, float(voxel_scale)
        # the volume solves' preconditioner: 'jacobi' (each unknown's diagonal), 'block' (per voxel and material, the
        # inverse of the (1 + C) x (1 + C) block coupling its density and strain components), or 'fft' (Jacobi on the
        # densities; for the strain, the exact inverse of a translation-invariant stand-in for its Hessian block, applied
        # by FFT: use it with the equilibrium prior, whose stiff coupling of neighbouring voxels slows the others)
        assert precond in ('jacobi', 'block', 'fft'), precond
        self.precond = precond
        # support=True: MONSTR's support prior in place of the weak strain ridge (strain held at 0 outside the sample)
        self.support = support
        self.R = fit.M.R
        self.V = V.clone()
        self.E = (E[..., None] if E.dim() == 4 else E).clone()
        self.C = self.E.shape[-1]
        n = V.shape[0]
        w = torch.ones(proj.V, 1, device=V.device) if weights is None else torch.as_tensor(weights, device=V.device, dtype=V.dtype)
        assert w.shape == (proj.V, self.C), (w.shape, proj.V, self.C)
        self.w_ray = w.repeat_interleave(getattr(proj, 'rays_per_view', n * n), 0)  # (P, C): rays are view-major
        self.proj = proj.with_q(self.R * (1 + self.C))
        vs = self.voxel_scale
        self.lam_E = 0.0 if support else vs ** 3 * 0.5 / STRAIN_PRIOR_SD ** 2    # negative log-likelihood scale
        self.smooth_E = 0.0 if smooth_sd is None else vs / smooth_sd ** 2
        # density smoothness, per material: 1 / sd^2 with sd = DENSITY_SMOOTH_REL x the median density over the
        # voxels above half the material's maximum
        self.smooth_V = torch.zeros(self.R, device=V.device, dtype=V.dtype)
        self.sigma_V = torch.full((self.R,), float('inf'), device=V.device, dtype=V.dtype)
        if density_smooth:
            own = self.material_voxels(V)
            for r in range(self.R):
                v = V[..., r]; typ = v[own[..., r]].median() if bool(own[..., r].any()) else torch.ones(())
                self.sigma_V[r] = float(DENSITY_SMOOTH_REL * typ)
                self.smooth_V[r] = vs / float(DENSITY_SMOOTH_REL * typ) ** 2
        self.elastic = elastic if (elastic is not None and self.C == 6) else None
        self.traction = bool(traction) and self.elastic is not None
        self.eq_w = vs / eq_sd ** 2
        self._wts = None
        # hold_strain=True keeps the strain where it is (the volume steps move the density only), e.g. for the first
        # iterations, so the spectrum and the density settle before the strain is fitted
        self.hold_strain = False
        self.cg_iters = cg_iters
        self.damp = 1e-3
        self.last_step_se = float('inf')     # the last volume step's RMS in standard errors (_step_se), for a stopping rule
        self.sync()

    # ---- projections
    def _fwd(self, V, VE):
        """W (P, R) and the view-weighted U (P, R) from a density-like and a (density x strain)-like volume."""
        n, R, C = V.shape[0], self.R, self.C
        out = self.proj.fwd(torch.cat([V, VE.reshape(n, n, n, R * C)], -1))
        return out[:, :R], (out[:, R:].reshape(-1, R, C) * self.w_ray[:, None, :]).sum(-1)

    def _back(self, gW, gU):
        """The adjoint of _fwd: volumes (n, n, n, R) and (n, n, n, R, C)."""
        P, R, C = gW.shape[0], self.R, self.C
        b = self.proj.back(torch.cat([gW, (gU[:, :, None] * self.w_ray[:, None, :]).reshape(P, R * C)], 1))
        return b[..., :R], b[..., R:].reshape(*b.shape[:3], R, C)

    def rays(self, V, E):
        W, U = self._fwd(V, V[..., None] * E)
        eps = torch.where(W > EPS_W_MIN, U / W.clamp_min(EPS_W_MIN), torch.zeros_like(U))
        return W.clamp_min(0), eps

    def sync(self):
        self.fit.W, self.fit.eps = self.rays(self.V, self.E)
        self._synced = (self.V, self.V._version, self.E, self.E._version, self.fit.W, self.fit.W._version,
                        self.fit.eps, self.fit.eps._version)

    def _is_synced(self):
        """True when the fit's projected maps are this volume's: nothing changed since the last sync."""
        s = getattr(self, "_synced", None)
        return s is not None and s[0] is self.V and s[1] == self.V._version and s[2] is self.E and s[3] == self.E._version \
            and s[4] is self.fit.W and s[5] == self.fit.W._version and s[6] is self.fit.eps and s[7] == self.fit.eps._version

    # ---- priors
    @staticmethod
    def laplacian(E):
        """Sum over the six neighbours of (E_i - E_j), with no neighbours past the volume's faces (first 3 axes)."""
        out = torch.zeros_like(E)
        for d in range(3):
            df = E.diff(dim=d)                                                      # E_(k+1) - E_k
            n = E.shape[d]
            out.narrow(d, 0, n - 1).sub_(df); out.narrow(d, 1, n - 1).add_(df)
        return out

    def _dens_wts(self, V):
        """Per-axis neighbour weights of the q-GGMRF density prior at V (its surrogate curvature rho'(d) / d per
        neighbour pair), or None for the quadratic prior."""
        if self.density_prior == 'quadratic' or not bool(torch.isfinite(self.sigma_V).any()):
            return None
        from mbirtorch.qggmrf import b_tilde_by_definition
        out = []
        for d in range(3):
            df = V.diff(dim=d)
            w = torch.stack([2 * b_tilde_by_definition(df[..., r], float(self.sigma_V[r]), QGGMRF_P, QGGMRF_Q, QGGMRF_T)
                             if torch.isfinite(self.sigma_V[r]) else torch.zeros_like(df[..., r]) for r in range(self.R)], -1)
            out.append(self.voxel_scale * w)
        return out

    def dens_lap(self, x, wts):
        """The density prior's curvature (or, at x = V, its gradient) applied to x: a Laplacian, weighted per pair."""
        if wts is None:
            return self.smooth_V * self.laplacian(x)
        out = torch.zeros_like(x)
        for d in range(3):
            df = x.diff(dim=d) * wts[d]; k = x.shape[d]
            out.narrow(d, 0, k - 1).sub_(df); out.narrow(d, 1, k - 1).add_(df)
        return out

    def dens_diag(self, wts):
        """The diagonal of dens_lap: per voxel, the sum of its pairs' weights."""
        if wts is None:
            return 6 * self.smooth_V
        out = torch.zeros(*wts[0].shape[:0], *self.V.shape, device=self.V.device, dtype=self.V.dtype)
        for d in range(3):
            k = self.V.shape[d]
            out.narrow(d, 0, k - 1).add_(wts[d]); out.narrow(d, 1, k - 1).add_(wts[d])
        return out

    def _dens_prior_value(self, V):
        if self.density_prior == 'quadratic':
            sq = lambda X: sum((d.double() ** 2).sum((0, 1, 2)) for d in (X.diff(dim=0), X.diff(dim=1), X.diff(dim=2)))
            return 0.5 * float((self.smooth_V.double() * sq(V)).sum())
        tot = 0.0
        for r in range(self.R):
            sx = float(self.sigma_V[r])
            if not np.isfinite(sx):
                continue
            for d in range(3):
                a = V[..., r].diff(dim=d).double().abs().clamp_min(QGGMRF_T * sx * 1e-7)
                ds = (a / (QGGMRF_T * sx)) ** (QGGMRF_Q - QGGMRF_P)
                tot += float((a ** QGGMRF_P / (QGGMRF_P * sx ** QGGMRF_P) * ds / (1 + ds)).sum())
        return self.voxel_scale * tot

    def material_voxels(self, V=None):
        """(n, n, n, R) bool: each material's own voxels, by mask_method (see __init__); with mask_freeze, the mask of
        the first call is kept."""
        method, freeze = getattr(self, 'mask_method', 'max'), getattr(self, 'mask_freeze', False)
        if freeze and getattr(self, '_mask', None) is not None:
            return self._mask
        V = self.V if V is None else V
        if method == 'max':
            m = V > 0.5 * V.amax((0, 1, 2), keepdim=True)
        else:
            m = torch.zeros_like(V, dtype=torch.bool)
            for r in range(V.shape[-1]):
                v = V[..., r]
                if float(v.max()) <= 0:
                    continue
                if method == 'median':
                    m[..., r] = v > MASK_MEDIAN_REL * v[v > MASK_FLOOR_REL * v.max()].median()
                else:
                    m[..., r] = v > _otsu_threshold(v[v > 0])
        if freeze:
            self._mask = m
        return m

    def _eq_mask(self):
        """Per material, where the equilibrium prior applies, (n, n, n, R): with the central stencil, the voxels whose
        six neighbours all hold the material; with the corner stencil, the corners (indexed by the voxel at their low
        side) whose eight surrounding voxels all hold it."""
        inside = self.material_voxels()
        self._eq_inside = inside                                                    # for the traction-free faces
        m = inside.clone()
        if getattr(self, 'eq_stencil', 'central') == 'corner':
            for shift in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)):
                m &= torch.roll(inside, tuple(-v for v in shift), (0, 1, 2))
            for d in range(3):
                m.narrow(d, m.shape[d] - 1, 1).zero_()
            return m
        for d in range(3):
            m &= torch.roll(inside, 1, d) & torch.roll(inside, -1, d)
            m.narrow(d, 0, 1).zero_(); m.narrow(d, m.shape[d] - 1, 1).zero_()
        return m                                                                    # (n, n, n, R)

    @staticmethod
    def _stress(e, nu):
        """Stress / Young's modulus from strain (isotropic elasticity), in COMPONENTS order xx, yy, zz, xy, yz, xz."""
        lam_, mu_ = nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))
        tr = e[..., 0] + e[..., 1] + e[..., 2]
        return [lam_ * tr + 2 * mu_ * e[..., k] for k in range(3)] + [2 * mu_ * e[..., k] for k in (3, 4, 5)]

    def _traction(self, E):
        """Traction / Young's modulus on the sample's free faces, (n, n, n, R, 3 normals, 3): for each volume axis, at a
        material's voxels with an empty neighbour along it (no material at all), the stress times that axis' unit
        normal. A face against another material is bonded, not free, so it carries no traction condition."""
        out = []
        inside = self._eq_inside
        occupied = inside.any(-1)
        for r, (Em, nu) in enumerate(self.elastic):
            sxx, syy, szz, sxy, syz, sxz = self._stress(E[..., r, :], nu)
            ins = inside[..., r]
            faces = []
            for ax, t in ((1, (sxx, sxy, sxz)), (0, (sxy, syy, syz)), (2, (sxz, syz, szz))):   # volume axes (y, x, z)
                face = ins & ~(torch.roll(occupied, 1, ax) & torch.roll(occupied, -1, ax))
                faces.append(torch.stack(t, -1) * face[..., None])
            out.append(torch.stack(faces, -2))
        return torch.stack(out, -3)

    def _eq_residual(self, E, mask):
        """The equilibrium prior's residual: div(sigma) / Young's modulus on the interior, and with traction=True the
        face tractions, flattened per voxel."""
        div = self._divergence(E, mask).reshape(*E.shape[:3], -1)
        if not self.traction:
            return div
        return torch.cat([div, self._traction(E).reshape(*E.shape[:3], -1)], -1)

    def _divergence(self, E, mask):
        """div(sigma) / Young's modulus on the mask, (n, n, n, R, 3), by eq_stencil; volume axes (y, x, z), E in
        COMPONENTS order."""
        out = []
        if getattr(self, 'eq_stencil', 'central') == 'corner':
            def D(a, ax):                       # at the corner on the high side of each voxel (see __init__)
                f = torch.roll(a, -1, ax) - a
                o1, o2 = (d for d in range(3) if d != ax)
                return 0.25 * (f + torch.roll(f, -1, o1) + torch.roll(f, -1, o2) + torch.roll(f, (-1, -1), (o1, o2)))
        else:
            D = lambda a, ax: 0.5 * (torch.roll(a, -1, ax) - torch.roll(a, 1, ax))
        for r, (Em, nu) in enumerate(self.elastic):
            lam_, mu_ = nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))       # Lame constants / Young's modulus
            e = E[..., r, :]; tr = e[..., 0] + e[..., 1] + e[..., 2]
            sxx, syy, szz = (lam_ * tr + 2 * mu_ * e[..., k] for k in range(3))
            sxy, syz, sxz = (2 * mu_ * e[..., k] for k in (3, 4, 5))
            dx, dy, dz = (lambda a: D(a, 1)), (lambda a: D(a, 0)), (lambda a: D(a, 2))
            f = torch.stack([dx(sxx) + dy(sxy) + dz(sxz), dx(sxy) + dy(syy) + dz(syz), dx(sxz) + dy(syz) + dz(szz)], -1)
            out.append(f * mask[..., r, None])
        return torch.stack(out, -2)

    def prior(self, V, E, mask=None):
        sq = lambda X: sum((d.double() ** 2).sum((0, 1, 2)) for d in (X.diff(dim=0), X.diff(dim=1), X.diff(dim=2)))
        tot = (self.lam_E * float((E.double() ** 2).sum()) + 0.5 * self.smooth_E * float(sq(E).sum())
               + self._dens_prior_value(V))
        if self.elastic is not None:
            mask = self._eq_mask() if mask is None else mask
            tot += 0.5 * self.eq_w * float((self._eq_residual(E, mask).double() ** 2).sum())
        return tot

    def data_loss(self, V=None, E=None):
        """The deviance of all rays (no priors) for the volumes V, E, the current ones by default (then the fit's
        remembered value when a pass has computed it at this state)."""
        if V is None and E is None and self._is_synced():
            return self.fit.data_loss()
        V = self.V if V is None else V; E = self.E if E is None else E
        W, eps = self.rays(V, E)
        cache = self.fit._state(self.fit.g); tot = 0.0
        with torch.no_grad():
            for sl in self.fit.chunks():
                tot += float(self.fit.ray_loss(sl, W[sl], eps[sl], cache).sum())
        return all_sum_float(tot, V.device)

    def loss(self, V=None, E=None, mask=None):
        return self.data_loss(V, E) + self.prior(self.V if V is None else V, self.E if E is None else E, mask)

    # ---- the Gauss-Newton step
    def volume_step(self):
        """One Gauss-Newton step on (V, E). Returns the step's Newton decrement per volume parameter (noise units)."""
        fit, R, C = self.fit, self.R, self.C
        self.sync()
        cache = fit._state(fit.g)
        P = fit.T.shape[0]
        G = torch.zeros(P, 2 * R, device=fit.T.device, dtype=torch.float64)
        F = torch.zeros(P, 2 * R, 2 * R, device=fit.T.device, dtype=torch.float64)
        devs = []
        with torch.no_grad():
            for sl in fit.chunks():
                G[sl], F[sl] = fit.ray_system(sl, cache, deviance=devs)
        fit.remember_data_loss(all_sum_float(sum(devs), fit.T.device))          # the deviance at this synced state
        G, F = G.float(), F.float()
        V, E = self.V, self.E
        mask = self._eq_mask() if self.elastic is not None else None
        eq = (lambda X: self.eq_w * torch.func.vjp(lambda Y: self._eq_residual(Y, mask), X)[1](self._eq_residual(X, mask))[0]) \
            if self.elastic is not None else (lambda X: torch.zeros_like(X))
        bGW, bGU = self._back(G[:, :R], G[:, R:])
        self._wts = wts = self._dens_wts(V)
        gV = bGW + (E * bGU).sum(-1) + self.dens_lap(V, wts)
        gE = V[..., None] * bGU + 2 * self.lam_E * E + self.smooth_E * self.laplacian(E) + eq(E)
        free = self._free_density(V, gV)                                            # V's active bounds held
        fE = self.support_mask().expand_as(E) & (not self.hold_strain)              # strain free inside the support
        # Jacobi preconditioner from the ray Fisher diagonals (for the strain parts, each view's weight squared)
        dWW = torch.diagonal(F[:, :R, :R], dim1=1, dim2=2); dUU = torch.diagonal(F[:, R:, R:], dim1=1, dim2=2)
        dWU = torch.diagonal(F[:, :R, R:], dim1=1, dim2=2)
        P_ = dWW.shape[0]
        bWW = self.proj.back(torch.cat([dWW, torch.zeros(P_, R * C, device=dWW.device)], 1))[..., :R]
        bUU = self.proj.back(torch.cat([torch.zeros_like(dWW), (dUU[:, :, None] * self.w_ray[:, None, :] ** 2).reshape(P_, R * C)], 1))[..., R:].reshape(*V.shape[:3], R, C)
        bWU = self.proj.back(torch.cat([torch.zeros_like(dWW), (dWU[:, :, None] * self.w_ray[:, None, :]).reshape(P_, R * C)], 1))[..., R:].reshape(*V.shape[:3], R, C)
        eq_diag = 12.0 * self.eq_w if self.elastic is not None else 0.0          # rough scale of the equilibrium curvature
        dV_ = bWW + (E * E * bUU).sum(-1) + 2 * (E * bWU).sum(-1) + self.dens_diag(wts)
        dE_ = V[..., None] ** 2 * bUU + 2 * self.lam_E + 6 * self.smooth_E + eq_diag

        def H(xV, xE):
            zW, zU = self._fwd(xV, E * xV[..., None] + V[..., None] * xE)
            y = torch.einsum('pab,pb->pa', F, torch.cat([zW, zU], 1))
            bW, bU = self._back(y[:, :R], y[:, R:])
            return ((bW + (E * bU).sum(-1) + self.dens_lap(xV, wts)) * free,
                    (V[..., None] * bU + 2 * self.lam_E * xE + self.smooth_E * self.laplacian(xE) + eq(xE)) * fE)

        for _ in range(6):
            lam_ = self.damp
            pV, pE = dV_ * (1 + lam_) + 1e-12, dE_ * (1 + lam_)
            if self.precond == 'block':
                prec = self._block_preconditioner(F, lam_, dV_, dE_, free, fE, mask)
            elif self.precond == 'fft':
                prec = self._fft_preconditioner(F, lam_, pV, dE_, free, fE)
            else:
                prec = lambda rV, rE: (rV / pV, rE / pE)
            bV, bE = -gV * free, -gE * fE
            xV, xE = torch.zeros_like(V), torch.zeros_like(E)
            rV, rE = bV.clone(), bE.clone()
            zV, zE = prec(rV, rE)
            dV, dE = zV.clone(), zE.clone()
            rz = float((rV * zV).sum() + (rE * zE).sum())
            for _ in range(self.cg_iters):
                hV, hE = H(dV, dE)
                hV = hV + lam_ * dV_ * dV; hE = hE + lam_ * dE_ * dE
                a = rz / max(float((dV * hV).sum() + (dE * hE).sum()), 1e-30)
                xV += a * dV; xE += a * dE; rV -= a * hV; rE -= a * hE
                zV, zE = prec(rV, rE)
                rz_new = float((rV * zV).sum() + (rE * zE).sum())
                dV, dE = zV + (rz_new / rz) * dV, zE + (rz_new / rz) * dE; rz = rz_new
            dec = float(-(gV * free * xV).sum() - (gE * fE * xE).sum())            # Newton decrement^2 (deviance)
            l0 = self.loss(mask=mask)
            for alpha in (1.0, 0.5, 0.25):
                Vn = (V + alpha * xV).clamp(0, self.v_max); En = E + alpha * xE
                d1 = self.data_loss(Vn, En)
                l1 = d1 + self.prior(Vn, En, mask)
                if l1 < l0:
                    self.last_step_se = self._step_se(alpha * alpha * dec)
                    self.V, self.E = Vn, En; self.sync(); fit.remember_data_loss(d1)
                    self.damp = max(self.damp / 3, 1e-6)
                    return math.sqrt(max(dec, 0.0) / (V.numel() + E.numel())), l0 - l1
            self.damp *= 10
        self.last_step_se = 0.0
        return float('nan'), 0.0

    def normalize_fractions(self):
        """With fraction=True: rescale each material so its median volume fraction over its own voxels is 1 (a typical
        interior voxel holds all of the material), the inverse factor on its spectrum (FitB.rescale_material). Lowering
        every fraction while raising the spectrum leaves the data unchanged and the bound at 1 cannot stop it; this
        fixes that scale, as recentring fixes the strain's. Use it after a volume step. Returns the factors."""
        if not np.isfinite(self.v_max):
            return []
        own = self.material_voxels(); out = []
        with torch.no_grad():
            for r in range(self.R):
                if not bool(own[..., r].any()):
                    out.append(1.0); continue
                m = float(self.V[..., r][own[..., r]].median())
                if m > 0:
                    self.V[..., r] = (self.V[..., r] / m).clamp(0, self.v_max)
                    self.fit.rescale_material(r, m)
                out.append(m)
        self.sync()
        return out

    def _free_density(self, V, gV):
        """The densities a step may move: not those at a bound (0, or v_max with fraction=True) whose gradient points
        outward."""
        return ~(((V <= 0) & (gV > 0)) | ((V >= self.v_max) & (gV < 0)))

    def _step_se(self, quad):
        """The root mean square of a volume step in standard errors, from its length in the Hessian's metric (quad =
        x^T H x, with all correlations): sqrt(quad / n), n the parameters of the voxels holding material (density above
        half its maximum), for a stopping rule. The empty voxels barely move and would dilute the mean."""
        n_in = int(self.material_voxels().sum()) * (1 + self.C)
        return math.sqrt(max(quad, 0.0) / max(n_in, 1))

    def _prior_symbol(self):
        """(n, n, n, R, C, C) complex: the Fourier symbol of the strain priors' Hessian (the weak ridge, the smoothness
        prior, and the equilibrium prior with its mask taken as everywhere, where its operator is periodic), from its
        responses to unit impulses at the volume's centre. The traction-free faces are left out (faces only)."""
        R, C, V = self.R, self.C, self.V
        n = V.shape[0]; c = (n // 2,) * 3
        full = torch.ones(n, n, n, R, dtype=torch.bool, device=V.device)
        div = lambda Y: self._divergence(Y, full).reshape(*Y.shape[:3], -1)
        out = torch.zeros(n, n, n, R, C, C, dtype=torch.complex128, device=V.device)
        for r in range(R):
            for k in range(C):
                X = torch.zeros(n, n, n, R, C, device=V.device, dtype=V.dtype); X[c + (r, k)] = 1.0
                resp = 2 * self.lam_E * X + self.smooth_E * self.laplacian(X)
                if self.elastic is not None:
                    resp = resp + self.eq_w * torch.func.vjp(div, X)[1](div(X))[0]
                resp = torch.roll(resp[..., r, :], shifts=tuple(-v for v in c), dims=(0, 1, 2)).double()
                out[..., r, :, k] = torch.fft.fftn(resp, dim=(0, 1, 2))
        return out

    def _fft_preconditioner(self, F, lam_, pV, dE_, fV, fE):
        """The 'fft' preconditioner: Jacobi (pV) on the densities; on the strain, per spatial frequency, the inverse of
        the priors' exact Fourier symbol plus the data's strain block (the ray Fisher information through each view's
        weights) averaged over the material's voxels, plus the damping. The equilibrium prior is stiff and couples the
        components of neighbouring voxels, so a diagonal preconditioner leaves its smooth modes to converge slowly.
        Returns z = P(rV, rE)."""
        R, C, V = self.R, self.C, self.V
        n = V.shape[0]; P_ = F.shape[0]; w = self.w_ray
        dUU = torch.diagonal(F[:, R:, R:], dim1=1, dim2=2)
        bUU = torch.stack([self.proj.back(torch.cat([torch.zeros(P_, R, device=V.device), (dUU[:, :, None] * (w[:, k:k + 1] * w)[:, None, :]).reshape(P_, R * C)], 1))
                           [..., R:].reshape(n, n, n, R, C) for k in range(C)], -2)              # (n, n, n, R, C, C)
        sym = self._prior_symbol()
        own = self.material_voxels()
        for r in range(R):
            ins = own[..., r]
            if not bool(ins.any()):
                continue
            B = (V[..., r, None, None] ** 2 * bUU[..., r, :, :])[ins].double().mean(0)
            dmp = lam_ * dE_[..., r, :][ins].double().mean(0)
            sym[..., r, :, :] += (B + torch.diag(dmp)).to(sym.dtype)
        inv = torch.linalg.inv(sym)

        def apply(rV, rE):
            z = torch.fft.ifftn(torch.einsum('xyzrab,xyzrb->xyzra', inv, torch.fft.fftn(rE.double(), dim=(0, 1, 2))), dim=(0, 1, 2))
            return rV / pV * fV, z.real.to(rE.dtype) * fE
        return apply

    def _block_preconditioner(self, F, lam_, dV_, dE_, fV, fE, mask):
        """The 'block' preconditioner: per voxel and material, the zero-lag block of the volume Hessian over (V, E_1..C)
        (the data term through the ray Fisher blocks of that material, the priors' zero-lag terms, the equilibrium
        prior's zero-lag block from an impulse response) plus the damping, inverted. Returns z = P(rV, rE)."""
        R, C, V, E = self.R, self.C, self.V, self.E
        n = V.shape[0]; P_ = F.shape[0]; w = self.w_ray
        dWW = torch.diagonal(F[:, :R, :R], dim1=1, dim2=2); dUU = torch.diagonal(F[:, R:, R:], dim1=1, dim2=2)
        dWU = torch.diagonal(F[:, :R, R:], dim1=1, dim2=2)
        b = self.proj.back(torch.cat([dWW, (dWU[:, :, None] * w[:, None, :]).reshape(P_, R * C)], 1))
        bWW, bWU = b[..., :R], b[..., R:].reshape(n, n, n, R, C)
        bUU = torch.stack([self.proj.back(torch.cat([torch.zeros_like(dWW), (dUU[:, :, None] * (w[:, k:k + 1] * w)[:, None, :]).reshape(P_, R * C)], 1))
                           [..., R:].reshape(n, n, n, R, C) for k in range(C)], -2)              # (n, n, n, R, C, C)
        B = torch.zeros(n, n, n, R, 1 + C, 1 + C, device=V.device, dtype=V.dtype)
        EbUU = torch.einsum('xyzrk,xyzrkl->xyzrl', E, bUU)
        B[..., 0, 0] = bWW + 2 * (E * bWU).sum(-1) + (EbUU * E).sum(-1) + self.dens_diag(self._wts)
        B[..., 0, 1:] = V[..., None] * (bWU + EbUU); B[..., 1:, 0] = B[..., 0, 1:]
        B[..., 1:, 1:] = V[..., None, None] ** 2 * bUU + (2 * self.lam_E + 6 * self.smooth_E) * torch.eye(C, device=V.device)
        if self.elastic is not None:
            # the equilibrium prior's zero-lag block, from its response to unit impulses at an interior voxel
            for r in range(R):
                deep = mask[..., r].clone()
                for d in range(3):
                    deep &= torch.roll(mask[..., r], 1, d) & torch.roll(mask[..., r], -1, d)
                idx = deep.nonzero()
                if len(idx) == 0:
                    continue
                j = tuple(int(v) for v in idx[len(idx) // 2])
                blk = torch.zeros(C, C, device=V.device, dtype=V.dtype)
                for k in range(C):
                    X = torch.zeros_like(E); X[j + (r, k)] = 1.0
                    resp = self.eq_w * torch.func.vjp(lambda Y: self._eq_residual(Y, mask), X)[1](self._eq_residual(X, mask))[0]
                    blk[:, k] = resp[j + (r,)]
                B[..., r, 1:, 1:] += mask[..., r, None, None].to(V.dtype) * blk
        # damping as in the Jacobi version, then the held unknowns' rows and columns replaced by the identity
        B = B + lam_ * torch.diag_embed(torch.cat([dV_[..., None], dE_], -1))
        free = torch.cat([fV[..., None], fE], -1).to(V.dtype)                              # (n, n, n, R, 1 + C)
        B = B * free[..., :, None] * free[..., None, :] + torch.diag_embed(1 - free)
        B = B + 1e-12 * torch.eye(1 + C, device=V.device)
        Binv = torch.linalg.inv(B.double()).to(V.dtype)

        def apply(rV, rE):
            z = torch.einsum('xyzrab,xyzrb->xyzra', Binv, torch.cat([rV[..., None], rE], -1)) * free
            return z[..., 0], z[..., 1:]
        return apply

    def mount_step(self, refine_rigid=None, damp=1e-2):
        """One Gauss-Newton step on the mounts of a MultiProjector, for robustness to poorly calibrated view metadata:
        per series its three rotations (degrees), three shifts (voxels) and its centre-of-rotation offset (detector
        channels). refine_rigid lists the series whose rotations and shifts move (default: all but the first, which fixes
        the sample frame); every series' offset moves. Derivatives by central differences of the series' projections
        (MOUNT_FD), steps capped at MOUNT_MAX_STEP, then a backtracking line search on the loss. Returns the accepted
        changes per series."""
        proj, fit, R, C = self.proj, self.fit, self.R, self.C
        refine_rigid = list(range(1, len(proj.parts))) if refine_rigid is None else list(refine_rigid)
        self.sync()
        cache = fit._state(fit.g)
        P = fit.T.shape[0]
        G = torch.zeros(P, 2 * R, device=fit.T.device, dtype=torch.float64)
        F = torch.zeros(P, 2 * R, 2 * R, device=fit.T.device, dtype=torch.float64)
        with torch.no_grad():
            for sl in fit.chunks():
                G[sl], F[sl] = fit.ray_system(sl, cache)
        n = self.V.shape[0]
        vols = torch.cat([self.V, (self.V[..., None] * self.E).reshape(n, n, n, R * C)], -1)
        bounds = np.cumsum([0] + proj.counts)
        to_wu = lambda d, w: torch.cat([d[:, :R], (d[:, R:].reshape(-1, R, C) * w[:, None, :]).sum(-1)], 1)

        def state(s_):
            (perm, signs, sh, ro), p = proj.parts[s_]
            return list(ro) + list(sh) + [proj.offsets[s_]]

        def put(s_, th):
            proj.set_mount(s_, rot=th[:3], shift=th[3:6], offset=th[6])

        steps = []
        for s_ in range(len(proj.parts)):
            sl = slice(int(bounds[s_]), int(bounds[s_ + 1])); w = self.w_ray[sl]
            th0 = state(s_)
            names = (list(range(6)) if s_ in refine_rigid else []) + [6]
            cols = []
            for k in names:
                h = MOUNT_FD[0] if k < 3 else MOUNT_FD[1] if k < 6 else MOUNT_FD[2]
                tp, tm = list(th0), list(th0); tp[k] += h; tm[k] -= h
                put(s_, tp); a_ = proj.project_series(s_, vols)
                put(s_, tm); b_ = proj.project_series(s_, vols)
                cols.append(to_wu((a_ - b_) / (2 * h), w))
            put(s_, th0)
            J = torch.stack(cols, 2).double()                                         # (Ps, 2R, k)
            g = all_sum(torch.einsum('pa,pak->k', G[sl], J))
            Fm = all_sum(torch.einsum('pak,pab,pbl->kl', J, F[sl], J))
            A = Fm + damp * torch.diag(torch.diagonal(Fm)) + 1e-12 * torch.eye(len(names), device=Fm.device, dtype=Fm.dtype)
            d = -torch.linalg.solve(A, g)
            cap = torch.tensor([MOUNT_MAX_STEP[0] if k < 3 else MOUNT_MAX_STEP[1] if k < 6 else MOUNT_MAX_STEP[2] for k in names],
                               device=d.device, dtype=d.dtype)
            d = d * min(1.0, float((cap / d.abs().clamp_min(1e-30)).min()))
            steps.append((s_, names, th0, d.cpu().numpy()))
        l0 = self.loss()

        def apply(alpha):
            for s_, names, th0, d in steps:
                th = list(th0)
                for k, dk in zip(names, alpha * d):
                    th[k] += float(dk)
                put(s_, th)
        mounts = lambda: [dict(shift=list(tr[2]), rot=list(tr[3]), offset=proj.offsets[k]) for k, (tr, _) in enumerate(proj.parts)]
        for alpha in (1.0, 0.5, 0.25):
            apply(alpha); self.sync()
            l1 = self.loss()
            if l1 < l0:
                return dict(alpha=alpha, loss_decrease=l0 - l1, mounts=mounts())
        apply(0.0); self.sync()
        return dict(alpha=0.0, loss_decrease=0.0, mounts=mounts())

    def recenter_strain(self, full=False):
        """Move each material's mean isotropic strain into its lattice parameter, so the strain is measured from the
        sample's mean lattice and carries no uniform offset. Use it after a volume step. The predicted data are unchanged
        to second order in the strain. A pinned lattice parameter (equal bounds) moves with its bounds; a free one, fitted
        by the global steps, moves within its bounds. Returns the offsets moved, one per material.
        full=True (strain tensor only) also removes the mean deviatoric strain, so every component averages to zero over
        the material: a body with no external load has zero mean stress, hence zero mean elastic strain. Unlike the
        isotropic part, the deviatoric mean changes the predicted data, so this is a constraint, not a choice of gauge."""
        M, R, C = self.fit.M, self.R, self.C
        moved = []
        with torch.no_grad():
            own = self.material_voxels()
            for r in range(R):
                V = self.V[..., r]
                w = own[..., r].to(V.dtype) * V                                 # the material's own voxels, density weighted
                if float(w.sum()) <= 0:
                    moved.append(0.0); continue
                iso = self.E[..., r, :3].mean(-1) if C == 6 else self.E[..., r, 0]
                d = float((w * iso).sum() / w.sum())
                if C == 6:
                    self.E[..., r, :3] -= d
                    if full:                                                   # the deviatoric mean, per component
                        m = (w[..., None] * self.E[..., r, :]).sum((0, 1, 2)) / w.sum()
                        self.E[..., r, :] -= m
                else:
                    self.E[..., r, 0] -= d
                ia = M.slices[r]['a']
                a_new = float(self.fit.g[ia]) * (1 + d)
                if float(M.lb[ia]) == float(M.ub[ia]):
                    M.lb[ia] = a_new; M.ub[ia] = a_new
                else:
                    a_new = min(max(a_new, float(M.lb[ia])), float(M.ub[ia]))
                self.fit.g[ia] = a_new
                moved.append(d)
        self.sync()
        return moved

    def support_mask(self):
        """(n, n, n, R, 1): where each material's strain is free (its 3 x 3 x 3 mean density above SUPPORT_REL of the
        maximum), or everywhere without the support prior."""
        if not self.support:
            return torch.ones_like(self.V)[..., None].bool()
        Vs = torch.nn.functional.avg_pool3d(self.V.permute(3, 0, 1, 2)[None], 3, stride=1, padding=1,
                                            count_include_pad=False)[0].permute(1, 2, 3, 0)
        return (Vs > SUPPORT_REL * Vs.amax((0, 1, 2), keepdim=True))[..., None]

    def joint_step(self, idx, cross='ray'):
        """One Gauss-Newton step on the volumes (V, E) and the global parameters idx together: the cross information
        between them comes from the rays (FitB.joint_system), back-projected into one volume pair per global parameter;
        the joint system is solved by preconditioned CG (Jacobi on the volumes, the exact inverse of the free global
        block), then a projected backtracking line search on the full loss. Returns (decrement per parameter, loss
        decrease). cross='ray' keeps the cross information per ray (memory ~ rays x 2R x n_global) and applies it on
        the projections the volume block already makes; cross='volume' back-projects it into one volume pair per global
        parameter (memory ~ n_global x voxels x R (1 + C)), the first implementation, kept for comparison."""
        fit, R, C = self.fit, self.R, self.C
        M = fit.M
        self.sync()
        G, Fw, Cx, gg, Fg = fit.joint_system(idx)
        G, Fw = G.float(), Fw.float()
        V, E = self.V, self.E
        mask = self._eq_mask() if self.elastic is not None else None
        eq = (lambda X: self.eq_w * torch.func.vjp(lambda Y: self._eq_residual(Y, mask), X)[1](self._eq_residual(X, mask))[0]) \
            if self.elastic is not None else (lambda X: torch.zeros_like(X))
        bGW, bGU = self._back(G[:, :R], G[:, R:])
        self._wts = wts = self._dens_wts(V)
        gV = bGW + (E * bGU).sum(-1) + self.dens_lap(V, wts)
        gE = V[..., None] * bGU + 2 * self.lam_E * E + self.smooth_E * self.laplacian(E) + eq(E)
        fV = self._free_density(V, gV); fE = self.support_mask().expand_as(E) & (not self.hold_strain)
        # the global parameters that move: informative and not held at a bound
        gi = fit.g[idx]; lb, ub = M.lb[idx], M.ub[idx]
        d = torch.diagonal(Fg)
        fg = (d > 1e-12 * d.max()) & ~((gi <= lb) & (gg > 0)) & ~((gi >= ub) & (gg < 0))
        gg = gg * fg
        fgf = fg.float()
        if cross == 'volume':                       # the cross columns, one volume pair per global parameter
            cV, cE = [], []
            for j in range(len(idx)):
                bW, bU = self._back(Cx[:, :R, j].contiguous(), Cx[:, R:, j].contiguous())
                cV.append((bW + (E * bU).sum(-1)) * fV); cE.append(V[..., None] * bU * fE)
            cV, cE = torch.stack(cV), torch.stack(cE)
            cV, cE = cV * fgf[:, None, None, None, None], cE * fgf[:, None, None, None, None, None]
        Fg = Fg * (fg[:, None] & fg[None, :]) + torch.diag((~fg).double())
        # Jacobi preconditioner for the volumes (as volume_step)
        dWW = torch.diagonal(Fw[:, :R, :R], dim1=1, dim2=2); dUU = torch.diagonal(Fw[:, R:, R:], dim1=1, dim2=2)
        dWU = torch.diagonal(Fw[:, :R, R:], dim1=1, dim2=2); P_ = dWW.shape[0]
        bWW = self.proj.back(torch.cat([dWW, torch.zeros(P_, R * C, device=dWW.device)], 1))[..., :R]
        bUU = self.proj.back(torch.cat([torch.zeros_like(dWW), (dUU[:, :, None] * self.w_ray[:, None, :] ** 2).reshape(P_, R * C)], 1))[..., R:].reshape(*V.shape[:3], R, C)
        bWU = self.proj.back(torch.cat([torch.zeros_like(dWW), (dWU[:, :, None] * self.w_ray[:, None, :]).reshape(P_, R * C)], 1))[..., R:].reshape(*V.shape[:3], R, C)
        eq_diag = 12.0 * self.eq_w if self.elastic is not None else 0.0
        dV_ = bWW + (E * E * bUU).sum(-1) + 2 * (E * bWU).sum(-1) + self.dens_diag(wts)
        dE_ = V[..., None] ** 2 * bUU + 2 * self.lam_E + 6 * self.smooth_E + eq_diag

        def H(xV, xE, xg):
            xgf = xg.float() * fgf
            zW, zU = self._fwd(xV * fV, E * (xV * fV)[..., None] + V[..., None] * (xE * fE))
            z = torch.cat([zW, zU], 1)
            y = torch.einsum('pab,pb->pa', Fw, z)
            if cross == 'ray':                      # the cross block applied on the same rays: no extra projections
                y = y + torch.einsum('pan,n->pa', Cx, xgf)
            bW, bU = self._back(y[:, :R], y[:, R:])
            hV = (bW + (E * bU).sum(-1) + self.dens_lap(xV, wts)) * fV
            hE = (V[..., None] * bU + 2 * self.lam_E * xE + self.smooth_E * self.laplacian(xE) + eq(xE)) * fE
            if cross == 'ray':
                hg = (Fg @ xg) + (all_sum(torch.einsum('pan,pa->n', Cx, z)) * fgf).double()
            else:
                hV = hV + torch.einsum('j,j...->...', xgf, cV)
                hE = hE + torch.einsum('j,j...->...', xgf, cE)
                hg = (Fg @ xg) + (torch.einsum('j...,...->j', cV, xV) + torch.einsum('j...,...->j', cE, xE)).double()
            return hV, hE, hg

        def total(Vn, En, gn):
            g_keep = fit.g; fit.g = gn
            try:
                return self.loss(Vn, En, mask=mask) + float(fit.spline_penalty(gn))
            finally:
                fit.g = g_keep

        l0 = total(V, E, fit.g)
        for _ in range(6):
            lam_ = self.damp
            Ag = Fg + lam_ * torch.diag(torch.diagonal(Fg))
            Ag_inv = torch.linalg.inv(Ag)
            pV, pE = dV_ * (1 + lam_) + 1e-12, dE_ * (1 + lam_)
            if self.precond in ('block', 'fft'):
                pvol = self._block_preconditioner(Fw, lam_, dV_, dE_, fV, fE, mask) if self.precond == 'block' \
                    else self._fft_preconditioner(Fw, lam_, pV, dE_, fV, fE)
                prec = lambda rV, rE, rg: (*pvol(rV, rE), Ag_inv @ rg)
            else:
                prec = lambda rV, rE, rg: (rV / pV, rE / pE, Ag_inv @ rg)
            bV, bE, bg = -gV * fV, -gE * fE, -gg
            xV, xE, xg = torch.zeros_like(V), torch.zeros_like(E), torch.zeros_like(gg)
            rV, rE, rg = bV.clone(), bE.clone(), bg.clone()
            zV, zE, zg = prec(rV, rE, rg)
            dV, dE, dg = zV.clone(), zE.clone(), zg.clone()
            rz = float((rV * zV).sum() + (rE * zE).sum() + (rg * zg).sum()); rz0 = rz
            for _ in range(JOINT_CG_ITERS):
                hV, hE, hg = H(dV, dE, dg)
                hV = hV + lam_ * dV_ * dV; hE = hE + lam_ * dE_ * dE; hg = hg + lam_ * torch.diagonal(Fg) * dg
                den = float((dV * hV).sum() + (dE * hE).sum() + (dg * hg).sum())
                if den <= 0:
                    break
                a = rz / den
                xV += a * dV; xE += a * dE; xg += a * dg; rV -= a * hV; rE -= a * hE; rg -= a * hg
                zV, zE, zg = prec(rV, rE, rg)
                rz_new = float((rV * zV).sum() + (rE * zE).sum() + (rg * zg).sum())
                b = rz_new / max(rz, 1e-30)
                dV, dE, dg = zV + b * dV, zE + b * dE, zg + b * dg; rz = rz_new
            self.last_cg = math.sqrt(max(rz, 0.0) / max(rz0, 1e-300))                 # the CG's final relative residual
            dec = float(-(gV * fV * xV).sum() - (gE * fE * xE).sum() - (gg * xg).sum())
            for alpha in (1.0, 0.5, 0.25):
                Vn = (V + alpha * xV).clamp(0, self.v_max); En = E + alpha * xE
                gn = fit.g.clone(); gn[idx] = gn[idx] + alpha * xg
                gn = torch.maximum(torch.minimum(gn, M.ub), M.lb)
                l1 = total(Vn, En, gn)
                if l1 < l0:
                    self.V, self.E, fit.g = Vn, En, gn; self.sync()
                    self.damp = max(self.damp / 3, 1e-6)
                    return math.sqrt(max(dec, 0.0) / (V.numel() + E.numel() + len(idx))), l0 - l1
            self.damp *= 10
        return float('nan'), 0.0


def warmup(T, lam, d, lattices, a0, proj, device='cuda', n_spline=8, n_contam=2, chunk=1024, a_range=0.01,
           d0_from_data=False, anchor_scale=False, inflation=1.0, density_smooth=False, weights=None, elastic=None,
           eq_sd=EQUILIBRIUM_SD, cg_iters=CG_ITERS, kappa_prior=None, support=False, ray_mask=None, precond='jacobi',
           density_prior='quadratic', voxel_scale=1.0, kernel='two_scale', kernel_values=None, traction=False,
           smooth_sd=STRAIN_SMOOTH_SD, mask_method='max', mask_freeze=False, fraction=False, eq_stencil='corner',
           noise=None, detector_tiles=(1, 1), shape_model=None):
    """The pipeline's start with given lattice types: returns a TomoFit. warm_start searches each lattice parameter
    within a0 * (1 +- a_range). The lattice parameters are then fixed: at a0 (a known stress-free reference), or with
    d0_from_data at warm_start's values (strain relative to the sample's mean lattice). With anchor_scale, each
    spectrum is pinned at one wavelength to its warm-up value, which removes the density-spectrum scale trade-off.
    inflation is the variance inflation of the counts (count correlation; it scales the contamination thresholds);
    density_smooth adds the density smoothness prior. weights (views, C) select the strain model (None: a scalar
    dilation; beam_geometry tensor weights: the strain tensor) and elastic the equilibrium prior (see TomoFit).
    ray_mask (rays,) bool leaves the rays where it is False out of the reconstruction and the likelihood (FitB).
    kernel: the resolution kernel's form (model.Hybrid2); kernel_values: its 8 parameters from a calibration (for
    example on a Ni cylinder), held fixed; None fits the kernel from the warm start's estimate. traction: traction-free faces in the
    equilibrium prior (TomoFit); smooth_sd: the strain smoothness prior's standard deviation (None: no smoothness prior);
    mask_method, mask_freeze: how the material's own voxels are found (TomoFit). eq_stencil: the equilibrium prior's
    stencil (TomoFit). fraction: densities as volume
    fractions (TomoFit): each material's warm-start density is divided by its full density (the median over its own
    voxels, by mask_method), and its spectrum's linear parameters (edge heights, smooth part, spline) and contamination
    attenuations are multiplied by it, so the predicted data are unchanged. noise='measure' measures the counts' noise on
    the rays that miss the sample (noise.measure_noise) and fits with the dose divided by the measured dispersion and
    the measured inflation in place of d and inflation; None uses d and inflation as given. detector_tiles: the
    detector's sensors as (rows, channels) tiles, each with its own open-beam level when air rays are found.
    shape_model='texture3d' gives each material a March-Dollase texture (strength and axis, model.Hybrid2.texture_factor):
    each view's edges carry its own factor (needs the strain-tensor weights, which hold the views' beam directions); it
    starts as a random powder. shape_model='cubic4' gives each cubic material its lowest-order texture instead (9
    coefficients, linear; model.harmonics4), also starting as a random powder."""
    from mbirtorch import hsnt
    noise_info = None
    if noise == 'measure':
        from .noise import measure_noise
        rpv = getattr(proj, 'rays_per_view', proj.n * proj.n); side = int(round(math.sqrt(rpv)))
        noise_info = measure_noise(T, d, (T.shape[0] // rpv, side, side), None if ray_mask is None else ~ray_mask.bool(),
                                   detector_tiles)
        d = np.asarray(d, dtype=np.float64) / noise_info['dispersion']; inflation = noise_info['inflation']
    elif noise is not None:
        raise ValueError(f"noise must be None or 'measure', not {noise!r}")
    R = len(lattices)
    dev = str(T.device) if T.is_cuda else None   # a GPU tensor is factored on its own GPU, without a host copy
    if ray_mask is None:
        W0, H0, _ = hsnt.dehydrate(T, 'transmission', num_materials=R, device=dev, verbose=0)
        W0 = W0.reshape(-1, R)
    else:                                       # the spectra from the kept rays only; left-out rays start at zero
        keep = ray_mask.cpu().numpy().astype(bool)
        Wk, H0, _ = hsnt.dehydrate(T[ray_mask.to(T.device)], 'transmission', num_materials=R, device=dev, verbose=0)
        W0 = np.zeros((T.shape[0], R), dtype=Wk.dtype); W0[keep] = Wk.reshape(-1, R)
    lats = [(lt, (a * (1 - a_range), a * (1 + a_range))) for lt, a in zip(lattices, a0)]
    # with the rays split over processes, the first process's spectra start every process (the same model everywhere);
    # each process keeps its own H0 to express its maps in that model's spectra
    H0_local = H0
    H0, = broadcast_objects([H0])
    out = [None]
    if rank() == 0:
        th_, fd_, _, _, res_ = warm_start(H0.astype(np.float64), lam, n_bragg=R, lattices=lats, device=device)
        out = [([t.cpu() for t in th_], fd_, res_)]
    thetas, found, resolution = broadcast_objects(out)[0]
    # warm_start orders its components by its own criterion: pair each given lattice with the found component of the
    # same type and the nearest lattice parameter
    from scipy.optimize import linear_sum_assignment
    cost = np.array([[abs(float(th[0]) / a - 1) if lf == lt else np.inf for (lf, _), th in zip(found, thetas)]
                     for lt, a in zip(lattices, a0)])
    assert len(found) == R and np.isfinite(cost).any(axis=1).all(), (found, lattices)
    _, order = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e9))
    found = [found[j] for j in order]; thetas = [thetas[j] for j in order]
    assert all(np.isfinite(cost[i, j]) for i, j in enumerate(order)), (found, lattices)
    targets = []
    for (lt, ar), th in zip(found, thetas):
        tgt = BraggSpectrum(lam, lattice=lt, a_range=ar, n_spline=8, resolution=resolution)(th.cpu()).detach().numpy()
        targets.append(tgt / max(np.abs(tgt).mean(), 1e-12) * float(np.abs(H0).mean()))
    targets = np.stack(targets)
    if d0_from_data:
        a0 = [float(th[0]) for th in thetas]
    M = Hybrid2(lam, d, lats, n_spline=n_spline, n_contam=n_contam, device=device, two_scale=True, kernel=kernel,
                shape_model=shape_model)
    g = torch.zeros(M.n_global, device=device, dtype=torch.float64)
    g[M.res_sl[0]:M.res_sl[1]] = torch.tensor([0.0, resolution[0], 1e-5, resolution[1], 0.1, 0.02, 0.0, 0.0], dtype=torch.float64)
    if kernel_values is not None:                       # a calibrated kernel, held fixed
        kv = torch.as_tensor(np.asarray(kernel_values, dtype=np.float64), device=device)
        g[M.res_sl[0]:M.res_sl[1]] = kv; M.lb[M.res_sl[0]:M.res_sl[1]] = kv; M.ub[M.res_sl[0]:M.res_sl[1]] = kv
    mu_bar = float(np.mean(targets))
    for j in range(M.J):
        for r in range(M.R):
            g[M.kap_sl[0] + j * M.R + r] = mu_bar * (0.1 + 0.4 * j)
    for r in range(R):
        ls_init_material(M, g, r, targets[r], a0[r])
        M.lb[M.slices[r]['a']] = M.ub[M.slices[r]['a']] = a0[r]                    # d0 reference
        if shape_model == 'texture3d':                                             # a random powder to start
            x0 = M.slices[r]['x'][0]
            g[x0] = 1.0; g[x0 + 1] = TEXTURE_AXIS_START[0]; g[x0 + 2] = TEXTURE_AXIS_START[1]
        elif shape_model == 'cubic4':                                              # a random powder: no degree-4 term
            g[M.slices[r]['x'][0]:M.slices[r]['x'][1]] = 0.0
    Mx = np.linalg.lstsq(targets.T, H0_local.T.astype(np.float64), rcond=None)[0].T
    Wr = torch.as_tensor(np.clip(W0 @ Mx, 0, None), device=device, dtype=torch.float32)
    # the pipeline's per-ray fits of the maps to the initial spectra (no strain yet)
    fit0 = FitB(M, T, g, Wr, torch.zeros(M.J, M.K, device=device), chunk=chunk)
    for _ in range(INIT_PIXEL_STEPS):
        fit0.pixel_step()
    Wr = fit0.W; del fit0
    # least-squares reconstruction of the ray maps (CGLS), clipped to >= 0
    n = proj.n; pr = proj.with_q(R)
    x = torch.zeros(n, n, n, R, device=device)
    rm = torch.ones(Wr.shape[0], 1, device=device) if ray_mask is None else ray_mask.to(device, Wr.dtype)[:, None]
    r_ = (Wr - pr.fwd(x)) * rm; s = pr.back(r_); p = s.clone(); gam = float((s * s).sum())
    for _ in range(INIT_CG_ITERS):
        q = pr.fwd(p) * rm; al = gam / max(all_sum_float(float((q * q).sum()), q.device), 1e-30)
        x += al * p; r_ -= al * q; s = pr.back(r_); gnew = float((s * s).sum())
        p = s + (gnew / gam) * p; gam = gnew
    V = x.clamp_min(0)
    fit = FitB(M, T, g, Wr, torch.zeros(M.J, M.K, device=device), eps=torch.zeros_like(Wr), chunk=chunk, inflation=inflation,
               ray_mask=ray_mask)
    if shape_model in ('texture3d', 'cubic4'):                                     # each view's beam, each ray's view
        from .geometry import directions_from_weights
        assert weights is not None, 'a view-dependent texture needs the strain-tensor weights (they hold the beam directions)'
        rpv = getattr(proj, 'rays_per_view', proj.n * proj.n)
        M.view_dirs = directions_from_weights(weights)
        fit.ray_view = torch.arange(T.shape[0] // rpv, device=T.device).repeat_interleave(rpv)
    if kappa_prior is not None:
        # the standard deviation as a multiple of the warm-up spectra's mean attenuation (kappa's own units)
        fit.kappa_prior = kappa_prior * float(np.abs(targets).mean())
    if anchor_scale:
        # pin each spectrum at the fine grid's middle to its warm-up value: the stiffness is ANCHOR_STIFFNESS times
        # the largest diagonal entry of the spectrum parameters' Fisher information, so the pin dominates the data
        # along the scale (the resolution kernel is left out: its derivatives are costly and it does not set the scale)
        with torch.no_grad():
            k = M.Kf // 2; targets = M.spectra(g)[:, k].clone()
        fit.scale_anchor = (k, targets, 0.0)
        idx = [i for i in range(M.n_global) if float(M.ub[i]) > float(M.lb[i]) and not (M.res_sl[0] <= i < M.res_sl[1])]
        _, F0 = fit.global_system(idx)
        fit.scale_anchor = (k, targets, ANCHOR_STIFFNESS * float(torch.diagonal(F0).max()))
    if fraction:
        probe = object.__new__(TomoFit); probe.V = V; probe.mask_method = 'median' if mask_method == 'max' else mask_method
        probe.mask_freeze = False; probe._mask = None
        own = probe.material_voxels()
        with torch.no_grad():
            for r in range(R):
                full = float(V[..., r][own[..., r]].median()) if bool(own[..., r].any()) else 1.0
                V[..., r] = V[..., r] / full
                fit.rescale_material(r, full)
            V = V.clamp(0, VOLUME_FRACTION_MAX)
    C = 1 if weights is None else np.asarray(weights).shape[1]
    E0 = torch.zeros(*V.shape, C, device=device, dtype=V.dtype)
    return TomoFit(fit, proj, V, E0, weights=weights, density_smooth=density_smooth, elastic=elastic, eq_sd=eq_sd,
                   cg_iters=cg_iters, support=support, precond=precond, density_prior=density_prior,
                   voxel_scale=voxel_scale, traction=traction, smooth_sd=smooth_sd, mask_method=mask_method,
                   mask_freeze=mask_freeze, fraction=fraction, eq_stencil=eq_stencil), dict(found=[(lt, float(th[0])) for (lt, _), th in zip(found, thetas)],
                                                           resolution=[float(v) for v in resolution], traction=traction, smooth_sd=smooth_sd,
                                                           noise=noise_info)


def newton_step(F, g, rcond=1e-8):
    """F^-1 g on the directions the data determine: a Jacobi-scaled eigendecomposition, keeping the eigenvalues above
    rcond times the largest. The global parameters' Fisher diagonal spans many orders of magnitude (kernel widths
    against edge heights), and some parameters carry no information at all (the absorption of a contamination
    component that is off, an edge outside the measured range), so a plain solve is dominated by those directions."""
    d = torch.diagonal(F)
    s = torch.where(d > rcond * d.max(), 1.0 / torch.sqrt(d.clamp_min(torch.finfo(F.dtype).tiny)), torch.zeros_like(d))
    lam, Q = torch.linalg.eigh(s[:, None] * F * s[None, :])
    keep = lam > rcond * lam.max()
    return s * (Q[:, keep] @ ((Q[:, keep].T @ (s * g)) / lam[keep]))

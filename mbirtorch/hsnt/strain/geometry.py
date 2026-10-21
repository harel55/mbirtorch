"""Beam directions of the package's parallel-beam projector, for strain tomography (the longitudinal ray transform:
a ray measures n . eps . n, n its direction). Volumes are (rows y, cols x, slices z) with the rotation axis along z.

beam_directions(angles) finds n = (n_x, n_y) for each view numerically, from where two off-centre voxels land on
the detector: a voxel at x projects at the channel offset x t_x, one at y at y t_y, with t the in-plane detector
direction; n is t turned by 90 degrees (its sign does not matter, since n . eps . n is even in n).
view_weights(angles) gives the weights of the six strain components (xx, yy, zz, xy, yz, xz; tensor shears)."""
import numpy as np
import torch

OFFSET = 4                                  # voxels from the centre for the probe voxels


def beam_directions(angles, n=16):
    import mbirtorch
    model = mbirtorch.ParallelBeamModel((len(angles), 1, n), np.asarray(angles))
    c = n // 2
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = []
    for dy, dx in ((0, OFFSET), (OFFSET, 0)):
        v = torch.zeros(n, n, 1, device=dev); v[c + dy, c + dx, 0] = 1.0
        v0 = torch.zeros(n, n, 1, device=dev); v0[c, c, 0] = 1.0
        s = model.forward_project(v, output_sharded=True)[:, 0, :].double()
        s0 = model.forward_project(v0, output_sharded=True)[:, 0, :].double()
        ch = torch.arange(n, device=s.device, dtype=torch.float64)
        t.append(((s * ch).sum(1) / s.sum(1) - (s0 * ch).sum(1) / s0.sum(1)) / OFFSET)
    tx, ty = t                                              # the detector direction's x and y parts
    norm = torch.sqrt(tx ** 2 + ty ** 2)
    tx, ty = tx / norm, ty / norm
    return torch.stack([-ty, tx], 1).cpu().numpy()          # n = t turned by 90 degrees


def view_weights(angles):
    """(views, 6) weights of the strain components (xx, yy, zz, xy, yz, xz; tensor shears) in n . eps . n, for a
    rotation about z (n_z = 0, so zz, yz and xz carry no weight)."""
    nd = beam_directions(angles)
    z = np.zeros(len(nd))
    return np.stack([nd[:, 0] ** 2, nd[:, 1] ** 2, z, 2 * nd[:, 0] * nd[:, 1], z, z], 1)


if __name__ == '__main__':
    th = np.linspace(0, np.pi, 8, endpoint=False)
    nd = beam_directions(th)
    for a_, d in zip(np.degrees(th), nd):
        print(f'angle {a_:6.1f} deg: n = ({d[0]:+.3f}, {d[1]:+.3f})  |n| = {np.hypot(*d):.3f}')


# The three rotation series of a sample, as axis permutations of the sample volume (layout: rows y, cols x, slices z):
# a series' volume is the sample volume permuted by PERMS[s] (its rows, cols, slices are these sample axes), so its
# rotation axis is the projector's slice axis. Each is a cyclic permutation of (x, y, z), hence a proper rotation.
PERMS = dict(z=(0, 1, 2), x=(2, 0, 1), y=(1, 2, 0))


def series_weights(series):
    """(total views, 6) strain-component weights for a list of (axis, angles): each view's beam direction, found in its
    series' frame (beam_directions: n' = (along cols, along rows)), mapped to the sample's (x, y, z) components. axis is
    a PERMS key or a measured mount (perm, signs, shift) as in tomo_hybrid.MultiProjector."""
    out = []
    dim_to_xyz = {0: 1, 1: 0, 2: 2}                     # sample volume dims (rows y, cols x, slices z) -> x, y, z index
    for axis, angles in series:
        perm, signs = (PERMS[axis], (1, 1, 1)) if isinstance(axis, str) else (tuple(axis[0]), tuple(axis[1]))
        nd = beam_directions(angles)                    # (views, 2): components along the series' cols and rows
        n3 = np.zeros((len(nd), 3))
        n3[:, dim_to_xyz[perm[1]]] = signs[1] * nd[:, 0]   # the series' cols are sample dim perm[1] (flipped if < 0)
        n3[:, dim_to_xyz[perm[0]]] = signs[0] * nd[:, 1]   # its rows are sample dim perm[0]
        nx, ny, nz = n3.T
        out.append(np.stack([nx * nx, ny * ny, nz * nz, 2 * nx * ny, 2 * ny * nz, 2 * nx * nz], 1))
    return np.concatenate(out)


def directions_from_weights(w):
    """The beam directions (views, 3) in the sample frame (x, y, z), up to sign, from series_weights' strain-component
    weights (n_x^2, n_y^2, n_z^2, 2 n_x n_y, 2 n_y n_z, 2 n_x n_z): each view's largest component is taken positive and
    the others' signs come from their products with it."""
    w = np.asarray(w, dtype=np.float64)
    d = np.sqrt(np.clip(w[:, :3], 0.0, None))
    pair = {(0, 1): 3, (1, 2): 4, (0, 2): 5}
    for v in range(len(w)):
        k = int(np.argmax(d[v]))
        for j in range(3):
            if j != k and w[v, pair[tuple(sorted((k, j)))]] < 0:
                d[v, j] = -d[v, j]
    return d

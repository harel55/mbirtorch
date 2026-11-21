"""Outward surface normals of a voxelized solid, from local plane fits to its exposed voxel faces.

This is a port of the surface-normal estimate of MONSTR (cabouman/monstr, reconstruction_3d._compute_surface_normals_3d).
Each exposed face gets the normal of a plane fitted by principal component analysis to the centres of nearby exposed
faces with similar orientation, so a staircase on a curved or oblique surface gets the normal of the surface it
approximates.  Each boundary voxel gets the normal of its dominant cluster of face normals, so corners stay sharp.
"""
import numpy as np

PLANE_FIT_HOPS = 4              # the faces within this many shared-vertex steps of a face are candidates for its plane
PLANE_FIT_DISTANCE = 0.75       # a candidate face's centre must lie within this many voxels of the fitted plane
PLANE_FIT_DOT = 0.85            # a candidate face's one-ring normal must have at least this dot product with the seed's
CLUSTER_ANGLE_DEG = 35.0        # a voxel's face normals within this angle of a cluster's mean join that cluster
CLUSTER_DOMINANCE = 1.5         # the largest cluster sets the voxel's normal when it is this many times the next one
_EPS = 1e-12

# Each face: its name, its outward normal (x, y, z) and the index offset of the neighbour it faces, in (z, y, x) order.
_FACES = (('+x', (1.0, 0.0, 0.0), (0, 0, 1)), ('-x', (-1.0, 0.0, 0.0), (0, 0, -1)),
          ('+y', (0.0, 1.0, 0.0), (0, 1, 0)), ('-y', (0.0, -1.0, 0.0), (0, -1, 0)),
          ('+z', (0.0, 0.0, 1.0), (1, 0, 0)), ('-z', (0.0, 0.0, -1.0), (-1, 0, 0)))


def _unit(v):
    n = float(np.linalg.norm(v))
    return np.zeros(3) if n <= _EPS else np.asarray(v, dtype=np.float64) / n


def _face_geometry(z, y, x, k):
    """The centre (x, y, z) and the four corner vertices of face k of voxel (z, y, x)."""
    axis = (0, 0, 1, 1, 2, 2)[k]                      # the face's normal axis, in (x, y, z) order
    lo = np.array([x, y, z], dtype=np.float64)
    c = lo + 0.5
    c[axis] = lo[axis] + (1.0 if k % 2 == 0 else 0.0)
    others = [a for a in range(3) if a != axis]
    verts = []
    for da in (0, 1):
        for db in (0, 1):
            v = [int(x), int(y), int(z)]
            v[axis] += 1 if k % 2 == 0 else 0
            v[others[0]] += da
            v[others[1]] += db
            verts.append(tuple(v))
    return c, verts


def plane_fit_normals(mask):
    """Return the outward unit normals (n0, n1, n2, 3) of a boolean solid ``mask`` of shape (n0, n1, n2), in the
    (x, y, z) order of the mask's axes (2, 1, 0).  Voxels without an exposed face get zero."""
    m = np.asarray(mask, dtype=bool)
    nz, ny, nx = m.shape
    out = np.zeros(m.shape + (3,))
    centres, raw, verts, owner = [], [], [], []
    for z, y, x in np.argwhere(m):
        for k, (_, normal, (dz, dy, dx)) in enumerate(_FACES):
            qz, qy, qx = z + dz, y + dy, x + dx
            if 0 <= qz < nz and 0 <= qy < ny and 0 <= qx < nx and m[qz, qy, qx]:
                continue
            c, v = _face_geometry(z, y, x, k)
            centres.append(c); raw.append(normal); verts.append(v); owner.append((int(z), int(y), int(x)))
    if not centres:
        return out
    centres, raw = np.asarray(centres), np.asarray(raw)
    by_vertex = {}
    for f, vs in enumerate(verts):
        for v in vs:
            by_vertex.setdefault(v, []).append(f)
    neighbours = []
    for f, vs in enumerate(verts):
        s = set()
        for v in vs:
            s.update(by_vertex[v])
        s.discard(f)
        neighbours.append(np.fromiter(sorted(s), dtype=np.int64))
    provisional = np.array([_unit(raw[f] + raw[nb].sum(0)) for f, nb in enumerate(neighbours)])
    bad = np.linalg.norm(provisional, axis=1) <= _EPS
    provisional[bad] = raw[bad]

    def plane_normal(points, hint):
        if len(points) < 3:
            return _unit(hint)
        n = _unit(np.linalg.svd(points - points.mean(0), full_matrices=False)[2][-1])
        if np.linalg.norm(n) <= _EPS:
            n = _unit(hint)
        return -n if n @ hint < 0 else n

    def ball(f0):
        seen, frontier, head = {f0}, [(f0, 0)], 0
        while head < len(frontier):
            f, d = frontier[head]; head += 1
            if d < PLANE_FIT_HOPS:
                for nb in neighbours[f]:
                    if int(nb) not in seen:
                        seen.add(int(nb)); frontier.append((int(nb), d + 1))
        return np.array([f for f, _ in frontier], dtype=np.int64)

    def face_normal(f):
        seed = provisional[f]
        cand = ball(f)
        dots = provisional[cand] @ seed
        acc = cand[dots >= PLANE_FIT_DOT]
        if acc.size < 3:
            acc = cand[dots > 0]
        if acc.size < 3:
            acc = cand
        for _ in range(3):
            if acc.size < 3:
                break
            n = plane_normal(centres[acc], seed)
            dist = np.abs((centres[cand] - centres[acc].mean(0)) @ n)
            new = cand[(dist <= PLANE_FIT_DISTANCE) & (dots >= PLANE_FIT_DOT)]
            if new.size < 3:
                new = cand[(dist <= PLANE_FIT_DISTANCE) & (dots > 0)]
            if new.size < 3 or np.array_equal(new, acc):
                break
            acc = new
        n = _unit(plane_normal(centres[acc], seed)) if acc.size >= 3 else _unit(seed)
        return n if np.linalg.norm(n) > _EPS else raw[f]

    by_voxel = {}
    for f, o in enumerate(owner):
        by_voxel.setdefault(o, []).append(f)
    cos_cluster = np.cos(np.deg2rad(CLUSTER_ANGLE_DEG))
    for (z, y, x), faces in by_voxel.items():
        clusters = []
        for n in (face_normal(f) for f in faces):
            for cl in clusters:
                if n @ _unit(np.sum(cl, 0)) >= cos_cluster:
                    cl.append(n); break
            else:
                clusters.append([n])
        clusters.sort(key=len, reverse=True)
        if len(clusters) == 1 or len(clusters[0]) >= CLUSTER_DOMINANCE * len(clusters[1]):
            out[z, y, x] = _unit(np.sum(clusters[0], 0))
        else:
            out[z, y, x] = _unit(np.sum([_unit(np.sum(cl, 0)) for cl in clusters], 0))
    return out

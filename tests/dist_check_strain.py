"""Check that the strain fit gives the same steps with its rays split over processes (gloo, CPU) as in one process.
Run directly: python tests/dist_check_strain.py  (not collected by pytest; it spawns processes)."""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mbirtorch.hsnt.strain.phantom import make_fe_cube, FE  # noqa: E402
from mbirtorch.hsnt.strain.tomo import MultiProjector, warmup  # noqa: E402
from mbirtorch.hsnt.strain.geometry import series_weights  # noqa: E402

N = 12
VIEWS = 6
MOUNTS = [((0, 1, 2), (1, 1, 1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), ((1, 2, 0), (1, 1, 1), (0.2, -1.0, 0.8), (0.0, 0.0, 0.0)),
          ((2, 0, 1), (-1, 1, -1), (0.6, 0.1, 0.7), (0.0, 0.0, 0.0))]


def build(ph, views, T):
    ang = ph['angles'][views]
    sr = [(m, ang) for m in MOUNTS]
    proj = MultiProjector(N, sr, 2, channel_offsets=[0.0, 0.0, 0.0], ror_mask=False)
    tf, _ = warmup(T, ph['lam'], ph['dose_per_bin'], [FE['lattice']], [FE['a']], proj, device='cpu', a_range=0.02,
                   d0_from_data=True, inflation=1.0, density_smooth=True, weights=series_weights(sr))
    return tf


def rays_of(views):
    return np.concatenate([(k * VIEWS + views)[:, None] * N * N + np.arange(N * N)[None] for k in range(3)]).ravel()


def steps(tf, state):
    V, E, g, f, sV = state
    tf.V, tf.E = V.clone(), E.clone(); tf.fit.g = g.clone(); tf.fit.f = f.clone(); tf.smooth_V = sV.clone(); tf.damp = 1e-3
    tf.sync()
    out = [tf.loss()]
    # the volume gradient and one data-term Hessian product at the start, to compare directly
    fit, R = tf.fit, tf.R
    cache = fit._state(fit.g)
    G = torch.cat([fit.ray_system(sl, cache)[0] for sl in fit.chunks()]).float()
    Fr = torch.cat([fit.ray_system(sl, cache)[1] for sl in fit.chunks()]).float()
    bW, bU = tf._back(G[:, :R], G[:, R:])
    x = torch.rand(tf.V.shape, generator=torch.Generator().manual_seed(3))
    zW, zU = tf._fwd(x, tf.E * x[..., None])
    y = torch.einsum('pab,pb->pa', Fr, torch.cat([zW, zU], 1))
    hW, hU = tf._back(y[:, :R], y[:, R:])
    out += [float(bW.norm()), float(bU.norm()), float(hW.norm()), float(hU.norm())]
    tf.volume_step(); out.append(tf.loss())
    tf.fit.global_step(1e-3); out.append(tf.loss())
    tf.fit.contamination_atoms_step(); out.append(tf.loss())
    tf.mount_step(); out.append(tf.loss())
    M = tf.fit.M
    return out, tf.V.clone(), tf.fit.g.clone()


def worker(r, world, store, ph, state, ref):
    dist.init_process_group('gloo', init_method=f'file://{store}', rank=r, world_size=world)
    torch.set_num_threads(2)
    views = np.arange(r, VIEWS, world)
    tf = build(ph, views, ph['T'][torch.as_tensor(rays_of(views))])
    out, V, g = steps(tf, state)
    if r == 0:
        print('losses, one process:   ', [f'{v:.8g}' for v in ref[0]])
        print('losses, split rays:    ', [f'{v:.8g}' for v in out])
        print('max relative loss difference %.2e; volume %.2e; globals %.2e' % (
            max(abs(a - b) / abs(b) for a, b in zip(out, ref[0])),
            float((V - ref[1]).norm() / ref[1].norm()), float((g - ref[2]).norm() / ref[2].norm())))
    dist.destroy_process_group()


if __name__ == '__main__':
    torch.manual_seed(0)
    ph = make_fe_cube(n=N, angles=np.linspace(0, np.pi, VIEWS), mounts=MOUNTS, levels=5, device='cpu', dose=2000.0)
    tf = build(ph, np.arange(VIEWS), ph['T'])
    state = (tf.V.clone(), tf.E.clone(), tf.fit.g.clone(), tf.fit.f.clone(), tf.smooth_V.clone())
    ref = steps(tf, state)
    store = os.path.join(tempfile.mkdtemp(), 'store')
    mp.spawn(worker, args=(2, store, ph, state, ref), nprocs=2, join=True)

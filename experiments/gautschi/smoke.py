"""Smoke test for the HPC gate pipeline: report the environment and the pinned commit, run a 4k-pixel NNAL solve, write JSON."""
import argparse, os, sys, json, time, subprocess, numpy as np, torch
root = os.environ.get('PROJECT_ROOT', os.environ.get('MBIRTORCH_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, root)
from mbirtorch.hsnt import generate_hyper_data, nnal_factorization, stable_nnal
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='.'); ap.add_argument('--basis', default=None); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
info = dict(host=os.uname().nodename, python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            project_root=root, project_commit=os.environ.get('HPC_PROJECT_COMMIT'), jobs_commit=os.environ.get('HPC_COMMIT'),
            job_id=os.environ.get('SLURM_JOB_ID'))
basis = np.load(a.basis or os.path.expanduser('~/mbirjax/experiments/hsnt/binaries/material_basis.npy'))
np.random.seed(0); proj, _, _ = generate_hyper_data(basis, num_angles=1, detector_rows=64, detector_columns=64, dosage_rate=3.0,
                                                    material_density={"Ni": 0.25, "Cu": 0.25, "Al": 0.75}, noisy=True, verbose=0)
T = torch.tensor(np.exp(-np.nan_to_num(proj)).reshape(-1, proj.shape[-1]), dtype=torch.float32, device='cuda')
torch.cuda.synchronize(); t0 = time.perf_counter()
W, H, it = nnal_factorization(T, method='joint_newton', num_materials=3, max_steps=300, rel_tol=1e-8)
torch.cuda.synchronize(); info.update(steps=int(it), sec=time.perf_counter() - t0, nnal=stable_nnal(W.double() @ H.double(), T.double()).item())
json.dump(info, open(os.path.join(a.out, 'smoke.json'), 'w'), indent=1)
print(json.dumps(info, indent=1)); print('SMOKEDONE')

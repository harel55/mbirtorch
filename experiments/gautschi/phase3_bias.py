"""Phase 3 on Gautschi: does the H bias remedy pay at 262k-1M pixels? Per size: early-stopped, converged MLE,
unconstrained-W spectra (W>=0 re-solved), support-selected spectra (bootstrap available as 'boot'). Spectral + map SNR, fp64 loss, time. Branch bias-correction."""
import argparse, os, sys, time, json, numpy as np
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # 1M px at 63.6 GB peak fragmented: 16.7 GB reserved-unallocated at the OOM
import torch
os.environ.setdefault('MBIRTORCH_ROOT', os.path.expanduser('~/mbirtorch'))
sys.path.insert(0, os.environ['MBIRTORCH_ROOT'])
from mbirtorch.hsnt import generate_hyper_data, nnal_factorization, bias_corrected_spectra, unconstrained_spectra, support_selected_spectra, stable_nnal
ap = argparse.ArgumentParser()
ap.add_argument('--out', default='.'); ap.add_argument('--basis', default=None)
ap.add_argument('--sides', default='512,724,1024'); ap.add_argument('--doses', default='3,30'); ap.add_argument('--ranks', default='3,4')
ap.add_argument('--methods', default='early,mle,uncw,supp'); ap.add_argument('--n-sim', type=int, default=32); ap.add_argument('--smoke', action='store_true')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
basis_np = np.load(a.basis or os.path.expanduser('~/mbirjax/experiments/hsnt/binaries/material_basis.npy')); D = {"Ni": 0.25, "Cu": 0.25, "Al": 0.75}; NM = 3
sides = [64, 128] if a.smoke else [int(x) for x in a.sides.split(',')]
def gen(side, seed, dose):
    np.random.seed(seed); proj, _, gt = generate_hyper_data(basis_np, num_angles=1, detector_rows=side, detector_columns=side, dosage_rate=dose, material_density=D, noisy=True, verbose=0)
    T = torch.tensor(np.exp(-np.nan_to_num(proj)).reshape(-1, proj.shape[-1]), dtype=torch.float32, device='cuda'); del proj
    Bt = torch.tensor(basis_np, dtype=torch.float64, device='cuda'); G = torch.tensor(gt.reshape(-1, gt.shape[-1]), dtype=torch.float64, device='cuda')
    Wt = torch.linalg.lstsq(Bt.T, G.T)[0].T.clamp(min=0); del G
    return T, Wt, Bt
def snr_db(x, y): return 20*np.log10(np.linalg.norm(x)/max(np.linalg.norm(x-y), 1e-300))
def score(W, H, Wt, Bt, fit_px=262144):
    """Spectral SNR via the gauge that best expresses the true spectra in H's row space; map SNR via the
    gauge fitted in W's column space on a pixel subsample (a 262k x 3 least squares) and applied to all pixels."""
    H = H.double(); th = torch.linalg.lstsq(H.T, Bt.T)[0].T; S = (th @ H).cpu().numpy(); b = Bt.cpu().numpy()
    # strided so the subsample spans the whole image: a contiguous block misses whole materials on this row-wise phantom
    idx = torch.arange(0, W.shape[0], max(1, -(-W.shape[0] // fit_px)), device=W.device)
    A = torch.linalg.lstsq(W[idx].double(), Wt[idx])[0]
    Mfit = torch.cat([W[s:s+65536].double() @ A for s in range(0, W.shape[0], 65536)]).cpu().numpy(); mt = Wt.cpu().numpy()
    return float(np.mean([snr_db(b[i], S[i]) for i in range(NM)])), float(np.mean([snr_db(mt[:, i], Mfit[:, i]) for i in range(NM)]))
def nnal64(W, H, T, chunk=65536):
    Hd = H.double(); return sum(stable_nnal(W[s:s+chunk].double() @ Hd, T[s:s+chunk].double()).item() for s in range(0, T.shape[0], chunk))
def sync(): torch.cuda.synchronize(); return time.perf_counter()
rows = []; out_json = os.path.join(a.out, 'phase3_bias.json')
def emit(**r):
    rows.append(r); json.dump(rows, open(out_json, 'w'), indent=1)
    print(f"  {r['method']:6s} spectra {r['spectra']:6.2f} maps {r['maps']:5.2f} NNAL {r['nnal']:16.1f} {r['sec']:7.1f}s peak {r['peak_GB']:.1f} GB {r.get('notes','')}", flush=True)
print(f'{torch.cuda.get_device_name()}  torch {torch.__version__}', flush=True)
for dose in [float(x) for x in a.doses.split(',')]:
    for side in sides:
      T, Wt, Bt = gen(side, 129, dose); P = side*side
      for R in [int(x) for x in a.ranks.split(',')]:
        print(f'\nP = {P:,}  dose {dose:g}  rank {R}', flush=True)
        base = dict(P=P, dose=dose, rank=R)
        torch.cuda.reset_peak_memory_stats(); t0 = sync(); W, H, it = nnal_factorization(T, method='joint_newton', num_materials=R, max_steps=600, rel_tol=1e-10); tm = sync()-t0
        pk = lambda: torch.cuda.max_memory_allocated()/2**30
        sh, sm = score(W, H, Wt, Bt); emit(**base, method='mle', spectra=sh, maps=sm, nnal=nnal64(W, H, T), sec=tm, peak_GB=pk(), steps=it)
        if 'early' in a.methods:
            torch.cuda.reset_peak_memory_stats(); t0 = sync(); W6, H6, i6 = nnal_factorization(T, method='joint_newton', num_materials=R, max_steps=600, rel_tol=1e-6); t = sync()-t0
            sh, sm = score(W6, H6, Wt, Bt); emit(**base, method='early', spectra=sh, maps=sm, nnal=nnal64(W6, H6, T), sec=t, peak_GB=pk(), steps=i6); del W6, H6
        if 'uncw' in a.methods:
            torch.cuda.reset_peak_memory_stats(); t0 = sync(); Wc, Hu, su = unconstrained_spectra(T, W, H); t = sync()-t0
            sh, sm = score(Wc, Hu, Wt, Bt); emit(**base, method='uncw', spectra=sh, maps=sm, nnal=nnal64(Wc, Hu, T), sec=tm+t, peak_GB=pk(), steps=su, notes=f'+{t/tm:.2f}x MLE'); del Wc, Hu
        if 'supp' in a.methods and R <= 6:
            torch.cuda.reset_peak_memory_stats(); t0 = sync(); Ws, Hs, supp, st = support_selected_spectra(T, W, H, dose); t = sync()-t0
            sh, sm = score(Ws, Hs, Wt, Bt); emit(**base, method='supp', spectra=sh, maps=sm, nnal=nnal64(Ws, Hs, T), sec=tm+t, peak_GB=pk(), steps=st, notes=f'mean |S| {supp.sum(1).double().mean().item():.2f}, +{t/tm:.2f}x MLE'); del Ws, Hs, supp
        if 'boot' in a.methods:
            torch.cuda.reset_peak_memory_stats(); t0 = sync(); Wb, Hb, nb = bias_corrected_spectra(T, W.clone(), H.clone(), dose, correction='bootstrap', n_sim=a.n_sim, max_outer=10, chunk=65536); t = sync()-t0
            sh, sm = score(Wb, Hb, Wt, Bt); emit(**base, method='boot', spectra=sh, maps=sm, nnal=nnal64(Wb, Hb, T), sec=tm+t, peak_GB=pk(), steps=nb, notes=f'{nb} outer, +{t/tm:.1f}x MLE'); del Wb, Hb
        del W, H; torch.cuda.empty_cache()
      del T, Wt; torch.cuda.empty_cache()
print('PHASE3DONE', flush=True)

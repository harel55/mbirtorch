"""Phase 2 on Gautschi: stream_factorization at 1e7-1e8 pixels on tiles generated up front, in parallel.

T is never materialised as one array. Each chunk is a deterministic set of tiles
regenerated from their seeds, cached in host RAM while the total fits under
--cache-gb and otherwise as float32 memmaps on scratch. Generation runs first, in
forked numpy-only workers -- before CUDA is touched -- with the worker count set
by the CPU set and the job's memory limit. Reports per-pass wall clock (tiles
cached, so generation is excluded), the spectral SNR of H, and the relative KKT
residual of H after every pass.
"""
import argparse, os, sys, time, json, resource, numpy as np, torch
os.environ.setdefault('MBIRTORCH_ROOT', os.path.expanduser('~/mbirtorch'))
sys.path.insert(0, os.environ['MBIRTORCH_ROOT'])
from mbirtorch.hsnt import generate_hyper_data, stream_factorization
ap = argparse.ArgumentParser()
ap.add_argument('--pixels', type=int, default=10_000_000); ap.add_argument('--chunk', type=int, default=524_288)
ap.add_argument('--tile', type=int, default=512); ap.add_argument('--passes', type=int, default=4)
ap.add_argument('--dose', type=float, default=3.0); ap.add_argument('--warmup', type=int, default=262144)
ap.add_argument('--out', default='.'); ap.add_argument('--basis', default=None); ap.add_argument('--smoke', action='store_true')
ap.add_argument('--polish-dtype', default=None, choices=[None, 'float32', 'float64'], help='dtype for the polish passes (warm-up stays float32); float64 costs ~2x on H100 and removes the float32 elementwise ceiling on H')
ap.add_argument('--unconstrained-w', action='store_true', help='polish H with the bound on the pixel coefficients dropped (removes the truncation bias of the spectra); W >= 0 re-solved in a final pass')
ap.add_argument('--kkt-tol', type=float, default=None, help='also stop the polish once the relative projected gradient of H (printed after every pass regardless) falls below this')
ap.add_argument('--gen-workers', type=int, default=0, help='processes for tile generation; 0 = as many as the CPU set and the memory limit allow')
ap.add_argument('--keep-tiles', action='store_true', help='keep the scratch tile cache after the run (default: delete it; at 1e8 pixels it is ~480 GB)')
ap.add_argument('--cache-gb', type=float, default=60.0, help='keep generated chunks in host RAM up to this many GB; beyond it, spill to <out>/tiles as memmaps')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if a.smoke: a.pixels, a.chunk, a.tile, a.warmup, a.passes = 16384, 4096, 64, 4096, 2
basis_path = a.basis or os.path.expanduser('~/mbirjax/experiments/hsnt/binaries/material_basis.npy')
basis = np.load(basis_path); D = {"Ni": 0.25, "Cu": 0.25, "Al": 0.75}; R = 3; K = basis.shape[1]
tile_px = a.tile*a.tile; tiles_per_chunk = max(1, a.chunk // tile_px); chunk_px = tiles_per_chunk*tile_px
n_chunks = max(1, a.pixels // chunk_px)

# ---- tile generation, numpy only, before CUDA ---------------------------------------------
GEN_BYTES_PER_ELEMENT = 48   # peak host bytes per (pixel x bin) in one gen_tile call: float64 intermediates inside generate_hyper_data plus the exp and float32 copy (measured locally)

def tile_seed(i, t):
    return 10_000 + i*tiles_per_chunk + t          # unchanged from the serial version: identical data for the same --chunk/--tile

def gen_tile(seed):
    """One float32 tile (tile^2 x K) from its seed. numpy only, so it runs in a forked worker."""
    np.random.seed(seed)
    proj, _, _ = generate_hyper_data(basis, num_angles=1, detector_rows=a.tile, detector_columns=a.tile,
                                     dosage_rate=a.dose, material_density=D, noisy=True, verbose=0)
    return np.exp(-np.nan_to_num(proj)).reshape(-1, proj.shape[-1]).astype(np.float32)

def memory_limit_bytes():
    """Physical RAM, or the cgroup limit Slurm enforces on this job if that is smaller."""
    lim = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    if os.environ.get('SLURM_MEM_PER_NODE', '').isdigit():
        lim = min(lim, int(os.environ['SLURM_MEM_PER_NODE'])*2**20)
    try:
        for line in open('/proc/self/cgroup'):
            parts = line.strip().split(':', 2)
            if len(parts) != 3: continue
            ctrl, path = parts[1], parts[2]
            if ctrl == '': base, fname = '/sys/fs/cgroup', 'memory.max'                          # cgroup v2
            elif 'memory' in ctrl.split(','): base, fname = '/sys/fs/cgroup/memory', 'memory.limit_in_bytes'   # cgroup v1
            else: continue
            while True:                                        # the limit may sit on a parent cgroup
                try:
                    v = open(os.path.join(base + path, fname)).read().strip()
                    if v.isdigit() and int(v) < 2**60: lim = min(lim, int(v))
                except OSError: pass
                if path in ('', '/'): break
                path = os.path.dirname(path)
    except OSError: pass
    return lim

class TileStore:
    """Indexable sequence of CPU chunks, generated once up front."""
    def __init__(self):
        self.ram = {}; self.bytes_per_chunk = chunk_px*K*4
        self.use_ram = n_chunks*self.bytes_per_chunk <= a.cache_gb*2**30
        self.dir = os.path.join(a.out, f'tiles_{a.pixels}_{a.tile}_{a.dose:g}')
        if not self.use_ram: os.makedirs(self.dir, exist_ok=True)
        print(f'tile cache: {"host RAM" if self.use_ram else "scratch memmap at "+self.dir} '
              f'({n_chunks*self.bytes_per_chunk/2**30:.1f} GB total)', flush=True)
    def __len__(self): return n_chunks
    def path(self, i): return os.path.join(self.dir, f'chunk_{i:05d}.npy')
    def has(self, i): return i in self.ram if self.use_ram else os.path.exists(self.path(i))
    def put(self, i, arr):
        if self.use_ram: self.ram[i] = torch.from_numpy(arr)
        else: np.save(self.path(i), arr)
    def __getitem__(self, i):
        return self.ram[i] if self.use_ram else torch.from_numpy(np.load(self.path(i), mmap_mode='r').copy())
    def generate_missing(self):
        import multiprocessing as mp
        todo = [i for i in range(n_chunks) if not self.has(i)]
        if not todo:
            print(f'all {n_chunks} chunks already cached', flush=True); return 0.0, 0
        cpus = len(os.sched_getaffinity(0)); lim = memory_limit_bytes()
        per_worker = (GEN_BYTES_PER_ELEMENT + 4)*tile_px*K                    # generation peak + its result waiting in the parent
        cache = n_chunks*self.bytes_per_chunk if self.use_ram else 0
        budget = lim - cache - 2*self.bytes_per_chunk - 8*2**30                # cache, one chunk being assembled (2x during concat), torch/CUDA margin
        workers = a.gen_workers or int(max(1, min(cpus, budget // per_worker)))
        print(f'generating {len(todo)} chunks ({len(todo)*tiles_per_chunk} tiles of {a.tile}^2) with {workers} workers: '
              f'{cpus} CPUs, memory limit {lim/2**30:.0f} GB, cache {cache/2**30:.0f} GB, ~{per_worker/2**30:.1f} GB per worker', flush=True)
        t0 = time.perf_counter(); seeds = [tile_seed(i, t) for i in todo for t in range(tiles_per_chunk)]
        parts = []; done = 0; report = max(1, len(todo)//10)
        with mp.get_context('fork').Pool(workers) as pool:                    # forked before CUDA is initialised: workers never touch the GPU
            for arr in pool.imap(gen_tile, seeds):                             # ordered, so chunks fill in sequence
                parts.append(arr)
                if len(parts) == tiles_per_chunk:
                    self.put(todo[done], np.concatenate(parts, 0) if len(parts) > 1 else parts[0]); parts = []; done += 1
                    if done % report == 0: print(f'  {done}/{len(todo)} chunks, {time.perf_counter()-t0:.0f} s', flush=True)
        return time.perf_counter()-t0, workers

store = TileStore()
gen_sec, workers = store.generate_missing()
worker_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/2**20
print(f'{n_chunks} chunks x {chunk_px:,} px = {n_chunks*chunk_px:,} pixels, K={K}, dose {a.dose}; '
      f'T would be {n_chunks*chunk_px*K*4/2**30:.0f} GB -- never materialised. '
      f'generation {gen_sec:.0f} s with {workers} workers, worker peak {worker_peak:.1f} GB', flush=True)

# ---- CUDA from here on ---------------------------------------------------------------------
dev = 'cuda'; B = torch.tensor(basis, dtype=torch.float32, device=dev)
print(f'{torch.cuda.get_device_name()} {torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB', flush=True)
def snr(H):
    Hf = H.float(); th = torch.linalg.lstsq(Hf.T, B.T)[0].T; s_ = (th @ Hf).cpu().numpy()
    return float(np.mean([20*np.log10(np.linalg.norm(basis[i])/max(np.linalg.norm(basis[i]-s_[i]), 1e-300)) for i in range(R)]))
pd = getattr(torch, a.polish_dtype) if a.polish_dtype else None
results = []
for passes in (range(0, a.passes+1, max(1, a.passes//2)) if a.passes > 0 else [0]):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); st = {}
    Wc, H, np_ = stream_factorization(store, R, max_passes=passes, rel_tol=1e-8, warmup_pixels=a.warmup, verbose=True,
                                      polish_dtype=pd, kkt_tol=a.kkt_tol, stats=st, nonneg_W=not a.unconstrained_w)
    torch.cuda.synchronize(); el = time.perf_counter()-t0
    r = dict(passes=np_, sec=el, snr=snr(H), loss=st.get('loss'), kkt=st.get('kkt'),
             peak_GB=torch.cuda.max_memory_allocated()/2**30, host_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    kkt_last = r['kkt'][-1] if r['kkt'] else float('nan')
    print(f'passes {np_}: {el:8.1f}s (tiles cached), H spectral SNR {r["snr"]:.2f} dB, KKT residual {kkt_last:.2e}, '
          f'GPU peak {r["peak_GB"]:.1f} GB, host peak {r["host_GB"]:.1f} GB', flush=True)
    results.append(r); del Wc
tag = (a.polish_dtype or 'float32') + ('_uncw' if a.unconstrained_w else '')
json.dump(dict(args=vars(a), generation=dict(sec=gen_sec, workers=workers, worker_peak_GB=worker_peak), results=results),
          open(os.path.join(a.out, f'phase2_stream_{a.pixels}_{tag}.json'), 'w'), indent=1)
if not store.use_ram and not a.keep_tiles:
    import shutil; n_files = len(os.listdir(store.dir)); shutil.rmtree(store.dir, ignore_errors=True)
    print(f'removed scratch tile cache {store.dir} ({n_files} files); pass --keep-tiles to retain it', flush=True)
elif not store.use_ram:
    print(f'kept scratch tile cache at {store.dir} (--keep-tiles)', flush=True)
print('PHASE2DONE', flush=True)

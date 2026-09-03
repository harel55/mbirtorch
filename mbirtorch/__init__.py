"""mbirtorch: model-based iterative reconstruction for computed tomography, in PyTorch.

The package reconstructs volumes from tomographic measurements for several
scanner geometries.  The public API takes numpy arrays and returns numpy
arrays by default; pass ``output_sharded=True`` to get the device tensor
instead.  All available GPUs are used automatically.
"""

__version__ = "0.0.2"

# ── persistent torch.compile cache ────────────────────────────────────────────
# The inductor cache directory defaults to /tmp/torchinductor_<user>, which the
# OS may clean; pin it to a stable per-user location so compiled artifacts
# survive across processes and reboots.  The FX-graph cache is what makes a NEW
# PROCESS reuse prior compilations; enable it explicitly for torch versions
# where it is not the default.  setdefault keeps both overridable per-run via
# the environment.  Both settings take effect only if mbirtorch is imported
# before torch triggers its first compile, which any import-mbirtorch-first
# program satisfies.  Dynamo TRACING still runs per process (the cache skips
# inductor codegen, not tracing), so a cold process keeps a small residual
# warmup.  ``mbirtorch.clear_cache()`` removes the whole ~/.mbirtorch directory
# (see utilities.py).
import os as _os

_os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                       _os.path.expanduser("~/.mbirtorch/torch_cache"))
_os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

from .parallel_beam import ParallelBeamModel, recon_simple_parallel
from .cone_beam import ConeBeamModel, recon_simple_cone
from .translation_model import TranslationModel
from .multiaxis_parallel import MultiAxisParallelModel, MultiAxisParallelBeamModel
from .denoising import QGGMRFDenoiser
from .tomography_model import TomographyModel
from .autograd import (TorchProjector, forward_project_differentiable,
                       back_project_differentiable)
from .vcd_utils import (gen_weights, gen_weights_mar, gen_full_indices,
                        gen_pixel_partition, gen_set_of_pixel_partitions,
                        gen_partition_sequence, get_2d_ror_mask)
from .denoising import median_filter3d
from .qggmrf import (qggmrf_gradient_and_hessian_at_indices, get_b_from_nbr_wts,
                     b_tilde_by_definition, qggmrf_loss)
from .utilities import (generate_3d_shepp_logan_low_dynamic_range, clear_cache,
                        makedirs, load_data_hdf5, save_data_hdf5,
                        export_recon_hdf5, import_recon_hdf5,
                        build_model, download_and_extract,
                        copy_ct_model, stitch_arrays,
                        get_ct_model, generate_demo_data,
                        generate_3d_shepp_logan_reference, gen_cube_phantom,
                        gen_translation_vectors, gen_translation_phantom,
                        get_helical_half_rotation_slice_range,
                        merge_log_files)
from .memory_stats import get_memory_stats

# __all__ is the DECLARED public surface, and autodoc honors it: every name here is
# documented by ``automodule:: mbirtorch :members:``.  It is deliberately narrower than
# the import list above -- the VCD and qGGMRF helpers stay importable as attributes
# (mbirtorch.gen_full_indices still works, and the tests use that spelling) but are not
# promised as public API.  The same goes for gen_cube_phantom and
# get_helical_half_rotation_slice_range: they are importable from the package, and the
# tests call them there, but the docs do not carry them.
__all__ = [
    "ParallelBeamModel", "ConeBeamModel", "TranslationModel",
    "MultiAxisParallelModel", "TomographyModel", "QGGMRFDenoiser",
    "recon_simple_parallel", "recon_simple_cone",
    "TorchProjector", "forward_project_differentiable",
    "back_project_differentiable", "gen_weights", "gen_weights_mar",
    "median_filter3d", "download_and_extract", "build_model",
    "save_data_hdf5", "load_data_hdf5", "export_recon_hdf5",
    "import_recon_hdf5",
    "generate_3d_shepp_logan_low_dynamic_range", "clear_cache",
    "get_memory_stats", "SliceViewer", "VolumeStack", "slice_viewer",
    "stitch_arrays", "get_ct_model", "copy_ct_model",
    "generate_demo_data", "generate_3d_shepp_logan_reference",
    # Documented hsnt and vcls names; these resolve lazily through __getattr__.
    "hyper_denoise", "dehydrate", "rehydrate", "import_hsnt_data_hdf5",
    "create_hsnt_metadata", "export_hsnt_data_hdf5", "generate_hyper_data",
    "get_opt_views", "show_image_with_projection_rays",
]

# ── lazy exports (PEP 562) ───────────────────────────────────────────────────
# The viewer names resolve on first attribute access so that a headless
# `import mbirtorch` never imports matplotlib; most mbirtorch runs (batch
# recons, tests) never open a viewer.  The preprocess, hsnt, and vcls
# modules resolve the same way, so `import mbirtorch` never pays for their
# dependency stacks (preprocess: osqp pulls scipy.sparse, plus cv2 and
# tifffile; hsnt: scikit-learn and h5py, with matplotlib imported lazily inside
# its plotting functions; vcls: the model layer and tqdm).  Both spellings keep
# working -- `mbirtorch.hsnt` resolves here, and
# `import mbirtorch.hsnt` is an ordinary submodule import -- and the
# star-exported FUNCTION names (mbirtorch.dehydrate, mbirtorch.get_opt_views,
# ...) resolve through _LAZY_NAMES, so the public surface is exactly what eager
# star imports would give; only WHEN each module loads changes.
_VIEWER_EXPORTS = ("SliceViewer", "VolumeStack", "slice_viewer")

_LAZY_MODULES = ("preprocess", "hsnt", "vcls")

# The names exposed at package level via `from .hsnt import *` and
# `from .vcls import *`, mapped to their owning module. hsnt is a package with
# an explicit __all__ (mbirtorch/hsnt/__init__.py); this table lists the subset
# promoted to the top level, so a new top-level name gets a line here.
_LAZY_NAMES = {
    'hyper_denoise': 'hsnt', 'dehydrate': 'hsnt', 'rehydrate': 'hsnt',
    'import_hsnt_data_hdf5': 'hsnt', 'create_hsnt_metadata': 'hsnt',
    'export_hsnt_data_hdf5': 'hsnt', 'generate_hyper_data': 'hsnt',
    'subsample_R_gamma': 'vcls', 'max_abs_neighbor_diff': 'vcls',
    'get_opt_views': 'vcls', 'compute_view_basis_functions': 'vcls',
    'compute_cov_matrix': 'vcls', 'compute_vcl': 'vcls',
    'compute_opt_angle_subset': 'vcls', 'get_2d_subsampling_indices': 'vcls',
    'show_image_with_projection_rays': 'vcls', 'reorder_by_priority': 'vcls',
    # The blue-noise pattern (a 382 KB array literal), loaded on first use.
    'bn256': 'bn256',
}


def __getattr__(name):
    import importlib
    if name in _VIEWER_EXPORTS:
        from . import view_utils
        value = getattr(view_utils, name)
        globals()[name] = value  # cache: later accesses skip this hook
        return value
    if name in _LAZY_MODULES:
        value = importlib.import_module('.' + name, __name__)
        globals()[name] = value
        return value
    if name in _LAZY_NAMES:
        module = importlib.import_module('.' + _LAZY_NAMES[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

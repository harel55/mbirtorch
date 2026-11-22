"""This package performs model-based iterative reconstruction (MBIR) for computed tomography.

mbirtorch is written in PyTorch and supports several scanner geometries.  By default, the
public functions take and return numpy arrays.  With ``output_sharded=True``, a function returns
the tensor on the GPUs instead.  mbirtorch uses all available GPUs automatically.
"""

__version__ = "0.2.0"

# These lines put the torch.compile caches under ~/.mbirtorch.  They take effect only if
# mbirtorch is imported before torch compiles anything.
import os as _os

_os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                       _os.path.expanduser("~/.mbirtorch/torch_cache"))
_os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
# Triton keeps a separate cache for its kernels, so this line puts that cache there too.
_os.environ.setdefault("TRITON_CACHE_DIR",
                       _os.path.expanduser("~/.mbirtorch/triton_cache"))

# This call warns when torch uses one CPU thread on a machine with more cores.
from ._host_threads import warn_if_threads_unused as _warn_if_threads_unused
_warn_if_threads_unused()

from typing import TYPE_CHECKING

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
from .utilities import (gen_shepp_logan_3d, clear_cache,
                        makedirs, load_data_hdf5, save_data_hdf5,
                        export_recon_hdf5, import_recon_hdf5,
                        build_model, download_and_extract,
                        copy_ct_model, stitch_arrays, save_volume_as_gif,
                        get_ct_model, gen_demo_data, gen_demo_data_4d,
                        gen_shepp_logan_3d_reference, gen_cube_phantom,
                        gen_translation_vectors,
                        get_helical_half_rotation_slice_range,
                        merge_log_files)
from .memory_stats import get_memory_stats
from .resources import estimate_resources

# autodoc documents exactly the names in __all__.
__all__ = [
    "ParallelBeamModel", "ConeBeamModel", "TranslationModel",
    "MultiAxisParallelModel", "TomographyModel", "QGGMRFDenoiser",
    "recon_simple_parallel", "recon_simple_cone",
    "TorchProjector", "forward_project_differentiable",
    "back_project_differentiable", "gen_weights", "gen_weights_mar",
    "median_filter3d", "download_and_extract", "build_model",
    "save_data_hdf5", "load_data_hdf5", "export_recon_hdf5",
    "import_recon_hdf5",
    "gen_shepp_logan_3d", "clear_cache",
    "get_memory_stats", "estimate_resources", "SliceViewer", "VolumeStack", "slice_viewer",
    "GeometryScene", "GeometryFigure", "geometry_viewer",
    "SliceViewer4D", "VolumeStack4D", "slice_viewer4d",
    "stitch_arrays", "get_ct_model", "copy_ct_model", "save_volume_as_gif",
    "MACE4DModel", "temporal_filter_matrix", "apply_temporal_filter",
    "gen_demo_data", "gen_demo_data_4d", "gen_shepp_logan_3d_reference",
    # Python imports these names on first access through __getattr__.
    "hyper_denoise", "dehydrate", "rehydrate", "import_hsnt_data_hdf5",
    "export_hsnt_data_hdf5", "generate_hyper_data",
    "synthetic_material_basis", "gen_material_phantom", "generate_hyper_sinogram",
    "get_opt_views", "show_image_with_projection_rays",
]

# Python imports these names on first access (PEP 562).  As a result, importing mbirtorch
# does not import matplotlib or the dependencies of preprocess, hsnt, and vcls.
_VIEWER_EXPORTS = ("SliceViewer", "VolumeStack", "slice_viewer",
                   "GeometryScene", "GeometryFigure", "geometry_viewer",
                   "SliceViewer4D", "VolumeStack4D", "slice_viewer4d")

_LAZY_MODULES = ("preprocess", "hsnt", "vcls", "mace")

# This dict maps each package-level name to the module that defines it.  A new public
# function in one of these modules needs a line here.
_LAZY_NAMES = {
    'hyper_denoise': 'hsnt', 'dehydrate': 'hsnt', 'rehydrate': 'hsnt',
    'import_hsnt_data_hdf5': 'hsnt',
    'export_hsnt_data_hdf5': 'hsnt', 'generate_hyper_data': 'hsnt',
    'synthetic_material_basis': 'hsnt', 'gen_material_phantom': 'hsnt',
    'generate_hyper_sinogram': 'hsnt',
    'subsample_R_gamma': 'vcls', 'max_abs_neighbor_diff': 'vcls',
    'get_opt_views': 'vcls', 'compute_view_basis_functions': 'vcls',
    'compute_cov_matrix': 'vcls', 'compute_vcl': 'vcls',
    'compute_opt_angle_subset': 'vcls', 'get_2d_subsampling_indices': 'vcls',
    'show_image_with_projection_rays': 'vcls', 'reorder_by_priority': 'vcls',
    # The blue noise pattern is a 382 KB array that is loaded on first use.
    'bn256': 'bn256',
    # The function mace() is available as mbirtorch.mace.mace, not at the package level.
    'MACE': 'mace', 'Task': 'mace', 'ForwardProxAgent': 'mace',
    'QGGMRFDenoiserAgent': 'mace', 'HyperplaneAgent': 'mace',
    'resolve_device_pool': 'mace',
    'MACE4DModel': 'mace4d',
    'temporal_filter_matrix': 'mace4d', 'apply_temporal_filter': 'mace4d',
}

# These imports are for editors and type checkers only.  tests/test_lazy_exports.py checks
# them.
if TYPE_CHECKING:
    from . import preprocess, hsnt, vcls, mace
    from .view_utils import (SliceViewer, VolumeStack, slice_viewer,
                             GeometryScene, GeometryFigure, geometry_viewer,
                             SliceViewer4D, VolumeStack4D, slice_viewer4d)
    from .mace import (MACE, Task, ForwardProxAgent, QGGMRFDenoiserAgent, HyperplaneAgent,
                       resolve_device_pool)
    from .mace4d import MACE4DModel, temporal_filter_matrix, apply_temporal_filter
    from .hsnt import (hyper_denoise, dehydrate, rehydrate,
                       import_hsnt_data_hdf5,
                       export_hsnt_data_hdf5, generate_hyper_data,
                       synthetic_material_basis, gen_material_phantom,
                       generate_hyper_sinogram)
    from .vcls import (subsample_R_gamma, max_abs_neighbor_diff, get_opt_views,
                       compute_view_basis_functions, compute_cov_matrix,
                       compute_vcl, compute_opt_angle_subset,
                       get_2d_subsampling_indices,
                       show_image_with_projection_rays, reorder_by_priority)
    from .bn256 import bn256


def __getattr__(name):
    import importlib
    if name in _VIEWER_EXPORTS:
        from . import view_utils
        value = getattr(view_utils, name)
        globals()[name] = value  # Later accesses do not call __getattr__.
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

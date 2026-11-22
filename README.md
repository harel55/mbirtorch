# mbirtorch

[![CI](https://github.com/cabouman/mbirtorch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cabouman/mbirtorch/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/mbirtorch/badge/?version=latest)](https://mbirtorch.readthedocs.io/en/latest/)

MBIRTorch: Model-Based Iterative Reconstruction (MBIR) for tomographic reconstruction using [PyTorch](https://pytorch.org/). 

Features include:
* Multiple geometries:  parallel beam, cone beam (including curved detector and helical), translation mode, and multi-axis parallel. 
* 4D reconstruction: a time sequence of volumes from a single continuous scan, using multi-agent consensus equilibrium (MACE).
* Preprocessing routines for NSI and Zeiss scanners, plus geometry calibration.
* Utilities for metal artifact reduction and stripe removal.  
* Hyperspectral neutron data: maximum-likelihood dehydration into component maps and spectra, denoising line ([documentation](https://mbirtorch.readthedocs.io/en/latest/usr_hsnt.html)).
* Interactive slice and geometry viewers.
* Informative demos and extensive documentation. 
* Seamless operation on 1 or more GPUs, Mac MPS, or CPU. 
* Compiled torch and Triton kernels for efficiency.

Available on PyPI via 
```bash
pip install mbirtorch
```

Reconstruct in one line:
```python
import mbirtorch
recon, recon_dict = mbirtorch.recon_simple_parallel(sinogram, angles)
```

Full documentation at [https://mbirtorch.readthedocs.io/](https://mbirtorch.readthedocs.io/)

## Citation

Please use the following BibTeX citation when referencing this software.

```bibtex
@misc{mbirtorch,
  title = {{MBIRTorch}: {H}igh-performance tomographic reconstruction using {PyTorch}},
  author = {Gregery T. Buzzard and Charles A. Bouman and Jingsong Lin and Ziyun Li},
  howpublished = {Software library available from \url{https://github.com/cabouman/mbirtorch}},
  note = {Version 0.2.0},
  year = 2026
}
```

GitHub's "Cite this repository" button on the repository page generates this
citation from `CITATION.cff`.

mbirtorch is a PyTorch port of [MBIRJAX](https://github.com/cabouman/mbirjax);
please also cite it when referencing the underlying methods.

"""Experimental: strain tomography with the hybrid Bragg-edge model.

The modules fit hyperspectral neutron transmission data from one or more rotation series with a physical model of
each material's Bragg-edge spectrum, and reconstruct density and strain-tensor volumes.  The API is experimental and
may change.

- ``bragg``: Bragg-edge spectra, reflections and the warm start that finds lattice parameters.
- ``model``: the hybrid spectral model (``Hybrid2``) and its per-ray fit (``FitB``).
- ``geometry``: beam directions and strain-component weights of the views.
- ``tomo``: projectors for one or several rotation series and the tomographic fit (``TomoFit``, ``warmup``).

The package holds no simulation code: the strain phantoms live with the experiment drivers.
"""

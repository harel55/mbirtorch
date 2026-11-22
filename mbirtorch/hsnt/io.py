"""This module reads and writes the hsnt HDF5 file.

A file holds either hyperspectral data or dehydrated data.  The root of the file carries these
attributes: ``format`` ("mbirtorch-hsnt"), ``format_version`` (1), ``mbirtorch_version``,
``written`` (the date and time), ``kind`` ("hyperspectral" or "dehydrated"), ``dataset_type``
("attenuation" or "transmission", hyperspectral only), ``domain`` ("sinogram" or "recon"), ``name``
(optional), and ``model`` (optional, the parameters of the tomography model as JSON, from which
``mbirtorch.build_model`` rebuilds the model).  A hyperspectral
file holds the float32 dataset ``data`` with the spectral axis last.  A dehydrated file holds the
float32 datasets ``maps``, with the materials on the last axis, and ``spectra``, with shape
(materials, bins).  Either file may hold the dataset ``wavelengths``, in angstroms, which carries a
``units`` attribute.
"""
import datetime
import json

import h5py
import numpy as np

FORMAT = "mbirtorch-hsnt"
FORMAT_VERSION = 1
DATASET_TYPES = ("attenuation", "transmission")
DOMAINS = ("sinogram", "recon")
# These are the units of the optional arrays.  The reader refuses any other unit.
UNITS = {"wavelengths": "angstrom"}


def export_hsnt_data_hdf5(filename, data, dataset_type=None, *, domain="sinogram", name=None, wavelengths=None,
                          model=None):
    """This function writes hyperspectral data or dehydrated data to an hsnt HDF5 file.
    The spectral axis of hyperspectral data is always the last axis.

    Args:
        filename (str): The path of the file to write.
        data (numpy.ndarray or list): Hyperspectral data as an array with the spectral axis last,
            or dehydrated data as the list ``[maps, spectra]`` that ``dehydrate`` returns.
        dataset_type (str, optional): 'attenuation' or 'transmission'.  Required for
            hyperspectral data.  Not allowed for dehydrated data, which is always attenuation.
        domain (str, optional): 'sinogram' when the pixel axes are (views, rows, cols), or
            'recon' when they are the axes of a reconstructed volume.  Defaults to 'sinogram'.
        name (str, optional): A name for the dataset.  Defaults to None.
        wavelengths (array, optional): The wavelength of each bin, in angstroms.  Defaults to None.
        model (TomographyModel, optional): The model of the scan.  Its parameters are stored,
            and ``import_hsnt_data_hdf5`` rebuilds the model from them.  When given, the pixel
            axes of the data must match the model's sinogram shape or reconstruction shape, as
            ``domain`` says.  Defaults to None.
    """
    from .. import __version__
    from ..utilities import _json_default
    dehydrated = isinstance(data, (list, tuple))
    if dehydrated:
        if len(data) != 2:
            raise ValueError(f"dehydrated data is the list [maps, spectra]; got {len(data)} elements")
        if dataset_type is not None:
            raise ValueError("dehydrated data is always attenuation; do not pass dataset_type")
        maps, spectra = (np.asarray(a, dtype=np.float32) for a in data)
        if spectra.ndim != 2 or maps.ndim < 2 or maps.shape[-1] != spectra.shape[0]:
            raise ValueError(f"maps must have shape (..., materials) and spectra (materials, bins); got "
                             f"{maps.shape} and {spectra.shape}")
    else:
        if dataset_type not in DATASET_TYPES:
            raise ValueError(f"dataset_type must be one of {DATASET_TYPES}; got {dataset_type!r}")
        data = np.asarray(data, dtype=np.float32)
        if data.ndim < 2:
            raise ValueError(f"hyperspectral data needs at least two axes, with the bins last; got shape {data.shape}")

    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}; got {domain!r}")
    if model is not None:
        pixel_axes = tuple((maps if dehydrated else data).shape[:-1])
        expected = tuple(model.get_params("sinogram_shape" if domain == "sinogram" else "recon_shape"))
        if pixel_axes != expected:
            raise ValueError(f"the pixel axes of the data are {pixel_axes}, but the model's {domain} shape is "
                             f"{expected}")

    with h5py.File(filename, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["mbirtorch_version"] = __version__
        f.attrs["written"] = datetime.datetime.now().isoformat(timespec="seconds")
        f.attrs["kind"] = "dehydrated" if dehydrated else "hyperspectral"
        f.attrs["domain"] = domain
        if dehydrated:
            f.create_dataset("maps", data=maps)
            f.create_dataset("spectra", data=spectra)
        else:
            f.attrs["dataset_type"] = dataset_type
            f.create_dataset("data", data=data)
        if name is not None:
            f.attrs["name"] = str(name)
        if model is not None:
            f.attrs["model"] = json.dumps(model.get_all_params(), default=_json_default)
        for key, value in (("wavelengths", wavelengths),):
            if value is not None:
                d = f.create_dataset(key, data=np.asarray(value, dtype=np.float64))
                d.attrs["units"] = UNITS[key]


def import_hsnt_data_hdf5(filename):
    """This function reads an hsnt HDF5 file written by ``export_hsnt_data_hdf5``.
    The spectral axis of hyperspectral data is always the last axis.

    Args:
        filename (str): The path of the file to read.

    Returns:
        list: The list ``[data, metadata]``.  For a hyperspectral file, ``data`` is the array.
            For a dehydrated file, ``data`` is the list ``[maps, spectra]``.  ``metadata`` is a
            dict with the keys ``dataset_type`` (None for a dehydrated file), ``domain``, ``name``,
            ``wavelengths``, ``model``, ``mbirtorch_version``, and ``written``.
            ``model`` is the tomography model of the scan, rebuilt from the file.  A key that
            the file does not hold has the value None.
    """
    with h5py.File(filename, "r") as f:
        if f.attrs.get("format") != FORMAT:
            raise ValueError(f"{filename} is not an hsnt file: the root has no 'format' attribute of "
                             f"'{FORMAT}'.  Files written before version 1 of the format are not supported.")
        version = int(f.attrs["format_version"])
        if version > FORMAT_VERSION:
            raise ValueError(f"{filename} has format version {version}, written by mbirtorch "
                             f"{f.attrs.get('mbirtorch_version')}; this mbirtorch reads versions up to "
                             f"{FORMAT_VERSION}")
        kind = f.attrs["kind"]
        if kind == "dehydrated":
            data = [f["maps"][()], f["spectra"][()]]
        elif kind == "hyperspectral":
            data = f["data"][()]
        else:
            raise ValueError(f"{filename} has an unknown kind {kind!r}")
        metadata = {key: f.attrs.get(key) for key in ("dataset_type", "domain", "name", "mbirtorch_version",
                                                      "written")}
        for key, units in UNITS.items():
            metadata[key] = None
            if key in f:
                found = f[key].attrs.get("units")
                if found != units:
                    raise ValueError(f"{filename}: {key} has units {found!r}; expected {units!r}")
                metadata[key] = f[key][()]
        metadata["model"] = None
        if "model" in f.attrs:
            from ..utilities import _json_object_hook, build_model
            metadata["model"] = build_model(*json.loads(f.attrs["model"], object_hook=_json_object_hook))
    return [data, metadata]

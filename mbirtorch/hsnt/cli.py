"""This module is the command line program ``mbirtorch-hsnt``, which runs the hsnt dehydration.
The program also runs as ``python -m mbirtorch.hsnt``.

    inspect    load a dataset, run the checks, and print what a solve would see (no GPU needed)
    convert    write a TIFF stack (with its open beam) or an HDF5 dataset in the hsnt HDF5 layout
    dehydrate  fit X = W H and write W and H in the dehydrated layout, with plots and a JSON report
    rehydrate  multiply a dehydrated file back into hyperspectral data (all bins or a range)
    denoise    dehydrate and rehydrate in one run

Dehydration fits the data X as the product W H of a few spectra H and their maps W.  The number
of spectra is the rank.  The program estimates the rank by likelihood ratio tests unless the user
gives ``--rank``.  An input is either an HDF5 file in the hsnt layout or a directory with one
TIFF image per wavelength bin.  A TIFF stack of counts needs ``--open-beam``.  Any subcommand
with ``-h`` shows the common options.  With ``--help-all``, it shows every option.
"""
import argparse
import json
import logging
import os
import re
import sys
import warnings

import numpy as np

from ..utilities import makedirs
from .io import _written_atomically
from .loading import (INPUT_TYPES, InputError, _columns_in_range, _converted_path, _file_source_bins,
                      _selected_metadata, convert_to_hdf5, load_dataset)
from .outputs import component_check, fit_quality, write_dehydrated, write_denoised
from .rank import _estimate_rank

log = logging.getLogger("mbirtorch.hsnt")

_MAX_PLOTTED_VIEWS = 4


def _parse_slice(text, name):
    """This function parses the text 'START:STOP' into the tuple (start, stop).  Each value is an
    int, or None when its part of the text is empty."""
    if text is None:
        return None
    m = re.fullmatch(r"(-?\d*):(-?\d*)", text.strip())
    if not m:
        raise InputError(f"--{name} expects START:STOP (Python slice), got {text!r}")
    return (int(m.group(1)) if m.group(1) else None, int(m.group(2)) if m.group(2) else None)


def _box_arg(text):
    """This function parses one word of --background-boxes.  The word is either a preset name or
    a box 'Y0:Y1,X0:X1' in pixels of the full resolution image."""
    if re.fullmatch(r"[a-z][a-z0-9-]*", text):
        return text
    m = re.fullmatch(r"(\d+):(\d+),(\d+):(\d+)", text.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"expected a box Y0:Y1,X0:X1 or a preset name, got {text!r}")
    return tuple(int(v) for v in m.groups())


def _background_spec(words):
    """This function returns the background_boxes argument from the words given on the command
    line."""
    if not words:
        return None
    names = [w for w in words if isinstance(w, str)]
    if names and len(words) > 1:
        raise InputError(f"--background-boxes takes one preset name ({names[0]}) or boxes, not both")
    return names[0] if names else list(words)


def _tiles_arg(text):
    m = re.fullmatch(r"(\d+)x(\d+)", text.strip().lower())
    if not m or int(m.group(1)) < 1 or int(m.group(2)) < 1:
        raise argparse.ArgumentTypeError(f"expected ROWSxCOLS, e.g. 2x2, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def _rank_arg(text):
    if text.lower() == "auto":
        return "auto"
    if not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer or 'auto', got {text!r}")
    return int(text)


def _number_arg(kind, low, inclusive):
    """This function returns an argparse type that accepts a number greater than ``low``.  If
    ``inclusive`` is True, the type also accepts ``low``."""
    def parse(text):
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected {'an integer' if kind is int else 'a number'}, got "
                                             f"{text!r}") from None
        if not (value >= low if inclusive else value > low):
            raise argparse.ArgumentTypeError(f"expected a value {'>=' if inclusive else '>'} {low}, got {text!r}")
        return value
    return parse


_positive_int = _number_arg(int, 0, inclusive=False)
_nonneg_int = _number_arg(int, 0, inclusive=True)
_positive_float = _number_arg(float, 0.0, inclusive=False)
_nonneg_float = _number_arg(float, 0.0, inclusive=True)


def _dose_arg(text):
    """This function parses --dose.  It returns a positive number, or the path of a file that
    holds the open-beam count of each bin."""
    try:
        float(text)
    except ValueError:
        if not os.path.isfile(text):
            raise argparse.ArgumentTypeError(f"expected a positive number or a file of per-bin counts, got {text!r}")
        return text
    return _positive_float(text)


def _read_dose_file(path):
    """This function reads the open-beam count of each bin from a .npy file or a text file.  It
    returns a 1-D float64 array."""
    try:
        d = np.load(path, allow_pickle=False) if path.lower().endswith(".npy") else np.loadtxt(path, ndmin=1)
    except (OSError, ValueError) as e:
        raise InputError(f"--dose {path}: cannot read a per-bin dose ({e})") from None
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 1:
        raise InputError(f"--dose {path}: expected one value per bin, got an array of shape {d.shape}")
    return d


def _pool_arg(text):
    if text.lower() == "auto":
        return "auto"
    if not text.isdigit():
        raise argparse.ArgumentTypeError(f"expected 'auto' or a block size (0 disables), got {text!r}")
    return int(text)


def _load(args, log_checks=True):
    """This function runs load_dataset with the options of the command.  If ``log_checks`` is
    True, the function logs the results of the data checks."""
    dose = _read_dose_file(args.dose) if isinstance(args.dose, str) else args.dose
    ds = load_dataset(args.input, open_beam=args.open_beam, input_type=args.input_type, dataset=args.dataset,
                      dose=dose, views=_parse_slice(args.views, "views"),
                      wave_range=_parse_slice(args.wave_range, "wave-range"), wave_bin=args.wave_bin,
                      downsample=args.downsample,
                      background_boxes=_background_spec(args.background_boxes),
                      background_tiles=args.background_tiles, open_beam_smoothing=args.open_beam_smoothing)
    log.info("loaded in %.1f s", ds.info["load_seconds"])
    for c in ds.checks if log_checks else []:
        getattr(log, {"ok": "info", "warn": "warning", "error": "error"}[c.level])("check: %s", c.message)
    return ds


def _device(name):
    """This function checks a device name and returns the device to use.  The name is 'auto',
    'cpu', 'cuda', or 'cuda:N'."""
    import torch
    from ._device import _default_device
    m = re.fullmatch(r"auto|cpu|cuda(?::(\d+))?", name)
    if not m:
        raise InputError(f"--device {name}: expected auto, cpu, cuda or cuda:N")
    if name.startswith("cuda") and int(m.group(1) or 0) >= torch.cuda.device_count():
        raise InputError(f"--device {name}: {torch.cuda.device_count()} CUDA device(s) available")
    name = _default_device(name)
    if name == "cpu":
        log.warning("running on the CPU: a large solve takes much longer than on a GPU")
    return name


def plan_memory(ds, device, mode, chunk_pixels, spectra="mle", compile_mode="auto"):
    """This function chooses between a full solve and a streamed solve for the loaded data on
    `device`.  The function returns (mode, chunk_pixels, warmup_pixels, note)."""
    from ._fit import _plan
    return _plan(ds.pixels, ds.bins, device, spectra, mode, chunk_pixels, compile_mode)


def _penalty_arg(text):
    if text.lower() == "auto":
        return "auto"
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto' or a multiple of log(bins), got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"the penalty must be nonnegative, got {text!r}")
    return value


def _load_basis(path, ds):
    """This function reads the subspace_basis of a dehydrated file.  The function checks that the
    basis matches the loaded data, and then returns it."""
    import h5py
    from .denoise import _check_basis
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--basis: file not found: {path}")
    with h5py.File(path, "r") as f:
        if not all(k in f for k in ("subspace_data", "subspace_basis", "dataset_type")):
            raise InputError(f"--basis {path}: not a dehydrated file (needs subspace_data, subspace_basis and "
                             "dataset_type)")
        H = f["subspace_basis"][()]
        has_bins = "bin_indices" in f
        bins = _file_source_bins(f, H.shape[-1]) if H.ndim == 2 else None
        background = f.attrs.get("background")
    if has_bins and bins is not None and not np.array_equal(bins, ds.bin_indices):
        raise InputError(f"--basis {path}: its spectra cover source bins {bins[0]}..{bins[-1]} ({len(bins)}), the "
                         f"data {ds.bin_indices[0]}..{ds.bin_indices[-1]} ({ds.bins}); load the data with the same "
                         "--wave-range and --wave-bin")
    try:
        H = _check_basis(H, ds.bins, None, "mle")
    except ValueError as e:
        raise InputError(f"--basis {path}: {e}") from None
    if not has_bins:
        warnings.warn(f"--basis {path} records no source bins; only its number of bins ({H.shape[1]}) is checked")

    def calibration(bg):
        return None if bg is None else (bg["name"], [list(b) for b in bg["boxes"]], list(bg["tiles"]))

    fitted = calibration(json.loads(background)) if background is not None else None
    here = calibration(ds.info.get("background"))
    if fitted != here:
        warnings.warn(f"--basis {path} was fitted on data with another background calibration (there: "
                      f"{fitted[0] if fitted else 'none'}; here: {here[0] if here else 'none'}): a per-bin factor "
                      "between the two biases the maps")
    return H


def solve(ds, args, device):
    """This function runs the factorization and estimates the spectra.  The function returns the
    numpy arrays W and H and a report."""
    from ._fit import _fit, _fit_fixed_basis
    args.fit_report = {}                                # main uses this memory plan for advice if the solve fails
    if args.basis:
        W, H, rep = _fit_fixed_basis(ds.T, args.basis_H, device=device, mode=args.mode, chunk_pixels=args.chunk_pixels,
                                     max_steps=args.max_steps, rel_tol=args.rel_tol, compile_mode=args.compile,
                                     report=args.fit_report, weights=ds.bin_weights)
        rep.update(rank=args.rank_value, rank_note=args.rank_note, rank_search=None, basis=os.path.abspath(args.basis))
        args.fit_report = {"solved": True}
        return W, H, rep
    if args.spectra == "support" and ds.dose is None:
        raise InputError("support selection needs the dose (open-beam counts per pixel and bin): pass --dose, or "
                         "give --open-beam with a TIFF stack of counts")
    if args.rank_value > min(ds.pixels, ds.bins):
        raise InputError(f"--rank {args.rank_value} exceeds min(pixels, bins) = {min(ds.pixels, ds.bins)}")
    W, H, rep = _fit(ds.T, args.rank_value, spectra=args.spectra, dose=ds.dose, penalty=args.support_penalty,
                     wald_screen=args.wald_screen, device=device, mode=args.mode,
                     chunk_pixels=args.chunk_pixels, max_steps=args.max_steps, rel_tol=args.rel_tol,
                     max_passes=args.max_passes, compile_mode=args.compile, report=args.fit_report,
                     weights=ds.bin_weights)
    rep.update(rank=args.rank_value, rank_note=args.rank_note, rank_search=args.rank_detail)
    args.fit_report = {"solved": True}                  # a later device failure happens after the solve
    return W, H, rep


def _output_path(output, default_name):
    """This function returns the output path.  If -o ends in .h5 or .hdf5, the path is -o.
    Otherwise the path is ``default_name`` in the -o directory."""
    if output is not None and output.lower().endswith((".h5", ".hdf5")):
        path = output
    else:
        path = os.path.join(output or ".", default_name)
    makedirs(path)
    return path


def _check_outputs(paths, inputs, overwrite):
    """This function raises FileExistsError if an output is also an input.  It also raises
    FileExistsError if an output exists and --overwrite is not given."""
    existing = [p for p in paths if os.path.exists(p)]
    for p in existing:
        if any(i and os.path.exists(i) and os.path.samefile(p, i) for i in inputs):
            raise FileExistsError(f"the output {p} is an input of this run; choose another output")
    if existing and not overwrite:
        raise FileExistsError(f"{', '.join(existing)} already exist{'s' if len(existing) == 1 else ''}; pass "
                              "--overwrite to replace")


def _out_type(ds, args):
    if args.as_type:
        return args.as_type
    return "attenuation" if ds.dataset_type in ("counts", "attenuation") else "transmission"


def _run_attrs(ds, rep, args, **extra):
    """This function returns the HDF5 attributes that record the data source and the settings."""
    return dict(source=ds.source, input_type=ds.dataset_type, mode=rep["mode"],
                spectra="given basis" if args.basis else args.spectra, support_penalty=str(args.support_penalty),
                downsample=args.downsample, wave_bin=args.wave_bin, dose=-1.0 if ds.dose is None else float(ds.dose),
                mbirtorch_hsnt_cli="1", **({"basis": os.path.abspath(args.basis)} if args.basis else {}),
                **({"background": json.dumps(ds.info["background"])} if "background" in ds.info else {}),
                **({"open_beam_smoothing": json.dumps(ds.info["open_beam_smoothing"])}
                   if "open_beam_smoothing" in ds.info else {}),
                **extra)


def write_report(base, ds, rep, args, outputs):
    path = base + "_report.json"
    report = dict(input=ds.source, input_type=ds.dataset_type, spatial_shape=list(ds.spatial_shape), pixels=ds.pixels,
                  bins=ds.bins, dose=ds.dose,
                  args={k: v for k, v in vars(args).items() if k not in ("func", "basis_H")},
                  checks=[dict(level=c.level, message=c.message) for c in ds.checks], info=ds.info, result=rep,
                  outputs=outputs)
    with _written_atomically(path) as tmp, open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    log.info("wrote %s", path)
    return report


def _log_fit(rep, args):
    comp = rep["components"]
    if comp["proportional_pairs"]:
        i, j, cos = max(comp["proportional_pairs"], key=lambda p: p[2])
        log.warning("components %d and %d have nearly proportional spectra (cosine %.4f; %d such pair(s)): the data "
                    "do not fix how that spectral shape is split between their maps, so a smaller rank describes the "
                    "data as well%s. The mean-pixel spectrum in the outputs is unaffected.", i, j, cos,
                    len(comp["proportional_pairs"]), "; refit the basis at a smaller rank" if args.basis else
                    "" if args.rank_detail else "; run without --rank to estimate it")
    log.info("components: max spectral cosine %.3f, max map correlation %.2f; per-row noise (relative to level) %s",
             comp["max_spectral_cosine"], comp["max_map_correlation"],
             ", ".join(f"{x:.3f}" for x in comp["row_noise_rel"]))
    q = rep["fit"]
    if "reduced_chi2" not in q:
        log.info("fit: relative residual in transmission %.4g (no dose, so no chi-square)", q["relative_residual"])
        return
    chi2 = q["reduced_chi2"]
    if 0.8 <= chi2 <= 1.3:
        verdict = "at the Poisson noise level"
    elif chi2 > 1.3:
        verdict = "above the noise level: the rank may be too small, the model misspecified or the dose overestimated"
    else:
        verdict = "below the noise level: the dose is underestimated, or the noise model overstates the variance"
    log.info("fit: reduced chi-square %.3f (%s); relative residual in transmission %.4g", chi2, verdict,
             q["relative_residual"])
    if chi2 > 2.0:
        log.warning("reduced chi-square %.2f is far from 1; check the rank (%s) and the dose", chi2, args.rank_note)
    elif chi2 < 0.5:
        log.warning("reduced chi-square %.2f is far from 1; check the dose, to which it is proportional (a component "
                    "more lowers it by only about 1 / bins)", chi2)


def cmd_inspect(args):
    import torch
    ds = _load(args, log_checks=False)
    st = ds.info["T_stats"]
    V, rows, cols = ds.spatial_shape
    print(f"\n{ds.source}\n  type {ds.dataset_type}; {V} view(s) x {rows} x {cols} pixels x {ds.bins} bins (source "
          f"bins {ds.bin_indices[0]}..{ds.bin_indices[-1]}); {ds.T.nbytes / 2**30:.2f} GiB as float32")
    print(f"  T: min {st['min']:.4g}  median {st['median']:.4g}  mean {st['mean']:.4g}  max {st['max']:.4g}"
          + (" (sampled)" if st["sampled"] else "") + f"; zeros {st['zero']:.2%}, above 1: {ds.info['above_one']:.2%}")
    print(f"  dose: {'unknown' if ds.dose is None else f'{ds.dose:.4g} counts per pixel and bin'}"
          + ("" if ds.bin_weights is None else f" (the mean of the per-bin dose of --dose {args.dose})"))
    for c in ds.checks:
        print(f"  [{c.level:5s}] {c.message}")
    if args.estimate_rank:
        _, note, d = _estimate_rank(ds.T, ds.spatial_shape, _device(args.device), max_rank=args.max_rank,
                                    pool=args.rank_pool, weights=ds.bin_weights)
        gains = ", ".join(f"{r}: {g:,.0f}" for r, g in zip(range(2, d["max_rank"] + 1), d["gains"]))
        print(f"  {note}; effective dose {d['effective_dose']:.3g}; gains by component (deciding test): {gains}; "
              f"threshold {d['threshold']:,.0f}")
    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        plan_memory(ds, dev, "auto", None)
    print()
    return 0


def cmd_convert(args):
    if isinstance(args.dose, str):
        raise InputError(f"--dose {args.dose}: convert records a scalar dose only; give the per-bin dose to "
                         "dehydrate or denoise")
    out = args.output or _converted_path(args.input)
    _check_outputs([out], [args.input, *(args.open_beam or [])], args.overwrite)
    out, checks, _ = convert_to_hdf5(args.input, output=out, open_beam=args.open_beam,
                                     input_type=args.input_type, dataset=args.dataset, dose=args.dose,
                                     views=_parse_slice(args.views, "views"),
                                     wave_range=_parse_slice(args.wave_range, "wave-range"), wave_bin=args.wave_bin,
                                     downsample=args.downsample, as_type=args.as_type,
                                     memory_budget_mib=args.memory_budget, workers=args.workers,
                                     progress=not args.quiet,
                                     background_boxes=_background_spec(args.background_boxes),
                                     background_tiles=args.background_tiles,
                                     open_beam_smoothing=args.open_beam_smoothing)
    for c in checks:
        getattr(log, {"ok": "info", "warn": "warning", "error": "error"}[c.level])("check: %s", c.message)
    n_err = sum(c.level == "error" for c in checks)
    print(out + (f"  ({n_err} data check(s) had errors)" if n_err else ""))
    return 0


def _resolve_rank(ds, args, device):
    """This function sets args.rank_value, args.rank_note, and args.rank_detail.  The rank comes
    from --basis, from --rank N, or from the likelihood ratio estimate."""
    if args.basis:
        args.basis_H = _load_basis(args.basis, ds)
        R = args.basis_H.shape[0]
        if args.rank not in ("auto", R):
            raise InputError(f"--rank {args.rank} disagrees with the {R} spectra of --basis {args.basis}")
        if args.spectra != "mle":
            raise InputError(f"--spectra {args.spectra} re-estimates the spectra, which --basis holds fixed")
        args.rank_value, args.rank_note, args.rank_detail = R, f"rank {R} from the basis in {args.basis}", None
    elif args.rank == "auto":
        args.rank_value, args.rank_note, args.rank_detail = _estimate_rank(ds.T, ds.spatial_shape, device,
                                                                           max_rank=args.max_rank, pool=args.rank_pool,
                                                                           weights=ds.bin_weights)
        log.info("%s; pass --rank N to override", args.rank_note)
    else:
        args.rank_value, args.rank_note, args.rank_detail = args.rank, f"rank {args.rank} given", None


def _pipeline(args, denoise):
    """This function runs the dehydrate or denoise command.  The function loads the data, solves,
    and writes the outputs."""
    ds = _load(args)
    device = _device(args.device)
    stem = os.path.splitext(os.path.basename(os.path.normpath(args.input)))[0]
    base = os.path.splitext(_output_path(args.output, stem + ".h5"))[0]
    names = (([] if denoise and args.no_dehydrated else ["_dehydrated.h5"]) + (["_denoised.h5"] if denoise else [])
             + ["_report.json"] + ([] if args.no_plots else ["_spectra.png", "_maps.png"]))
    _check_outputs([base + n for n in names], [args.input, args.basis, *(args.open_beam or [])], args.overwrite)
    _resolve_rank(ds, args, device)
    if args.dry_run:
        plan_memory(ds, device, args.mode, args.chunk_pixels, "basis" if args.basis else args.spectra, args.compile)
        print(f"dry run: data loaded and checked, {args.rank_note}; no solve. Output base: {base}")
        return 0
    W, H, rep = solve(ds, args, device)
    R, out_type, outputs = H.shape[0], _out_type(ds, args), []
    if not (denoise and args.no_dehydrated):          # the result is written before any diagnostic runs
        outputs.append(write_dehydrated(base + "_dehydrated.h5", W.reshape(*ds.spatial_shape, R), H, out_type,
                                        ds.bin_indices, _run_attrs(ds, rep, args, rank=R, loss=rep["loss_final"]),
                                        ds.metadata))
    per_bin = ds.dose_per_bin if ds.bin_weights is None else ds.dose * ds.bin_weights
    rep["fit"] = fit_quality(ds.T, W, H, ds.dose, device, per_bin, ds.open_beam_observations)
    rep["components"] = component_check(W, H)
    _log_fit(rep, args)
    if denoise:
        outputs.append(write_denoised(base + "_denoised.h5", ds.spatial_shape, W, H, out_type, ds.bin_indices,
                                      _run_attrs(ds, rep, args, loss=rep["loss_final"]), ds.metadata))
    write_report(base, ds, rep, args, outputs)
    if not args.no_plots:
        from .plots import plot_factorization
        outputs += plot_factorization(base, ds.source, W.reshape(*ds.spatial_shape, R), H, ds.bin_indices,
                                      max_views=_MAX_PLOTTED_VIEWS, wavelengths=ds.metadata.get("wavelengths"))
        log.info("wrote %s and %s", *outputs[-2:])
        write_report(base, ds, rep, args, outputs)
    n_err = sum(c.level == "error" for c in ds.checks)
    chi2 = rep["fit"].get("reduced_chi2")
    how = "from --basis" if args.basis else "estimated" if args.rank_detail else "given"
    print(f"done: rank {R} ({how}), {ds.pixels:,} pixels x {ds.bins} bins, "
          f"{rep['mode']} solve in {rep['solve_seconds']} s, loss {rep['loss_final']:.6g}"
          + (f", reduced chi-square {chi2:.2f}" if chi2 is not None else "") + f"; outputs at {base}_*"
          + (f"; {n_err} data check(s) had errors" if n_err else ""))
    return 0


def cmd_dehydrate(args):
    return _pipeline(args, denoise=False)


def cmd_denoise(args):
    return _pipeline(args, denoise=True)


def cmd_rehydrate(args):
    """This function rehydrates a dehydrated file.  The output covers all bins or the bins in
    --wave-range."""
    import h5py
    from .io import import_hsnt_data_hdf5
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"input not found: {args.input}")
    data, meta = import_hsnt_data_hdf5(args.input)
    if not isinstance(data, list):
        raise InputError(f"{args.input}: not a dehydrated file (needs subspace_data, subspace_basis and "
                         "dataset_type); run `mbirtorch-hsnt dehydrate` first")
    W4, H, dtype = data
    K_file = H.shape[1]
    if W4.ndim == 3:
        W4 = W4[None]
    if W4.ndim != 4:
        raise InputError(f"subspace_data has shape {W4.shape}; expected (views, rows, cols, rank) or (rows, cols, "
                         "rank)")
    if W4.shape[-1] != H.shape[0]:
        raise InputError(f"rank mismatch: subspace_data has {W4.shape[-1]} components, subspace_basis {H.shape[0]} "
                         "rows")
    with h5py.File(args.input, "r") as f:
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
        file_bins = _file_source_bins(f, K_file)
    views = _parse_slice(args.views, "views")
    if views:
        W4 = W4[slice(*views)]
    kept = _columns_in_range(file_bins, _parse_slice(args.wave_range, "wave-range"))
    H, bin_indices = H[:, kept.start:kept.stop], file_bins[kept.start:kept.stop]
    if H.shape[1] == 0:
        raise InputError(f"--wave-range {args.wave_range} selects none of the file's source bins "
                         f"{file_bins[0]}..{file_bins[-1]}")
    V, rows, cols, R = W4.shape
    out_type = args.as_type or dtype
    log.info("%s: %d component(s), %d view(s) x %d x %d pixels, %d of %d bins -> %s", args.input, R, V, rows, cols,
             H.shape[1], K_file, out_type)
    stem = re.sub(r"_dehydrated$", "", os.path.splitext(os.path.basename(args.input))[0])
    path = _output_path(args.output, stem + "_rehydrated.h5")
    _check_outputs([path], [args.input], args.overwrite)
    attrs = {k: v for k, v in attrs.items() if k not in ("rank", "rehydrated")}
    attrs.update(dehydrated_source=os.path.abspath(args.input), rehydrated_bins=f"{bin_indices[0]}..{bin_indices[-1]}")
    meta = {k: v for k, v in meta.items() if v is not None and k not in ("dataset_type", "dataset_modality")}
    write_denoised(path, (V, rows, cols), W4.reshape(-1, R).astype(np.float32), H.astype(np.float32), out_type,
                   bin_indices, attrs, _selected_metadata(meta, views, kept, 1, 1))
    print(path)
    return 0


_EXAMPLES = """Examples:
  mbirtorch-hsnt inspect data.h5
  mbirtorch-hsnt inspect sample_tifs/ --open-beam open_beam/ --estimate-rank
  mbirtorch-hsnt convert sample_tifs/ --open-beam open_beam/ --wave-bin 4 -o sample.h5
  mbirtorch-hsnt dehydrate sample.h5 -o results/                    # number of components estimated
  mbirtorch-hsnt dehydrate sample_tifs/ --open-beam open_beam/ --rank 2 --downsample 2 --wave-bin 4 -v
  mbirtorch-hsnt rehydrate results/sample_dehydrated.h5 --wave-range 100:200 -o results/
  mbirtorch-hsnt denoise sample.h5 -o results/                      # denoised data + dehydrated file

Each subcommand's -h lists the options most runs need; --help-all lists every option.
"""

_SPECTRA_HELP = ("how the component spectra are estimated. mle (default): the spectra that best fit the measured "
                 "counts. unconstrained: removes a bias the best fit has at low dose; worth it with many pixels (on "
                 "a test phantom 7-12 dB at a million pixels, up to 2 dB at 65,000). support: works out which "
                 "components each pixel contains, which removes the same bias and zeroes the background of the maps; "
                 "needs the dose (an open beam or --dose)")


class _Options:
    """This class is used to add arguments to the parser.  The class hides advanced arguments
    unless the user asks for the full help."""

    def __init__(self, show_all):
        self.show_all = show_all

    def add(self, group, *flags, advanced=False, **kw):
        if advanced and not self.show_all:
            kw["help"] = argparse.SUPPRESS
        group.add_argument(*flags, **kw)

    def input(self, sp):
        sp.add_argument("input", help="HDF5 file (hsnt layout), a directory with one TIFF per wavelength bin, or a "
                        "directory of such directories, one per view")
        g = sp.add_argument_group("input")
        self.add(g, "--open-beam", nargs="+", metavar="DIR", help="open-beam TIFF stack(s), needed when the TIFFs "
                 "hold counts, and shared by all views; a directory of observation subdirectories is averaged over "
                 "them")
        self.add(g, "--dose", type=_dose_arg, metavar="D|FILE",
                 help="open-beam counts per pixel and source bin, when no open beam or converted file gives it; or "
                      "a .npy or text file with one count per bin of the input (before --wave-range), when the open "
                      "beam differs between bins: the fit then weights each bin by its count")
        self.add(g, "--input-type", choices=("auto",) + INPUT_TYPES, default="auto",
                 help="what the values are (default: inferred)")
        self.add(g, "--dataset", metavar="GROUP", help="HDF5 group holding 'data' (default: found automatically)")
        g = sp.add_argument_group("selection")
        self.add(g, "--views", metavar="A:B", help="views of 4-D HDF5 data or of a directory of view directories "
                 "(default: all)")
        self.add(g, "--wave-range", metavar="A:B", help="source wavelength bins (default: all)")
        self.add(g, "--wave-bin", type=_positive_int, default=1, metavar="N",
                 help="group N adjacent bins (counts are summed, transmissions averaged)")
        self.add(g, "--downsample", type=_positive_int, default=1, metavar="S", help="keep every S-th row and column")
        g = sp.add_argument_group("calibration")
        self.add(g, "--background-boxes", type=_box_arg, nargs="+", metavar="BOX",
                 help="boxes free of the sample, each Y0:Y1,X0:X1 in full-resolution pixels: in each bin, each "
                      "detector tile's transmission is divided by its boxes', which corrects a sample run and an open "
                      "beam of different exposure. Or an instrument preset: ornl-snap (ORNL SNAP only: the corner box "
                      "of each 256 x 256 chip)")
        self.add(g, "--background-tiles", type=_tiles_arg, metavar="RxC", advanced=True,
                 help="detector tiles calibrated separately, each by the boxes whose centers it holds (default: the "
                      "preset's, else 1x1)")
        self.add(g, "--open-beam-smoothing", type=int, default=0, metavar="W",
                 help="smooth the averaged open beam in each bin with a W x W window (the square root of the outer "
                      "product of two Hamming windows, normalized), W odd and at least 3 (default 0, none; the ORNL "
                      "SNAP preprocessing used 3), leaving out pixels with no count in any selected bin; with one "
                      "open-beam observation the variance reduction is assumed, not measured")

    def run(self, sp, device=False, dry_run=False):
        g = sp.add_argument_group("run")
        if device:
            self.add(g, "--device", default="auto", metavar="auto|cpu|cuda|cuda:N",
                     help="compute device (default: cuda if available)")
        if dry_run:
            self.add(g, "--dry-run", action="store_true", help="load, check and plan, then stop")
        self.add(g, "-v", "--verbose", action="count", default=0, help="-v for INFO (default), -vv for DEBUG")
        self.add(g, "-q", "--quiet", action="store_true", help="warnings and errors only")
        self.add(g, "--help-all", action="store_true", help="show every option, including solver, memory, "
                 "rank-test and support-selection settings")
        g = sp.add_argument_group("advanced: logging")
        self.add(g, "--log-file", advanced=True, help="also write the log here")

    def rank_test(self, sp):
        g = sp.add_argument_group("advanced: rank test")
        self.add(g, "--max-rank", type=_positive_int, default=6, advanced=True,
                 help="largest number of components the estimate considers (default 6)")
        self.add(g, "--rank-pool", type=_pool_arg, default="auto", metavar="auto|B|0", advanced=True,
                 help="also test on B x B pooled pixels and take the larger rank (default: B chosen so pooled pixels "
                      "hold about 64 counts per bin; 0 disables)")

    def solve(self, sp, denoise):
        g = sp.add_argument_group("output")
        self.add(g, "-o", "--output", metavar="PATH", help="output directory (created if needed; default: current "
                 "directory), or a .h5 path whose stem names the files")
        self.add(g, "--as-type", choices=("attenuation", "transmission"),
                 help="quantity stored in the outputs (default: the input's)")
        if denoise:
            self.add(g, "--no-dehydrated", action="store_true",
                     help="write only the denoised data, not the dehydrated file")
        self.add(g, "--no-plots", action="store_true", help="skip the PNG plots of the maps and spectra")
        self.add(g, "--overwrite", action="store_true", help="replace outputs that already exist")
        g = sp.add_argument_group("model")
        self.add(g, "--rank", "-r", type=_rank_arg, default="auto", metavar="N",
                 help="number of components, about the number of distinct materials (default: estimated from the "
                      "data)")
        self.add(g, "--spectra", choices=("mle", "unconstrained", "support"), default="mle", help=_SPECTRA_HELP)
        self.add(g, "--basis", metavar="FILE", help="fit only the maps, for the spectra (subspace_basis) of this "
                 "dehydrated file, e.g. one fitted on a few views of the same scan; the bins must match")
        self.run(sp, device=True, dry_run=True)
        self.rank_test(sp)
        g = sp.add_argument_group("advanced: support selection")
        self.add(g, "--support-penalty", type=_penalty_arg, default="auto", metavar="auto|F", advanced=True,
                 help="charge per selected component, F x log(bins) nats, or 'auto' (default): 0.5 up to 10 counts "
                      "per bin in the median pixel, rising to 2 at 100. On a test phantom 2 did as well or slightly "
                      "better at 3 to 10 counts; below 3 it was up to 8.6 dB worse on one spectrum and up to 3.4 dB "
                      "better on another")
        self.add(g, "--wald-screen", type=_nonneg_float, default=0.0, metavar="F", advanced=True,
                 help="skip single-component fits below F x penalty of Wald statistic in the full fit (0 = off; "
                      "trades rare-component recall for time)")
        g = sp.add_argument_group("advanced: solver")
        self.add(g, "--max-steps", type=_positive_int, default=1000, advanced=True,
                 help="largest number of solver steps in a full solve (default 1000)")
        self.add(g, "--rel-tol", type=_nonneg_float, default=1e-8, advanced=True,
                 help="relative loss change per step; the solve stops after five steps in a row below it "
                      "(default 1e-8)")
        self.add(g, "--compile", choices=("auto", "on", "off"), default="auto", advanced=True,
                 help="compile the solver kernels: auto (default) on CUDA for data of 5e8 entries or more (about "
                      "417k pixels at 1200 bins, where a first compiled call starts to pay for its compile); stream "
                      "mode compiles only with 'on', and the rank estimate always runs uncompiled")
        g = sp.add_argument_group("advanced: memory")
        self.add(g, "--mode", choices=("auto", "full", "stream"), default="auto", advanced=True,
                 help="full solve on the device or streamed by chunks of pixels (default: full when its estimated "
                      "working set, compiled or not, is within 80%% of the available memory)")
        self.add(g, "--chunk-pixels", type=_positive_int, advanced=True,
                 help="pixels per chunk in stream mode (default: from available memory); halve it if the device runs "
                      "out of memory")
        self.add(g, "--max-passes", type=_nonneg_int, default=5, advanced=True,
                 help="stream mode: polish passes over the data after the fit on a pixel subsample, stopping early on "
                      "the first pass that changes the loss by at most --rel-tol, relatively; 0 keeps that fit "
                      "(default 5). More passes trade time for SNR: on a 1M-pixel test phantom at 3 counts per bin, 40 "
                      "passes brought the streamed spectra within 0.5 dB of the fit solved whole, at 4-6x the time")


def build_parser(show_all=False):
    """This function builds the command line parser.  If ``show_all`` is True, the help lists the
    advanced options."""
    opt = _Options(show_all)
    p = argparse.ArgumentParser(prog="mbirtorch-hsnt", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_EXAMPLES)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("inspect", help="load, check and describe a dataset (no solve)")
    opt.input(s)
    g = s.add_argument_group("number of components")
    opt.add(g, "--estimate-rank", action="store_true",
            help="estimate the number of components (runs a few small fits)")
    opt.run(s, device=True)
    opt.rank_test(s)
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("convert", help="write a TIFF stack (and open beam) or an HDF5 dataset as an HDF5 dataset in "
                                       "the hsnt layout, streamed in blocks of bins")
    opt.input(s)
    g = s.add_argument_group("output")
    opt.add(g, "-o", "--output", metavar="PATH", help="output .h5 (default: <input>_converted.h5)")
    opt.add(g, "--as-type", choices=("attenuation", "transmission"), default="attenuation",
            help="stored quantity (default attenuation)")
    opt.add(g, "--overwrite", action="store_true", help="replace the output if it already exists")
    opt.run(s)
    g = s.add_argument_group("advanced: streaming")
    opt.add(g, "--memory-budget", type=_positive_float, default=256, metavar="MiB", advanced=True,
            help="working memory for the blocks (default 256 MiB)")
    opt.add(g, "--workers", type=_positive_int, metavar="N", advanced=True,
            help="threads decoding TIFF images of a block (default: min(8, CPUs))")
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("dehydrate", help="fit component maps and spectra and write them in the dehydrated layout, with "
                                         "plots and a report")
    opt.input(s)
    opt.solve(s, denoise=False)
    s.set_defaults(func=cmd_dehydrate)

    s = sub.add_parser("rehydrate",
                       help="multiply a dehydrated file back into hyperspectral data (all bins or a range)")
    s.add_argument("input", help="a dehydrated .h5 (subspace_data, subspace_basis, dataset_type), as written by "
                                 "dehydrate")
    g = s.add_argument_group("selection")
    opt.add(g, "--wave-range", metavar="A:B", help="source wavelength bins to rehydrate (default: all)")
    opt.add(g, "--views", metavar="A:B", help="views to rehydrate (default: all)")
    g = s.add_argument_group("output")
    opt.add(g, "-o", "--output", metavar="PATH", help="output directory (default: current directory), or a .h5 "
                                                     "path; default name <stem>_rehydrated.h5")
    opt.add(g, "--as-type", choices=("attenuation", "transmission"),
            help="quantity to write (default: the file's dataset_type)")
    opt.add(g, "--overwrite", action="store_true", help="replace the output if it already exists")
    opt.run(s)
    s.set_defaults(func=cmd_rehydrate)

    s = sub.add_parser("denoise", help="dehydrate and rehydrate: write the denoised hyperspectral data in the hsnt "
                                       "HDF5 layout")
    opt.input(s)
    opt.solve(s, denoise=True)
    s.set_defaults(func=cmd_denoise)
    return p


# These CUDA library errors can occur instead of the allocator's out of memory error when
# the device runs out of memory.
_DEVICE_MEMORY_FAILURES = ("CUBLAS_STATUS_EXECUTION_FAILED", "CUBLAS_STATUS_ALLOC_FAILED",
                           "CUBLAS_STATUS_NOT_INITIALIZED", "CUSOLVER_STATUS_ALLOC_FAILED",
                           "CUSOLVER_STATUS_EXECUTION_FAILED", "CUSOLVER_STATUS_INTERNAL_ERROR",
                           "CUSOLVER_STATUS_NOT_INITIALIZED", "CUDA error: out of memory")


def _is_device_out_of_memory(e):
    """This function returns True if ``e`` is an out of memory error of the device."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(e, RuntimeError) and any(s in str(e) for s in _DEVICE_MEMORY_FAILURES)


def _out_of_memory_message(e, rep):
    """This function returns the error message for a device that ran out of memory.  The message
    gives the memory plan of the solve and the options that reduce memory."""
    torch = sys.modules.get("torch")
    lines = str(e).strip().splitlines()
    what = ("ran out of memory" if torch is not None and isinstance(e, torch.cuda.OutOfMemoryError)
            else "failed, most likely out of memory")
    text = f"error: the device {what} ({lines[0] if lines else type(e).__name__})"
    if rep.get("solved"):
        return text + " after the solve, in the diagnostics: try --device cpu"
    if not rep.get("memory_plan"):
        return text + ": try --mode stream or a smaller --chunk-pixels, --downsample or --wave-bin, or --device cpu"
    text += f" in a {rep['mode']} solve; plan: {rep['memory_plan']}"
    if rep["mode"] == "stream":
        return text + f". Try half the chunk, --chunk-pixels {max(1, rep['chunk_pixels'] // 2)}, or --device cpu"
    return text + ". Try --mode stream, a smaller --downsample or --wave-bin, or --device cpu"


def main(argv=None):
    """This function runs the command line program.  Bad input makes the program exit with a
    message.  With -vv, the program prints the traceback instead."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--help-all" in argv:                    # argparse's -h prints the full help with every option
        build_parser(show_all=True).parse_args([a if a != "--help-all" else "-h" for a in argv])
    args = build_parser().parse_args(argv)
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose >= 2 else logging.INFO)
    handlers = [logging.StreamHandler(sys.stderr)] + ([logging.FileHandler(args.log_file)] if args.log_file else [])
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)
    saved = log.level, log.propagate, warnings.showwarning
    log.setLevel(level)
    log.propagate = False
    warnings.showwarning = lambda message, *rest, **kw: log.warning("%s", message)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    except Exception as e:          # Input and resource errors print one line.  Other errors print a traceback.
        if level <= logging.DEBUG:
            raise
        if _is_device_out_of_memory(e):
            raise SystemExit(_out_of_memory_message(e, getattr(args, "fit_report", {}))) from None
        if isinstance(e, (InputError, OSError, MemoryError)):
            raise SystemExit(f"error: {e}") from None
        raise
    finally:
        log.setLevel(saved[0])
        log.propagate = saved[1]
        warnings.showwarning = saved[2]
        for h in handlers:
            log.removeHandler(h)
            h.close()


if __name__ == "__main__":
    sys.exit(main())

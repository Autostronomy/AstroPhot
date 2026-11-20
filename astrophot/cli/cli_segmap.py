# =============================================================================
# Fit all objects identified in a segmentation map
#
# This is a quick script to fit all the objects identified in a segmentation map
# using a single model type. The script will load the target image, mask,
# psf, and variance image (if available) and fit the models to the target image.
# Each object is fit on its own (plus a sky level), with the other objects in
# its window masked.
#
# Run this script with:
# ~$ astrophot segmap target.fits segmap.fits [OPTIONS]
# or, for a copy made with --dump:
# ~$ python segmap_astrophot_model.py target.fits segmap.fits [OPTIONS]
# =============================================================================

import argparse
import sys
from pathlib import Path

import astrophot as ap
import numpy as np
import yaml
from astropy.io import fits
from astropy.wcs import WCS


def hdu_id(value):
    """An HDU may be given by index (0, 1, ...) or by extension name (SCI)."""
    try:
        return int(value)
    except ValueError:
        return value


def read_image(path, hdu, dtype, option):
    """Read a 2D image from a FITS file as a native byte order numpy array."""
    path = Path(path)
    if not path.is_file():
        sys.exit(f"error: {option} file '{path}' does not exist")
    with fits.open(path) as hdul:
        try:
            data = hdul[hdu].data
            header = hdul[hdu].header
        except (IndexError, KeyError):
            sys.exit(f"error: '{path}' has no HDU {hdu!r} (it has {len(hdul)} HDUs)")
        if data is None or data.ndim != 2:
            shape = None if data is None else data.shape
            sys.exit(
                f"error: HDU {hdu!r} of '{path}' is not a 2D image (shape {shape}). "
                f"Use {option}_hdu to select the HDU which holds the image."
            )
        return np.array(data, dtype=dtype), header


def to_serializable(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.array(value)
    return value.item() if value.shape == () else value.tolist()


def collect_model_parameters(model):
    """Value and uncertainty of each parameter of the model, fitted or fixed."""
    summary = {}
    for name in model.parameter_specs:
        param = getattr(model, name)
        summary[name] = {
            "value": to_serializable(param.npvalue),
            "uncertainty": (to_serializable(param.uncertainty) if param.dynamic else None),
            "dynamic": bool(param.dynamic),
        }
    return summary


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Fit a model to a series of targets in an image using AstroPhot.",
    )

    # Core Parameters
    parser.add_argument(
        "target_file",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the target FITS file",
    )
    parser.add_argument(
        "segmap_file",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the segmentation map FITS file",
    )
    parser.add_argument(
        "--cat",
        type=Path,
        default=None,
        help="Path to a catalogue yaml file with initial parameters for each segmentation map id",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="astrophot_model",
        help="Prefix name used for the models and the output files. "
        "Only letters, digits and underscores",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Directory to write the output files to, created if needed. "
        "Default: current directory",
    )
    parser.add_argument("--psf", type=Path, default=None, help="Path to the PSF FITS file")
    parser.add_argument(
        "--psf_upsample",
        type=int,
        default=1,
        help="PSF upsampling factor for convolution (int)",
    )
    parser.add_argument("--zeropoint", type=float, default=None, help="Magnitude zeropoint")
    parser.add_argument(
        "--pixelscale",
        type=float,
        default=None,
        help="Pixel size in arcsec. If given, the WCS in the FITS header is ignored and "
        "coordinates are measured from the centre of the lower left pixel. "
        "Default: use the header WCS, or 1 if the header has none",
    )
    parser.add_argument(
        "--initial_sky",
        type=float,
        default=None,
        help="Initial sky value for the I0 parameter (flux per arcsec^2)",
    )
    parser.add_argument(
        "--sky_locked", action="store_true", help="Lock the sky model during fitting"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="sersic_galaxy_model",
        help="Type of AstroPhot model to fit. Replace spaces with underscores, e.g. 'sersic_galaxy_model' or 'exponential_galaxy_model'",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level for fitting output")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="File to also write the messages of AstroPhot to. Default: no log file",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump this file to the current directory as 'segmap_astrophot_model.py' for editing and running",
    )

    # Window Parameters
    parser.add_argument(
        "--window_expand_scale",
        type=float,
        default=1.0,
        help="Scale factor to expand windows from segmentation map for final fit",
    )
    parser.add_argument(
        "--window_expand_border",
        type=int,
        default=0,
        help="Number of pixels to expand windows from segmentation map for final fit",
    )
    parser.add_argument(
        "--window_min_size",
        type=int,
        default=1,
        help="Minimum area, in pixels, of a segment's bounding box for it to be fit. "
        "Checked before the window is expanded",
    )
    parser.add_argument(
        "--filter_ids",
        type=int,
        nargs="+",
        default=[],
        help="List of segmentation map ids to exclude from fitting. "
        "Id 0 is the background and is always excluded",
    )

    # Variance and Mask Parameters
    parser.add_argument(
        "--variance",
        type=str,
        default=None,
        help="Path to the variance FITS file, or 'auto' to estimate the variance from "
        "the image. Default: a variance of 1 for every pixel",
    )
    parser.add_argument(
        "--variance_hdu",
        type=hdu_id,
        default=0,
        help="HDU index or name for variance data",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Path to the mask FITS file, non-zero pixels are ignored in the fit",
    )
    parser.add_argument(
        "--mask_hdu", type=hdu_id, default=0, help="HDU index or name for mask data"
    )

    # Extra Parameters
    parser.add_argument(
        "--no_save_images",
        action="store_false",
        dest="save_images",
        help="Disable saving the model and residual images",
    )
    parser.add_argument(
        "--no_save_cov",
        action="store_false",
        dest="save_covariance_matrix",
        help="Disable saving the covariance matrix",
    )
    parser.add_argument(
        "--target_hdu",
        type=hdu_id,
        default=0,
        help="HDU index or name for target image data",
    )
    parser.add_argument(
        "--segmap_hdu",
        type=hdu_id,
        default=0,
        help="HDU index or name for segmentation map data",
    )
    parser.add_argument("--psf_hdu", type=hdu_id, default=0, help="HDU index or name for PSF data")
    parser.add_argument(
        "--sky_model_type",
        type=str,
        default="flat",
        help="Type of sky model to fit, options include 'flat' or 'plane'",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=50,
        help="Maximum number of Levenberg-Marquardt iterations for fitting",
    )

    args = parser.parse_args(argv)

    if args.dump:
        dump_file = args.outdir / Path("segmap_astrophot_model.py")
        dump_file.write_text(
            "# This file was generated by dumping the `astrophot segmap` script. You can edit and run this file directly.\n\n"
            + Path(__file__).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print(f"Dumped to {dump_file}")
        if args.target_file is None:
            return

    if args.target_file is None or args.segmap_file is None:
        parser.error("the following arguments are required: target_file, segmap_file")
    if not args.name.isidentifier():
        parser.error(
            f"--name '{args.name}' must only have letters, digits and underscores, "
            "and not start with a digit."
        )

    ap.set_logger(filename=args.log, stdout=args.verbose > 0)

    # Load Target Data
    # ---------------------------------------------------------------------
    if args.verbose > 0:
        print("Loading target image...")
    target_data, target_header = read_image(
        args.target_file, args.target_hdu, np.float64, "--target"
    )

    # Load Variance Data
    # ---------------------------------------------------------------------
    variance_data = None
    bad_variance = None
    if args.variance == "auto":
        variance_data = "auto"
    elif args.variance is not None:
        if args.verbose > 0:
            print("Loading variance image...")
        variance_data, _ = read_image(args.variance, args.variance_hdu, np.float64, "--variance")
        if variance_data.shape != target_data.shape:
            sys.exit(
                f"error: variance shape {variance_data.shape} does not match "
                f"target shape {target_data.shape}"
            )
        # Pixels without a usable variance are masked rather than given infinite weight
        bad_variance = ~np.isfinite(variance_data) | (variance_data <= 0)
        variance_data[bad_variance] = 1.0

    # Load Mask Data
    # ---------------------------------------------------------------------
    mask_data = None
    if args.mask is not None:
        if args.verbose > 0:
            print("Loading mask image...")
        mask_data, _ = read_image(args.mask, args.mask_hdu, bool, "--mask")
        if mask_data.shape != target_data.shape:
            sys.exit(
                f"error: mask shape {mask_data.shape} does not match "
                f"target shape {target_data.shape}"
            )
    if bad_variance is not None and bad_variance.any():
        if args.verbose > 0:
            print(f"Masking {bad_variance.sum()} pixels with zero, negative or NaN variance")
        mask_data = bad_variance if mask_data is None else mask_data | bad_variance

    # Load Segmentation Map
    # ---------------------------------------------------------------------
    if args.verbose > 0:
        print("Loading segmentation map...")
    segmap_data, _ = read_image(args.segmap_file, args.segmap_hdu, np.int64, "--segmap")
    if segmap_data.shape != target_data.shape:
        sys.exit(
            f"error: segmentation map shape {segmap_data.shape} does not match "
            f"target shape {target_data.shape}"
        )

    # Load Catalogue of Initial Parameters
    # ---------------------------------------------------------------------
    override_init_params = {}
    if args.cat is not None:
        if args.verbose > 0:
            print("Loading catalogue of initial parameters...")
        if not args.cat.is_file():
            sys.exit(f"error: --cat file '{args.cat}' does not exist")
        with open(args.cat, "r", encoding="utf-8") as cat_file:
            catalogue = yaml.safe_load(cat_file) or {}
        try:
            override_init_params = {int(key): dict(value) for key, value in catalogue.items()}
        except (AttributeError, TypeError, ValueError):
            sys.exit(
                f"error: '{args.cat}' should map each segmentation map id (an integer) "
                "to a dictionary of parameters"
            )

    # Load PSF
    # ---------------------------------------------------------------------
    psf_data = None
    if args.psf is not None:
        if args.verbose > 0:
            print("Loading PSF...")
        psf_data, _ = read_image(args.psf, args.psf_hdu, np.float64, "--psf")
        if psf_data.shape[0] % 2 == 0 or psf_data.shape[1] % 2 == 0:
            sys.exit(
                f"error: the PSF image must have an odd number of pixels on each axis so "
                f"that it has a central pixel, got {psf_data.shape}"
            )
        psf_data = ap.PSFImage(data=psf_data, upsample=args.psf_upsample)

    # Make Target
    # ---------------------------------------------------------------------
    target_wcs = WCS(target_header)
    if args.pixelscale is not None:
        coordinates = {"pixelscale": args.pixelscale}
    elif target_wcs.has_celestial:
        coordinates = {"wcs": target_wcs}
    else:
        if args.verbose > 0:
            print("No WCS in the target header, using a pixelscale of 1 (see --pixelscale)")
        coordinates = {"pixelscale": 1.0}
    target = ap.TargetImage(
        data=target_data,
        zeropoint=args.zeropoint,
        variance=variance_data,
        mask=mask_data,
        psf=psf_data,
        **coordinates,
    )

    # Initialization from segmap
    # ---------------------------------------------------------------------
    if args.verbose > 0:
        print("Parsing segmentation map")
    skip_ids = (0, *args.filter_ids)
    windows = ap.utils.initialize.windows_from_segmentation_map(segmap_data, skip_index=skip_ids)

    windows = ap.utils.initialize.filter_windows(
        windows,
        min_area=args.window_min_size,
        image=target,
    )
    windows = ap.utils.initialize.scale_windows(
        windows,
        image=target,
        expand_scale=args.window_expand_scale,
        expand_border=args.window_expand_border,
    )
    windows = {int(index): window for index, window in windows.items()}
    if len(windows) == 0:
        sys.exit("error: no objects to fit in the segmentation map")
    unused = sorted(set(override_init_params) - set(windows))
    if unused and args.verbose > 0:
        if len(unused) <= 10:
            print(f"Note: catalogue ids {unused} are not among the objects being fit")
        else:
            print(f"Note: {len(unused)} catalogue ids are not being fit")

    centers = ap.utils.initialize.centroids_from_segmentation_map(
        segmap_data, target, skip_index=skip_ids
    )
    if "galaxy" in args.model_type:
        PAs = ap.utils.initialize.PA_from_segmentation_map(
            segmap_data, target, centers, skip_index=skip_ids
        )
        qs = ap.utils.initialize.q_from_segmentation_map(
            segmap_data, target, centers, skip_index=skip_ids
        )
    else:
        PAs = None
        qs = None
    init_params = {}
    for window in windows:
        init_params[window] = {"center": centers[window]}
        if "galaxy" in args.model_type:
            init_params[window]["PA"] = PAs[window]
            init_params[window]["q"] = qs[window]
        if window in override_init_params:
            init_params[window].update(override_init_params[window])

    # Create and fit Models
    # ---------------------------------------------------------------------
    results = {}
    model_hdus = [fits.PrimaryHDU()]
    residual_hdus = [fits.PrimaryHDU()]
    cov_matrices = {}
    for window in windows:
        name = f"{args.name}_{window}"
        if args.verbose > 0:
            print(f"Fitting model for window {window}...")
        try:
            W = ap.Window(windows[window], target)
            indices = target.get_indices(W)
            subtarget = target[W]
            # Mask interlopers, the transpose puts the segmap on the (i, j) axes of the window
            sub_segmap = segmap_data.T[indices]
            interlopers = (sub_segmap != 0) & (sub_segmap != window)
            subtarget.mask = ap.backend.to_numpy(subtarget.mask) | interlopers.T
            # Sky model
            sky = ap.Model(
                name="sky",
                model_type=args.sky_model_type + " sky model",
                target=subtarget,
                I0=args.initial_sky,
            )
            if args.sky_locked:
                sky.to_static()
            # Primary object model
            object_model = ap.Model(
                name=f"{args.model_type}_{window}",
                model_type=args.model_type.replace("_", " "),
                target=subtarget,
                **init_params[window],
            )
            # Combine into group model for fitting
            model = ap.Model(
                name=name,
                model_type="group model",
                target=subtarget,
                models=[sky, object_model],
            )
            # Fitting
            model.initialize()
            result = ap.fit.LM(model, verbose=args.verbose - 1, max_iter=args.max_iter).fit()
            # Collect results
            results[name] = {
                "parameters": collect_model_parameters(object_model),
                "total_flux": to_serializable(object_model.total_flux()),
                "total_flux_uncertainty": to_serializable(object_model.total_flux_uncertainty()),
                "sky_parameters": collect_model_parameters(sky),
                "message": result.message,
            }
            if args.zeropoint is not None:
                results[name]["total_magnitude"] = to_serializable(object_model.total_magnitude())
                results[name]["total_magnitude_uncertainty"] = to_serializable(
                    object_model.total_magnitude_uncertainty()
                )
            if args.save_covariance_matrix:
                cov_matrices[name] = ap.backend.to_numpy(result.covariance_matrix)
            if args.save_images:
                # The headers hold the WCS of each window
                model_image = model()
                residual_image = subtarget - model_image
                model_hdus.append(
                    fits.ImageHDU(
                        ap.backend.to_numpy(model_image.data),
                        header=fits.Header(model_image.fits_info()),
                        name=f"{name}_model",
                    )
                )
                residual_hdus.append(
                    fits.ImageHDU(
                        ap.backend.to_numpy(residual_image.data),
                        header=fits.Header(residual_image.fits_info()),
                        name=f"{name}_residual",
                    )
                )
        except Exception as error:  # one bad object should not stop the rest
            results[name] = {"message": f"fit failed, {type(error).__name__}: {error}"}
            if args.verbose > 0:
                print(f"Fit failed for window {window}, {type(error).__name__}: {error}")

    # Save outputs
    # ---------------------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.save_covariance_matrix:
        np.savez(args.outdir / f"{args.name}_covariance_matrix.npz", **cov_matrices)

    if args.save_images:
        fits.HDUList(model_hdus).writeto(
            args.outdir / f"{args.name}_model_images.fits", overwrite=True
        )
        fits.HDUList(residual_hdus).writeto(
            args.outdir / f"{args.name}_residual_images.fits", overwrite=True
        )

    with open(args.outdir / f"{args.name}_parameters.yaml", "w", encoding="utf-8") as output_file:
        yaml.dump(results, output_file, default_flow_style=False)


if __name__ == "__main__":
    main()

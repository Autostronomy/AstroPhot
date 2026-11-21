# =============================================================================
# Fit a single model to a target image
#
# This is a quick script to fit a single AstroPhot model to a target image. The
# script will load the target image, mask, psf, and variance image (if
# available) and fit the model to the target image. The script will save the
# fitted parameters, model image, residual image, and covariance matrix.
# This script is intended for quick easy fits, and as a starting point to build
# a more complex analysis.
#
# Run this script with:
# ~$ astrophot single target_image.fits [OPTIONS]
# or, for a copy made with --dump:
# ~$ python single_astrophot_model.py target_image.fits [OPTIONS]
# =============================================================================

import argparse
import ast
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


def parse_value(text):
    """Convert a command line string to a number, list, bool, dict, etc. if possible."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text  # keep it as a string


def parse_model_params(tokens):
    """
    Parse the arguments which argparse did not recognize into a dictionary of
    model parameters. Accepts ``--key value``, ``--key=value``, ``--key v1 v2``
    (collected into a list) and ``--key`` (set to True).
    """
    collected = {}
    key = None
    for token in tokens:
        if token.startswith("--"):
            key, has_value, text = token[2:].partition("=")
            collected[key] = [parse_value(text)] if has_value else []
        elif key is None:
            sys.exit(
                f"error: unexpected argument '{token}'. Give the target file first, "
                "then the options."
            )
        else:
            collected[key].append(parse_value(token))
    return {
        key: True if len(values) == 0 else values[0] if len(values) == 1 else values
        for key, values in collected.items()
    }


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
        description="Fit a model to a target image using AstroPhot.",
        epilog="Any other option is passed to the model as an initial parameter, "
        "for example: --n 2 --center 51.2 48.7 --q 0.7",
        allow_abbrev=False,  # so that model parameters like --n are not mistaken for --name
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
        "--window",
        type=int,
        default=None,
        nargs=4,
        metavar=("IMIN", "IMAX", "JMIN", "JMAX"),
        help="Window to fit, in 0-indexed pixels, given as: --window imin imax jmin jmax. "
        "i is the horizontal axis (FITS axis 1) and j the vertical axis (FITS axis 2), "
        "the upper limits are exclusive. Default: the whole image",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="astrophot_model",
        help="Name of the model, also the prefix of the output files. "
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
        help="PSF upsampling factor for convolution (int >0). Model is evaluated at this upsampling resolution then downsampled to the target resolution for likelihood calculation.",
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
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Amount of printed output: 0 for none, 1 for the progress of the fit, "
        "2 to add debugging messages",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="File to also write the messages of AstroPhot to. Default: no log file",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump this file to the current directory as 'single_astrophot_model.py' for editing and running",
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
        default=100,
        help="Maximum number of Levenberg-Marquardt iterations for fitting",
    )

    # Parse known arguments, leave the rest for the dynamic dictionary
    args, unknown = parser.parse_known_args(argv)

    # Parse the arbitrary parameters for the model
    initial_params = parse_model_params(unknown)

    if args.dump:
        args.outdir.mkdir(parents=True, exist_ok=True)
        dump_file = args.outdir / Path("single_astrophot_model.py")
        dump_file.write_text(
            "# This file was generated by dumping the `astrophot single` script. You can edit and run this file directly.\n\n"
            + Path(__file__).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print(f"Dumped to {dump_file}")
        if args.target_file is None:
            return

    if args.target_file is None:
        parser.error("the following arguments are required: target_file")
    if not args.target_file.is_file():
        parser.error(f"target file '{args.target_file}' does not exist.")
    if not args.name.isidentifier():
        parser.error(
            f"--name '{args.name}' must only have letters, digits and underscores, "
            "and not start with a digit."
        )
    if args.name == "sky_model":
        parser.error("--name 'sky_model' is the name of the sky in the parameters file.")

    # The messages go to the screen unless --verbose is 0, and to the --log file if there is one
    ap.set_logger(
        filename=args.log, stdout=args.verbose > 0, level="DEBUG" if args.verbose > 1 else "INFO"
    )

    # Load Target Data
    # ---------------------------------------------------------------------
    ap.config.logger.info("Loading target image...")
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
        ap.config.logger.info("Loading variance image...")
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
        ap.config.logger.info("Loading mask image...")
        mask_data, _ = read_image(args.mask, args.mask_hdu, bool, "--mask")
        if mask_data.shape != target_data.shape:
            sys.exit(
                f"error: mask shape {mask_data.shape} does not match "
                f"target shape {target_data.shape}"
            )
    if bad_variance is not None and bad_variance.any():
        ap.config.logger.info(
            f"Masking {bad_variance.sum()} pixels with zero, negative or NaN variance"
        )
        mask_data = bad_variance if mask_data is None else mask_data | bad_variance

    # Check the fitting window
    # ---------------------------------------------------------------------
    window = None
    if args.window is not None:
        window = (
            max(args.window[0], 0),
            min(args.window[1], target_data.shape[1]),
            max(args.window[2], 0),
            min(args.window[3], target_data.shape[0]),
        )
        if window[0] >= window[1] or window[2] >= window[3]:
            sys.exit(
                f"error: --window {args.window} has no pixels inside the image, which is "
                f"{target_data.shape[1]} (i) by {target_data.shape[0]} (j) pixels"
            )

    # Load PSF
    # ---------------------------------------------------------------------
    psf_data = None
    if args.psf is not None:
        ap.config.logger.info("Loading PSF...")
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
        if "CRTAN1" in target_header and "CRTAN2" in target_header:
            # an image saved by AstroPhot, which can be shifted on the tangent plane
            coordinates["crtan"] = (target_header["CRTAN1"], target_header["CRTAN2"])
    else:
        ap.config.logger.info(
            "No WCS in the target header, using a pixelscale of 1 (see --pixelscale)"
        )
        coordinates = {"pixelscale": 1.0}
    target = ap.TargetImage(
        data=target_data,
        zeropoint=args.zeropoint,
        variance=variance_data,
        mask=mask_data,
        psf=psf_data,
        **coordinates,
    )

    # Create Model
    # ---------------------------------------------------------------------
    model_object = ap.Model(
        name=args.name,
        model_type=args.model_type.replace("_", " "),
        target=target,
        **initial_params,
        window=window,
    )

    model_sky = ap.Model(
        name=f"{args.name}_sky",
        model_type=args.sky_model_type + " sky model",
        target=target,
        I0=args.initial_sky,
        window=window,
    )

    if args.sky_locked:
        model_sky.to_static()

    model = ap.Model(
        name=f"{args.name}_and_sky",
        model_type="group model",
        target=target,
        models=[model_sky, model_object],
    )

    # Fit the model
    # ---------------------------------------------------------------------
    ap.config.logger.info("Initializing model...")
    model.initialize()
    ap.config.logger.info("Fitting model...")
    result = ap.fit.LM(model, max_iter=args.max_iter).fit()

    # Report Total Magnitude or Flux
    # ----------------------------------------------------------------------
    total_flux = to_serializable(model_object.total_flux())
    total_flux_uncertainty = to_serializable(model_object.total_flux_uncertainty())
    total_magnitude = None
    total_magnitude_uncertainty = None
    if args.zeropoint is not None:
        total_magnitude = to_serializable(model_object.total_magnitude())
        total_magnitude_uncertainty = to_serializable(model_object.total_magnitude_uncertainty())
    ap.config.logger.info(str(model))
    if args.zeropoint is not None:
        ap.config.logger.info(
            f"Total Magnitude: {total_magnitude} +- {total_magnitude_uncertainty}"
        )
    else:
        ap.config.logger.info(f"Total Flux: {total_flux} +- {total_flux_uncertainty}")

    # Save Results
    # ----------------------------------------------------------------------
    output_summary = {
        model_object.name: {
            "model_type": model_object.model_type,
            "parameters": collect_model_parameters(model_object),
            "total_flux": total_flux,
            "total_flux_uncertainty": total_flux_uncertainty,
            "total_magnitude": total_magnitude,
            "total_magnitude_uncertainty": total_magnitude_uncertainty,
            "note": "Total flux/mag is within the fitting window, not extended to infinity",
            "message": result.message,
        },
        "sky_model": {
            "model_type": model_sky.model_type,
            "parameters": collect_model_parameters(model_sky),
        },
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outdir / f"{args.name}_parameters.yaml", "w", encoding="utf-8") as output_file:
        yaml.dump(output_summary, output_file, default_flow_style=False)

    if args.save_images:
        model().save(args.outdir / f"{args.name}_model_image.fits")
        (target[model.window] - model()).save(args.outdir / f"{args.name}_residual_image.fits")

    if args.save_covariance_matrix:
        np.save(
            args.outdir / f"{args.name}_covariance_matrix.npy",
            ap.backend.to_numpy(result.covariance_matrix),
        )


if __name__ == "__main__":
    main()

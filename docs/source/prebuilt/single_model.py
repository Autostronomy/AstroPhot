# =============================================================================
# Fit a single model to a target image
#
# This is a quick script to fit a single AstroPhot model to a target image. The
# script will load the target image, mask, psf, and variance image (if
# available) and fit the model to the target image. The script will save the
# model image, residual image, and covariance matrix to the current directory.
# This script is intended for quick easy fits, and as a starting point to build
# a more complex analysis.
#
# Run this script with:
# >>> python single_model_fit.py --target_file <required>.fits [OPTIONS]
# =============================================================================

import astrophot as ap
import numpy as np
from astropy.io import fits
import argparse
import ast


def parse_arbitrary_args(unknown_args):
    """
    Parses unrecognized command-line arguments into a dictionary.
    Attempts to cast values to native Python types (int, float, list, etc.).
    """
    params = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg[2:]
            # Handle --key=value format
            if "=" in key:
                key, val_str = key.split("=", 1)
            # Handle --key value format
            elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                val_str = unknown_args[i + 1]
                i += 1
            else:
                val_str = "True"  # Assume a boolean flag if no value is provided

            # Safely evaluate strings into numbers, lists, or booleans
            try:
                val = ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                val = val_str  # Fallback to keeping it as a string

            params[key] = val
        i += 1
    return params


def main():
    parser = argparse.ArgumentParser(description="Fit a model to a target image using AstroPhot.")

    # Core Parameters
    parser.add_argument("target_file", type=str, help="Path to the target FITS file")
    parser.add_argument(
        "--window",
        type=str,
        default=None,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Window border for the fit in pixel coordinates, given as: --window xmin xmax ymin ymax",
    )
    parser.add_argument(
        "--name", type=str, default="object_name", help="Name used for saving files"
    )
    parser.add_argument("--psf_file", type=str, default=None, help="Path to the PSF FITS file")
    parser.add_argument(
        "--psf_upsample", type=int, default=1, help="PSF upsampling factor for convolution (int)"
    )
    parser.add_argument("--zeropoint", type=float, default=None, help="Magnitude zeropoint")
    parser.add_argument(
        "--initial_sky",
        type=float,
        default=None,
        help="Initial sky value for the I0 parameter",
    )
    parser.add_argument(
        "--sky_locked", action="store_true", help="Lock the sky model during fitting"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="sersic_galaxy_model",
        help="Type of AstroPhot model to fit. Replace spaces with underscores, e.g. 'sersic_galaxy_model' or 'exponential_disk_model'",
    )

    # Variance and Mask Parameters
    parser.add_argument(
        "--variance_file", type=str, default=None, help="Path to the variance FITS file"
    )
    parser.add_argument(
        "--variance_hdu", type=int, default=0, help="FITS file index for variance data"
    )
    parser.add_argument("--mask_file", type=str, default=None, help="Path to the mask FITS file")
    parser.add_argument("--mask_hdu", type=int, default=0, help="FITS file index for mask data")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level for fitting output")

    # Extra Parameters
    parser.add_argument(
        "--no_save_images",
        action="store_false",
        dest="save_images",
        help="Disable saving the model image",
    )
    parser.add_argument(
        "--no_save_cov",
        action="store_false",
        dest="save_covariance_matrix",
        help="Disable saving the covariance matrix",
    )
    parser.add_argument(
        "--target_hdu", type=int, default=0, help="FITS file index for target image data"
    )
    parser.add_argument("--psf_hdu", type=int, default=0, help="FITS file index for PSF data")
    parser.add_argument(
        "--sky_model_type",
        type=str,
        default="flat",
        help="Type of sky model to fit, options include 'flat' or 'plane'",
    )

    # Parse known arguments, leave the rest for the dynamic dictionary
    args, unknown = parser.parse_known_args()

    # Parse the arbitrary parameters for the model
    initial_params = parse_arbitrary_args(unknown)

    if args.verbose == 0:
        ap.config.set_logging_output(stdout=False, filename=None)

    # Load Target Data
    # ---------------------------------------------------------------------
    if args.verbose > 0:
        print("Loading target image...")
    with fits.open(args.target_file) as hdu:
        target_data = np.array(hdu[args.target_hdu].data, dtype=np.float64)

    # Load Variance Data
    # ---------------------------------------------------------------------
    variance_data = None
    if args.variance_file is not None:
        if args.verbose > 0:
            print("Loading variance image...")
        with fits.open(args.variance_file) as hdu:
            variance_data = np.array(hdu[args.variance_hdu].data, dtype=np.float64)

    # Load Mask Data
    # ---------------------------------------------------------------------
    mask_data = None
    if args.mask_file is not None:
        if args.verbose > 0:
            print("Loading mask image...")
        with fits.open(args.mask_file) as hdu:
            mask_data = np.array(hdu[args.mask_hdu].data)

    # Load PSF
    # ---------------------------------------------------------------------
    psf_data = None
    if args.psf_file is not None:
        if args.verbose > 0:
            print("Loading PSF...")
        with fits.open(args.psf_file) as hdu:
            psf_data = np.array(hdu[args.psf_hdu].data, dtype=np.float64)
        psf_data = ap.PSFImage(data=psf_data, upsample=args.psf_upsample)

    # Make Target
    # ---------------------------------------------------------------------
    target = ap.TargetImage(
        data=target_data,
        zeropoint=args.zeropoint,
        variance=variance_data,
        mask=mask_data,
        psf=psf_data,
    )

    # Create Model
    # ---------------------------------------------------------------------
    model_object = ap.Model(
        name=args.name,
        model_type=args.model_type.replace("_", " "),
        target=target,
        **initial_params,
        window=args.window,
    )

    model_sky = ap.Model(
        name="sky",
        model_type=args.sky_model_type + " sky model",
        target=target,
        I0=args.initial_sky,
        window=args.window,
    )

    if args.sky_locked:
        model_sky.to_static()

    model = ap.Model(
        name="astrophot_model",
        model_type="group model",
        target=target,
        models=[model_sky, model_object],
    )

    # Fit the model
    # ---------------------------------------------------------------------
    if args.verbose > 0:
        print("Initializing model...")
    model.initialize()
    if args.verbose > 0:
        print("Fitting model...")
    result = ap.fit.LM(model, verbose=args.verbose).fit()

    # Report Total Magnitude or Flux
    # ----------------------------------------------------------------------
    if args.verbose > 0:
        print(model)
        if args.zeropoint is not None:
            totmag = model_object.total_magnitude().detach().cpu().numpy()
            totmag_err = model_object.total_magnitude_uncertainty().detach().cpu().numpy()
            print(f"Total Magnitude: {totmag} +- {totmag_err}")
        else:
            totflux = model_object.total_flux().detach().cpu().numpy()
            totflux_err = model_object.total_flux_uncertainty().detach().cpu().numpy()
            print(f"Total Flux: {totflux} +- {totflux_err}")

    # Save Results
    # ----------------------------------------------------------------------
    model.save_state(f"{args.name}_parameters.hdf5")
    if args.save_images:
        model().save(f"{args.name}_model_image.fits")
        (target[model.window] - model()).save(f"{args.name}_residual_image.fits")

    if args.save_covariance_matrix:
        np.save(
            f"{args.name}_covariance_matrix.npy", result.covariance_matrix.detach().cpu().numpy()
        )


if __name__ == "__main__":
    main()

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
# >>> python single_model_cli.py target_image.fits [OPTIONS]
# =============================================================================

import astrophot as ap
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import argparse
import ast

try:
    import yaml
except ImportError:
    from astropy.io.misc import yaml


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


def to_serializable(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.array(value)
    return value.item() if value.shape == () else value.tolist()


def collect_model_parameters(model):
    params = model.dynamic_params
    names = tuple(param.name for param in params)
    values = tuple(to_serializable(param.npvalue) for param in params)
    uncertainties = tuple(to_serializable(param.uncertainty) for param in params)
    return {
        name: {"value": val, "uncertainty": unc}
        for name, val, unc in zip(names, values, uncertainties)
    }


def main():
    parser = argparse.ArgumentParser(description="Fit a model to a target image using AstroPhot.")

    # Core Parameters
    parser.add_argument("target_file", type=str, help="Path to the target FITS file")
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        nargs=4,
        metavar=("IMIN", "IMAX", "JMIN", "JMAX"),
        help="Window border for the fit in pixel coordinates, given as: --window imin imax jmin jmax",
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
        help="Disable saving the model and residual images",
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
    target_wcs = WCS(fits.getheader(args.target_file, args.target_hdu))
    target = ap.TargetImage(
        data=target_data,
        wcs=target_wcs,
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
    total_flux = to_serializable(model_object.total_flux())
    total_flux_uncertainty = to_serializable(model_object.total_flux_uncertainty())
    total_magnitude = None
    total_magnitude_uncertainty = None
    if args.zeropoint is not None:
        total_magnitude = to_serializable(model_object.total_magnitude())
        total_magnitude_uncertainty = to_serializable(model_object.total_magnitude_uncertainty())
    if args.verbose > 0:
        print(model)
        if args.zeropoint is not None:
            print(f"Total Magnitude: {total_magnitude} +- {total_magnitude_uncertainty}")
        else:
            print(f"Total Flux: {total_flux} +- {total_flux_uncertainty}")

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
        },
        "sky_model": {
            "model_type": model_sky.model_type,
            "parameters": collect_model_parameters(model_sky),
        },
    }
    if len(model_object.static_params) > 0:
        output_summary[model_object.name]["static_parameters"] = [
            param.name for param in model_object.static_params
        ]

    with open(f"{args.name}_parameters.yaml", "w", encoding="utf-8") as output_file:
        yaml.dump(output_summary, output_file, default_flow_style=False)

    if args.save_images:
        model().save(f"{args.name}_model_image.fits")
        (target[model.window] - model()).save(f"{args.name}_residual_image.fits")

    if args.save_covariance_matrix:
        np.save(
            f"{args.name}_covariance_matrix.npy", result.covariance_matrix.detach().cpu().numpy()
        )


if __name__ == "__main__":
    main()

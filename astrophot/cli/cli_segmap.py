# =============================================================================
# Fit all objects identified in a segmentation map
#
# This is a quick script to fit all the objects identified in a segmentation map
# using a single model type. The script will load the target image, mask,
# psf, and variance image (if available) and fit the models to the target image.
#
# Run this script with:
# ~$ python segmap_astrophot_model.py target.fits segmap.fits [OPTIONS]
# =============================================================================

import astrophot as ap
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import argparse

try:
    import yaml
except ImportError:
    from astropy.io.misc import yaml


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
    parser = argparse.ArgumentParser(
        description="Fit a model to a series of targets in an image using AstroPhot."
    )

    # Core Parameters
    parser.add_argument("target_file", type=str, help="Path to the target FITS file")
    parser.add_argument("segmap_file", type=str, help="Path to the segmentation map FITS file")
    parser.add_argument(
        "--cat",
        type=str,
        default=None,
        help="Path to a catalogue yaml file with initial parameters for each segmentation map id",
    )
    parser.add_argument(
        "--name", type=str, default="astrophot_model", help="Prefix name used for models"
    )
    parser.add_argument("--psf", type=str, default=None, help="Path to the PSF FITS file")
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
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level for fitting output")
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump this file to the current directory as 'single_astrophot_model.py' for editing and running",
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
        help="Minimum size of windows to include in the fit. Before expanding. Number of pixels.",
    )
    parser.add_argument(
        "--filter_ids",
        type=int,
        nargs="+",
        default=[0],
        help="List of segmentation map ids to exclude from fitting. Default is [0] to exclude the background",
    )

    # Variance and Mask Parameters
    parser.add_argument("--variance", type=str, default=None, help="Path to the variance FITS file")
    parser.add_argument(
        "--variance_hdu", type=int, default=0, help="FITS file index for variance data"
    )
    parser.add_argument("--mask", type=str, default=None, help="Path to the mask FITS file")
    parser.add_argument("--mask_hdu", type=int, default=0, help="FITS file index for mask data")

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
    parser.add_argument(
        "--segmap_hdu", type=int, default=0, help="FITS file index for segmentation map data"
    )
    parser.add_argument("--psf_hdu", type=int, default=0, help="FITS file index for PSF data")
    parser.add_argument(
        "--sky_model_type",
        type=str,
        default="flat",
        help="Type of sky model to fit, options include 'flat' or 'plane'",
    )

    # Parse known arguments, leave the rest for the dynamic dictionary
    args = parser.parse_args()

    if args.dump:
        with open("segmap_astrophot_model.py", "w", encoding="utf-8") as f:
            f.write(
                "# This file was generated by dumping the segmap_model_cli.py script. You can edit and run this file directly.\n\n"
            )
            with open(__file__, "r", encoding="utf-8") as original:
                f.write(original.read())
        print("Dumped to segmap_astrophot_model.py")

    # Load Target Data
    # ---------------------------------------------------------------------
    if args.verbose > 0:
        print("Loading target image...")
    with fits.open(args.target_file) as hdu:
        target_data = np.array(hdu[args.target_hdu].data, dtype=np.float64)

    # Load Variance Data
    # ---------------------------------------------------------------------
    variance_data = None
    if args.variance is not None:
        if args.verbose > 0:
            print("Loading variance image...")
        with fits.open(args.variance) as hdu:
            variance_data = np.array(hdu[args.variance_hdu].data, dtype=np.float64)

    # Load Mask Data
    # ---------------------------------------------------------------------
    mask_data = None
    if args.mask is not None:
        if args.verbose > 0:
            print("Loading mask image...")
        with fits.open(args.mask) as hdu:
            mask_data = np.array(hdu[args.mask_hdu].data)

    # Load PSF
    # ---------------------------------------------------------------------
    psf_data = None
    if args.psf is not None:
        if args.verbose > 0:
            print("Loading PSF...")
        with fits.open(args.psf) as hdu:
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

    # Load Segmentation Map
    # ---------------------------------------------------------------------
    if args.verbose > 0:
        print("Loading segmentation map...")
    with fits.open(args.segmap_file) as hdu:
        segmap_data = np.array(hdu[args.segmap_hdu].data, dtype=np.int32)

    # Load Catalogue of Initial Parameters
    # ---------------------------------------------------------------------
    override_init_params = {}
    if args.cat is not None:
        if args.verbose > 0:
            print("Loading catalogue of initial parameters...")
        with open(args.cat, "r", encoding="utf-8") as cat_file:
            override_init_params = yaml.safe_load(cat_file)

    # Initialization from segmap
    # ---------------------------------------------------------------------
    if args.verbose > 0:
        print("Parsing segmentation map")
    windows = ap.utils.initialize.windows_from_segmentation_map(
        segmap_data, skip_index=args.filter_ids
    )

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

    centers = ap.utils.initialize.centroids_from_segmentation_map(segmap_data, target)
    if "galaxy" in args.model_type:
        PAs = ap.utils.initialize.PA_from_segmentation_map(segmap_data, target, centers)
        qs = ap.utils.initialize.q_from_segmentation_map(segmap_data, target, centers)
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
    model_images = {}
    residual_images = {}
    cov_matrices = {}
    for window in windows:
        if args.verbose > 0:
            print(f"Fitting model for window {window}...")
        W = ap.Window(windows[window], target)
        subtarget = target[W]
        sub_segmap = segmap_data[target.get_indices(W)]
        # Mask interlopers
        subtarget.mask = np.logical_or(
            subtarget.mask, ~(np.logical_or(sub_segmap == 0, sub_segmap == window)).T
        )
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
            name=f"{args.name}_{window}",
            model_type="group model",
            target=subtarget,
            models=[sky, object_model],
        )
        # Fitting
        model.initialize()
        result = ap.fit.LM(model, verbose=args.verbose - 1).fit()
        # Collect results
        results[model.name] = {
            "parameters": collect_model_parameters(object_model),
            "total_flux": to_serializable(object_model.total_flux()),
            "total_flux_uncertainty": to_serializable(object_model.total_flux_uncertainty()),
            "sky_parameters": collect_model_parameters(sky),
            "message": result.message,
        }
        if args.save_covariance_matrix:
            cov_matrices[model.name] = to_serializable(result.covariance_matrix)
        if args.save_images:
            model_images[model.name] = to_serializable(model().data)
            residual_images[model.name] = to_serializable((subtarget - model()).data)
        if args.zeropoint is not None:
            results[model.name]["total_magnitude"] = to_serializable(object_model.total_magnitude())
            results[model.name]["total_magnitude_uncertainty"] = to_serializable(
                object_model.total_magnitude_uncertainty()
            )

    # Save outputs
    # ---------------------------------------------------------------------
    if args.save_covariance_matrix:
        np.savez(f"{args.name}_covariance_matrix.npz", **cov_matrices)

    if args.save_images:
        hdul = fits.HDUList()
        for name, image in model_images.items():
            hdul.append(fits.ImageHDU(data=image, name=f"{name}_model"))
        hdul.writeto(f"{args.name}_model_images.fits", overwrite=True)
        hdul = fits.HDUList()
        for name, image in residual_images.items():
            hdul.append(fits.ImageHDU(data=image, name=f"{name}_residual"))
        hdul.writeto(f"{args.name}_residual_images.fits", overwrite=True)

    with open(f"{args.name}_parameters.yaml", "w", encoding="utf-8") as output_file:
        yaml.dump(results, output_file, default_flow_style=False)

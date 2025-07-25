# =============================================================================
# Fit a single model to a target image
#
# This is a quick script to fit a single model to a target image. You should set
# the parameters under PARAMETERS to be appropriate for your data. The script
# will load the target image, mask, psf, and variance image (if available) and
# fit the model to the target image. The script will save the model image,
# residual image, and covariance matrix to the current directory. This script is
# intended for quick easy fits, users more comfortable with configuration file
# style behavior, and as a starting point to build a more complex analysis.
#
# Run this script with:
# >>> python single_model_fit.py
# =============================================================================

import astrophot as ap
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# PARAMETERS
######################################################################
name = "object_name"  # used for saving files
target_file = "<required>.fits"  # can be a numpy array instead
psf_file = None  # "<path to psf>.fits" # can be a numpy array instead
zeropoint = 22.5  # mag
initial_params = None  # e.g. {"center": [3, 3], "q": {"value": 0.8, "locked": True}}
window = None  # None to fit whole image, otherwise (xmin,xmax,ymin,ymax) pixels
initial_sky = None  # If None, sky will be estimated
sky_locked = False
model_type = "sersic galaxy model"
# Extra parameters
######################################################################
save_model_image = True
save_residual_image = True
save_covariance_matrix = True
target_hdu = 0  # FITS file index for image data
psf_hdu = 0
sky_model_type = "flat sky model"
######################################################################

# load target
# ---------------------------------------------------------------------
print("loading target")
target = ap.TargetImage(
    filename=target_file,
    hduext=target_hdu,
    zeropoint=zeropoint,
)

# PSF
if isinstance(psf_file, str):
    print("loading psf")
    hdu = fits.open(psf_file)
    psf_data = np.array(hdu[psf_hdu].data, dtype=np.float64)
    target.psf = target.psf_image(data=psf_data)
elif psf_file is None:
    psf = None
else:
    target.psf = target.psf_image(data=psf_file)

# Create Model
# ---------------------------------------------------------------------
model_object = ap.Model(
    name=name,
    model_type=model_type,
    target=target,
    psf_convolve=True if psf_file is not None else False,
    **initial_params,
    window=window,
)
model_sky = ap.Model(
    name="sky",
    model_type=sky_model_type,
    target=target,
    I=initial_sky if initial_sky is not None else {},
    window=window,
)
if sky_locked:
    model_sky.to_static()
model = ap.Model(
    name="astrophot model",
    model_type="group model",
    target=target,
    models=[model_sky, model_object],
)

# Fit the model
# ---------------------------------------------------------------------
print("Initializing model")
model.initialize()
print("Fitting model")
result = ap.fit.LM(model, verbose=1).fit()

# Report Results
# ----------------------------------------------------------------------
print(model)
totmag = model_object.total_magnitude().detach().cpu().numpy()
totmag_err = model_object.total_magnitude_uncertainty().detach().cpu().numpy()
print(f"Total Magnitude: {totmag} +- {totmag_err}")

model.save_state(f"{name}_parameters.hdf5")
if save_model_image:
    model().save(f"{name}_model_image.fits")
    fig, ax = plt.subplots()
    ap.plots.model_image(fig, ax, model)
    plt.savefig(f"{name}_model_image.jpg")
    plt.close()
    if hasattr(model_object, "radial_model"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ap.plots.radial_light_profile(fig, ax, model_object)
        plt.savefig(f"{name}_radial_light_profile.jpg")
        plt.close()
if save_residual_image:
    (target - model()).save(f"{name}_residual_image.fits")
    fig, ax = plt.subplots()
    ap.plots.residual_image(fig, ax, model, normalize_residuals=True)
    plt.savefig(f"{name}_residual_image.jpg")
    plt.close()

if save_covariance_matrix:
    np.save(f"{name}_covariance_matrix.npy", result.covariance_matrix.detach().cpu().numpy())
    fig, ax = ap.plots.covariance_matrix(
        result.covariance_matrix.detach().cpu().numpy(),
        model.parameters.vector_values().detach().cpu().numpy(),
        model.parameters.vector_names(),
    )
    fig.suptitle("Parameter Covariance")
    plt.savefig(f"{name}_covariance_matrix.pdf")
    plt.close()

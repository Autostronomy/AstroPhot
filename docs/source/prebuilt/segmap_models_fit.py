# =============================================================================
# Fit all objects identified in a segmentation map
#
# This is a quick script to fit all the objects identified in a segmentation map
# using a single model type. You should set the parameters under PARAMETERS to
# be appropriate for your data. The script will load the target image, mask,
# psf, and variance image (if available) and fit the models to the target image.
#
# Run this script with:
# >>> python segmap_models_fit.py
# =============================================================================

import astrophot as ap
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# PARAMETERS
######################################################################
name = "field_name"  # used for saving files
target_file = "<required>.fits"  # can be a numpy array instead
segmap_file = "<required>.fits"  # can be a numpy array instead
mask_file = None  # "<path to mask>.fits" # can be a numpy array instead
psf_file = None  # "<path to psf>.fits" # can be a numpy array instead
variance_file = None  # "<path to variance>.fits" # can be a numpy array or "auto" instead
pixelscale = 0.1  # arcsec/pixel
zeropoint = 22.5  # mag
initial_sky = None  # If None, sky will be estimated
sky_fixed = False
model_type = "sersic galaxy model"  # model type for segmap entries
segmap_filter = {}
primary_key = 1  # object number in segmentation map, use None to have no primary object
primary_model_type = "sersic galaxy model"
primary_initial_params = None  # e.g. {"center": [3, 3], "q": {"value": 0.8, "locked": True}}
primary_window = None  # None to fit whole image, otherwise ((xmin,xmax),(ymin,ymax)) pixels
save_model_image = True
save_residual_image = True
save_covariance_matrix = True
######################################################################

# load target and segmentation map
# ---------------------------------------------------------------------
print("loading target and segmentation map")
if isinstance(target_file, str):
    hdu = fits.open(target_file)
    target_data = np.array(hdu[0].data, dtype=np.float64)
else:
    target_data = target_file

if isinstance(segmap_file, str):
    hdu = fits.open(segmap_file)
    segmap_data = np.array(hdu[0].data, dtype=np.int32)
else:
    segmap_data = segmap_file

# load mask, variance, and psf
# ---------------------------------------------------------------------
# Mask
if isinstance(mask_file, str):
    print("loading mask")
    hdu = fits.open(mask_file)
    mask_data = np.array(hdu[0].data, dtype=bool)
elif mask_file is None:
    mask_data = None
else:
    mask_data = mask_file
# Variance
if isinstance(variance_file, str) and not variance_file == "auto":
    print("loading variance")
    hdu = fits.open(variance_file)
    variance_data = np.array(hdu[0].data, dtype=np.float64)
elif variance_file is None:
    variance_data = None
else:
    variance_data = variance_file
# PSF
if isinstance(psf_file, str):
    print("loading psf")
    hdu = fits.open(psf_file)
    psf_data = np.array(hdu[0].data, dtype=bool)
    psf = ap.image.PSF_Image(
        data=psf_data,
        pixelscale=pixelscale,
    )
elif psf_file is None:
    psf = None
else:
    psf = ap.image.PSF_Image(
        data=psf_file,
        pixelscale=pixelscale,
    )

# Create target object
# ---------------------------------------------------------------------
target = ap.image.Target_Image(
    data=target_data,
    pixelscale=pixelscale,
    zeropoint=zeropoint,
    mask=mask_data,
    psf=psf,
    variance=variance_data,
)

# Create Models
# ---------------------------------------------------------------------
models = []
models.append(
    ap.models.AstroPhot_Model(
        name="sky",
        model_type="flat sky model",
        target=target,
        parameters={"F": initial_sky} if initial_sky is not None else {},
    )
)
windows = ap.utils.initialize.windows_from_segmentation_map(segmap_data)
windows = ap.utils.initialize.scale_windows(
    windows, image_shape=target_data.shape, expand_scale=2, expand_border=10
)
centers = ap.utils.initialize.centroids_from_segmentation_map(segmap_data, target_data)
if "galaxy" in model_type:
    PAs = ap.utils.initialize.PA_from_segmentation_map(segmap_data, target_data, centers)
    qs = ap.utils.initialize.q_from_segmentation_map(segmap_data, target_data, centers, PAs)

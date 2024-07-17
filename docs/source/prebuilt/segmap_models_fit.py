# =============================================================================
# Fit all objects identified in a segmentation map
#
# This is a quick script to fit all the objects identified in a segmentation map
# using a single model type. You should set the parameters under PARAMETERS to
# be appropriate for your data. The script will load the target image, mask,
# psf, and variance image (if available) and fit the models to the target image.
#
# First a fit will be run on tight windows exactly enclosing the segmentations
# for each object. Then the windows will be expanded by the set factors and the
# fit will be run again. This is more stable than fitting the expanded windows
# from the start since it reduces the effects of overlap
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
variance_file = None  # "<path to variance>.fits" # or numpy array or "auto"
pixelscale = 0.1  # arcsec/pixel
zeropoint = 22.5  # mag
initial_sky = None  # If None, sky will be estimated. Recommended to set manually
sky_locked = False
model_type = "sersic galaxy model"  # model type for segmap entries
segmap_filter = {}  # in pixels or ADU: min_size, min_area, min_flux
segmap_filter_ids = []  # list of segmap ids to remove from fit
segmap_override_init_params = {}  # Override some initial parameters for segmap models
primary_key = None  # segmentation map id, use None to have no primary object
primary_name = "primary object"  # name for primary object
primary_model_type = "spline galaxy model"
primary_initial_params = {}  # {"center": [3, 3], "q": {"value": 0.8, "locked": True}}
# Extra parameters
######################################################################
save_model_image = True
save_residual_image = True
target_hdu = 0  # FITS file index for image data
segmap_hdu = 0
mask_hdu = 0
variance_hdu = 0
psf_hdu = 0
window_expand_scale = 2  # Windows from segmap will be expanded by this factor
window_expand_border = 10  # Windows from segmap will be expanded by this number of pixels
sky_model_type = "flat sky model"
print_all_models = True
######################################################################

# load target and segmentation map
# ---------------------------------------------------------------------
print("loading target and segmentation map")
if isinstance(target_file, str):
    hdu = fits.open(target_file)
    target_data = np.array(hdu[target_hdu].data, dtype=np.float64)
else:
    target_data = target_file

if isinstance(segmap_file, str):
    hdu = fits.open(segmap_file)
    segmap_data = np.array(hdu[segmap_hdu].data, dtype=np.int32)
else:
    segmap_data = segmap_file

# load mask, variance, and psf
# ---------------------------------------------------------------------
# Mask
if isinstance(mask_file, str):
    print("loading mask")
    hdu = fits.open(mask_file)
    mask_data = np.array(hdu[mask_hdu].data, dtype=bool)
elif mask_file is None:
    mask_data = None
else:
    mask_data = mask_file
# Variance
if isinstance(variance_file, str) and not variance_file == "auto":
    print("loading variance")
    hdu = fits.open(variance_file)
    variance_data = np.array(hdu[variance_hdu].data, dtype=np.float64)
elif variance_file is None:
    variance_data = None
else:
    variance_data = variance_file
# PSF
if isinstance(psf_file, str):
    print("loading psf")
    hdu = fits.open(psf_file)
    psf_data = np.array(hdu[psf_hdu].data, dtype=bool)
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

# Initialization from segmap
# ---------------------------------------------------------------------
print("Parsing segmentaiton map")
windows = ap.utils.initialize.windows_from_segmentation_map(segmap_data)
if len(segmap_filter) > 0:
    windows = ap.utils.initialize.filter_windows(
        windows,
        **segmap_filter,
        image=target_data,
    )

for ids in segmap_filter_ids:
    del windows[ids]
centers = ap.utils.initialize.centroids_from_segmentation_map(segmap_data, target_data)
if "galaxy" in model_type:
    PAs = ap.utils.initialize.PA_from_segmentation_map(segmap_data, target_data, centers)
    qs = ap.utils.initialize.q_from_segmentation_map(segmap_data, target_data, centers, PAs)
else:
    PAs = None
    qs = None
init_params = {}
for window in windows:
    init_params[window] = {
        "center": np.array(centers[window]) * pixelscale,
    }
    if "galaxy" in model_type:
        init_params[window]["PA"] = PAs[window]
        init_params[window]["q"] = qs[window]
    init_params[window].update(segmap_override_init_params)

# Create Models
# ---------------------------------------------------------------------
print("Creating models")
models = []
models.append(
    ap.models.AstroPhot_Model(
        name="sky",
        model_type=sky_model_type,
        target=target,
        parameters={"F": initial_sky} if initial_sky is not None else {},
        locked=sky_locked,
    )
)
primary_model = None
for window in windows:
    if primary_key is not None and window == primary_key:
        print(primary_name, window)
        if "center" not in primary_initial_params:
            primary_initial_params["center"] = init_params[window]["center"]
        if (
            "PA" not in primary_initial_params
            and PAs is not None
            and "galaxy" in primary_model_type
        ):
            primary_initial_params["PA"] = PAs[window]
        if "q" not in primary_initial_params and qs is not None and "galaxy" in primary_model_type:
            primary_initial_params["q"] = qs[window]
        model = ap.models.AstroPhot_Model(
            name=primary_name,
            model_type=primary_model_type,
            target=target,
            parameters=primary_initial_params,
            window=windows[window],
        )
        primary_model = model
    else:
        print(window)
        model = ap.models.AstroPhot_Model(
            name=f"{model_type} {window}",
            model_type=model_type,
            target=target,
            window=windows[window],
            parameters=init_params[window],
        )
    models.append(model)
model = ap.models.AstroPhot_Model(
    name=f"{name} model",
    model_type="group model",
    target=target,
    models=models,
)

# Fit the model
# ---------------------------------------------------------------------
print("Initializing model")
model.initialize()
print("Fitting model")
result = ap.fit.Iter(model, verbose=1).fit()
print("expanding windows")
windows = ap.utils.initialize.scale_windows(
    windows,
    image_shape=target_data.shape,
    expand_scale=window_expand_scale,
    expand_border=window_expand_border,
)
for i, window in enumerate(windows):
    models[i + 1].window = windows[window]
print("Fitting round 2")
result = ap.fit.Iter(model, verbose=1).fit()
# result.update_uncertainty() coming soon

# Report Results
# ----------------------------------------------------------------------
if not sky_locked:
    print(models[0].parameters)

if not primary_model is None:
    print(primary_model.parameters)
    totflux = primary_model.total_flux().detach().cpu().numpy()
    print(f"Total Magnitude: {zeropoint - 2.5 * np.log10(totflux)}")
    if hasattr(primary_model, "radial_model"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ap.plots.radial_light_profile(fig, ax, primary_model)
        plt.savefig(f"{name}_radial_light_profile.jpg")
        plt.close()

if print_all_models:
    segmap_params = []
    for segmodel in models[1:]:
        if segmodel.name == primary_name:
            continue
        print(segmodel.parameters)
        totflux = segmodel.total_flux().detach().cpu().numpy()
        segmap_params.append(
            [segmodel.name, totflux]
            + list(segmodel.parameters.vector_values().detach().cpu().numpy())
        )
    with open(f"{name}_segmap_params.csv", "w") as f:
        f.write("Name,Total Flux," + ",".join(segmodel.parameters.vector_names()) + "\n")
        flat_params = segmodel.parameters.flat(False, False).values()
        f.write(
            "string,mag," + ",".join(p.units for p in flat_params for _ in range(p.size)) + "\n"
        )
        for row in segmap_params:
            f.write(",".join([str(x) for x in row]) + "\n")

model.save(f"{name}_parameters.yaml")
if save_model_image:
    model().save(f"{name}_model_image.fits")
    fig, ax = plt.subplots()
    ap.plots.model_image(fig, ax, model)
    plt.savefig(f"{name}_model_image.jpg")
    plt.close()
if save_residual_image:
    (target - model()).save(f"{name}_residual_image.fits")
    fig, ax = plt.subplots()
    ap.plots.residual_image(fig, ax, model, normalize_residuals=True)
    plt.savefig(f"{name}_residual_image.jpg")
    plt.close()

# Download the data
from astropy.io import fits
hdu = fits.open("https://www.legacysurvey.org/viewer/\
fits-cutout?ra=36.3684&dec=-25.6389&size=700&layer=\
ls-dr9&pixscale=0.262&bands=r")
hdu.writeto("ESO479-G1.fits", overwrite = True)

# Image features
ap_target_file = "ESO479-G1.fits"
ap_target_pixelscale = 0.262
ap_target_zeropoint = 22.5

# Main galaxy model
ap_model_ESO479_G1 = {"model_type": "spline galaxy model"}

# Interloper galaxies
ap_model_sub1 = {"model_type": "sersic galaxy model",
                 "window": [[480, 590],[555, 665]]}
ap_model_sub2 = {"model_type": "sersic galaxy model",
                 "window": [[572, 630],[534, 611]]}
ap_model_sub3 = {"model_type": "sersic galaxy model",
                 "window": [[183, 240],[0, 46]]}
ap_model_sub4 = {"model_type": "sersic galaxy model",
                 "window": [[103, 167], [557, 610]]}
ap_model_sub5 = {
    "model_type": "sersic galaxy model",
    "window": [[336, 385], [15, 65]],
    "parameters": {
        "center": [95.,10.], # arcsec from bottom corner
        "q": 0.9, # b / a
        "PA": 2.85, # radians
        "n": 1., # sersic index
        "Re": 0.6, # arcsec
        "Ie": 1.3, # log10(flux / arcsec^2)
    }}
 
# Optimizer
ap_optimizer_kwargs = {"verbose": 1, "max_iter": 100}

# Output
ap_saveto_model = "spline_model.yaml"
ap_saveto_model_image = "spline_model_image.fits"
ap_saveto_model_residual = "spline_model_residual.fits"

# This is for demo purposes, in general you would have the data saved
# separately. That said, it shows how you can put arbitrary python
# code in a config file!
######################################################################
from astropy.io import fits
hdu = fits.open("https://www.legacysurvey.org/viewer/fits-cutout?ra=36.3684&dec=-25.6389&size=700&layer=ls-dr9&pixscale=0.262&bands=r")
hdu.writeto("simpleconfig_ESO479-G1.fits", overwrite = True)


# This is the actual config file
######################################################################
ap_target_file = "simpleconfig_ESO479-G1.fits"
ap_target_pixelscale = 0.262
ap_target_zeropoint = 22.5

ap_model_sky = {"model_type": "flat sky model"}
ap_model_ESO479_G1 = {"model_type": "sersic galaxy model"}

ap_optimizer_kwargs = {"verbose": 1}

from .autoprof_node import make_AP_Process
import numpy as np
from autoprof.utils.image_operations import StarFind
from autoprof.image import PSF_Image

@make_AP_Process("gaussian psf")
def gaussian_psf(state):
    """
    fit a PSF to the image as a gaussian
    """

    edge_mask = np.zeros(IMG.shape, dtype=bool)
    edge_mask[
        int(IMG.shape[0] / 5.0) : int(4.0 * IMG.shape[0] / 5.0),
        int(IMG.shape[1] / 5.0) : int(4.0 * IMG.shape[1] / 5.0),
    ] = True
    stars = StarFind(
        IMG - results["background"],
        fwhm_guess,
        results["background noise"],
        edge_mask,
        maxstars=50,
    )
    if len(stars["fwhm"]) <= 10:
        logging.error(
            "%s: unable to detect enough stars! PSF results not valid, using 1 arcsec estimate psf of %f"
            % (options["ap_name"], fwhm_guess)
        )
        return IMG, {"psf fwhm": fwhm_guess}

    def_clip = 0.1
    while np.sum(stars["deformity"] < def_clip) < max(10, len(stars["fwhm"]) / 2):
        def_clip += 0.1
    psf = np.median(stars["fwhm"][stars["deformity"] < def_clip])

    psf_size = int(psf*3)
    psf_size += 1 - (psf_size % 2)
    XX, YY = np.meshgrid(np.linspace(-(psf_size-1)/2, (psf_size-1)/2, psf_size),
                         np.linspace(-(psf_size-1)/2, (psf_size-1)/2, psf_size)
    )
    psf_img = np.exp(-0.5*(XX**2 + YY**2)/(psf / (2*np.sqrt(2*np.log(2))))**2)
    psf_img /= np.sum(psf_img)
    state.data.update_psf(PSF_Image(pixelscale = state.data.image.pixelscale, image = psf_img))
    
    return state

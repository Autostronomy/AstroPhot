from .autoprof_node import make_AP_Process

@make_AP_Process("gaussian psf")
def gaussian_psf(state):
    """
    fit a PSF to the image as a gaussian
    """
    return state

from .autoprof_node import make_AP_Process

@make_AP_Process("psf image")
def psf_image(state):
    """
    fit a PSF to the image by stacking many images of stars 
    """
    return state

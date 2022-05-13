from .autoprof_node import make_AP_Process

@make_AP_Process("psf image")
def sigma_image(state):
    """
    determine the uncertainty on each pixel. This can come from an inputted sigma image, or by assesing the noise in the image and guessing approximate uncertainties. 
    """
    return state

from .autoprof_node import make_AP_Process
from autoprof.image import AP_Image

@make_AP_Process("load image")
def load_image(state):
    """
    Load the image from memory which is to be fit by the model(s)
    """

    raise NotImplementedError('doesnt yet load an image')
    state.data.update_image(AP_Image(pixelscale = state.options['ap_pixelscale'], filename = state.options['ap_image_file']))
    
    return state

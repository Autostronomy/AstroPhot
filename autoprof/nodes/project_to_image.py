from .autoprof_node import make_AP_Process
from autoprof.image import Model_Image

@make_AP_Process("project to image")
def project_to_image(state):
    """
    Construct a full model image from the various individual models in the state.
    """

    model_image = Model_Image(pixelscale = state.data.pixelscale, shape = state.data.image.shape)
    for model in state.models:
        model.convolve_psf()
        model_image.add(model)
    state.results['model image'] = model_image
    
    return state

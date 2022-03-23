from ..autoprof_node import make_AP_Process

@make_AP_Process("resample image")
def resample_image(state):
    """
    Resample the image at a different resolution.
    """
    return state

from ..autoprof_node import make_AP_Process

@make_AP_Process("load image")
def load_image(state):
    """
    Load the image from memory which is to be fit by the model(s)
    """
    return state

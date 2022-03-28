from .autoprof_node import make_AP_Process

@make_AP_Process("initialize models")
def initialize_models(state):
    """
    Initialize the model parameters using basic fitting to the image.
    """

    state.models.initialize()
    
    return state

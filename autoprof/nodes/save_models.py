from ..autoprof_node import make_AP_Process

@make_AP_Process("save models")
def save_models(state):
    """
    Save the models to memory with the fitted parameters.
    """
    return state

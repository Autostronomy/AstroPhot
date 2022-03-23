from .autoprof_node import make_AP_Process

@make_AP_Process("sample models")
def sample_models(state):
    """
    Create a model image based on the current set of model parameters.
    """
    return state

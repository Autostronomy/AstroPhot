from ..autoprof_node import make_AP_Process

@make_AP_Process("update parameters")
def update_parameters(state):
    """
    Update the model parameters with a step that minimizes the loss using information from previous parameter-loss pairs.
    """
    return state

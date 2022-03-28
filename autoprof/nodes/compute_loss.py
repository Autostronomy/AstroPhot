from .autoprof_node import make_AP_Process

@make_AP_Process("compute loss")
def compute_loss(state):
    """
    For the relevant models, have them compute an updated loss metric for their latest set of parameters.
    """

    for model in state.models:
        if model.locked:
            continue
        model.compute_loss()
    
    return state

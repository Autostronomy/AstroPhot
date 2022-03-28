from .autoprof_node import make_AP_Process

@make_AP_Process("stocastic k-delta update parameters")
def stocastic_kdelta_update_parameters(state):
    """
    Update the model parameters with a step that minimizes the loss using information from previous parameter-loss pairs.
    Stocastic kdelta operates by computing the gradient on the hyperplane defined by "k" samples of the loss function. A
    stocastic update is included to ensure the full parameter space can be explored even when k is less than dimensions
    plus one.
    """
    return state

from .autoprof_node import make_AP_Process

@make_AP_Process("random update parameters")
def random_update_parameters(state):
    """
    Update the model parameters with a step that minimizes the loss using information from previous parameter-loss pairs.
    Using a window of k previous loss samples, select the best performing set of parameters and perform a random step in
    the parameter space.
    """

    for model in state.models:
        if model.locked:
            continue
        N = np.argmin(model.get_loss(slice(0,5)))
        parameters = model.get_parameters_representation()
        new_parameters = {}
        for p in parameters:
            new_parameters[p] = parameters[p] + np.random.normal(scale = 0.1)
        model.set_parameters_representation(parameters)
    
    return state

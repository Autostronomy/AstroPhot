import os
from .autoprof_node import make_AP_Process

@make_AP_Process("save models")
def save_models(state):
    """
    Save the models to memory with the fitted parameters.
    """

    saveto = os.path.join(state.options['ap_saveto'], state.options['ap_name'] + '.prof')
    with open(saveto, 'w') as f:
        for model in state.models:
            model_string = model.save()
            f.write(model_string)
            f.write('\n')
     
    return state

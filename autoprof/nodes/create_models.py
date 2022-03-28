from .autoprof_node import make_AP_Process
from autoprof.models import *

@make_AP_Process("create models")
def create_models(state):
    """
    Create models and add them to the state based on options specified by the user.
    """

    if 'ap_models' in state.options:
        for m in state.options['ap_models']:
            state.models.add_model(name = m, **state.options['ap_models'][m])
    else:
        for n, m in [('sky', 'flat sky'), ('galaxy', 'nonparametric ellipse')]:
            state.models.add_model(name = n, model = m)
    
    return state

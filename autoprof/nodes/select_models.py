from .autoprof_node import make_AP_Process

@make_AP_Process("select models")
def select_models(state):
    """
    Select which models have the highest loss and are most in need of updated parameters.
    This way computation power isn't wasted on models which have already converged.

    For models which overlap it is important to update both even if one has converged given their covariance.
    """
    
    return state

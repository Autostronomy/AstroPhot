from ..autoprof_node import make_AP_Process

@make_AP_Process("project to image")
def project_to_image(state):
    """
    Construct a full model image from the various individual models in the state.
    """
    return state

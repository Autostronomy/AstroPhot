from ..autoprof_node import make_AP_Decision

@make_AP_Decision("stop iteration")
def stop_iteration(state):
    """
    When certain conditions are met, stop the iteration proceedure and return the fitted (hopefully converged) models.
    """
    if False:
        return 'start'
    return 'end'

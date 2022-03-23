from ..autoprof_node import make_AP_Process

@make_AP_Process("quality checks")
def quality_checks(state):
    """
    identify and flag any failed fits.
    """
    return state

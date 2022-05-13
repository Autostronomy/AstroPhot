from .autoprof_node import make_AP_Decision
from autoprof.utils.exceptions import AP_StopPipeline

@make_AP_Decision("stop pipeline")
def stop_pipeline(state):
    """
    When certain conditions are met, stop the iteration proceedure and return the fitted (hopefully converged) models.
    """
    raise AP_StopPipeline

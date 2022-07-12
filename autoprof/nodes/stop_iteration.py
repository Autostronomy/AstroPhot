from flow import Decision, FlowExitChart
import numpy as np

class Stop_Iteration(Decision):
    """
    Determine if the fit loop should end or return for another cycle.
    """

    def action(self, state):

        # All models locked
        if all(model.locked for model in state.models):
            state.models.unlock_models()
            return "End"          
        
        # Too many iterations
        if state.models.iteration > state.options.max_iterations:
            state.models.unlock_models()
            return "End"
        
        return "Start"

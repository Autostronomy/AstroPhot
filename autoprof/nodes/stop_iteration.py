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
        
        # for model in state.models:
        #     # not enough iterations
        #     if model.iteration < 100:
        #         if model.user_locked:
        #             continue
        #         else:
        #             break
        #     # not yet converged
        #     if np.any(np.abs(np.array(model.loss_history[:9]) - model.loss)/model.loss) > 1e-2:
        #         break
        # else:
        #     # all checks passed, all models must have converged
        #     state.models.unlock_models()
        #     return "End"
        return "Start"

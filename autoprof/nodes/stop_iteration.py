from flow import Decision, FlowExitChart
import numpy as np

class Stop_Iteration(Decision):
    """
    Determine if the fit loop should end or return for another cycle.
    """

    def action(self, state):

        for model in state.models:
            # not enough iterations
            if model.iteration < 100:
                if model.user_locked:
                    continue
                else:
                    break
            # Too many iterations
            if model.iteration > 100:
                return "End"
            # not yet converged
            if np.any(np.abs(np.array(model.loss_history[:99]) - model.loss)/model.loss) > 1e-2:
                break
        else:
            # all checks passed, all models must have converged
            return "End"
        return "Start"

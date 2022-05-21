from flow import Decision, FlowExitChart

class Stop_Iteration(Decision):
    """
    Determine if the fit loop should end or return for another cycle.
    """

    def action(self, state):

        for model in state.models:
            # not enough iterations
            if model.iteration < 100 or model.user_locked:
                continue
            # Too many iterations
            if model.iteration > 1000:
                return self.forward[1]
            # not yet converged
            if np.all(np.abs(model.loss[:99] - model.loss[0])/model.loss[0]) > 1e-2:
                break
        else:
            # all checks passed, all models must have converged
            return self.forward[1]

        return self.forward[0]

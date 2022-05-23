from flow import Process

class Save_Models(Process):
    """
    Call each model to save it's fitted parameter values.
    """

    def action(self, state):
        state.models.save_models(state.options["ap_saveto"])
        return state

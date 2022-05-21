from flow import Process

class Sample_Models(Process):
    """
    Create a model image based on the current set of model parameters.
    """

    def action(self, state):
        state.models.sample_models()
        return state


from flow import Process

class Initialize_Models(Process):
    """
    Initialize the model parameters using basic fitting to the image.
    """

    def action(self, state):
        state.models.initialize()
        return state

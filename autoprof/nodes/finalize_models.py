from flow import Process
import numpy as np

class Finalize_Models(Process):
    """
    Finalize the models if needed.
    """

    def action(self, state):
        state.data.initialize_model_image()
        for model in state.models:
            model.finalize()
        return state

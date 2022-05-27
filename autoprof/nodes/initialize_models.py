from flow import Process
import numpy as np

class Initialize_Models(Process):
    """
    Initialize the model parameters using basic fitting to the image.
    """

    def action(self, state):
        state.data.initialize_model_image()
        for model in state.models:
            model.initialize(state.data.target - state.data.model_image)
            model.sample_model()
            state.data.model_image += model.model_image
        return state

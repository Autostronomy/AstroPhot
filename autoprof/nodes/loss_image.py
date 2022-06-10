from flow import Process
from scipy.stats import iqr
import numpy as np

class Loss_Image(Process):
    """
    Compute a loss image by comparing the data image and model image
    """

    def action(self, state):

        target_area = state.data.target[state.data.model_image.window]
        state.data.loss_image = target_area - state.data.model_image
        state.data.loss_image.data = np.abs(state.data.loss_image.data) # / (0.5 + np.abs(target_area.data) + iqr(target_area.data, rng = (16,84))/2)

        return state

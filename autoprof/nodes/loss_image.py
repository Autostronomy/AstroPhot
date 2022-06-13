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
        state.data.loss_image.data = state.data.loss_image.data**2 / state.data.variance_image[state.data.model_image.window].data

        return state

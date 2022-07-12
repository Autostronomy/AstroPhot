from flow import Process
from scipy.stats import iqr
from autoprof.image import AP_Image
import numpy as np

class Loss_Image(Process):
    """
    Compute a loss image by comparing the data image and model image
    """

    def action(self, state):

        target_area = state.data.target[state.data.model_image.window]
        state.data.residual_image = target_area - state.data.model_image
        state.data.loss_image = target_area.blank_copy()
        state.data.loss_image.data = state.data.residual_image.data**2 / state.data.variance_image[state.data.model_image.window].data

        return state

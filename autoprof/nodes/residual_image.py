from flow import Process
from scipy.stats import iqr
from autoprof.image import AP_Image
import numpy as np

class Residual_Image(Process):
    """
    Compute a loss image by comparing the data image and model image
    """

    def action(self, state):

        target_area = state.data.target[state.data.model_image.window]
        state.data.residual_image = target_area - state.data.model_image

        return state

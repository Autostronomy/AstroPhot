from flow import Process
from scipy.stats import iqr
import numpy as np

class Variance_Image(Process):
    """
    Compute a loss image by comparing the data image and model image
    """

    def action(self, state):
        
        state.data.update_variance(np.abs(state.data.target.data) + (iqr(state.data.target.data, rng = (16,84))/2)**2, pixelscale = state.data.target.pixelscale, origin = state.data.target.origin) 

        return state

from flow import Process
import numpy as np
from autoprof.utils.convolution import fft_convolve

class Global_PSF(Process):
    """
    Apply PSF blurring for the entire model image.
    """

    def action(self, state):

        state.data.model_image.data = fft_convolve(state.data.model_image.data, state.data.psf.data)

        return state

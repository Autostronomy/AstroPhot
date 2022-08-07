from flow import Process
from astropy.io import fits
import os
import numpy as np

class Save_Models(Process):
    """
    Call each model to save it's fitted parameter values.
    """

    def action(self, state):
        print("saving models")
        state.models.save_models()
        header = fits.Header()
        hdul = fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(state.data.model_image.data)])
        hdul.writeto(
            os.path.join(state.options.save_path, state.options.name + '_model.fits'),
            overwrite=True,
        )
        
        return state

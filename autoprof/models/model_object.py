try:
    import cPickle as pickle
except:
    import pickle
from autoprof.image import Model_Image
from .parameter_object import Parameter
import numpy as np
from copy import deepcopy

class Model(object):

    PSF_mode = "none"
    model_type = "model"
    parameter_specs = {"center": {"units": "pix"}}
    
    def __init__(self, name, image, window, locked = None, **kwargs):

        self.name = name
        self.set_image(image)
        self.set_window(window)
        self.parameters = [{}]
        self.loss = []
        self.user_locked = locked
        self.update_locked(False)
        self.iteration = -1
        self.sampling_iteration = -2

        if "parameters" in kwargs:
            for p in kwargs["parameters"]:
                # If the user supplied a parameter object subclass, simply use that as is
                if isinstance(kwargs["parameters"][p], Parameter):
                    self.parameters[0][p] = kwargs["parameters"][p]
                    del self.parameter_specs[p]
                else: # if the user supplied parameter specifications, update the defaults
                    self.parameter_specs[p].update(kwargs["parameters"][p])
        self.parameters[0].update(dict((p, Parameter(p,self.parameter_specs[p])) for p in self.parameter_specs))

    # Initialization functions
    ######################################################################
    def set_image(self, image):
        self.image = image

    def set_window(self, window):
        self.window = window
        self.model_image = Model_Image(
            np.zeros((window[1][1] - window[1][0], window[0][1] - window[0][0])),
            pixelscale=self.image.pixelscale,
            origin = [window[1][0], window[0][0]],
        )
        self.model_image.clear_image()

    def update_window(self, window):
        raise NotImplimentedError("update window not yet available")

    def update_locked(self, locked):
        self.locked = self.user_locked or locked

    def initialize(self):
        # Use center of window if a center hasn't been set yet
        if self["center"].value is None:
            self["center"].set_value(
                np.array(
                    [
                        self.window[1][0] + (self.model_image.shape[1] - 1) / 2,
                        self.window[0][0] + (self.model_image.shape[0] - 1) / 2,
                    ]
                ),
                override_fixed=True,
            )

    # Fit loop functions
    ######################################################################
    def step_iteration(self):
        # Add a new set of parameters to the history that defaults to the most recent values
        self.parameters.insert(0, deepcopy(self.parameters[0]))
        self.iteration += 1

    def sample_model(self):
        # Don't bother resampling the model if nothing has been updated
        if self.iteration == self.sampling_iteration:
            return
        self.sampling_iteration = self.iteration
        # Reset the model image before filling it with updated parameters
        self.model_image.clear_image()

    def convolve_psf(self):
        # Skip PSF convolution if not required for this model
        if self.PSF_mode == "none":
            return

    def update_loss(self, loss_image):
        # Basic loss is the mean Chi^2 error in the window
        self.loss.insert(
            0,
            np.mean(
                loss_image[
                    self.window[1][0] : self.window[1][1],
                    self.window[0][0] : self.window[0][1],
                ]
            ),
        )

    # Interface Functions
    ######################################################################
    def get_loss(self, index=0):
        # Return the loss for the requested iteration
        return self.loss[index]

    def get_parameters(self, index=0, exclude_fixed = False):
        # Return all parameters for a given iteration
        if not exclude_fixed:
            return self.parameters[index]
        return_parameters = {}
        for p in self.parameters[index]:
            # Skip fixed parameters since they cannot be updated anyway
            if self.parameters[index][p].fixed:
                continue
            # Return representation which is valid in [-inf, inf] range
            return_parameters[p] = self.parameters[index][p]
        return return_parameters

    def __get__(self, key, index=0):
        # Get the parameter for an optionally specified iteration
        return self.parameters[index][key]

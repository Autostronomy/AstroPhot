try:
    import cPickle as pickle
except:
    import pickle
from autoprof.image import Model_Image
from .parameter_object import Parameter
import numpy as np
from scipy.stats import iqr
from copy import deepcopy

class Model(object):

    model_type = "model"
    parameter_specs = {
        "center_x": {"units": "pix", "uncertainty": 0.2},
        "center_y": {"units": "pix", "uncertainty": 0.2}
    }

    # Hierarchy variables
    PSF_mode = "none" # FFT, superresolve
    sample_mode = "direct" # integrate
    learning_rate = 0.01
    average = np.median
    scatter = lambda v: iqr(v, rng=(31.731 / 2, 100 - 31.731 / 2)) / 2.0
    interpolate = "lanczos"
    
    def __init__(self, name, state, image, window = None, locked = None, **kwargs):

        self.name = name
        self.state = state
        self.set_image(image)
        self.set_window(window)
        self.parameters = {}
        self.parameter_history = []
        self.loss = None
        self.loss_history = []
        self.user_locked = locked
        self.update_locked(False)
        self.iteration = -1
        self.sampling_iteration = -2
        self.sample_points = None # used for integrating below pixel resolution

        if "psf window" in kwargs:
            self.psf_window = kwargs["psf window"]
        else:
            self.psf_window = None

        self.parameter_specs = self.build_parameter_specs()
        if "parameters" in kwargs:
            for p in kwargs["parameters"]:
                # If the user supplied a parameter object subclass, simply use that as is
                if isinstance(kwargs["parameters"][p], Parameter):
                    self.parameters[p] = kwargs["parameters"][p]
                    del self.parameter_specs[p]
                else: # if the user supplied parameter specifications, update the defaults
                    self.parameter_specs[p].update(kwargs["parameters"][p])
        self.parameters.update(dict((p, Parameter(p,self.parameter_specs[p])) for p in self.parameter_specs))

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

    @classmethod
    def build_parameter_specs(cls):
        parameter_specs = {}
        for base in cls.__bases__:
            try:
                parameter_specs.update(base.build_parameter_specs())
            except AttributeError:
                pass
        parameter_specs.update(cls.parameter_specs)
        return parameter_specs

    def initialize(self):
        # Use center of window if a center hasn't been set yet
        if self["center_x"].value is None:
            self["center_x"].set_value(self.window[1][0] + (self.model_image.shape[1] - 1) / 2, override_fixed = True)
        if self["center_y"].value is None:
            self["center_y"].set_value(self.window[0][0] + (self.model_image.shape[0] - 1) / 2, override_fixed = True)

        COM = center_of_mass((self["center_x"].value,self["center_y"].value), self.image)
        self["center_x"].set_value(COM[0])
        self["center_y"].set_value(COM[1])
        
    # Fit loop functions
    ######################################################################
    def step_iteration(self):
        # Add a new set of parameters to the history that defaults to the most recent values
        self.parameter_history.insert(0, deepcopy(self.parameters))
        self.loss_history.insert(0, deepcopy(self.loss))
        self.loss = None
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
        self.loss = np.mean(
            loss_image[
                self.window[1][0] : self.window[1][1],
                self.window[0][0] : self.window[0][1],
            ]
        )
        
    # Interface Functions
    ######################################################################
    def get_loss(self, index=0):
        # Return the loss for the requested iteration
        return self.loss[index]

    def get_loss_history(self, limit = np.inf):
        param_order = self.get_parameters(exclude_fixed = True).keys()
        params = []
        for i in range(min(limit, len(self.loss_history))):
            params_i = self.get_parameters(index = i, exclude_fixed = True)
            params.append(np.array([params_i[P] for P in param_order], dtype = Parameter))
        yield self.loss_history, params

    def get_parameters(self, index=None, exclude_fixed = False):
        # Pick if using current parameters, or parameter history
        if index is None:
            use_params = self.parameters
        else:
            use_params = self.parameter_history[index]
        
        # Return all parameters for a given iteration
        if not exclude_fixed:
            return use_params
        return_parameters = {}
        for p in use_params:
            # Skip currently fixed parameters since they cannot be updated anyway
            if self.parameters[p].fixed:
                continue
            # Return representation which is valid in [-inf, inf] range
            return_parameters[p] = use_params[p]
        return return_parameters

    def __get__(self, key, index=None):
        # Get the parameter for an optionally specified iteration
        if index is None:
            return self.parameters[key]
        else:
            return self.parameter_history[index][key]

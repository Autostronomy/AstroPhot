try:
    import cPickle as pickle
except:
    import pickle
from autoprof.image import Model_Image
from autoprof.utils.conversions.optimization import boundaries, inv_boundaries
import numpy as np


class Model(object):

    PSF_mode = "none"
    mode = "fitting"
    name = "model"
    parameter_names = ("center",)

    def __init__(self, **kwargs):

        if "load_file" in kwargs:
            self.load(kwargs["load_file"])
            return
        self.window = None
        self.image = None
        self.model_image = None
        self.parameters = [{}]
        self.loss = []
        self.user_locked = False
        self.user_fixed = set()
        self.locked = False
        self.fixed = set()
        self.limits = {}
        self.iteration = -1
        self.sampling_iteration = -2
        self.stage = "created"

        if "image" in kwargs:
            self.update_image(kwargs["image"])
        if "window" in kwargs:
            self.update_window(kwargs["window"])

        if "locked" in kwargs:
            self.user_locked = kwargs["locked"]
            self.update_fixed(lock=kwargs["locked"])
        if "fixed" in kwargs:
            self.user_fixed.update(kwargs["fixed"])
            for fix in kwargs["fixed"]:
                self.update_fixed(is_fixed=fix)
        if "limits" in kwargs:
            self.limits.update(kwargs["limits"])

        if "parameters" in kwargs:
            self.set_parameters(kwargs["parameters"], override_fixed=True)
        for p in self.parameter_names:
            if not p in self.get_parameters():
                self.set_value(p, None, override_fixed=True)

    def update_fixed(self, is_fixed=None, not_fixed=None, locked=None):
        # Lock model, can only unlock if user hasnt locked model
        if not locked is None:
            self.locked = self.user_locked or locked
        # Fix a single parameter
        if not is_fixed is None:
            self.fixed.add(is_fixed)
        # Un-fix a single parameter, only if user hasnt locked parameter
        if not (not_fixed is None or not_fixed in self.user_fixed):
            self.fixed.discard(not_fixed)

    def initialize(self):
        # If a window still hasn't been set, use the whole image
        if self.window is None:
            self.update_window(
                [[0, self.image.shape[1]], [0, self.image.shape[0]]]
            )
        self.stage = "initialized"

    def step_iteration(self):
        # Add a new set of parameters to the history that defaults to the most recent values
        self.parameters.insert(0, self.parameters[0])
        self.iteration += 1
        self.stage = "fitting"

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

    def convolve_psf(self):
        # Skip PSF convolution if not required for this model
        if self.PSF_mode == "none":
            return

    def sample_model(self):
        # Don't bother resampling the model if nothing has been updated
        if self.iteration == self.sampling_iteration:
            return
        self.sampling_iteration = self.iteration
        # Reset the model image before filling it with updated parameters
        self.model_image.clear_image()

    def set_image(self, image):
        self.image = image

    def update_window(self, window):
        old_window = self.window
        self.window = window
        self.model_image = Model_Image(
            np.zeros((window[1][1] - window[1][0], window[0][1] - window[0][0])),
            pixelscale=self.image.pixelscale,
            origin = [window[1][0], window[0][0]],
        )
        self.model_image.clear_image()

        # Use center of window if a center hasn't been set yet
        if self.get_value("center") is None:
            self.set_value(
                "center",
                np.array(
                    [
                        self.window[1][0] + (self.model_image.shape[1] - 1) / 2,
                        self.window[0][0] + (self.model_image.shape[0] - 1) / 2,
                    ]
                ),
                override_fixed=True,
            )

    def save(self):
        return str(self.parameters[0])

    def write(self, filename):
        with open(filename, "w") as f:
            pickle.dump(self, f)

    def load(self, filename):
        with open(filename, "r") as f:
            self.__dict__ = pickle.load(f)

    def get_loss(self, index=0):
        # Return the loss for the requested iteration
        return self.loss[index]

    def get_parameters_representation(self, index=0):
        # Return all the parameters for a given iteration.
        return_parameters = {}
        for p in self.parameters[index]:
            # Skip fixed parameters since they cannot be updated anyway
            if p in self.fixed:
                continue
            # Return representation which is valid in [-inf, inf] range
            return_parameters[p] = self.get_representation(p, index)

    def get_parameters(self, index=0):
        # Return all parameters for a given iteration, values in base
        # representation
        return self.parameters[index]

    def set_parameters_representation(self, parameters):
        # Set the value for a number of parameters
        for p in parameters:
            # Set the values using a representation which is valid in
            # [-inf, inf] range
            self.set_representation(p, parameters[p])

    def set_parameters(self, parameters, override_fixed=False):
        # Set the value for a number of parameters in their base
        # representation
        for p in parameters:
            self.set_value(p, parameters[p], override_fixed)

    def update_parameters_representation(self, parameters):
        # Like set_parameters_representation, except it adds the value
        # to the parameter instead of overwriting it.
        for p in parameters:
            self.add_representation(p, parameters[p])

    def get_value(self, key, index=0):
        # Return a parameter in its base representation
        return self.parameters[index][key]

    def set_value(self, key, value, override_fixed=False):
        # Set a parameter in its base representation
        if (self.locked or key in self.fixed) and not override_fixed:
            return
        self.parameters[0][key] = value

    def add_value(self, key, value, override_fixed=False):
        # Add to a parameter in its base representation
        if (self.locked or key in self.fixed) and not override_fixed:
            return
        self.parameters[0][key] += value

    def get_representation(self, key, index=0):
        # Get a representation of a parameter which is valid in the
        # [-inf, inf] range
        if key in self.limits:
            return boundaries(self.get_value(key, index), self.limits[key])
        else:
            return self.get_value(key, index)

    def set_representation(self, key, value, override_fixed=False):
        # Set a parameter value in a representation which is valid in
        # the [-inf, inf] range
        if key in self.limits:
            self.set_value(key, inv_boundaries(value, self.limits[key]))
        else:
            self.set_value(key, value)

    def add_representation(self, key, value):
        # Add value to a parameter in a representation which is valid
        # in the [-inf, inf] range
        self.set_representation(key, self.get_representation(key) + value)

    def __get__(self, key, index=0):
        # Get the value of a parameter for a specified iteration
        return self.get_value(key, index)

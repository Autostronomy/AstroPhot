try:
    import cPickle as pickle
except:
    import pickle
from autoprof.image import Model_Image
from autoprof.utils.initialize import center_of_mass
from autoprof.utils.conversions.coordinates import coord_to_index, index_to_coord
from autoprof.utils.convolution import direct_convolve, fft_convolve
from .parameter_object import Parameter
import numpy as np
from copy import deepcopy

class BaseModel(object):

    model_type = "model"
    parameter_specs = {
        "center_x": {"units": "pix", "uncertainty": 0.1},
        "center_y": {"units": "pix", "uncertainty": 0.1}
    }

    # Hierarchy variables
    psf_mode = "direct" # direct, FFT, superresolve
    psf_window_size = 100
    sample_mode = "direct" # integrate
    learning_rate = 0.1
    interpolate = "lanczos"
    
    def __init__(self, name, state, target, window = None, locked = None, **kwargs):

        self.name = name
        self.state = state
        self.set_target(target)
        self.set_window(window)
        self.parameters = {}
        self.parameter_history = []
        self.loss = None
        self.loss_history = []
        self.user_locked = locked
        self.update_locked(False)
        self.iteration = -1
        self.is_sampled = False
        self.is_convolved = False
        self.sample_points = None # used for integrating below pixel resolution

        if "psf_window_size" in kwargs:
            self.psf_window_size = kwargs["psf_window_size"]
            
        self.parameter_specs = self.build_parameter_specs()
        if "parameters" in kwargs:
            for p in kwargs["parameters"]:
                # If the user supplied a parameter object subclass, simply use that as is
                if isinstance(kwargs["parameters"][p], Parameter):
                    self.parameters[p] = kwargs["parameters"][p]
                    del self.parameter_specs[p]
                else: # if the user supplied parameter specifications, update the defaults
                    self.parameter_specs[p].update(kwargs["parameters"][p])
        self.parameters.update(dict((p, Parameter(p, **self.parameter_specs[p])) for p in self.parameter_specs))

    # Initialization functions
    ######################################################################
    def set_target(self, target):
        self.target = target

    def set_window(self, window):
        if window is None:
            window = [
                [0, self.target.shape[0]],
                [0, self.target.shape[1]],
            ]
        if isinstance(window, tuple) and isinstance(window[0], slice) and isinstance(window[1], slice):
            self.window = window
        else:
            self.window = (slice(window[1][0], window[1][1]), slice(window[0][0], window[0][1]))
        self.window_shape = (self.window[0].stop - self.window[0].start, self.window[1].stop - self.window[1].start)
        self.model_image = Model_Image(
            np.zeros(self.window_shape),
            pixelscale=self.target.pixelscale,
            origin = [self.window[0].start, self.window[1].start],#fixme handle target origin
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

    def initialize(self, target = None):
        # Use center of window if a center hasn't been set yet
        window_center = index_to_coord((self.window_shape[0] - 1) / 2, (self.window_shape[1] - 1) / 2, self.model_image)
        if self["center_x"].value is None:
            self["center_x"].set_value(window_center[0], override_fixed = True)
        if self["center_y"].value is None:
            self["center_y"].set_value(window_center[1], override_fixed = True)

        if self["center_x"].fixed and self["center_y"].fixed:
            return
        # Get the sub-image area corresponding to the model image
        target_area = target.get_image_area(self.model_image)
        # Convert center coordinates to target area array indices
        init_icenter = coord_to_index(self["center_x"].value, self["center_y"].value, target_area)
        # Compute center of mass in window
        COM = center_of_mass((init_icenter[1], init_icenter[0]), target_area.data)
        # Convert center of mass indices to coordinates
        COM_center = index_to_coord(COM[1], COM[0], target_area)
        # Set the new coordinates as the model center
        self["center_x"].set_value(COM_center[0])
        self["center_y"].set_value(COM_center[1])
        
    # Fit loop functions
    ######################################################################
    def step_iteration(self):
        # Add a new set of parameters to the history that defaults to the most recent values
        if not self.loss is None:
            self.parameter_history.insert(0, deepcopy(self.parameters))
            self.loss_history.insert(0, deepcopy(self.loss))
            self.loss = None
        self.iteration += 1
        self.is_sampled = False
        self.is_convolved = False
        
    def sample_model(self):
        # Don't bother resampling the model if nothing has been updated
        if self.is_sampled:
            return
        self.sampling_iteration = self.iteration
        # Reset the model image before filling it with updated parameters
        self.model_image.clear_image()
        self.is_sampled = True

    def convolve_psf(self):
        if self.is_convolved:
            return
        # Skip PSF convolution if not required for this model
        if self.psf_mode == "none":
            return

        icenter = coord_to_index(self["center_x"].value, self["center_y"].value, self.model_image)
        psf_window = (
            slice(max(0, int(icenter[0] - self.psf_window_size/2)), min(self.model_image.shape[0], int(icenter[0] + self.psf_window_size/2))),
            slice(max(0, int(icenter[1] - self.psf_window_size/2)), min(self.model_image.shape[1], int(icenter[1] + self.psf_window_size/2))),
        )
        
        if self.psf_mode == "direct":
            self.model_image.data[psf_window] = direct_convolve(self.model_image.data[psf_window], self.state.data.psf.data)    
        elif self.psf_mode == "fft":
            self.model_image.data[psf_window] = fft_convolve(self.model_image.data[psf_window], self.state.data.psf.data)
        self.is_convolved = True
        
    def compute_loss(self, loss_image):
        # Basic loss is the mean Chi^2 error in the window
        self.loss = np.mean(loss_image.get_image_area(self.model_image).data)
        
    # Interface Functions
    ######################################################################
    def get_loss(self, index=0):
        # Return the loss for the requested iteration
        if index is None:
            return self.loss
        else:
            return self.loss_history[index]

    def get_loss_history(self, limit = np.inf):
        param_order = self.get_parameters(exclude_fixed = True).keys()
        params = []
        loss_history = []
        for i in range(min(limit, max(1,len(self.loss_history)))):
            params_i = self.get_parameters(index = i if i > 0 else None, exclude_fixed = True)
            params.append(np.array([params_i[P] for P in param_order]))
            loss_history.append(self.loss_history[i] if len(self.loss_history) > 0 else self.loss)
        yield loss_history, params

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

    def save_model(self, fileobject):
        fileobject.write("\n" + "\n" + "*"*70 + "\n")
        fileobject.write(self.name + "\n")
        fileobject.write("*"*70 + "\n")
        for p in self.parameters:
            fileobject.write(f"{str(self.parameters[p])}\n")
    
    def __getitem__(self, key, index=None):
        # Get the parameter for an optionally specified iteration
        if index is None:
            return self.parameters[key]
        else:
            return self.parameter_history[index][key]            

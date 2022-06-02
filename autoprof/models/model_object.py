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
        "center_x": {"units": "arcsec", "uncertainty": 0.1},
        "center_y": {"units": "arcsec", "uncertainty": 0.1}
    }

    # modes: direct, direct+PSF, integrate, integrate+PSF, integrate+superPSF
    # Hierarchy variables
    sample_mode = "direct" # direct, integrate
    psf_mode = "none" # none, direct, FFT
    psf_window_size = 100
    integrate_window_size = 10
    integrate_factor = 5
    learning_rate = 0.1
    
    def __init__(self, name, state, target, window = None, locked = None, **kwargs):

        self._set_default_parameters()
        self.name = name
        self.state = state
        self.set_target(target)
        self.set_window(window)
        self.user_locked = locked
        self.update_locked(False)
            
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
        self._init_convert_input_units()
        
        # Set any user defined attributes for the model
        for kwarg in kwargs:
            # Skip parameters with special behaviour
            if kwarg in ["parameters"]:
                continue
            # Set the model parameter
            print("setting: ", kwarg)
            setattr(self, kwarg, kwargs[kwarg])

    def _init_convert_input_units(self):
        if self["center_x"].value is not None:
            physcenter = index_to_coord(np.nan, self["center_x"].value, self.target)
            self["center_x"].set_value(physcenter[0], override_fixed = True)
        if self["center_y"].value is not None:
            physcenter = index_to_coord(self["center_y"].value, np.nan, self.target)
            self["center_y"].set_value(physcenter[1], override_fixed = True)
            
    # Initialization functions
    ######################################################################    
    def initialize(self, target = None):
        # Use center of window if a center hasn't been set yet
        window_center = index_to_coord(self.window.shape[0] / 2, self.window.shape[1] / 2, self.model_image)
        if self["center_x"].value is None:
            self["center_x"].set_value(window_center[0], override_fixed = True)
        if self["center_y"].value is None:
            self["center_y"].set_value(window_center[1], override_fixed = True)

        if self["center_x"].fixed and self["center_y"].fixed:
            return

        if target is None:
            target = self.target

        # Get the sub-image area corresponding to the model image
        target_area = target[self.model_image]
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
    def sample_model(self, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image

        if sample_image is self.model_image:
            self.is_sampled = True
            # Reset the model image before filling it with updated values
            sample_image.clear_image()

    def integrate_model(self):
        if self.is_integrated:
            return
        if "integrate" not in self.sample_mode:
            return
        # Determine the on-sky window in which to integrate
        self.integrate_window = AP_Window(
            origin = (self["center_y"].value - self.integrate_window_size*self.model_image.pixelscale/2, self["center_x"].value - self.integrate_window_size*self.model_image.pixelscale/2),
            shape = (self.integrate_window_size*self.model_image.pixelscale, self.integrate_window_size*self.model_image.pixelscale)
        )

        # Determine the upsampled pixelscale 
        integrate_pixelscale = self.model_image.pixelscale / self.integrate_factor

        # Create a model image to store the high resolution samples
        self.model_integrate = Model_Image(
            np.zeros(integrate_window.shape // integrate_pixelscale),
            pixelscale = integrate_pixelscale,
            window = integrate_window,
        )

        # Evaluate the model at the fine sampling points
        self.sample_model(self.model_integrate)
        
        self.is_integrated = True
        
    def convolve_psf(self):
        # If already convolved, skip this step
        if self.is_convolved:
            return
        # Skip PSF convolution if not required for this model
        if "none" in self.psf_mode:
            return

        # Convert the model center to image coordinates
        psf_window = AP_Window(origin = (self["center_y"].value - self.psf_window_size*self.model_image.pixelscale/2, self["center_x"].value - self.psf_window_size*self.model_image.pixelscale/2),
                               shape = (self.psf_window_size*self.model_image.pixelscale, self.psf_window_size*self.model_image.pixelscale))

        # Perform the PSF convolution using the specified method
        psf_window_area = self.model_image[psf_window]
        if "direct" in self.psf_mode:
            psf_window_area.data = direct_convolve(psf_window_area.data, self.state.data.psf.data)    
        elif "fft" in self.psf_mode:
            psf_window_area.data = fft_convolve(psf_window_area.data, self.state.data.psf.data)
        else:
            raise ValueError(f"unrecognized psf_mode: {self.psf_mode}")

        if "integrate" in self.sample_mode:
            upsample_psf = self.state.data.psf.get_resolution(self.integrate_factor)
            if "direct" in self.psf_mode:
                self.model_integrate.data = direct_convolve(self.model_integrate.data, upsample_psf.data)
            elif 'fft' in self.psf_mode:
                self.model_integrate.data = fft_convolve(self.model_integrate.data, upsample_psf.data)                
                
        # Keep record that the image has been convolved
        self.is_convolved = True
        
    def add_integrated_model(self):
        if not self.is_integrated:
            return
        
        condensed = self.model_integrate.data.reshape(-1, integrate_factor, self.model_integrate.data.shape[0]//self.integrate_factor, self.integrate_factor).sum((-1,-3))
        self.model_image[self.integrate_window].data = condensed
        
    def compute_loss(self, loss_image):
        # If the image is locked, no need to compute the loss
        if self.locked:
            return
        # Basic loss is the mean Chi^2 error in the window
        self.loss = np.mean(loss_image[self.model_image].data)

        
    ######################################################################
    from ._model_methods import _set_default_parameters
    from ._model_methods import step_iteration
    from ._model_methods import set_target
    from ._model_methods import set_window
    from ._model_methods import scale_window
    from ._model_methods import update_locked
    from ._model_methods import build_parameter_specs
    from ._model_methods import get_loss
    from ._model_methods import get_loss_history
    from ._model_methods import get_parameters
    from ._model_methods import save_model
    from ._model_methods import __getitem__


    

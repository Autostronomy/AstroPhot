try:
    import cPickle as pickle
except:
    import pickle
from autoprof.image import Model_Image, AP_Window
from autoprof.utils.initialize import center_of_mass
from autoprof.utils.conversions.coordinates import coord_to_index, index_to_coord
from autoprof.utils.convolution import direct_convolve, fft_convolve
from autoprof.utils.image_operations import blockwise_sum
from .parameter_object import Parameter
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class BaseModel(object):

    model_type = "model"
    parameter_specs = {
        "center": {"units": "arcsec", "uncertainty": 0.1},
    }
    parameter_qualities = {
        "center": {"form": "array", "loss": "global"},
    }

    # Hierarchy variables
    sample_mode = "direct" 
    psf_mode = "none" # none, window/full, direct/fft
    loss_speed_factor = 1
    psf_window_size = 100
    integrate_mode = "none" # none, window, full
    integrate_window_size = 10
    integrate_factor = 5
    learning_rate = 0.1
    
    def __init__(self, name, target, window = None, locked = None, **kwargs):

        self._set_default_parameters()
        self.name = name
        self.set_target(target)
        self.set_window(window)
        self.user_locked = locked
        self.update_locked(False)
            
        self.parameter_specs = self.build_parameter_specs(kwargs.get("parameters", None))
        self.parameter_qualities = self.build_parameter_qualities()
        self.build_parameters()
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
        if self["center"].value is not None:
            physcenter = index_to_coord(self["center"][1].value, self["center"][0].value, self.target)
            self["center"].set_value(physcenter, override_fixed = True)
            
    # Initialization functions
    ######################################################################    
    def initialize(self, target = None):
        if target is None:
            target = self.target
        # Get the sub-image area corresponding to the model image
        target_area = target[self.model_image]
        
        # Use center of window if a center hasn't been set yet
        window_center = index_to_coord(self.model_image.data.shape[0] / 2, self.model_image.data.shape[1] / 2, self.model_image)
        if self["center"].value is None:
            self["center"].set_value(window_center, override_fixed = True)

        if self["center"].fixed:
            return

        # Convert center coordinates to target area array indices
        init_icenter = coord_to_index(self["center"][0].value, self["center"][1].value, target_area)
        # Compute center of mass in window
        COM = center_of_mass((init_icenter[1], init_icenter[0]), target_area.data)
        if np.any(np.array(COM) < 0) or np.any(np.array(COM) >= np.array(target_area.data.shape)):
            print("center of mass failed, using center of window")
            return
        # Convert center of mass indices to coordinates
        COM_center = index_to_coord(COM[1], COM[0], target_area)
        # Set the new coordinates as the model center
        self["center"].set_value(COM_center)

    def finalize(self):
        pass
        
    # Fit loop functions
    ######################################################################
    def evaluate_model(self, X, Y, image):
        return 0
    
    def sample_model(self, sample_image = None, psf = None):
        if sample_image is None:
            sample_image = self.model_image

        # no need to resample image if already done
        if self.is_sampled and sample_image is self.model_image:
            return
        
        if sample_image is self.model_image:
            # Reset the model image before filling it with updated values
            self.model_image.clear_image()
            
        working_window = deepcopy(sample_image.window)

        if "none" not in self.psf_mode and psf is not None:# fixme could make shifting smarter for psf window
            self.center_shift = self["center"].get_value() % 1.
            working_window.shift_origin(self.center_shift)

            if "full" in self.psf_mode or not (self.psf_window + psf.get_resolution(sample_image.pixelscale).border) <= working_window:
                working_window += psf.get_resolution(sample_image.pixelscale).border

        working_image = Model_Image(data = np.zeros(working_window.get_shape(sample_image.pixelscale)), pixelscale = sample_image.pixelscale, window = working_window)
            
        if not "full" in self.integrate_mode:
            X, Y = working_image.get_coordinate_meshgrid(self["center"][0].value, self["center"][1].value)
            M = self.evaluate_model(X, Y, working_image)
            working_image += M
            
        if "none" not in self.integrate_mode:
            self.integrate_model(working_image)
            
        if "none" not in self.psf_mode and psf is not None:
            self.convolve_psf(working_image, psf)
            working_image.shift_origin(-self.center_shift) # fixme get sign right
            self.center_shift = np.zeros(self["center"].get_value().shape)
        sample_image += working_image
        if sample_image is self.model_image:
            self.is_sampled = True

            
    def integrate_model(self, working_image):        
        # Determine the on-sky window in which to integrate
        if self.integrate_window.overlap_frac(working_image.window) == 0:
            return
        
        # Only need to evaluate integration within working image
        if "window" in self.integrate_mode:
            working_window = self.integrate_window & working_image.window
        elif "full" in self.integrate_mode:
            working_window = working_image.window
        else:
            raise ValueError(f"Unrecognized integration mode: {self.integrate_mode}, must include 'window' or 'full', or be 'none'.")    
            
        # Determine the upsampled pixelscale 
        integrate_pixelscale = working_image.pixelscale / self.integrate_factor

        # Build an image to hold the integration data
        integrate_image = Model_Image(np.zeros(np.array(working_image[working_window].data.shape) * self.integrate_factor), integrate_pixelscale, window = working_window)
        
        # Evaluate the model at the fine sampling points
        X, Y = integrate_image.get_coordinate_meshgrid(self["center"][0].value, self["center"][1].value)
        integrate_image.data = self.evaluate_model(X, Y, integrate_image)
        
        # Replace the image data where the integration has been done
        working_image.replace(working_window, blockwise_sum(integrate_image.data, (self.integrate_factor, self.integrate_factor)))
        
        
    def convolve_psf(self, working_image, psf = None):# fixme move out of model to utils?
        if psf is None:
            raise ValueError("A PSF is needed to convolve!")
        
        if self.psf_window.overlap_frac(working_image.window) == 0:
            return
        
        psf_image = psf.get_resolution(working_image.pixelscale)

        if "full" in self.psf_mode:
            working_window = working_image.window
        elif "window" in self.psf_mode:
            # Convert the model center to image coordinates
            working_window = (self.psf_window + psf_image.border) & working_image.window
        print(self.center_shift, working_window)
        # Perform the convolution according to the requested method
        if "direct" in self.psf_mode:
            convolution = direct_convolve(working_image[working_window].data, psf_image.data)
        elif "fft" in self.psf_mode:
            convolution = fft_convolve(working_image[working_window].data, psf_image.data)
        elif max(psf_image.data.shape) < 20 or max(working_image[working_window].data.shape) < 100:
            convolution = direct_convolve(working_image[working_window].data, psf_image.data)
        else:
            convolution = fft_convolve(working_image[working_window].data, psf_image.data)
            
        # Cut the 1/2PSF border from the data
        if "window" in self.psf_mode:
            convolution = convolution[psf_image.border_int[0]:-psf_image.border_int[0],psf_image.border_int[1]:-psf_image.border_int[1]]
            working_window -= psf_image.border 

        # Replace the corresponding pixels with the convolved image
        working_image.replace(working_window, convolution)    

    def evaluate_loss(self, residual_image, variance_image):
        return np.mean(residual_image[self.window].data[::self.loss_speed_factor,::self.loss_speed_factor]**2 / variance_image[self.window].data)
    
    def compute_loss(self, data):
        # If the image is locked, no need to compute the loss
        if self.locked:
            return
        # Basic loss is the mean Chi^2 error in the window
        self.loss = self.evaluate_loss(data.residual_image, data.variance_image)

    def compute_loss_derivative(self, data):
        # If the image is locked, no need to compute the loss
        if self.locked:
            return
        # fixme add basic numerical derivative
        
    ######################################################################
    from ._model_methods import _set_default_parameters
    from ._model_methods import step_iteration
    from ._model_methods import set_target
    from ._model_methods import set_window
    from ._model_methods import scale_window
    from ._model_methods import integrate_window
    from ._model_methods import psf_window
    from ._model_methods import update_locked
    from ._model_methods import build_parameter_specs
    from ._model_methods import build_parameter_qualities
    from ._model_methods import build_parameters
    from ._model_methods import get_parameters
    from ._model_methods import get_loss
    from ._model_methods import get_parameter_history
    from ._model_methods import get_loss_history
    from ._model_methods import get_history
    from ._model_methods import save_model
    from ._model_methods import __getitem__
    from ._model_methods import __str__


    

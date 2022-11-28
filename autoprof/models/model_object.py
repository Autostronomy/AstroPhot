from autoprof.image import Model_Image, AP_Window
from autoprof.utils.initialize import center_of_mass
from autoprof import plots
from autoprof.utils.conversions.coordinates import coord_to_index, index_to_coord
import numpy as np
import torch
from torch.nn.functional import conv2d
from copy import deepcopy
from .core_model import AutoProf_Model
import matplotlib.pyplot as plt

__all__ = ["BaseModel"]

class BaseModel(AutoProf_Model):
    """This is the basis for almost any model which represents a single
    object, or parametric form.  Subclassing models must define their
    parameters, initialization, and model evaluation
    functions. Otherwise, most function operate generally.

    """
    
    model_type = "model"
    parameter_specs = {
        "center": {"units": "arcsec", "uncertainty": 0.1},
    }
    parameter_order = ("center",)

    # Hierarchy variables
    psf_mode = "none" # none, window/full, direct/fft
    psf_window_size = 50
    integrate_mode = "window" # none, window, full
    integrate_window_size = 10
    integrate_factor = 10

    # settings
    special_kwargs = ["parameters", "filename", "model_type"]
    
    def __init__(self, name, target = None, window = None, locked = False, **kwargs):
        super().__init__(name, target, window, locked, **kwargs)
        
        self._set_default_parameters()
        self.target = target
        self.fit_window = window
        self._user_locked = locked
        self._locked = self._user_locked
        self.parameter_vector_len = None
        
        # Set any user defined attributes for the model
        for kwarg in kwargs:
            # Skip parameters with special behaviour
            if kwarg in self.special_kwargs:
                continue
            # Set the model parameter
            print("setting: ", kwarg)
            setattr(self, kwarg, kwargs[kwarg])

        self.parameter_specs = self.build_parameter_specs(kwargs.get("parameters", None))
        with torch.no_grad():
            self.build_parameters()
            if isinstance(kwargs.get("parameters", None), torch.Tensor):
                self.step(kwargs["parameters"])
            
        if "filename" in kwargs:
            self.load(kwargs["filename"])
            
    # Initialization functions
    ######################################################################
    @torch.no_grad()
    def initialize(self):
        """Determine initial values for the center coordinates. This is done
        with a local center of mass search which iterates by finding
        the center of light in a window, then iteratively updates
        until the iterations move by less than a pixel.

        """
        # Get the sub-image area corresponding to the model image
        target_area = self.target[self.fit_window]
        
        # Use center of window if a center hasn't been set yet
        if self["center"].value is None:
            self["center"].set_value(self.model_image.center, override_locked = True)
        else:
            return

        if self["center"].locked:
            return

        # Convert center coordinates to target area array indices
        init_icenter = coord_to_index(self["center"].value[0].detach().cpu().item(), self["center"].value[1].detach().cpu().item(), target_area)
        # Compute center of mass in window
        COM = center_of_mass((init_icenter[0], init_icenter[1]), target_area.data.detach().cpu().numpy())
        if np.any(np.array(COM) < 0) or np.any(np.array(COM) >= np.array(target_area.data.shape)):
            print("center of mass failed, using center of window")
            return
        # Convert center of mass indices to coordinates
        COM_center = index_to_coord(COM[0], COM[1], target_area)
        # Set the new coordinates as the model center
        self["center"].value = COM_center + np.random.normal(loc = 0, scale = 0.3*self.target.pixelscale, size = len(COM_center))
            
    # Fit loop functions
    ######################################################################
    def evaluate_model(self, image):
        """Evaluate the model on every pixel in the given image. The
        basemodel object simply returns zeros, this function should be
        overloaded by subclasses.

        """
        return torch.zeros_like(image.data) # do nothing in base model
    
    def sample(self, sample_image = None):
        """Evaluate the model on the space covered by an image object. This
        function properly calls integration methods and PSF
        convolution. This should not be overloaded except in special
        cases.

        """
        
        if sample_image is None:
            sample_image = self.model_image
        if self.is_sampled and sample_image is self.model_image:
            return
        if sample_image is self.model_image:
            sample_image.clear_image()

        # Check that psf and integrate modes line up
        if "window" in self.psf_mode:
            if "window" in self.integrate_mode:
                assert self.integrate_window_size <= self.psf_window_size
            assert "full" not in self.integrate_mode
        working_window = sample_image.window.make_copy()
        if "full" in self.psf_mode:
            working_window += self.target.psf_border
            center_shift = torch.round(self["center"].value/sample_image.pixelscale - 0.5)*sample_image.pixelscale - (self["center"].value - 0.5*sample_image.pixelscale)
            working_window.shift_origin(center_shift)
            
        working_image = Model_Image(pixelscale = sample_image.pixelscale, window = working_window, dtype = self.dtype, device = self.device)
        if "full" not in self.integrate_mode:
            working_image.data += self.evaluate_model(working_image)

        if "full" in self.psf_mode:
            self.integrate_model(working_image)
            working_image.data = conv2d(working_image.data.view(1,1,*working_image.data.shape), self.target.psf.view(1,1,*self.target.psf.shape), padding = "same")[0][0]
            working_image.shift_origin(-center_shift)
            working_image.crop(*self.target.psf_border_int)
        elif "window" in self.psf_mode:
            sub_window = self.psf_window.make_copy()
            sub_window += self.target.psf_border
            center_shift = torch.round(self["center"].value/sample_image.pixelscale - 0.5)*sample_image.pixelscale - (self["center"].value - 0.5*sample_image.pixelscale)
            sub_window.shift_origin(center_shift)
            sub_image = Model_Image(pixelscale = sample_image.pixelscale, window = sub_window, dtype = self.dtype, device = self.device)
            sub_image.data = self.evaluate_model(sub_image)
            self.integrate_model(sub_image)
            sub_image.data = conv2d(sub_image.data.view(1,1,*sub_image.data.shape), self.target.psf.view(1,1,*self.target.psf.shape), padding = "same")[0][0]
            sub_image.shift_origin(-center_shift)
            sub_image.crop(*self.target.psf_border_int)
            working_image.replace(sub_image)
        else:
            self.integrate_model(working_image)

        sample_image += working_image
        if sample_image is self.model_image:
            self.is_sampled = True
            
    def integrate_model(self, working_image):
        """Sample the model at a higher resolution than the given image, then
        integrate the super resolution up to the image
        resolution. This should not be overloaded except in very
        special circumstances.

        """
        # if True:
        #     return
        # Determine the on-sky window in which to integrate
        try:
            if "none" in self.integrate_mode or self.integrate_window.overlap_frac(working_image.window) <= 0.:
                return
        except AssertionError:
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
        integrate_image = Model_Image(pixelscale = integrate_pixelscale, window = working_window, dtype = self.dtype, device = self.device)
        # Evaluate the model at the fine sampling points
        X, Y = integrate_image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        integrate_image.data = self.evaluate_model(integrate_image)
        
        # Replace the image data where the integration has been done
        working_image.replace(working_window,
                              torch.sum(integrate_image.data.view(
                                  working_window.get_shape(working_image.pixelscale)[1],
                                  self.integrate_factor,
                                  working_window.get_shape(working_image.pixelscale)[0],
                                  self.integrate_factor
                              ), dim = (1,3))
        )

    def get_state(self):
        state = super().get_state()
        state["window"] = self.fit_window.get_state()
        if "parameters" not in state:
            state["parameters"] = {}
        for P in self.parameters:
            state["parameters"][P] = self[P].get_state()
        return state
    def load(self, filename = "AutoProf.yaml"):
        state = AutoProf_Model.load(filename)
        self.name = state["name"]
        self.fit_window = AP_Window(**state["window"])
        for key in state["parameters"]:
            self[key].update_state(state["parameters"][key])
        return state
    
    # Extra background methods for the basemodel
    ######################################################################
    from ._model_methods import _set_default_parameters
    from ._model_methods import set_fit_window
    from ._model_methods import fit_window
    from ._model_methods import scale_window
    from ._model_methods import target
    from ._model_methods import integrate_window
    from ._model_methods import psf_window
    from ._model_methods import step_iteration
    from ._model_methods import locked
    from ._model_methods import build_parameter_specs
    from ._model_methods import build_parameters
    from ._model_methods import get_parameters_representation
    from ._model_methods import get_parameters_value
    from ._model_methods import __getitem__
    from ._model_methods import __contains__
    from ._model_methods import __str__


    

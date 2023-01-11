from .core_model import AutoProf_Model
from ..image import Model_Image, Window
from .parameter_object import Parameter
from ..utils.initialize import center_of_mass
from ..utils.operations import fft_convolve_torch, fft_convolve_multi_torch
from ..utils.interpolate import _shift_Lanczos_kernel_torch
from ..utils.conversions.coordinates import coord_to_index, index_to_coord
from torch.autograd.functional import jacobian
from functools import partial
import numpy as np
import torch
from .. import AP_config

__all__ = ["BaseModel"]

class BaseModel(AutoProf_Model):
    """BaseModel(name, target, window, locked, **kwargs)

    This is the basis for almost any model which represents a single
    object, or parametric form.  Subclassing models must define their
    parameters, initialization, and model evaluation
    functions. See individual models for their behaviour.
    
    """

    # name of the model that AutoProf uses to identify
    model_type = "model"
    # Specifications for the model parameters including units, value, uncertainty, limits, locked, and cyclic
    parameter_specs = {
        "center": {"units": "arcsec", "uncertainty": 0.1},
    }
    # Fixed order of parameters for all methods that interact with the list of parameters
    _parameter_order = ("center",)

    # Technique and scope for PSF convolution
    psf_mode = "none" # none, window/full
    # size in pixels of the PSF convolution box
    psf_window_size = 50
    # Integration scope for model
    integrate_mode = "window" # none, window, full
    # size of the window in which to perform integration
    integrate_window_size = 10
    # Factor by which to upscale each dimension when integrating
    integrate_factor = 3 # number of pixels on one axis by which to supersample
    integrate_recursion_factor = 2 # relative size of windows between recursion levels (2 means each window will be half the size of the previous one)
    integrate_recursion_depth = 2 # number of recursion cycles to apply when integrating
    jacobian_mode = "full" # method to compute jacobian. "full" means full jacobian for all parameters at once (faster), "single" means one parameter at a time (less memory), "finite" means to use finite difference (minimum memory)

    # Parameters which are treated specially by the model object and should not be updated directly when initializing
    special_kwargs = ["parameters", "filename", "model_type"]
    
    def __init__(self, name, *args, **kwargs):        
        super().__init__(name, *args, **kwargs)
        
        self.parameters = {}
        # Set any user defined attributes for the model
        for kwarg in kwargs: # fixme move to core model?
            # Skip parameters with special behaviour
            if kwarg in self.special_kwargs:
                continue
            # Set the model parameter
            setattr(self, kwarg, kwargs[kwarg])

        self.parameter_specs = self.build_parameter_specs(kwargs.get("parameters", None))
        with torch.no_grad():
            self.build_parameters()
            if isinstance(kwargs.get("parameters", None), torch.Tensor):
                self.set_parameters(kwargs["parameters"])
            
        if "filename" in kwargs:
            self.load(kwargs["filename"])

    @property
    def parameters(self):
        try:
            return self._parameters
        except AttributeError:
            return {}
    @parameters.setter
    def parameters(self, val):
        self._parameters = val
        
    def parameter_order(self, override_locked = False):
        param_order = tuple()
        for P in  self.__class__._parameter_order:
            if self[P].locked and not override_locked:
                continue
            param_order = param_order + (P,)
        return param_order
            
    # Initialization functions
    ######################################################################
    @torch.no_grad()
    def initialize(self, target = None):
        """Determine initial values for the center coordinates. This is done
        with a local center of mass search which iterates by finding
        the center of light in a window, then iteratively updates
        until the iterations move by less than a pixel.

        """
        if target is None:
            target = self.target
        super().initialize(target)
        # Get the sub-image area corresponding to the model image
        target_area = target[self.window]
            
        # Use center of window if a center hasn't been set yet
        if self["center"].value is None:
            self["center"].set_value(self.window.center, override_locked = True)
        else:
            return

        if self["center"].locked:
            return

        # Convert center coordinates to target area array indices
        init_icenter = coord_to_index(self["center"].value[0], self["center"].value[1], target_area)
        # Compute center of mass in window
        COM = center_of_mass((init_icenter[0].detach().cpu().item(), init_icenter[1].detach().cpu().item()), target_area.data.detach().cpu().numpy())
        if np.any(np.array(COM) < 0) or np.any(np.array(COM) >= np.array(target_area.data.shape)):
            print("center of mass failed, using center of window")
            return
        # Convert center of mass indices to coordinates
        COM_center = index_to_coord(COM[0], COM[1], target_area)
        # Set the new coordinates as the model center
        self["center"].value = COM_center
            
    # Fit loop functions
    ######################################################################
    def evaluate_model(self, image):
        """Evaluate the model on every pixel in the given image. The
        basemodel object simply returns zeros, this function should be
        overloaded by subclasses.

        """
        return torch.zeros_like(image.data) # do nothing in base model

    def sample(self, sample_image = None, sample_window = None):
        """Evaluate the model on the space covered by an image object. This
        function properly calls integration methods and PSF
        convolution. This should not be overloaded except in special
        cases.

        Parameters:
            sample_image: An AutoProf Image object (likely a Model_Image) on which to evaluate the model values. Optional
            sample_window: A window within which to evaluate the model. Should only be used if a subset of the full image is needed.

        """
        # Image on which to evaluate model
        if sample_image is None:
            sample_image = self.make_model_image()
        # Window within which to evaluate model
        if sample_window is None:
            working_window = sample_image.window.make_copy()
        else:
            working_window = sample_window.make_copy() & sample_image.window
            
        if "window" in self.psf_mode:
            raise NotImplementedError("PSF convolution in sub-window not available yet")
            
        if "full" in self.psf_mode:
            # Add border for psf convolution edge effects, will be cropped out later
            working_window += self.target.psf_border
            # Determine the pixels scale at which to evalaute, this is smaller if the PSF is upscaled
            working_pixelscale = sample_image.pixelscale / self.target.psf_upscale
            # Sub pixel shift to align the model with the center of a pixel
            align = self.target.pixel_center_alignment()
            center_shift = self["center"].value - (torch.round(self["center"].value/working_pixelscale - align) + align)*working_pixelscale
            working_window.shift_origin(center_shift)
            # Make the image object to which the samples will be tracked
            working_image = Model_Image(pixelscale = working_pixelscale, window = working_window)
            # Evaluate the model at the current resolution
            working_image.data += self.evaluate_model(working_image)
            # If needed, super-resolve the image in areas of high curvature so pixels are properly sampled
            self.integrate_model(working_image, self.integrate_window(working_image, "center"), self.integrate_recursion_depth)
            # Convolve the PSF
            LL = _shift_Lanczos_kernel_torch(-center_shift[0]/working_image.pixelscale, -center_shift[1]/working_image.pixelscale, 3, AP_config.ap_dtype, AP_config.ap_device)
            shift_psf = torch.nn.functional.conv2d(self.target.psf.view(1,1,*self.target.psf.shape), LL.view(1,1,*LL.shape), padding = "same")[0][0]
            working_image.data = fft_convolve_torch(working_image.data, shift_psf/torch.sum(shift_psf), img_prepadded = True)
            # Shift image back to align with original pixel grid
            working_image.window.shift_origin(-center_shift)
            # Add the sampled/integrated/convolved pixels to the requested image
            sample_image += working_image.reduce(self.target.psf_upscale).crop(*self.target.psf_border_int)  
        else:
            # Create an image to store pixel samples
            working_image = Model_Image(pixelscale = sample_image.pixelscale, window = working_window)
            # Evaluate the model on the image
            working_image.data += self.evaluate_model(working_image)
            # Super-resolve and integrate where needed
            self.integrate_model(working_image, self.integrate_window(working_image, "pixel"), self.integrate_recursion_depth)
            # Add the sampled/integrated pixels to the requested image
            sample_image += working_image 
        
        return sample_image
            
            
    def integrate_model(self, working_image, window, depth = 2):
        """Sample the model at a higher resolution than the given image, then
        integrate the super resolution up to the image
        resolution.

        Parameters:
            working_image: the image on which to perform the model integration. Pixels in this image will be replaced with the integrated values
            window: A Window object within which to perform the integration.
            depth: recursion depth tracker. When called with depth = n, this function will call itself again with depth = n-1 until depth is 0 at which point it will exit without integrating.
        
        """
        if depth <= 0 or "none" in self.integrate_mode:
            return
        # Determine the on-sky window in which to integrate
        try:
            if window.overlap_frac(working_image.window) <= 0.:
                return
        except AssertionError:
            return
        # Only need to evaluate integration within working image
        working_window = window & working_image.window
        # Determine the upsampled pixelscale 
        integrate_pixelscale = working_image.pixelscale / self.integrate_factor
        # Build an image to hold the integration data
        integrate_image = Model_Image(pixelscale = integrate_pixelscale, window = working_window)
        # Evaluate the model at the fine sampling points
        integrate_image.data = self.evaluate_model(integrate_image)

        # If needed, recursively evaluates smaller windows
        recursive_shape = window.shape/integrate_pixelscale # get the number of pixels across the integrate window
        recursive_shape = torch.round(recursive_shape/self.integrate_recursion_factor).int() # divide window by recursion factor, ensure integer result
        window_align = torch.isclose((((window.center - integrate_image.origin)/integrate_image.pixelscale) % 1), torch.tensor(0.5, dtype = AP_config.ap_dtype, device = AP_config.ap_device), atol = 0.25)
        recursive_shape = (recursive_shape + 1 - (recursive_shape % 2) + 1 - window_align.to(dtype = torch.int32)) * integrate_pixelscale # ensure shape pairity is matched during recursion
        self.integrate_model(
            integrate_image,
            Window(
                center = window.center,
                shape = recursive_shape,
            ),
            depth = depth - 1,
        )
        # Replace the image data where the integration has been done
        working_image.replace(integrate_image.reduce(self.integrate_factor))

    @torch.no_grad()
    def jacobian_finite(self, parameters = None, as_representation = False, override_locked = False):
        """Compute the jacobian using regular finite differences. This uses
        none of the automatic differentiation available in pytorch and
        so is a bit slower and less precise. However, the memory
        footprint required is the absolute minimum for finite
        differences.

        """
        if parameters is not None:
            self.set_parameters(parameters, override_locked = override_locked, as_representation = as_representation)
        # fill out finite diff jacobian
        raise NotImplementedError("Finite differences jacobian not yet ready, but will be soon")
    
    def jacobian(self, parameters = None, as_representation = False, override_locked = False, flatten = False):
        if parameters is not None:
            self.set_parameters(parameters, override_locked = override_locked, as_representation = as_representation)

        if self.jacobian_mode == "full":
            full_jac = jacobian(
                partial(
                    self.full_sample,
                    as_representation = as_representation,
                    override_locked = override_locked,
                ),
                self.get_parameter_vector(
                    as_representation = as_representation,
                    override_locked = override_locked,
                ),
                strategy = "forward-mode",
                vectorize = True,
                create_graph = False,
            )
        elif self.jacobian_mode == "single":
            sub_jacs = []
            start = 0
            params = self.get_parameter_vector(
                as_representation = as_representation,
                override_locked = override_locked,
            )
            for P, V in zip(self.parameter_order(override_locked = override_locked),self.parameter_vector_len(override_locked = override_locked)):
                sub_jacs.append(
                    jacobian(
                        partial(
                            self.full_sample,
                            as_representation = as_representation,
                            override_locked = override_locked,
                            parameters_identity = (P,),
                        ),
                        params[start:start+V],
                        strategy = "forward-mode",
                        vectorize = True,
                        create_graph = False,                        
                    ).reshape(*tuple(self.window.get_shape_flip(self.target.pixelscale)), V)
                )
                start += V
            full_jac = torch.cat(sub_jacs,dim = 2)
        elif self.jacobian_mode == "finite":
            full_jac = self.jacobian_finite(
                parameters = parameters,
                as_representation = as_representation,
                override_locked = override_locked,
            )
        else:
            raise ValueError(f"Unrecognized jacobian mode for {self.name}: {self.jacobian_mode}")    
            
        if flatten:
            return full_jac.reshape(-1, np.sum(self.parameter_vector_len(override_locked = override_locked)))
        return full_jac
        
    def get_state(self):
        state = super().get_state()
        state["window"] = self.window.get_state()
        if "parameters" not in state:
            state["parameters"] = {}
        for P in self.parameters:
            state["parameters"][P] = self[P].get_state()
        return state
    def load(self, filename = "AutoProf.yaml"):
        state = AutoProf_Model.load(filename)
        self.name = state["name"]
        self.window = Window(**state["window"])
        for key in state["parameters"]:
            self[key].update_state(state["parameters"][key])
            self[key].to(dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        return state
    
    # Extra background methods for the basemodel
    ######################################################################
    from ._model_methods import integrate_window
    from ._model_methods import build_parameter_specs
    from ._model_methods import build_parameters
    from ._model_methods import __getitem__
    from ._model_methods import __contains__
    from ._model_methods import __str__


    

from functools import partial
from typing import Optional, Union
import io

from torch.autograd.functional import jacobian
import numpy as np
import torch

from .core_model import AutoProf_Model
from ..image import Model_Image, Window
from .parameter_object import Parameter
from ..utils.initialize import center_of_mass
from ..utils.operations import fft_convolve_torch, fft_convolve_multi_torch, selective_integrate
from ..utils.interpolate import _shift_Lanczos_kernel_torch
from ..utils.conversions.coordinates import coord_to_index, index_to_coord
from ._shared_methods import select_target
from .. import AP_config

__all__ = ["Component_Model"]


class Component_Model(AutoProf_Model):
    """Component_Model(name, target, window, locked, **kwargs)

    Component_Model is a base class for models that represent single
    objects or parametric forms. It provides the basis for subclassing
    models and requires the definition of parameters, initialization,
    and model evaluation functions. This class also handles
    integration, PSF convolution, and computing the Jacobian matrix.

    Attributes:
      parameter_specs (dict): Specifications for the model parameters.
      _parameter_order (tuple): Fixed order of parameters.
      psf_mode (str): Technique and scope for PSF convolution.
      psf_window_size (int): Size in pixels of the PSF convolution box.
      integrate_mode (str): Integration scope for the model.
      integrate_window_size (int): Size of the window in which to perform integration.
      integrate_factor (int): Factor by which to upscale each dimension when integrating.
      integrate_recursion_factor (int): Relative size of windows between recursion levels.
      integrate_recursion_depth (int): Number of recursion cycles to apply when integrating.
      jacobian_chunksize (int): Maximum size of parameter list before jacobian will be broken into smaller chunks.
      special_kwargs (list): Parameters which are treated specially by the model object and should not be updated directly.
      useable (bool): Indicates if the model is useable.

    Methods:
      initialize: Determine initial values for the center coordinates.
      sample: Evaluate the model on the space covered by an image object.
      jacobian: Compute the Jacobian matrix for this model.

    """

    # Specifications for the model parameters including units, value, uncertainty, limits, locked, and cyclic
    parameter_specs = {
        "center": {"units": "arcsec", "uncertainty": 0.1},
    }
    # Fixed order of parameters for all methods that interact with the list of parameters
    _parameter_order = ("center",)

    # Technique and scope for PSF convolution
    psf_mode = "none"  # none, window/full
    # size in pixels of the PSF convolution box
    psf_window_size = 50
    # Integration scope for model
    integrate_mode = "threshold"  # none, window, threshold

    # Size of the window in which to perform integration (window mode)
    integrate_window_size = 10 
    # Number of pixels on one axis by which to supersample (window mode)
    integrate_factor = 3  
    # Relative size of windows between recursion levels (2 means each window will be half the size of the previous one, window mode)
    integrate_recursion_factor = 2  
    # Number of recursion cycles to apply when integrating (window or threshold mode)
    integrate_recursion_depth = 3  
    # Threshold for triggering pixel integration (threshold mode)
    integrate_threshold = 1e-2 

    # Maximum size of parameter list before jacobian will be broken into smaller chunks, this is helpful for limiting the memory requirements to build a model, lower jacobian_chunksize is slower but uses less memory
    jacobian_chunksize = 10 

    # Parameters which are treated specially by the model object and should not be updated directly when initializing
    special_kwargs = ["parameters", "filename", "model_type"]
    useable = False

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        # Set any user defined attributes for the model
        for kwarg in kwargs:  # fixme move to core model?
            # Skip parameters with special behaviour
            if kwarg in self.special_kwargs:
                continue
            # Set the model parameter
            setattr(self, kwarg, kwargs[kwarg])

        self.parameter_specs = self.build_parameter_specs(
            kwargs.get("parameters", None)
        )
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

    def parameter_order(self, parameters_identity: Optional[tuple] = None):
        """Returns a tuple of names of the parameters in their set order."""
        param_order = tuple()
        for P in self.__class__._parameter_order:
            if self[P].locked:
                continue
            if parameters_identity is not None and not any(
                pid in parameters_identity for pid in self[P].identities
            ):
                continue
            param_order = param_order + (P,)
        return param_order

    # Initialization functions
    ######################################################################
    @torch.no_grad()
    @select_target
    def initialize(self, target: Optional["Target_Image"] = None):
        """Determine initial values for the center coordinates. This is done
        with a local center of mass search which iterates by finding
        the center of light in a window, then iteratively updates
        until the iterations move by less than a pixel.

        Args:
          target (Optional[Target_Image]): A target image object to use as a reference when setting parameter values

        """
        super().initialize(target)
        # Get the sub-image area corresponding to the model image
        target_area = target[self.window]

        # Use center of window if a center hasn't been set yet
        if self["center"].value is None:
            self["center"].set_value(self.window.center, override_locked=True)
        else:
            return

        if self["center"].locked:
            return

        # Convert center coordinates to target area array indices
        init_icenter = coord_to_index(
            self["center"].value[0], self["center"].value[1], target_area
        )
        # Compute center of mass in window
        COM = center_of_mass(
            (
                init_icenter[0].detach().cpu().item(),
                init_icenter[1].detach().cpu().item(),
            ),
            target_area.data.detach().cpu().numpy(),
        )
        if np.any(np.array(COM) < 0) or np.any(
            np.array(COM) >= np.array(target_area.data.shape)
        ):
            AP_config.ap_logger.warning("center of mass failed, using center of window")
            return
        # Convert center of mass indices to coordinates
        COM_center = index_to_coord(COM[0], COM[1], target_area)
        # Set the new coordinates as the model center
        self["center"].value = COM_center

    # Fit loop functions
    ######################################################################
    def evaluate_model(self, image: Union["Image", "Image_Header"], X: Optional[torch.Tensor] = None, Y: Optional[torch.Tensor] = None, **kwargs):
        """Evaluate the model on every pixel in the given image. The
        basemodel object simply returns zeros, this function should be
        overloaded by subclasses.

        Args:
          image (Image): The image defining the set of pixels on which to evaluate the model

        """
        if X is None or Y is None:
            X, Y = image.get_coordinate_meshgrid_torch(
                self["center"].value[0], self["center"].value[1]
            )
        return torch.zeros_like(X)  # do nothing in base model

    def sample(
        self,
        image: Optional["Image"] = None,
        window: Optional[Window] = None,
    ):
        """Evaluate the model on the space covered by an image object. This
        function properly calls integration methods and PSF
        convolution. This should not be overloaded except in special
        cases.

        This function is designed to compute the model on a given
        image or within a specified window. It takes care of sub-pixel
        sampling, recursive integration for high curvature regions,
        PSF convolution, and proper alignment of the computed model
        with the original pixel grid. The final model is then added to
        the requested image.

        Args:
          image (Optional[Image]): An AutoProf Image object (likely a Model_Image) 
                                     on which to evaluate the model values. If not 
                                     provided, a new Model_Image object will be created.
          window (Optional[Window]): A window within which to evaluate the model. 
                                   Should only be used if a subset of the full image 
                                   is needed. If not provided, the entire image will 
                                   be used.

        Returns:
          Image: The image with the computed model values.

        """
        # Image on which to evaluate model
        if image is None:
            image = self.make_model_image(window=window)

        # Window within which to evaluate model
        if window is None:
            working_window = image.window.copy()
        else:
            working_window = window.copy() & image.window

        if "window" in self.psf_mode:
            raise NotImplementedError("PSF convolution in sub-window not available yet")

        if "full" in self.psf_mode:
            # Add border for psf convolution edge effects, will be cropped out later
            working_window += self.target.psf_border
            # Determine the pixels scale at which to evalaute, this is smaller if the PSF is upscaled
            working_pixelscale = image.pixelscale / self.target.psf_upscale
            # Sub pixel shift to align the model with the center of a pixel
            align = self.target.pixel_center_alignment()
            center_shift = (
                self["center"].value
                - (
                    torch.round(self["center"].value / working_pixelscale - align)
                    + align
                )
                * working_pixelscale
            ).detach()
            working_window.shift_origin(center_shift)
            # Make the image object to which the samples will be tracked
            working_image = Model_Image(
                pixelscale=working_pixelscale, window=working_window
            )
            # Evaluate the model at the current resolution
            working_image.data += self.evaluate_model(image = working_image)
            # If needed, super-resolve the image in areas of high curvature so pixels are properly sampled
            if self.integrate_mode == "none":
                pass
            elif self.integrate_mode == "threshold":
                X, Y = working_image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
                selective_integrate(
                    X = X,
                    Y = Y,
                    data = working_image.data,
                    image_header = working_image.header,
                    eval_brightness = self.evaluate_model,
                    max_depth = self.integrate_recursion_depth,
                    integrate_threshold = self.integrate_threshold,
                )
            elif self.integrate_mode == "window":                
                self.window_integrate(
                    working_image,
                    self.integrate_window(working_image, "center"),
                    self.integrate_recursion_depth,
                )
            else:
                raise ValueError(f"{self.name} has unknown integration mode: {self.integrate_mode}")
            # Convolve the PSF
            LL = _shift_Lanczos_kernel_torch(
                -center_shift[0] / working_image.pixelscale,
                -center_shift[1] / working_image.pixelscale,
                3,
                AP_config.ap_dtype,
                AP_config.ap_device,
            )
            shift_psf = torch.nn.functional.conv2d(
                self.target.psf.view(1, 1, *self.target.psf.shape),
                LL.view(1, 1, *LL.shape),
                padding="same",
            )[0][0]
            working_image.data = fft_convolve_torch(
                working_image.data, shift_psf / torch.sum(shift_psf), img_prepadded=True
            )
            # Shift image back to align with original pixel grid
            working_image.window.shift_origin(-center_shift)
            # Add the sampled/integrated/convolved pixels to the requested image
            image += working_image.reduce(self.target.psf_upscale).crop(
                self.target.psf_border_int
            )
        else:
            # Create an image to store pixel samples
            working_image = Model_Image(
                pixelscale=image.pixelscale, window=working_window
            )
            # Evaluate the model on the image
            working_image.data += self.evaluate_model(image = working_image)
            # Super-resolve and integrate where needed
            if self.integrate_mode == "none":
                pass
            elif self.integrate_mode == "threshold":
                X, Y = working_image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
                selective_integrate(
                    X = X,
                    Y = Y,
                    data = working_image.data,
                    image_header = working_image.header,
                    eval_brightness = self.evaluate_model,
                    max_depth = self.integrate_recursion_depth,
                    integrate_threshold = self.integrate_threshold,
                )
            elif self.integrate_mode == "window":
                self.window_integrate(
                    working_image,
                    self.integrate_window(working_image, "pixel"),
                    self.integrate_recursion_depth,
                )
            else:
                raise ValueError(f"{self.name} has unknown integration mode: {self.integrate_mode}")
            # Add the sampled/integrated pixels to the requested image
            image += working_image

        return image

    def window_integrate(
        self, working_image: "Image", window: Window, depth: int = 2
    ):
        """Sample the model at a higher resolution than the given image, then
        integrate the super resolution up to the image resolution.

        This method improves the accuracy of the model evaluation by
        evaluating it at a finer resolution and integrating the
        results back to the original resolution. It recursively
        evaluates smaller windows in regions of high curvature until
        the specified recursion depth is reached.

        Args:
          working_image (Image): The image on which to perform the model
                                     integration. Pixels in this image will be
                                     replaced with the integrated values.
          window (Window): A Window object within which to perform the integration.
                           Specifies the region of interest for integration.
          depth (int, optional): Recursion depth tracker. When called with depth = n,
                                 this function will call itself again with depth = n-1
                                 until depth is 0, at which point it will exit without
                                 integrating further. Default value is 2.

        Returns:
          None: This method modifies the `working_image` in-place.

        """
        if depth <= 0 or "none" in self.integrate_mode:
            return
        # Determine the on-sky window in which to integrate
        try:
            if window.overlap_frac(working_image.window) <= 0.0:
                return
        except AssertionError:
            return
        # Only need to evaluate integration within working image
        working_window = window & working_image.window
        # Determine the upsampled pixelscale
        integrate_pixelscale = working_image.pixelscale / self.integrate_factor
        # Build an image to hold the integration data
        integrate_image = Model_Image(
            pixelscale=integrate_pixelscale, window=working_window
        )
        # Evaluate the model at the fine sampling points
        integrate_image.data = self.evaluate_model(integrate_image)

        # If needed, recursively evaluates smaller windows
        recursive_shape = (
            window.shape / integrate_pixelscale
        )  # get the number of pixels across the integrate window
        recursive_shape = torch.round(
            recursive_shape / self.integrate_recursion_factor
        ).int()  # divide window by recursion factor, ensure integer result
        window_align = torch.isclose(
            (
                ((window.center - integrate_image.origin) / integrate_image.pixelscale)
                % 1
            ),
            torch.tensor(0.5, dtype=AP_config.ap_dtype, device=AP_config.ap_device),
            atol=0.25,
        )
        recursive_shape = (
            recursive_shape
            + 1
            - (recursive_shape % 2)
            + 1
            - window_align.to(dtype=torch.int32)
        ) * integrate_pixelscale  # ensure shape pairity is matched during recursion
        self.window_integrate(
            integrate_image,
            Window(
                center=window.center,
                shape=recursive_shape,
            ),
            depth=depth - 1,
        )
        # Replace the image data where the integration has been done
        working_image.replace(integrate_image.reduce(self.integrate_factor))

    @torch.no_grad()
    def jacobian(
        self,
        parameters: Optional[torch.Tensor] = None,
        as_representation: bool = False,
        parameters_identity: Optional[tuple] = None,
        window: Optional[Window] = None,
        **kwargs,
    ):
        """Compute the Jacobian matrix for this model.

        The Jacobian matrix represents the partial derivatives of the
        model's output with respect to its input parameters. It is
        useful in optimization and model fitting processes. This
        method simplifies the process of computing the Jacobian matrix
        for astronomical image models and is primarily used by the
        Levenberg-Marquardt algorithm for model fitting tasks.

        Args:
          parameters (Optional[torch.Tensor]): A 1D parameter tensor to override the
                                               current model's parameters.
          as_representation (bool): Indicates if the parameters argument is
                                    provided as real values or representations
                                    in the (-inf, inf) range. Default is False.
          parameters_identity (Optional[tuple]): Specifies which parameters are to be
                                                 considered in the computation.
          window (Optional[Window]): A window object specifying the region of interest
                                     in the image.
          **kwargs: Additional keyword arguments.

        Returns:
          Jacobian_Image: A Jacobian_Image object containing the computed Jacobian matrix.

        """
        if window is None:
            window = self.window
        else:
            window = self.window & window
            
        # skip jacobian calculation if no parameters match criteria
        porder = self.parameter_order(parameters_identity=parameters_identity)
        if len(porder) == 0 or window.overlap_frac(self.window) <= 0:
            return self.target[window].jacobian_image()

        # Set the parameters if provided and check the size of the parameter list
        dochunk = False
        if parameters is not None:
            if len(parameters) > self.jacobian_chunksize:
                dochunk = True
            self.set_parameters(
                parameters,
                as_representation=as_representation,
                parameters_identity=parameters_identity,
            )
        else:
            if len(self.get_parameter_identity_vector(parameters_identity=parameters_identity)) > self.jacobian_chunksize:
                dochunk = True

        # If the parameter list is too large, apply the chunk jacobian analysis
        if dochunk:
            return self._chunk_jacobian(
                as_representation = as_representation,
                parameters_identity = parameters_identity,
                window = window,
                **kwargs,
            )            

        # Store the parameter identities
        if parameters_identity is None:
            pids = None
        else:
            pids = self.get_parameter_identity_vector(
                parameters_identity=parameters_identity,
            )
        # Compute the jacobian
        full_jac = jacobian(
            lambda P: self(
                image=None,
                parameters=P,
                as_representation=as_representation,
                parameters_identity=pids,
                window=window,
            ).data,
            self.get_parameter_vector(
                as_representation=as_representation,
                parameters_identity=parameters_identity,
            ).detach(),
            strategy="forward-mode",
            vectorize=True,
            create_graph=False,
        )

        # Store the jacobian as a Jacobian_Image object
        jac_img = self.target[window].jacobian_image(
            parameters=self.get_parameter_identity_vector(
                parameters_identity=parameters_identity,
            ),
            data=full_jac,
        )
        return jac_img

    @torch.no_grad()
    def _chunk_jacobian(
        self,
        as_representation: bool = False,
        parameters_identity: Optional[tuple] = None,
        window: Optional[Window] = None,
        **kwargs,
    ):
        """Evaluates the Jacobian in small chunks to reduce memory usage.

        For models with many parameters it can be prohibitive to build
        the full Jacobian in a single pass. Instead this function
        breaks the list of parameters into chunks as determined by
        `self.jacobian_chunksize` evaluates the Jacobian only for
        those, it then builds up the full Jacobian as a separate
        tensor. This is for internal use and should be called by the
        `self.jacobian` function when appropriate.

        """

        pids = self.get_parameter_identity_vector(
            parameters_identity=parameters_identity,
        )
        jac_img = self.target[window].jacobian_image(
            parameters=pids,
        )
        
        for ichunk in range(0,len(pids),self.jacobian_chunksize):
            jac_img += self.jacobian(
                parameters = None,
                as_representation = as_representation,
                parameters_identity = pids[ichunk:ichunk + self.jacobian_chunksize],
                window = window,
                **kwargs,
            )
            
        return jac_img
        

    def get_state(self):
        """Returns a dictionary with a record of the current state of the
        model.

        Specifically, the current parameter settings and the
        window for this model. From this information it is possible
        for the model to re-build itself lated when loading from
        disk. Note that the target image is not saved, this must be
        reset when loading the model.

        """
        state = super().get_state()
        state["window"] = self.window.get_state()
        if "parameters" not in state:
            state["parameters"] = {}
        for P in self.parameters:
            state["parameters"][P] = self[P].get_state()
        return state

    def load(self, filename: Union[str, dict, io.TextIOBase] = "AutoProf.yaml"):
        """Used to load the model from a saved state.

        Sets the model window to the saved value and updates all
        parameters with the saved information. This overrides the
        current parameter settings.

        Args:
          filename: The source from which to load the model parameters. Can be a string (the name of the file on disc), a dictionary (formatted as if from self.get_state), or an io.TextIOBase (a file stream to load the file from).

        """
        state = AutoProf_Model.load(filename)
        self.name = state["name"]
        self.window = Window(**state["window"])
        for key in state["parameters"]:
            self[key].update_state(state["parameters"][key])
            self[key].to(dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        return state

    # Extra background methods for the basemodel
    ######################################################################
    from ._model_methods import integrate_window
    from ._model_methods import build_parameter_specs
    from ._model_methods import build_parameters
    from ._model_methods import __getitem__
    from ._model_methods import __contains__
    from ._model_methods import __str__

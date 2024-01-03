from functools import partial
from typing import Optional, Union
import io

import numpy as np
import torch

from .core_model import AstroPhot_Model
from ..image import (
    Image,
    Model_Image,
    Window,
    PSF_Image,
    Target_Image,
    Target_Image_List,
    Image,
)
from ..param import Parameter_Node, Param_Unlock, Param_SoftLimits
from ..utils.initialize import center_of_mass
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ._shared_methods import select_target
from .. import AP_config
from ..errors import InvalidTarget

__all__ = ["Component_Model"]


class Component_Model(AstroPhot_Model):
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
      sampling_mode (str): Method for initial sampling of model. Can be one of midpoint, trapezoid, simpson. Default: midpoint
      sampling_tolerance (float): accuracy to which each pixel should be evaluated. Default: 1e-2
      integrate_mode (str): Integration scope for the model. One of none, threshold, full where threshold will select which pixels to integrate while full (in development) will integrate all pixels. Default: threshold
      integrate_max_depth (int): Maximum recursion depth when performing sub pixel integration.
      integrate_gridding (int): Amount by which to subdivide pixels when doing recursive pixel integration.
      integrate_quad_level (int): The initial quadrature level for sub pixel integration. Please always choose an odd number 3 or higher.
      softening (float): Softening length used for numerical stability and integration stability to avoid discontinuities (near R=0). Effectively has units of arcsec. Default: 1e-5
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
        "center": {"units": "arcsec", "uncertainty": [0.1, 0.1]},
    }
    # Fixed order of parameters for all methods that interact with the list of parameters
    _parameter_order = ("center",)

    # Scope for PSF convolution
    psf_mode = "none"  # none, full
    # Technique for PSF convolution
    psf_convolve_mode = "fft"  # fft, direct
    # Method to use when performing subpixel shifts. bilinear set by default for stability around pixel edges, though lanczos:3 is also fairly stable, and all are stable when away from pixel edges
    psf_subpixel_shift = "bilinear"  # bilinear, lanczos:2, lanczos:3, lanczos:5, none

    # Method for initial sampling of model
    sampling_mode = (
        "midpoint"  # midpoint, trapezoid, simpsons, quad:x (where x is a positive integer)
    )

    # Level to which each pixel should be evaluated
    sampling_tolerance = 1e-2

    # Integration scope for model
    integrate_mode = "threshold"  # none, threshold

    # Maximum recursion depth when performing sub pixel integration
    integrate_max_depth = 3

    # Amount by which to subdivide pixels when doing recursive pixel integration
    integrate_gridding = 5

    # The initial quadrature level for sub pixel integration. Please always choose an odd number 3 or higher
    integrate_quad_level = 3

    # Maximum size of parameter list before jacobian will be broken into smaller chunks, this is helpful for limiting the memory requirements to build a model, lower jacobian_chunksize is slower but uses less memory
    jacobian_chunksize = 10
    image_chunksize = 1000

    # Softening length used for numerical stability and/or integration stability to avoid discontinuities (near R=0)
    softening = 1e-3

    # Parameters which are treated specially by the model object and should not be updated directly when initializing
    special_kwargs = ["parameters", "filename", "model_type"]
    track_attrs = [
        "psf_mode",
        "psf_convolve_mode",
        "psf_subpixel_shift",
        "sampling_mode",
        "sampling_tolerance",
        "integrate_mode",
        "integrate_max_depth",
        "integrate_gridding",
        "integrate_quad_level",
        "jacobian_chunksize",
        "image_chunksize",
        "softening",
    ]
    useable = False

    def __init__(self, *, name=None, **kwargs):
        self._target_identity = None
        super().__init__(name=name, **kwargs)

        self.psf = None
        self.psf_aux_image = None

        # Set any user defined attributes for the model
        for kwarg in kwargs:  # fixme move to core model?
            # Skip parameters with special behaviour
            if kwarg in self.special_kwargs:
                continue
            # Set the model parameter
            setattr(self, kwarg, kwargs[kwarg])

        # If loading from a file, get model configuration then exit __init__
        if "filename" in kwargs:
            self.load(kwargs["filename"], new_name=name)
            return

        self.parameter_specs = self.build_parameter_specs(kwargs.get("parameters", None))
        with torch.no_grad():
            self.build_parameters()
            if isinstance(kwargs.get("parameters", None), torch.Tensor):
                self.parameters.value = kwargs["parameters"]

    def set_aux_psf(self, aux_psf, add_parameters=True):
        """Set the PSF for this model as an auxiliary psf model. This psf
        model will be resampled as part of the model sampling step to
        track changes made during fitting.

        Args:
          aux_psf: The auxiliary psf model
          add_parameters: if true, the parameters of the auxiliary psf model will become model parameters for this model as well.

        """

        self._psf = aux_psf

        if add_parameters:
            self.parameters.link(aux_psf.parameters)

    @property
    def psf(self):
        if self._psf is None:
            try:
                return self.target.psf
            except AttributeError:
                return None
        return self._psf

    @psf.setter
    def psf(self, val):
        if val is None:
            self._psf = None
        elif isinstance(val, PSF_Image):
            self._psf = val
        elif isinstance(val, AstroPhot_Model):
            self.set_aux_psf(val)
        else:
            self._psf = PSF_Image(data=val, pixelscale=self.target.pixelscale)
            AP_config.ap_logger.warning(
                "Setting PSF with pixel matrix, assuming target pixelscale is the same as "
                "PSF pixelscale. To remove this warning, set PSFs as an ap.image.PSF_Image "
                "or ap.models.AstroPhot_Model object instead."
            )

    # Initialization functions
    ######################################################################
    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(
        self,
        target: Optional["Target_Image"] = None,
        parameters: Optional[Parameter_Node] = None,
        **kwargs,
    ):
        """Determine initial values for the center coordinates. This is done
        with a local center of mass search which iterates by finding
        the center of light in a window, then iteratively updates
        until the iterations move by less than a pixel.

        Args:
          target (Optional[Target_Image]): A target image object to use as a reference when setting parameter values

        """
        super().initialize(target=target, parameters=parameters)
        # Get the sub-image area corresponding to the model image
        target_area = target[self.window]

        # Use center of window if a center hasn't been set yet
        if parameters["center"].value is None:
            with (
                Param_Unlock(parameters["center"]),
                Param_SoftLimits(parameters["center"]),
            ):
                parameters["center"].value = self.window.center
        else:
            return

        if parameters["center"].locked:
            return

        # Convert center coordinates to target area array indices
        init_icenter = target_area.plane_to_pixel(parameters["center"].value)

        # Compute center of mass in window
        COM = center_of_mass(
            (
                init_icenter[1].detach().cpu().item(),
                init_icenter[0].detach().cpu().item(),
            ),
            target_area.data.detach().cpu().numpy(),
        )
        if np.any(np.array(COM) < 0) or np.any(np.array(COM) >= np.array(target_area.data.shape)):
            AP_config.ap_logger.warning("center of mass failed, using center of window")
            return
        COM = (COM[1], COM[0])
        # Convert center of mass indices to coordinates
        COM_center = target_area.pixel_to_plane(
            torch.tensor(COM, dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        )

        # Set the new coordinates as the model center
        parameters["center"].value = COM_center

    # Fit loop functions
    ######################################################################
    def evaluate_model(
        self,
        X: Optional[torch.Tensor] = None,
        Y: Optional[torch.Tensor] = None,
        image: Optional[Image] = None,
        parameters: Parameter_Node = None,
        **kwargs,
    ):
        """Evaluate the model on every pixel in the given image. The
        basemodel object simply returns zeros, this function should be
        overloaded by subclasses.

        Args:
          image (Image): The image defining the set of pixels on which to evaluate the model

        """
        if X is None or Y is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]
        return torch.zeros_like(X)  # do nothing in base model

    def sample(
        self,
        image: Optional[Image] = None,
        window: Optional[Window] = None,
        parameters: Optional[Parameter_Node] = None,
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
          image (Optional[Image]): An AstroPhot Image object (likely a Model_Image)
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
            working_window = window.copy()

        # Parameters with which to evaluate the model
        if parameters is None:
            parameters = self.parameters

        if "window" in self.psf_mode:
            raise NotImplementedError("PSF convolution in sub-window not available yet")

        if "full" in self.psf_mode:
            if isinstance(self.psf, AstroPhot_Model):
                psf = self.psf(
                    parameters=parameters[self.psf.name],
                )
            else:
                psf = self.psf
            psf_upscale = torch.round(image.pixel_length / psf.pixel_length).int()
            # Add border for psf convolution edge effects, will be cropped out later
            working_window.pad_pixel(psf.psf_border_int)
            # Make the image object to which the samples will be tracked
            working_image = Model_Image(window=working_window)
            # Sub pixel shift to align the model with the center of a pixel
            if self.psf_subpixel_shift != "none":
                pixel_center = working_image.plane_to_pixel(parameters["center"].value)
                center_shift = pixel_center - torch.round(pixel_center)
                working_image.header.pixel_shift(center_shift)
            else:
                center_shift = None

            # Evaluate the model at the current resolution
            reference, deep = self._sample_init(
                image=working_image,
                parameters=parameters,
                center=parameters["center"].value,
            )
            # If needed, super-resolve the image in areas of high curvature so pixels are properly sampled
            deep = self._sample_integrate(
                deep, reference, working_image, parameters, parameters["center"].value
            )

            # update the image with the integrated pixels
            working_image.data += deep

            # Convolve the PSF
            self._sample_convolve(working_image, center_shift, psf, self.psf_subpixel_shift)

            # Shift image back to align with original pixel grid
            if self.psf_subpixel_shift != "none":
                working_image.header.pixel_shift(-center_shift)
            # Add the sampled/integrated/convolved pixels to the requested image
            working_image = working_image.reduce(psf_upscale).crop(psf.psf_border_int)

        else:
            # Create an image to store pixel samples
            working_image = Model_Image(pixelscale=image.pixelscale, window=working_window)
            # Evaluate the model on the image
            reference, deep = self._sample_init(
                image=working_image,
                parameters=parameters,
                center=parameters["center"].value,
            )
            # Super-resolve and integrate where needed
            deep = self._sample_integrate(
                deep,
                reference,
                working_image,
                parameters,
                center=parameters["center"].value,
            )
            # Add the sampled/integrated pixels to the requested image
            working_image.data += deep

        if self.mask is not None:
            working_image.data = working_image.data * torch.logical_not(self.mask)

        image += working_image

        return image

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, tar):
        if not (tar is None or isinstance(tar, Target_Image)):
            raise InvalidTarget("AstroPhot_Model target must be a Target_Image instance.")

        # If a target image list is assigned, pick out the target appropriate for this model
        if isinstance(tar, Target_Image_List) and self._target_identity is not None:
            for subtar in tar:
                if subtar.identity == self._target_identity:
                    usetar = subtar
                    break
            else:
                raise InvalidTarget(
                    f"Could not find target in Target_Image_List with matching identity "
                    f"to {self.name}: {self._target_identity}"
                )
        else:
            usetar = tar

        self._target = usetar

        # Remember the target identity to use
        try:
            self._target_identity = self._target.identity
        except AttributeError:
            pass

    def get_state(self, save_params=True):
        """Returns a dictionary with a record of the current state of the
        model.

        Specifically, the current parameter settings and the window for
        this model. From this information it is possible for the model to
        re-build itself lated when loading from disk. Note that the target
        image is not saved, this must be reset when loading the model.

        """
        state = super().get_state()
        state["window"] = self.window.get_state()
        if save_params:
            state["parameters"] = self.parameters.get_state()
        state["target_identity"] = self._target_identity
        if isinstance(self._psf, PSF_Image) or isinstance(self._psf, AstroPhot_Model):
            state["psf"] = self._psf.get_state()
        for key in self.track_attrs:
            if getattr(self, key) != getattr(self.__class__, key):
                state[key] = getattr(self, key)
        return state

    # Extra background methods for the basemodel
    ######################################################################
    from ._model_methods import radius_metric
    from ._model_methods import angular_metric
    from ._model_methods import _sample_init
    from ._model_methods import _sample_integrate
    from ._model_methods import _sample_convolve
    from ._model_methods import _integrate_reference
    from ._model_methods import _shift_psf
    from ._model_methods import build_parameter_specs
    from ._model_methods import build_parameters
    from ._model_methods import jacobian
    from ._model_methods import _chunk_jacobian
    from ._model_methods import _chunk_image_jacobian
    from ._model_methods import load

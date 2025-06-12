from typing import Optional

import numpy as np
import torch
from caskade import Param, forward

from .core_model import Model
from . import func
from ..image import (
    Model_Image,
    Window,
    PSF_Image,
    Target_Image,
    Target_Image_List,
    Image,
)
from ..utils.initialize import center_of_mass
from ..utils.decorators import ignore_numpy_warnings, default_internal, select_target
from .. import AP_config
from ..errors import InvalidTarget, SpecificationConflict

__all__ = ["Component_Model"]


class Component_Model(Model):
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
      usable (bool): Indicates if the model is usable.

    Methods:
      initialize: Determine initial values for the center coordinates.
      sample: Evaluate the model on the space covered by an image object.
      jacobian: Compute the Jacobian matrix for this model.

    """

    # Specifications for the model parameters including units, value, uncertainty, limits, locked, and cyclic
    _parameter_specs = Model._parameter_specs | {
        "center": {"units": "arcsec", "uncertainty": [0.1, 0.1]},
    }

    # Scope for PSF convolution
    psf_mode = "none"  # none, full
    # Method to use when performing subpixel shifts. bilinear set by default for stability around pixel edges, though lanczos:3 is also fairly stable, and all are stable when away from pixel edges
    psf_subpixel_shift = "lanczos:3"  # bilinear, lanczos:2, lanczos:3, lanczos:5, none

    # Method for initial sampling of model
    sampling_mode = "auto"  # auto (choose based on image size), midpoint, simpsons, quad:x (where x is a positive integer)

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
    usable = False

    def __init__(self, *, name=None, **kwargs):
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

        self.parameter_specs = self.build_parameter_specs(kwargs)
        for key in self.parameter_specs:
            setattr(self, key, Param(key, **self.parameter_specs[key]))

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
        elif isinstance(val, Model):
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
    def initialize(
        self,
    ):
        """Determine initial values for the center coordinates. This is done
        with a local center of mass search which iterates by finding
        the center of light in a window, then iteratively updates
        until the iterations move by less than a pixel.

        Args:
          target (Optional[Target_Image]): A target image object to use as a reference when setting parameter values

        """
        super().initialize()
        # Get the sub-image area corresponding to the model image
        target_area = self.target[self.window]

        # Use center of window if a center hasn't been set yet
        if self.center.value is None:
            self.center.value = target_area.center
        else:
            return

        # Compute center of mass in window
        COM = center_of_mass(target_area.data.npvalue)
        # Convert center of mass indices to coordinates
        COM_center = target_area.pixel_to_plane(
            *torch.tensor(COM, dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        )

        # Set the new coordinates as the model center
        self.center.value = COM_center

    # Fit loop functions
    ######################################################################
    @forward
    def brightness(
        self,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Evaluate the brightness of the model at the exact tangent plane coordinates requested."""
        return torch.zeros_like(x)  # do nothing in base model

    @forward
    def sample_image(self, image: Image):
        if self.sampling_mode == "auto":
            N = np.prod(image.data.shape)
            if N <= 100:
                sampling_mode = "quad:5"
            elif N <= 10000:
                sampling_mode = "simpsons"
            else:
                sampling_mode = "midpoint"
        else:
            sampling_mode = self.sampling_mode

        if sampling_mode == "midpoint":
            i, j = func.pixel_center_meshgrid(image.shape, AP_config.ap_dtype, AP_config.ap_device)
            x, y = image.pixel_to_plane(i, j)
            res = self.brightness(x, y)
            return func.pixel_center_integrator(res)
        elif sampling_mode == "simpsons":
            i, j = func.pixel_simpsons_meshgrid(
                image.shape, AP_config.ap_dtype, AP_config.ap_device
            )
            x, y = image.pixel_to_plane(i, j)
            res = self.brightness(x, y)
            return func.pixel_simpsons_integrator(res)
        elif sampling_mode.startswith("quad:"):
            order = int(self.sampling_mode.split(":")[1])
            i, j, w = func.pixel_quad_meshgrid(
                image.shape, AP_config.ap_dtype, AP_config.ap_device, order=order
            )
            x, y = image.pixel_to_plane(i, j)
            res = self.brightness(x, y)
            return func.pixel_quad_integrator(res, w)
        raise SpecificationConflict(
            f"Unknown integration mode {self.sampling_mode} for model {self.name}"
        )

    def shift_kernel(self, shift):
        if self.psf_subpixel_shift == "bilinear":
            return func.bilinear_kernel(shift[0], shift[1])
        elif self.psf_subpixel_shift.startswith("lanczos:"):
            order = int(self.psf_subpixel_shift.split(":")[1])
            return func.lanczos_kernel(shift[0], shift[1], order)
        elif self.psf_subpixel_shift == "none":
            return torch.tensor(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
        else:
            raise SpecificationConflict(
                f"Unknown PSF subpixel shift mode {self.psf_subpixel_shift} for model {self.name}"
            )

    @forward
    def sample(
        self,
        window: Optional[Window] = None,
        center=None,
    ):
        """Evaluate the model on the pixels defined in an image. This
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
        # Window within which to evaluate model
        if window is None:
            window = self.window

        if "window" in self.psf_mode:
            raise NotImplementedError("PSF convolution in sub-window not available yet")

        if "full" in self.psf_mode:
            psf = self.psf.image.value
            psf_upscale = torch.round(self.target.pixel_length / psf.pixel_length).int()
            psf_pad = np.max(psf.shape) // 2

            working_image = Model_Image(window=window, upsample=psf_upscale, pad=psf_pad)

            # Sub pixel shift to align the model with the center of a pixel
            if self.psf_subpixel_shift != "none":
                pixel_center = working_image.plane_to_pixel(center)
                pixel_shift = pixel_center - torch.round(pixel_center)
                center_shift = center - working_image.pixel_to_plane(torch.round(pixel_center))
                working_image.crtan = working_image.crtan.value + center_shift
            else:
                pixel_shift = torch.zeros_like(center)
                center_shift = torch.zeros_like(center)

            sample = self.sample_image(working_image)

            if self.integrate_mode == "threshold":
                sample = self.sample_integrate(sample, working_image)

            shift_kernel = self.shift_kernel(pixel_shift)
            working_image.data = func.convolve_and_shift(sample, shift_kernel, psf)
            working_image.crtan = working_image.crtan.value - center_shift

            working_image = working_image.crop(psf_pad).reduce(psf_upscale)

        else:
            working_image = Model_Image(window=window)
            sample = self.sample_image(working_image)
            if self.integrate_mode == "threshold":
                sample = self.sample_integrate(sample, working_image)
            working_image.data = sample

        if self.mask is not None:
            working_image.data = working_image.data * (~self.mask)

        return working_image

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
    from ._model_methods import build_parameter_specs
    from ._model_methods import jacobian
    from ._model_methods import load

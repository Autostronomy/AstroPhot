from typing import Optional

import numpy as np
import torch

from ..param import forward
from .base import Model
from . import func
from ..image import (
    ModelImage,
    TargetImage,
    Window,
    PSFImage,
)
from ..utils.initialize import recursive_center_of_mass
from ..utils.decorators import ignore_numpy_warnings
from .. import AP_config
from ..errors import SpecificationConflict, InvalidTarget
from .mixins import SampleMixin

__all__ = ["ComponentModel"]


class ComponentModel(SampleMixin, Model):
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
    _parameter_specs = {
        "center": {"units": "arcsec", "uncertainty": [0.1, 0.1], "shape": (2,)},
    }

    # Scope for PSF convolution
    psf_mode = "none"  # none, full
    # Method to use when performing subpixel shifts.
    psf_subpixel_shift = "lanczos:3"  # bilinear, lanczos:2, lanczos:3, lanczos:5, none
    # Softening length used for numerical stability and/or integration stability to avoid discontinuities (near R=0)
    softening = 1e-3  # arcsec

    _options = (
        "psf_mode",
        "psf_subpixel_shift",
        "softening",
    )
    usable = False

    def __init__(self, *args, psf=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.psf = psf

    @property
    def psf(self):
        if self._psf is None:
            return self.target.psf
        return self._psf

    @psf.setter
    def psf(self, val):
        if val is None:
            self._psf = None
        elif isinstance(val, PSFImage):
            self._psf = val
        elif isinstance(val, Model):
            self._psf = PSFImage(
                name="psf", data=val.target.data.value, pixelscale=val.target.pixelscale
            )
            self._psf.data = lambda p: p.psf_model().data.value
            self._psf.data.link("psf_model", val)
        else:
            self._psf = PSFImage(name="psf", data=val, pixelscale=self.target.pixelscale)
            AP_config.ap_logger.warning(
                "Setting PSF with pixel image, assuming target pixelscale is the same as "
                "PSF pixelscale. To remove this warning, set PSFs as an ap.image.PSF_Image "
                "or ap.models.PSF_Model object instead."
            )

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, tar):
        if tar is None:
            self._target = None
            return
        elif not isinstance(tar, TargetImage):
            raise InvalidTarget("AstroPhot Model target must be a Target_Image instance.")
        self._target = tar

    # Initialization functions
    ######################################################################
    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        """Determine initial values for the center coordinates. This is done
        with a local center of mass search which iterates by finding
        the center of light in a window, then iteratively updates
        until the iterations move by less than a pixel.

        Args:
          target (Optional[Target_Image]): A target image object to use as a reference when setting parameter values

        """
        target_area = self.target[self.window]

        # Use center of window if a center hasn't been set yet
        if self.center.value is None:
            self.center.dynamic_value = target_area.center
        else:
            return

        dat = np.copy(target_area.data.npvalue)
        if target_area.has_mask:
            mask = target_area.mask.detach().cpu().numpy()
            dat[mask] = np.nanmedian(dat[~mask])

        COM = recursive_center_of_mass(target_area.data.npvalue)
        if not np.all(np.isfinite(COM)):
            return
        COM_center = target_area.pixel_to_plane(
            *torch.tensor(COM, dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        )
        self.center.dynamic_value = COM_center

    def fit_mask(self):
        return torch.zeros_like(self.target[self.window].mask, dtype=torch.bool)

    @forward
    def transform_coordinates(self, x, y, center):
        return x - center[0], y - center[1]

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
            psf_upscale = torch.round(self.target.pixel_length / self.psf.pixel_length).int().item()
            psf_pad = np.max(self.psf.shape) // 2

            working_image = ModelImage(window=window, upsample=psf_upscale, pad=psf_pad)

            # Sub pixel shift to align the model with the center of a pixel
            if self.psf_subpixel_shift != "none":
                pixel_center = torch.stack(working_image.plane_to_pixel(*center))
                pixel_shift = pixel_center - torch.round(pixel_center)
                center_shift = center - torch.stack(
                    working_image.pixel_to_plane(*torch.round(pixel_center))
                )
                working_image.crtan = working_image.crtan.value + center_shift
            else:
                pixel_shift = torch.zeros_like(center)
                center_shift = torch.zeros_like(center)

            sample = self.sample_image(working_image)

            shift_kernel = self.shift_kernel(pixel_shift)
            working_image.data = func.convolve_and_shift(sample, shift_kernel, self.psf.data.value)
            working_image.crtan = working_image.crtan.value - center_shift

            working_image = working_image.crop([psf_pad]).reduce(psf_upscale)

        else:
            working_image = ModelImage(window=window)
            sample = self.sample_image(working_image)
            working_image.data = sample

        # Units from flux/arcsec^2 to flux
        working_image.data = working_image.data.value * working_image.pixel_area

        if self.mask is not None:
            working_image.data = working_image.data * (~self.mask)

        return working_image

from typing import Optional

import numpy as np
import torch

from ..param import forward
from .base import Model
from . import func
from ..image import (
    TargetImage,
    Window,
    PSFImage,
)
from ..utils.initialize import recursive_center_of_mass
from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from .. import config
from ..backend_obj import backend, ArrayLike
from ..errors import InvalidTarget
from .mixins import SampleMixin

__all__ = ("ComponentModel",)


@combine_docstrings
class ComponentModel(SampleMixin, Model):
    """Component of a model for an object in an image.

    This is a single component of an image model. It has a position on the sky
    determined by `center` and may or may not be convolved with a PSF to represent some data.

    **Parameters:**
    -  `center`: The center of the component in arcseconds [x, y] defined on the tangent plane.

    **Options:**
    -  `psf_convolve`: Whether to convolve the model with a PSF. (bool)

    """

    _parameter_specs = {"center": {"units": "arcsec", "shape": (2,)}}

    _options = ("psf_convolve",)

    usable = False

    def __init__(self, *args, psf=None, psf_convolve: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.psf = psf
        self.psf_convolve = psf_convolve

    @property
    def psf(self):
        if self._psf is None:
            return self.target.psf
        return self._psf

    @psf.setter
    def psf(self, val):
        try:
            del self._psf  # Remove old PSF if it exists
        except AttributeError:
            pass
        if val is None:
            self._psf = None
        elif isinstance(val, PSFImage):
            self._psf = val
            self.psf_convolve = True
        elif isinstance(val, Model):
            self._psf = val
            self.psf_convolve = True
        else:
            self._psf = self.target.psf_image(data=val)
            self.psf_convolve = True
        self._update_psf_upscale()

    def _update_psf_upscale(self):
        """Update the PSF upscale factor based on the current target pixel length."""
        if self.psf is None:
            self.psf_upscale = 1
        elif isinstance(self.psf, PSFImage):
            self.psf_upscale = int(np.round((self.target.pixelscale / self.psf.pixelscale).item()))
        elif isinstance(self.psf, Model):
            self.psf_upscale = int(
                np.round((self.target.pixelscale / self.psf.target.pixelscale).item())
            )
        else:
            raise TypeError(
                f"PSF must be a PSFImage or Model instance, got {type(self.psf)} instead."
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
            raise InvalidTarget("AstroPhot Model target must be a TargetImage instance.")
        try:
            del self._target  # Remove old target if it exists
        except AttributeError:
            pass
        self._target = tar
        try:
            self._update_psf_upscale()
        except AttributeError:
            pass

    # Initialization functions
    ######################################################################
    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        """Determine initial values for the center coordinates. This is done
        with a local center of mass search which iterates by finding
        the center of light in a window, then iteratively updates
        until the iterations move by less than a pixel.
        """
        if self.psf is not None and isinstance(self.psf, Model):
            self.psf.initialize()

        # Use center of window if a center hasn't been set yet
        if self.center.initialized:
            return

        target_area = self.target[self.window]
        dat = np.copy(backend.to_numpy(target_area.data))
        if target_area.has_mask:
            mask = backend.to_numpy(target_area.mask)
            dat[mask] = np.nanmedian(dat[~mask])

        COM = recursive_center_of_mass(dat)
        if not np.all(np.isfinite(COM)):
            return
        COM_center = target_area.pixel_to_plane(
            *backend.as_array(COM, dtype=config.DTYPE, device=config.DEVICE)
        )
        self.center.dynamic_value = COM_center

    def fit_mask(self):
        return backend.zeros_like(self.target[self.window].mask, dtype=backend.bool)

    @forward
    def transform_coordinates(self, x, y, center):
        return x - center[0], y - center[1]

    @forward
    def sample(
        self,
        window: Optional[Window] = None,
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

        **Args:**
        -  `window` (Optional[Window]): A window within which to evaluate the model.
                    By default this is the model's window.

        **Returns:**
        -  `Image` (ModelImage): The image with the computed model values.

        """
        # Window within which to evaluate model
        if window is None:
            window = self.window

        if self.psf_convolve:
            psf = self.psf() if isinstance(self.psf, Model) else self.psf

            working_image = self.target[window].model_image(
                upsample=self.psf_upscale, pad=psf.psf_pad
            )
            sample = self.sample_image(working_image)
            working_image._data = func.convolve(sample, psf.data)
            working_image = working_image.crop(psf.psf_pad).reduce(self.psf_upscale)

        else:
            working_image = self.target[window].model_image()
            working_image._data = self.sample_image(working_image)

        # Units from flux/arcsec^2 to flux, multiply by pixel area
        working_image.fluxdensity_to_flux()

        return working_image

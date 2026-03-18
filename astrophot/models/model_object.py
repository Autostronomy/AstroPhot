from typing import Optional, Union

import numpy as np
import torch

from ..param import forward
from .base import Model
from . import func
from ..image import TargetImage, Window, ModelImage, PSFImage
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

    _parameter_specs = {"center": {"units": "arcsec", "shape": (2,), "dynamic": True}}

    usable = False
    psf_convolve = True

    _options = ("psf_convolve",)

    def __init__(self, *args, psf=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.psf = psf
        self.saveattrs.add("window.extent")

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, tar):
        if tar is None:
            self._target = None
            return
        elif not isinstance(tar, TargetImage):
            raise InvalidTarget(
                f"AstroPhot {self.__class__.__name__} target must be a TargetImage instance."
            )
        try:
            del self._target  # Remove old target if it exists
        except AttributeError:
            pass
        self._target = tar

    @property
    def psf(self):
        if self._psf is None:
            return self.target.psf
        return self._psf

    @psf.setter
    def psf(self, psf):
        try:
            del self._psf  # Remove old psf if it exists
        except AttributeError:
            pass
        if psf is None:
            self._psf = None
        elif isinstance(psf, PSFImage):
            self._psf = psf
        elif isinstance(psf, Model):
            self._psf = psf
        else:
            self._psf = PSFImage(data=psf)
            config.logger.warning(
                f"PSF provided to {self.__class__.__name__} was not a PSFImage or Model instance, so it was converted to a PSFImage assuming no upsampling."
            )

    def _prep_psf(self):
        if not self.psf_convolve:
            return None, 1, 0
        psf = self.psf

        if isinstance(psf, PSFImage):
            return psf._data, psf.upsample, psf.pad
        if isinstance(psf, Model):
            return psf, psf.upsample, psf.pad
        return None, 1, 0

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
        dat = np.copy(backend.to_numpy(target_area._data))
        mask = backend.to_numpy(target_area._mask)
        dat[mask] = np.nanmedian(dat[~mask])

        COM = recursive_center_of_mass(dat)
        if not np.all(np.isfinite(COM)):
            return
        COM_center = target_area.pixel_to_plane(
            *backend.as_array(COM, dtype=config.DTYPE, device=config.DEVICE), ()
        )
        self.center.value = COM_center

    def fit_mask(self):
        return backend.zeros_like(self.target[self.window].mask, dtype=backend.bool)

    def _fit_mask(self):
        return backend.zeros_like(self.target[self.window]._mask, dtype=backend.bool)

    @forward
    def transform_coordinates(self, x, y, center):
        return x - center[0], y - center[1]

    @forward
    def pixel_brightness(self, i, j):
        """Evaluate the model at the pixel coordinates defined by i and j (of the target image)."""
        x, y = self.target.pixel_to_plane(i, j)
        return self.brightness(x, y)

    @forward
    def sample(
        self,
        I_: ArrayLike,
        J_: ArrayLike,
        psf: ArrayLike = None,
        crop: int = 0,
        downsample: int = 1,
    ):
        Z = self.pixel_brightness(I_, J_)
        Z = self._pixel_integrator(Z)
        I_, J_ = self._pixel_center_finder(I_, J_)
        Z = self._adaptive_integrator(Z, I_, J_, downsample)
        Z = Z * self.target.pixel_collecting_area(
            I_, J_, downsample
        )  # fixme, might not need downsample for pixel collecting area
        if psf is not None:
            if isinstance(psf, Model):
                psf = psf()._data
            if psf.shape != (1, 1):  # skip if identity PSF
                Z = func.convolve(Z, psf)
                Z = Z[crop : Z.shape[0] - crop, crop : Z.shape[1] - crop]
                Z = func.downsample(Z, downsample)
        return Z

    @forward
    def __call__(
        self,
        window: Optional[Window] = None,
    ) -> ModelImage:

        # Window within which to evaluate model
        if window is None:
            window = self.window
        else:
            window = window & self.window

        psf, upsample, pad = self._prep_psf()
        working_image = self.target.model_image(window)
        I, J = self._pixel_meshgridder(self.target, window, pad, upsample)

        # pixel_collecting_area: Units from flux/arcsec^2 to flux, multiply by pixel area
        working_image._data = self.sample(
            I,
            J,
            psf=psf,
            crop=pad,
            downsample=upsample,
        )
        return working_image

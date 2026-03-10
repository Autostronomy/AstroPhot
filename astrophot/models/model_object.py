from typing import Optional, Union

import numpy as np
import torch

from astrophot.image.model_image import ModelImage, ModelImageList

from ..param import forward, PSFParam
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

    _parameter_specs = {"center": {"units": "arcsec", "shape": (2,), "dynamic": True}}

    usable = False

    def __init__(self, *args, psf=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.psf = PSFParam(
            "psf",
            psf,
            shape=(None, None),
            description="Point Spread Function to convolve with this model",
        )
        self.saveattrs.add("window.extent")

    @property
    def psf(self):
        if self._psf is None:
            return self.target.psf
        elif isinstance(self._psf, Model):
            return self._psf()
        else:
            return self._psf

    def set_psf(self, psf):
        if psf is None:
            self.psf = None
        elif isinstance(psf, PSFImage):
            self.psf = psf
        elif isinstance(psf, Model):
            self._psf = psf
        else:
            self._psf = PSFImage(psf)

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
            *backend.as_array(COM, dtype=config.DTYPE, device=config.DEVICE)
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
    def sample(
        self,
        working_image: ModelImage,
        psf=None,
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
        assert (
            working_image.identity == self.target.identity
        ), "Model and target image must be matched (try `model.target.model_image()` to get a compatible model image)."
        sample = self.sample_image(working_image)
        if psf.shape != (1, 1):
            sample = func.convolve(sample, psf)

        return sample

    @forward
    def pixel_brightness(self, i, j):
        """Evaluate the model at the pixel coordinates defined by i and j (of the target image)."""
        x, y = self.target.pixel_to_plane(i, j)
        return self.brightness(x, y)

    @forward
    def sample(
        self,
        I: ArrayLike,
        J: ArrayLike,
        pixel_collecting_area: ArrayLike,
        crop: int = 0,
        downsample: int = 1,
        psf=None,
    ):
        Z = self.pixel_brightness(I, J)
        Z = self._pixel_integrator(Z)
        I, J = self._pixel_center_finder(I, J)
        Z = self._adaptive_integrator(Z, I, J)
        if psf.shape != (1, 1):
            Z = func.convolve(Z, psf)
            Z = Z[crop : Z.shape[0] - crop, crop : Z.shape[1] - crop]
            Z = func.downsample(Z, downsample)
        # fixme for sip this should technically be applied before PSF convolution (though effect is very very small)
        Z = Z * pixel_collecting_area
        return Z

    @forward
    def __call__(
        self,
        window: Optional[Window] = None,
        **kwargs,
    ) -> Union[ModelImage, ModelImageList]:

        # Window within which to evaluate model
        if window is None:
            window = self.window
        else:
            window = window & self.window

        working_image = self.target.model_image(window)
        I, J = self._pixel_meshgridder(self.target, window, self.psf.psf_pad, self.psf.upsample)
        # pixel_collecting_area: Units from flux/arcsec^2 to flux, multiply by pixel area
        working_image._data = self.sample(
            I,
            J,
            working_image.pixel_collecting_area,
            crop=self.psf.psf_pad,
            downsample=self.psf.upsample,
        )
        return working_image

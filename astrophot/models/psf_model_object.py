from typing import Optional, Tuple
import torch
from torch import Tensor
from caskade import forward

from .base import Model
from ..image import ModelImage, PSFImage, Window
from ..errors import InvalidTarget
from .mixins import SampleMixin
from ..backend_obj import backend, ArrayLike

__all__ = ["PSFModel"]


class PSFModel(SampleMixin, Model):
    """Prototype point source (typically a star) model, to be subclassed
    by other point source models which define specific behavior.

    PSF_Models behave differently than component models. For starters,
    their target image must be a PSF_Image object instead of a
    Target_Image object. PSF_Models also don't define a "center"
    variable since their center is always (0,0) just like a
    PSF_Image. A PSF_Model will never be convolved with a PSF_Model
    (that's it's job!), so a lot of the sampling method is simpler.

    """

    _parameter_specs = {
        "center": {"units": "arcsec", "value": (0.0, 0.0), "shape": (2,)},
    }
    _model_type = "psf"
    usable = False

    # The sampled PSF will be normalized to a total flux of 1 within the window
    normalize_psf = True

    # Parameters which are treated specially by the model object and should not be updated directly when initializing
    _options = ("normalize_psf",)

    def initialize(self):
        pass

    @forward
    def transform_coordinates(
        self, x: ArrayLike, y: ArrayLike, center: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        return x - center[0], y - center[1]

    # Fit loop functions
    ######################################################################
    @forward
    def sample(self, window: Optional[Window] = None) -> PSFImage:
        """Evaluate the model on the space covered by an image object. This
        function properly calls integration methods. This should not
        be overloaded except in special cases.

        This function is designed to compute the model on a given
        image or within a specified window. It takes care of sub-pixel
        sampling, recursive integration for high curvature regions,
        and proper alignment of the computed model with the original
        pixel grid. The final model is then added to the requested
        image.

        **Args:**
        -  `window` (Optional[Window]): A window within which to evaluate the model.
                                   Should only be used if a subset of the full image
                                   is needed. If not provided, the entire image will
                                   be used.

        **Returns:**
        -  `PSFImage`: The image with the computed model values.

        """
        # Create an image to store pixel samples
        working_image = self.target[self.window].model_image()
        working_image.data = self.sample_image(working_image)

        # normalize to total flux 1
        if self.normalize_psf:
            working_image.normalize()

        return working_image

    def fit_mask(self) -> ArrayLike:
        return backend.zeros_like(self.target[self.window].mask, dtype=backend.bool)

    @property
    def target(self):
        try:
            return self._target
        except AttributeError:
            return None

    @target.setter
    def target(self, target):
        if target is None:
            self._target = None
        elif not isinstance(target, PSFImage):
            raise InvalidTarget(f"Target for PSF_Model must be a PSF_Image, not {type(target)}")
        try:
            del self._target  # Remove old target if it exists
        except AttributeError:
            pass

        self._target = target

    @forward
    def __call__(self, window: Optional[Window] = None) -> ModelImage:
        return self.sample(window=window)

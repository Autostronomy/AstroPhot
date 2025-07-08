import torch
from caskade import forward

from .base import Model
from ..image import ModelImage, PSFImage
from ..errors import InvalidTarget
from .mixins import SampleMixin


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
    def transform_coordinates(self, x, y, center):
        return x - center[0], y - center[1]

    # Fit loop functions
    ######################################################################
    @forward
    def sample(self, window=None):
        """Evaluate the model on the space covered by an image object. This
        function properly calls integration methods. This should not
        be overloaded except in special cases.

        This function is designed to compute the model on a given
        image or within a specified window. It takes care of sub-pixel
        sampling, recursive integration for high curvature regions,
        and proper alignment of the computed model with the original
        pixel grid. The final model is then added to the requested
        image.

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
        # Create an image to store pixel samples
        working_image = self.target[self.window].model_image()
        working_image.data = self.sample_image(working_image)

        # normalize to total flux 1
        if self.normalize_psf:
            working_image.data = working_image.data / torch.sum(working_image.data)

        if self.mask is not None:
            working_image.data = working_image.data * (~self.mask)

        return working_image

    def fit_mask(self):
        return torch.zeros_like(self.target[self.window].mask, dtype=torch.bool)

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
    def __call__(self, window=None) -> ModelImage:
        return self.sample(window=window)

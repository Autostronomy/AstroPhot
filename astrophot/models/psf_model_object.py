from typing import Tuple
from caskade import forward

from .base import Model
from ..image import PSFImage
from ..errors import InvalidTarget
from .mixins import SampleMixin, GradMixin
from ..backend_obj import backend, ArrayLike

__all__ = ("PSFModel",)


class PSFModel(GradMixin, SampleMixin, Model):
    """Prototype point source (e.g., a star) model, to be subclassed
    by other point source models which define specific behavior.

    PSF models behave differently than component models. Their target image must
    be a ``PSFImage`` object instead of a ``TargetImage`` object. PSF models do
    not fit a free ``center`` parameter; their center is always ``(0, 0)`` in
    pixel coordinates, matching the convention of a ``PSFImage``. A PSF model is
    never convolved with another PSF model.

    Instead of units of arcsec for most length scales, PSFModel objects use
    `pix` units. This corresponds to the width of a pixel in the data target
    image; so if the PSFModel has an upsample factor of 2 then `1 pix`
    corresponds to two pixels in the image that the PSFModel outputs. This way,
    two PSFModels with different upsample factors, but applied to the same data
    target image, should still have the same parameter values for their shape
    parameters.

    :param center: Center of the PSF in pixel coordinates ``[x, y]``. Fixed at
        ``(0, 0)`` by default and not included in the fit.
    """

    _parameter_specs = {
        "center": {
            "units": "pix",
            "value": (0.0, 0.0),
            "shape": (2,),
            "dynamic": False,
            "description": "center of the PSF in pixel coordinates [x, y], fixed at (0,0)",
        },
    }
    _model_type = "psf"
    usable = False

    def initialize(self):
        pass

    @property
    def upsample(self):
        return self.target.upsample

    @property
    def pad(self):
        return self.target.pad

    @forward
    def transform_coordinates(
        self, x: ArrayLike, y: ArrayLike, center: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        return x - center[0], y - center[1]

    @forward
    def pixel_brightness(self, i, j):
        """Evaluate the model at the pixel coordinates defined by i and j (of
        the target image). For a PSF model, this is the same as the brightness
        since it is defined in pixel units."""
        return self.brightness(*self.target.mypixel_to_targpixel(i, j))

    def _prep_psf(self):
        return None, 1, 0

    # Fit loop functions
    ######################################################################
    @forward
    def sample(
        self,
        i: ArrayLike,
        j: ArrayLike,
        *args,
        **kwargs,
    ) -> PSFImage:
        """
        Sample the PSF model on the pixel grid defined by i and j.

        Depending on the model specification, this may involve supersampling for
        higher precision, or it may just be a direct evaluation of the model at
        the pixel centers. The output is the flux evaluated over the pixel grid
        at native resolution (for the PSFImage associated with this model.)

        :param i: 2D array of x-coordinates of pixel centers (or pre-upsampled
          according to the ``sampling_mode``) in pixel units.
        :param j: 2D array of y-coordinates of pixel centers (or pre-upsampled
          according to the ``sampling_mode``) in pixel units.

        :returns: 2D array (``Z``) of flux values at each pixel center, representing the
          PSF model evaluated at those coordinates.
        """
        Z = self.pixel_brightness(i, j)
        Z = self._pixel_integrator(Z)
        i, j = self._pixel_center_finder(i, j)
        Z = self._adaptive_integrator(Z, i, j, 1, self.pixel_brightness)
        return Z * self.target.pixel_area

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
            raise InvalidTarget(f"Target for PSFModel must be a PSFImage, not {type(target)}")
        try:
            del self._target  # Remove old target if it exists
        except AttributeError:
            pass

        self._target = target

    @forward
    def __call__(self, normalize_psf=True) -> PSFImage:
        working_image = self.target.model_image(self.window)
        i, j = self._pixel_meshgridder(self.target, self.window, 0, 1)
        working_image._data = self.sample(i, j)
        if normalize_psf:
            working_image.normalize()
        return working_image

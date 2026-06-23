import numpy as np

from .model_object import ComponentModel
from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from ..utils.interpolate import interp2d
from ..param import forward
from ..backend_obj import backend, ArrayLike
from ..utils.initialize import polar_decomposition
from . import func

__all__ = ["Pixelated"]


@combine_docstrings
class Pixelated(ComponentModel):
    """Model described by a grid of pixels on the sky (bilinear interpolation)

    This model represents an extended source on the sky as a grid of pixels with
    some brightness. The brightness at any point is determined by bilinear
    interpolation of the grid values. This is a very flexible model that can
    represent any source, but it is also computationally expensive to optimize
    the large number of free parameters.

    The PA and pixelscale are also parameters of this model, so one could alternately
    fix the pixels to some image and just fit the PA and pixelscale.

    :param I: the flux density at the center of each pixel, in units of flux/arcsec^2.
    :param PA: the position angle of the model, in radians.
    :param pixelscale: the pixel scale of the pixelated model, in arcsec per grid unit.
    :param scale: A scale factor to multiply the pixel values by. This times the pixel values in 'I' gives the total flux density in units of flux/arcsec^2.

    """

    _model_type = "pixelated"
    _parameter_specs = {
        "I": {
            "units": "flux/arcsec^2",
            "shape": (None, None),
            "dynamic": True,
            "description": "the flux density at the center of each pixel, in units of flux/arcsec^2",
        },
        "PA": {
            "units": "radians",
            "shape": (),
            "dynamic": False,
            "valid": (0, 2 * np.pi),
            "cyclic": True,
            "description": "the position angle of the model, in radians",
        },
        "pixelscale": {
            "units": "arcsec/grid-unit",
            "shape": (),
            "valid": (0, None),
            "dynamic": False,
            "description": "the pixel scale of the pixelated model, in arcsec per grid unit",
        },
        "scale": {
            "units": "unitless",
            "shape": (),
            "dynamic": False,
            "value": 1.0,
            "valid": (0, None),
            "description": "A scale factor to multiply the pixel values by. This times the pixel values in 'I' gives the total flux density in units of flux/arcsec^2",
        },
    }
    usable = True

    def __init__(self, *args, sampling_mode="upsample:3", integrate_mode="none", **kwargs):
        super().__init__(
            *args, sampling_mode=sampling_mode, integrate_mode=integrate_mode, **kwargs
        )

    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        if not self.pixelscale.initialized:
            self.pixelscale = self.target.pixelscale.item()

        if not self.scale.initialized:
            self.scale = 1.0

        if not self.I.initialized:
            target_area = self.target[self.window]
            self.I.value = backend.copy(target_area.data) / target_area.pixel_area

        if not self.PA.initialized:
            R, _ = polar_decomposition(self.target.CD.npvalue)
            self.PA.value = np.arccos(np.abs(R[0, 0]))

    @forward
    def transform_coordinates(
        self, x: ArrayLike, y: ArrayLike, PA: ArrayLike, pixelscale: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        x, y = super().transform_coordinates(x, y)
        x, y = func.rotate(-PA, x, y)
        return x / pixelscale, y / pixelscale

    @forward
    def brightness(self, x: ArrayLike, y: ArrayLike, I: ArrayLike, scale: ArrayLike) -> ArrayLike:
        x, y = self.transform_coordinates(x, y)
        I = I.T  # Transpose so I can use i,j indexing instead of j,i
        pixel_center = (I.shape[0] - 1) / 2, (I.shape[1] - 1) / 2
        return interp2d(I, x + pixel_center[0], y + pixel_center[1]) * scale

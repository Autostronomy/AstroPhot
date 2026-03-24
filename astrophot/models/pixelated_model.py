import torch
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

    The PA and scale are also parameters of this model, so one could alternately
    fix the pixels to some image and just fit the PA and scale.

    **Parameters:**
        - `I`: the total flux within each pixel, represented as the log of the flux.
        - `PA`: the position angle of the model, in radians.
        - `scale`: the scale of the model, in arcsec per grid unit.

    """

    _model_type = "pixelated"
    _parameter_specs = {
        "I": {"units": "flux/arcsec^2", "shape": (None, None), "dynamic": True},
        "PA": {"units": "radians", "shape": (), "dynamic": False},
        "scale": {"units": "arcsec/grid-unit", "shape": (), "dynamic": False},
    }
    usable = True

    def __init__(
        self, *args, scale=None, sampling_mode="upsample:3", integrate_mode="none", **kwargs
    ):
        super().__init__(
            *args, sampling_mode=sampling_mode, integrate_mode=integrate_mode, **kwargs
        )
        self.scale = scale

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        if not self.scale.initialized:
            self.scale = 2 * self.target.pixelscale.item()

        if not self.I.initialized:
            target_area = self.target[self.window]
            self.I.value = func.downsample(backend.copy(target_area.data), 2) / (
                target_area.pixel_area * 4
            )

        if not self.PA.initialized:
            R, _ = polar_decomposition(self.target.CD.npvalue)
            self.PA.value = np.arccos(np.abs(R[0, 0]))

    @forward
    def transform_coordinates(
        self, x: ArrayLike, y: ArrayLike, PA: ArrayLike, scale: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        x, y = super().transform_coordinates(x, y)
        x, y = func.rotate(-PA + np.pi / 2, x, y)
        return x / scale, y / scale

    @forward
    def brightness(self, x: ArrayLike, y: ArrayLike, I: ArrayLike) -> ArrayLike:
        x, y = self.transform_coordinates(x, y)
        pixel_center = (I.shape[0] - 1) / 2, (I.shape[1] - 1) / 2
        return interp2d(I, x + pixel_center[0], y + pixel_center[1])

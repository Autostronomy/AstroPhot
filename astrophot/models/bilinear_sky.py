from typing import Tuple
import numpy as np

from .sky_model_object import SkyModel
from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from ..utils.interpolate import interp2d
from ..param import forward
from ..backend_obj import backend, ArrayLike
from . import func
from ..utils.initialize import polar_decomposition

__all__ = ("BilinearSky",)


@combine_docstrings
class BilinearSky(SkyModel):
    """Sky background model using a coarse bilinear grid for the sky flux.

    This allows for modelling more complex sky surfaces, such as dust or
    galactic cirrus, without needing to specify a functional form. It is
    possible to specify a position angle and grid scale to control how it is
    oriented relative to the model target. By default it will just align with
    the image.

    :param I: sky brightness grid
    :param PA: position angle of the sky grid in radians.
    :param scale: scale of the sky grid in arcseconds per grid unit.

    """

    _model_type = "bilinear"
    _parameter_specs = {
        "I": {
            "units": "flux/arcsec^2",
            "shape": (None, None),
            "dynamic": True,
            "description": "sky brightness grid",
        },
        "PA": {
            "units": "radians",
            "shape": (),
            "dynamic": False,
            "description": "position angle of the sky grid in radians",
        },
        "scale": {
            "units": "arcsec/grid-unit",
            "shape": (),
            "dynamic": False,
            "description": "scale of the sky grid in arcseconds per grid unit",
        },
    }
    usable = True

    def __init__(self, *args, nodes: Tuple[int, int] = (3, 3), **kwargs):
        """Initialize the BilinearSky model with a grid of nodes."""
        super().__init__(*args, **kwargs)
        self.nodes = nodes

    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if self.I.initialized:
            self.nodes = tuple(self.I.value.shape)

        target_area = self.target[self.window]
        if not self.PA.initialized:
            R, _ = polar_decomposition(self.target.CD.npvalue)
            self.PA.value = np.arccos(np.abs(R[0, 0]))
        if not self.scale.initialized:
            self.scale.value = (
                self.target.pixelscale.item() * target_area._data.shape[0] / self.nodes[0]
            )

        if self.I.initialized:
            return

        dat = backend.to_numpy(target_area._data).copy()
        mask = backend.to_numpy(target_area._mask).copy()
        dat[mask] = np.nanmedian(dat)
        iS = dat.shape[0] // self.nodes[0]
        jS = dat.shape[1] // self.nodes[1]

        self.I.value = (
            np.median(
                dat[: iS * self.nodes[0], : jS * self.nodes[1]].reshape(
                    iS, self.nodes[0], jS, self.nodes[1]
                ),
                axis=(0, 2),
            )
            / self.target.pixel_area.item()
        )

    @forward
    def transform_coordinates(
        self, x: ArrayLike, y: ArrayLike, I: ArrayLike, PA: ArrayLike, scale: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        x, y = super().transform_coordinates(x, y)
        i, j = func.rotate(-PA, x, y)
        pixel_center = (I.shape[0] - 1) / 2, (I.shape[1] - 1) / 2
        return i / scale + pixel_center[0], j / scale + pixel_center[1]

    @forward
    def brightness(self, x: ArrayLike, y: ArrayLike, I: ArrayLike) -> ArrayLike:
        x, y = self.transform_coordinates(x, y)
        return interp2d(I, x, y)

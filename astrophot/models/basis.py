from typing import Union
import torch
import numpy as np

from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from ..utils.interpolate import interp2d
from .. import config
from .model_object import ComponentModel
from ..backend_obj import backend, ArrayLike
from ..errors import SpecificationConflict
from ..param import forward
from . import func
from ..utils.initialize import polar_decomposition

__all__ = ["BasisModel"]


@combine_docstrings
class BasisModel(ComponentModel):
    """Model described by a set of basis images.

    This model is composed of a set of basis images (think eigen decomposition
    or zernike polynomials) that are linearly combined with some weights to form
    the model image. The basis images are defined on a grid of coordinates,
    and the brightness at any point is determined by bilinear interpolation of
    the basis images. This is a very flexible model that can represent a wide
    range of sources, but depending on the number of basis elements it can become
    computationally expensive to optimize.

    :param weights: The weights of the basis set of images in units of flux.
    :param PA: the position angle of the model, in radians.
    :param scale: the scale of the model, in arcsec per grid unit.
    """

    _model_type = "basis"
    _parameter_specs = {
        "weights": {"units": "unitless", "shape": (None,), "dynamic": True},
        "PA": {"units": "radians", "shape": (), "dynamic": False},
        "scale": {"units": "arcsec/grid-unit", "shape": (), "dynamic": False},
    }
    usable = True

    def __init__(self, *args, basis: Union[str, ArrayLike] = "zernike:3", **kwargs):
        """Initialize the BasisModel with a basis set of images."""
        super().__init__(*args, **kwargs)
        self.basis = basis

    @property
    def basis(self):
        """The basis set of images used to form the model."""
        return self._basis

    @basis.setter
    def basis(self, value: Union[str, ArrayLike]):
        """Set the basis set of images."""
        if value is None:
            raise SpecificationConflict("BasisModel requires a basis set of images to be provided.")
        elif isinstance(value, str) and value.startswith("zernike:"):
            self._basis = value
        else:
            # Transpose since pytorch uses (j, i) indexing when (i, j) is more natural for coordinates
            self._basis = backend.transpose(
                backend.as_array(value, dtype=config.DTYPE, device=config.DEVICE), 2, 1
            )

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        if isinstance(self.basis, str) and self.basis.startswith("zernike:"):
            order = int(self.basis.split(":")[1])
            N = int(max(self.window.shape))
            N = N + 1 - N % 2
            self.basis = func.zernike_basis(order, N) / self.target.pixel_area

        if not self.weights.initialized:
            w = np.zeros(self.basis.shape[0])
            w[0] = 1.0
            self.weights.value = w

        if not self.PA.initialized:
            R, _ = polar_decomposition(self.target.CD.npvalue)
            self.PA.value = np.arccos(np.abs(R[0, 0]))

        if not self.scale.initialized:
            self.scale = self.target.pixelscale.item()

    @forward
    def transform_coordinates(
        self, x: ArrayLike, y: ArrayLike, PA: ArrayLike, scale: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        x, y = super().transform_coordinates(x, y)
        x, y = func.rotate(-PA + np.pi / 2, x, y)
        return x / scale, y / scale

    @forward
    def brightness(self, x: ArrayLike, y: ArrayLike, weights: ArrayLike) -> ArrayLike:
        x, y = self.transform_coordinates(x, y)
        wB = backend.sum(weights[:, None, None] * self.basis, dim=0)
        return interp2d(wB, x + wB.shape[0] // 2, y + wB.shape[1] // 2)

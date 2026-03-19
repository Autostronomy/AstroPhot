from typing import Union, Tuple
import torch
import numpy as np

from .psf_model_object import PSFModel
from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from ..utils.interpolate import interp2d
from .. import config
from ..backend_obj import backend, ArrayLike
from ..errors import SpecificationConflict
from ..param import forward
from . import func
from ..utils.initialize import polar_decomposition

__all__ = ["BasisPSF"]


@combine_docstrings
class PixelBasisPSF(PSFModel):
    """point source model which uses multiple images as a basis for the
    PSF as its representation for point sources. Using bilinear interpolation it
    will shift the PSF within a pixel to accurately represent the center
    location of a point source. There is no functional form for this object type
    as any image can be supplied. Bilinear interpolation is very fast and
    accurate for smooth models, so it is possible to do the expensive
    interpolation before optimization and save time.

    **Parameters:**
    -    `weights`: The weights of the basis set of images in units of flux.
    """

    _model_type = "basis"
    _parameter_specs = {"weights": {"units": "unitless", "shape": (None,), "dynamic": True}}
    usable = True

    def __init__(self, *args, basis: Union[str, ArrayLike] = "zernike:3", **kwargs):
        """Initialize the PixelBasisPSF model with a basis set of images."""
        super().__init__(*args, **kwargs)
        self.basis = basis

    @property
    def basis(self):
        """The basis set of images used to form the eigen point source."""
        return self._basis

    @basis.setter
    def basis(self, value: Union[str, ArrayLike]):
        """Set the basis set of images. If value is None, the basis is initialized to an empty tensor."""
        if value is None:
            raise SpecificationConflict(
                "PixelBasisPSF requires a basis set of images to be provided."
            )
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

    @forward
    def brightness(self, x: ArrayLike, y: ArrayLike, weights: ArrayLike) -> ArrayLike:
        x, y = self.transform_coordinates(x, y)
        wB = backend.sum(weights[:, None, None] * self.basis, dim=0)
        return interp2d(wB, x + wB.shape[0] // 2, y + wB.shape[1] // 2)

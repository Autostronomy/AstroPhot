from typing import Union
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

__all__ = ["PixelBasisPSF"]


@combine_docstrings
class PixelBasisPSF(PSFModel):
    """A point source defined by a linear combination of basis images.

    Point source model which uses multiple images as a basis for the PSF as its
    representation for point sources. Using bilinear interpolation it will shift
    the PSF within a pixel to accurately represent the center location of a
    point source. There is no functional form for this object type as any image
    can be supplied. Bilinear interpolation is very fast and accurate for smooth
    models, so it is possible to do the expensive interpolation before
    optimization and save time.

    The initialization of the weights is currently done by setting random
    values. This almost certainly produces a bad initial model. You may either
    set weights manually, or use a fitting step to get good starting weights.

    Note: The resulting PSF from the combined basis set will be normalized
        before being used as a PSF model, so the sum of the `weights` does not
        need to be restricted to any particular value.

    Note: It is possible for the basis elements to combine to give a PSF model
        that is negative in some areas. This is likely not desired, if this is a
        concern then use a non-negative basis and set the valid range of the
        weights to be `(0, None)`.

    :param weights: The weights of the basis set of images in units of flux.
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
            w = backend.as_array(
                1 / np.arange(1, self.basis.shape[0] + 1),
                dtype=config.DTYPE,
                device=config.DEVICE,
            )
            scale = backend.mean(self.target[self.window].data) / backend.mean(
                backend.sum(w[:, None, None] * self.basis, dim=0)
            )
            self.weights.value = w * scale

    @forward
    def brightness(self, x: ArrayLike, y: ArrayLike, weights: ArrayLike) -> ArrayLike:
        x, y = self.transform_coordinates(x, y)
        wB = backend.sum(weights[:, None, None] * self.basis, dim=0)
        u = self.target.upsample
        return interp2d(wB, x * u + wB.shape[0] // 2, y * u + wB.shape[1] // 2)

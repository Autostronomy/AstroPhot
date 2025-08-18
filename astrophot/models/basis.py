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
    -    `PA`: The position angle of the PSF in radians.
    -    `scale`: The scale of the PSF in arcseconds per grid unit.
    """

    _model_type = "basis"
    _parameter_specs = {
        "weights": {"units": "flux"},
        "PA": {"units": "radians", "shape": ()},
        "scale": {"units": "arcsec/grid-unit", "shape": ()},
    }
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
        target_area = self.target[self.window]
        if not self.PA.initialized:
            R, _ = polar_decomposition(self.target.CD.npvalue)
            self.PA.value = np.arccos(np.abs(R[0, 0]))
        if not self.scale.initialized:
            self.scale.value = self.target.pixelscale.item()
        if isinstance(self.basis, str) and self.basis.startswith("zernike:"):
            order = int(self.basis.split(":")[1])
            nm = func.zernike_n_m_list(order)
            N = int(
                target_area.data.shape[0] * self.target.pixelscale.item() / self.scale.value.item()
            )
            X, Y = np.meshgrid(
                np.linspace(-1, 1, N) * (N - 1) / N,
                np.linspace(-1, 1, N) * (N - 1) / N,
                indexing="ij",
            )
            R = np.sqrt(X**2 + Y**2)
            Phi = np.arctan2(Y, X)
            basis = []
            for n, m in nm:
                basis.append(func.zernike_n_m_modes(R, Phi, n, m))
            self.basis = np.stack(basis, axis=0)

        if not self.weights.initialized:
            w = np.zeros(self.basis.shape[0])
            w[0] = 1.0
            self.weights.dynamic_value = w

    @forward
    def transform_coordinates(
        self, x: ArrayLike, y: ArrayLike, PA: ArrayLike, scale: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        x, y = super().transform_coordinates(x, y)
        i, j = func.rotate(-PA, x, y)
        pixel_center = (self.basis.shape[1] - 1) / 2, (self.basis.shape[2] - 1) / 2
        return i / scale + pixel_center[0], j / scale + pixel_center[1]

    @forward
    def brightness(self, x: ArrayLike, y: ArrayLike, weights: ArrayLike) -> ArrayLike:
        x, y = self.transform_coordinates(x, y)
        return backend.sum(
            backend.vmap(lambda w, b: w * interp2d(b, x, y))(weights, self.basis), dim=0
        )

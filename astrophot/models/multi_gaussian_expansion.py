from typing import Optional, Tuple
import torch
import numpy as np

from .model_object import ComponentModel
from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from . import func
from .. import config
from ..backend_obj import backend, ArrayLike
from ..param import forward

__all__ = ["MultiGaussianExpansion"]


@combine_docstrings
class MultiGaussianExpansion(ComponentModel):
    """Model that represents a galaxy as a sum of multiple Gaussian
    profiles. The model is defined as:

    $$I(R) = \\sum_i {\\rm flux}_i * \\exp(-0.5*(R_i / \\sigma_i)^2) / (2 * \\pi * q_i * \\sigma_i^2)$$

    where $R_i$ is a radius computed using $q_i$ and $PA_i$ for that component. All components share the same center.

    **Parameters:**
    -    `q`: axis ratio to scale minor axis from the ratio of the minor/major axis b/a, this parameter is unitless, it is restricted to the range (0,1)
    -    `PA`: position angle of the semi-major axis relative to the image positive x-axis in radians, it is a cyclic parameter in the range [0,pi)
    -    `sigma`: standard deviation of each Gaussian
    -    `flux`: amplitude of each Gaussian
    """

    _model_type = "mge"
    _parameter_specs = {
        "q": {"units": "b/a", "valid": (0, 1)},
        "PA": {"units": "radians", "valid": (0, np.pi), "cyclic": True},
        "sigma": {"units": "arcsec", "valid": (0, None)},
        "flux": {"units": "flux"},
    }
    usable = True

    def __init__(self, *args, n_components: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if n_components is None:
            for key in ("q", "sigma", "flux"):
                if self[key].value is not None:
                    self.n_components = self[key].value.shape[0]
                    break
            else:
                self.n_components = 1
        else:
            self.n_components = int(n_components)

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        target_area = self.target[self.window]
        dat = backend.to_numpy(target_area.data).copy()
        if target_area.has_mask:
            mask = backend.to_numpy(target_area.mask)
            dat[mask] = np.median(dat[~mask])
        edge = np.concatenate((dat[:, 0], dat[:, -1], dat[0, :], dat[-1, :]))
        edge_average = np.nanmedian(edge)
        dat -= edge_average

        if not self.sigma.initialized:
            self.sigma.dynamic_value = np.logspace(
                np.log10(target_area.pixelscale.item() * 3),
                max(target_area.shape) * target_area.pixelscale.item() * 0.7,
                self.n_components,
            )
        if not self.flux.initialized:
            self.flux.dynamic_value = (np.sum(dat) / self.n_components) * np.ones(self.n_components)

        if self.PA.initialized or self.q.initialized:
            return

        x, y = target_area.coordinate_center_meshgrid()
        x = backend.to_numpy(x - self.center.value[0])
        y = backend.to_numpy(y - self.center.value[1])
        mu20 = np.median(dat * np.abs(x))
        mu02 = np.median(dat * np.abs(y))
        mu11 = np.median(dat * x * y / np.sqrt(np.abs(x * y) + self.softening**2))
        # mu20 = np.median(dat * x**2)
        # mu02 = np.median(dat * y**2)
        # mu11 = np.median(dat * x * y)
        M = np.array([[mu20, mu11], [mu11, mu02]])
        ones = np.ones(self.n_components)
        if not self.PA.initialized:
            if np.any(np.iscomplex(M)) or np.any(~np.isfinite(M)):
                self.PA.dynamic_value = ones * np.pi / 2
            else:
                self.PA.dynamic_value = (
                    ones * (0.5 * np.arctan2(2 * mu11, mu20 - mu02) - np.pi / 2) % np.pi
                )
        if not self.q.initialized:
            l = np.sort(np.linalg.eigvals(M))
            if np.any(np.iscomplex(l)) or np.any(~np.isfinite(l)):
                l = (0.7, 1.0)
            self.q.dynamic_value = ones * np.clip(np.sqrt(l[0] / l[1]), 0.1, 0.9)

    @forward
    def transform_coordinates(
        self, x: ArrayLike, y: ArrayLike, q: ArrayLike, PA: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        x, y = super().transform_coordinates(x, y)
        if np.prod(PA.shape) == 1:
            x, y = func.rotate(-(PA + np.pi / 2), x, y)
            x = x * backend.ones(
                (q.shape[0], *[1] * x.ndim), dtype=config.DTYPE, device=config.DEVICE
            )
            y = y * backend.ones(
                (q.shape[0], *[1] * y.ndim), dtype=config.DTYPE, device=config.DEVICE
            )
        else:
            x, y = backend.vmap(lambda pa: func.rotate(-(pa + np.pi / 2), x, y))(PA)
        y = backend.vmap(lambda q, y: y / q)(q, y)
        return x, y

    @forward
    def brightness(
        self, x: ArrayLike, y: ArrayLike, flux: ArrayLike, sigma: ArrayLike, q: ArrayLike
    ) -> ArrayLike:
        x, y = self.transform_coordinates(x, y)
        R = self.radius_metric(x, y)
        return backend.sum(
            backend.vmap(
                lambda A, r, sig, _q: (A / backend.sqrt(2 * np.pi * _q * sig**2))
                * backend.exp(-0.5 * (r / sig) ** 2)
            )(flux, R, sigma, q),
            dim=0,
        )

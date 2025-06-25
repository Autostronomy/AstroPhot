import torch
import numpy as np

from .model_object import ComponentModel
from ..utils.decorators import ignore_numpy_warnings
from . import func
from ..param import forward

__all__ = ["MultiGaussianExpansion"]


class MultiGaussianExpansion(ComponentModel):
    """Model that represents a galaxy as a sum of multiple Gaussian
    profiles. The model is defined as:

    I(R) = sum_i flux_i * exp(-0.5*(R_i / sigma_i)^2) / (2 * pi * q_i * sigma_i^2)

    where $R_i$ is a radius computed using $q_i$ and $PA_i$ for that component. All components share the same center.

    Parameters:
        q: axis ratio to scale minor axis from the ratio of the minor/major axis b/a, this parameter is unitless, it is restricted to the range (0,1)
        PA: position angle of the semi-major axis relative to the image positive x-axis in radians, it is a cyclic parameter in the range [0,pi)
        sigma: standard deviation of each Gaussian
        flux: amplitude of each Gaussian
    """

    _model_type = "mge"
    _parameter_specs = {
        "q": {"units": "b/a", "valid": (0, 1)},
        "PA": {"units": "radians", "valid": (0, np.pi), "cyclic": True},
        "sigma": {"units": "arcsec", "valid": (0, None)},
        "flux": {"units": "flux"},
    }
    usable = True

    def __init__(self, *args, n_components=None, **kwargs):
        super().__init__(*args, **kwargs)
        if n_components is None:
            for key in ("q", "sigma", "flux"):
                if self[key].value is not None:
                    self.n_components = self[key].value.shape[0]
                    break
            else:
                raise ValueError(
                    f"n_components must be specified when initial values is not defined."
                )
        else:
            self.n_components = int(n_components)

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        target_area = self.target[self.window]
        dat = target_area.data.npvalue.copy()
        if target_area.has_mask:
            mask = target_area.mask.detach().cpu().numpy()
            dat[mask] = np.median(dat[~mask])
        edge = np.concatenate((dat[:, 0], dat[:, -1], dat[0, :], dat[-1, :]))
        edge_average = np.nanmedian(edge)
        dat -= edge_average

        if self.sigma.value is None:
            self.sigma.dynamic_value = np.logspace(
                np.log10(target_area.pixel_length.item() * 3),
                max(target_area.shape) * target_area.pixel_length.item() * 0.7,
                self.n_components,
            )
            self.sigma.uncertainty = self.default_uncertainty * self.sigma.value
        if self.flux.value is None:
            self.flux.dynamic_value = (np.sum(dat) / self.n_components) * np.ones(self.n_components)
            self.flux.uncertainty = self.default_uncertainty * self.flux.value

        if not (self.PA.value is None or self.q.value is None):
            return

        x, y = target_area.coordinate_center_meshgrid()
        x = (x - self.center.value[0]).detach().cpu().numpy()
        y = (y - self.center.value[1]).detach().cpu().numpy()
        mu20 = np.median(dat * np.abs(x))
        mu02 = np.median(dat * np.abs(y))
        mu11 = np.median(dat * x * y / np.sqrt(np.abs(x * y)))
        # mu20 = np.median(dat * x**2)
        # mu02 = np.median(dat * y**2)
        # mu11 = np.median(dat * x * y)
        M = np.array([[mu20, mu11], [mu11, mu02]])
        ones = np.ones(self.n_components)
        if self.PA.value is None:
            if np.any(np.iscomplex(M)) or np.any(~np.isfinite(M)):
                self.PA.dynamic_value = ones * np.pi / 2
            else:
                self.PA.dynamic_value = (
                    ones * (0.5 * np.arctan2(2 * mu11, mu20 - mu02) - np.pi / 2) % np.pi
                )
        if self.q.value is None:
            l = np.sort(np.linalg.eigvals(M))
            if np.any(np.iscomplex(l)) or np.any(~np.isfinite(l)):
                l = (0.7, 1.0)
            self.q.dynamic_value = ones * np.clip(np.sqrt(l[0] / l[1]), 0.1, 0.9)

    @forward
    def total_flux(self, flux):
        return torch.sum(flux)

    @forward
    def transform_coordinates(self, x, y, q, PA):
        x, y = super().transform_coordinates(x, y)
        if PA.numel() == 1:
            x, y = func.rotate(-(PA + np.pi / 2), x, y)
            x = x.repeat(q.shape[0], *[1] * x.ndim)
            y = y.repeat(q.shape[0], *[1] * y.ndim)
        else:
            x, y = torch.vmap(lambda pa: func.rotate(-(pa + np.pi / 2), x, y))(PA)
        y = torch.vmap(lambda q, y: y / q)(q, y)
        return x, y

    @forward
    def brightness(self, x, y, flux, sigma, q):
        x, y = self.transform_coordinates(x, y)
        R = self.radius_metric(x, y)
        return torch.sum(
            torch.vmap(
                lambda A, r, sig, _q: (A / torch.sqrt(2 * np.pi * _q * sig**2))
                * torch.exp(-0.5 * (r / sig) ** 2)
            )(flux, R, sigma, q),
            dim=0,
        )

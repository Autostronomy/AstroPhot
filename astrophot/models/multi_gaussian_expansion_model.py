import torch
import numpy as np
from scipy.stats import iqr

from .psf_model_object import PSF_Model
from .model_object import Component_Model
from ._shared_methods import (
    select_target,
)
from ..utils.initialize import isophotes
from ..utils.angle_operations import Angle_COM_PA
from ..utils.conversions.coordinates import (
    Rotate_Cartesian,
)
from ..param import Param_Unlock, Param_SoftLimits, Parameter_Node
from ..utils.decorators import ignore_numpy_warnings, default_internal

__all__ = ["Multi_Gaussian_Expansion"]


class Multi_Gaussian_Expansion(Component_Model):
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

    model_type = f"mge {Component_Model.model_type}"
    parameter_specs = {
        "q": {"units": "b/a", "limits": (0, 1)},
        "PA": {"units": "radians", "limits": (0, np.pi), "cyclic": True},
        "sigma": {"units": "arcsec", "limits": (0, None)},
        "flux": {"units": "log10(flux)"},
    }
    _parameter_order = Component_Model._parameter_order + ("q", "PA", "sigma", "flux")
    usable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # determine the number of components
        for key in ("q", "sigma", "flux"):
            if self[key].value is not None:
                self.n_components = self[key].value.shape[0]
                break
        else:
            self.n_components = kwargs.get("n_components", 3)

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        target_area = target[self.window]
        target_dat = target_area.data.detach().cpu().numpy().copy()
        if target_area.has_mask:
            mask = target_area.mask.detach().cpu().numpy()
            target_dat[mask] = np.median(target_dat[np.logical_not(mask)])
        if parameters["sigma"].value is None:
            with Param_Unlock(parameters["sigma"]), Param_SoftLimits(parameters["sigma"]):
                parameters["sigma"].value = np.logspace(
                    np.log10(target_area.pixel_length.item() * 3),
                    max(target_area.shape.detach().cpu().numpy()) * 0.7,
                    self.n_components,
                )
                parameters["sigma"].uncertainty = (
                    self.default_uncertainty * parameters["sigma"].value
                )
        if parameters["flux"].value is None:
            with Param_Unlock(parameters["flux"]), Param_SoftLimits(parameters["flux"]):
                parameters["flux"].value = np.log10(
                    np.sum(target_dat[~mask]) / self.n_components
                ) * np.ones(self.n_components)
                parameters["flux"].uncertainty = 0.1 * parameters["flux"].value

        if not (parameters["PA"].value is None or parameters["q"].value is None):
            return
        edge = np.concatenate(
            (
                target_dat[:, 0],
                target_dat[:, -1],
                target_dat[0, :],
                target_dat[-1, :],
            )
        )
        edge_average = np.nanmedian(edge)
        edge_scatter = iqr(edge[np.isfinite(edge)], rng=(16, 84)) / 2
        icenter = target_area.plane_to_pixel(parameters["center"].value)

        if parameters["PA"].value is None:
            weights = target_dat - edge_average
            Coords = target_area.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]
            X, Y = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
            if target_area.has_mask:
                seg = np.logical_not(target_area.mask.detach().cpu().numpy())
                PA = Angle_COM_PA(weights[seg], X[seg], Y[seg])
            else:
                PA = Angle_COM_PA(weights, X, Y)

            with Param_Unlock(parameters["PA"]), Param_SoftLimits(parameters["PA"]):
                parameters["PA"].value = ((PA + target_area.north) % np.pi) * np.ones(
                    self.n_components
                )
                if parameters["PA"].uncertainty is None:
                    parameters["PA"].uncertainty = (5 * np.pi / 180) * torch.ones_like(
                        parameters["PA"].value
                    )  # default uncertainty of 5 degrees is assumed
        if parameters["q"].value is None:
            q_samples = np.linspace(0.2, 0.9, 15)
            try:
                pa = parameters["PA"].value.item()
            except:
                pa = parameters["PA"].value[0].item()
            iso_info = isophotes(
                target_area.data.detach().cpu().numpy() - edge_average,
                (icenter[1].detach().cpu().item(), icenter[0].detach().cpu().item()),
                threshold=3 * edge_scatter,
                pa=(pa - target.north),
                q=q_samples,
            )
            with Param_Unlock(parameters["q"]), Param_SoftLimits(parameters["q"]):
                parameters["q"].value = q_samples[
                    np.argmin(list(iso["amplitude2"] for iso in iso_info))
                ] * torch.ones(self.n_components)
                if parameters["q"].uncertainty is None:
                    parameters["q"].uncertainty = parameters["q"].value * self.default_uncertainty

    @default_internal
    def total_flux(self, parameters=None):
        return torch.sum(10 ** parameters["flux"].value)

    @default_internal
    def evaluate_model(self, X=None, Y=None, image=None, parameters=None, **kwargs):
        if X is None or Y is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]

        if parameters["PA"].value.numel() == 1:
            X, Y = Rotate_Cartesian(-(parameters["PA"].value - image.north), X, Y)
            X = X.repeat(parameters["q"].value.shape[0], *[1] * X.ndim)
            Y = torch.vmap(lambda q: Y / q)(parameters["q"].value)
        else:
            X, Y = torch.vmap(lambda pa: Rotate_Cartesian(-(pa - image.north), X, Y))(
                parameters["PA"].value
            )
            Y = torch.vmap(lambda q, y: y / q)(parameters["q"].value, Y)

        R = self.radius_metric(X, Y, image, parameters)
        return torch.sum(
            torch.vmap(
                lambda A, R, sigma, q: (A / (2 * np.pi * q * sigma**2))
                * torch.exp(-0.5 * (R / sigma) ** 2)
            )(
                image.pixel_area * 10 ** parameters["flux"].value,
                R,
                parameters["sigma"].value,
                parameters["q"].value,
            ),
            dim=0,
        )

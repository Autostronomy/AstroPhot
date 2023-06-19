import torch
import numpy as np

from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from .ray_model import Ray_Galaxy
from .wedge_model import Wedge_Galaxy
from .star_model_object import Star_Model
from ._shared_methods import (
    parametric_initialize,
    parametric_segment_initialize,
    select_target,
)
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.parametric_profiles import gaussian_torch, gaussian_np

__all__ = [
    "Gaussian_Galaxy",
    "Gaussian_SuperEllipse",
    "Gaussian_SuperEllipse_Warp",
    "Gaussian_FourierEllipse",
    "Gaussian_FourierEllipse_Warp",
    "Gaussian_Warp",
    "Gaussian_Star",
]


def _x0_func(model_params, R, F):
    return R[4], F[0]


def _wrap_gauss(R, sig, flu):
    return gaussian_np(R, sig, 10 ** flu)


class Gaussian_Galaxy(Galaxy_Model):
    """Basic galaxy model with Gaussian as the radial light profile. The
    gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    model_type = f"gaussian {Galaxy_Model.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0, None)},
        "flux": {"units": "log10(flux)"},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("sigma", "flux")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_gauss, ("sigma", "flux"), _x0_func
        )

    from ._shared_methods import gaussian_radial_model as radial_model


class Gaussian_SuperEllipse(SuperEllipse_Galaxy):
    """Super ellipse galaxy model with Gaussian as the radial light
    profile.The gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    model_type = f"gaussian {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0, None)},
        "flux": {"units": "log10(flux)"},
    }
    _parameter_order = SuperEllipse_Galaxy._parameter_order + ("sigma", "flux")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_gauss, ("sigma", "flux"), _x0_func
        )

    from ._shared_methods import gaussian_radial_model as radial_model


class Gaussian_SuperEllipse_Warp(SuperEllipse_Warp):
    """super ellipse warp galaxy model with a gaussian profile for the
    radial light profile. The gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    model_type = f"gaussian {SuperEllipse_Warp.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0, None)},
        "flux": {"units": "log10(flux)"},
    }
    _parameter_order = SuperEllipse_Warp._parameter_order + ("sigma", "flux")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_gauss, ("sigma", "flux"), _x0_func
        )

    from ._shared_methods import gaussian_radial_model as radial_model


class Gaussian_FourierEllipse(FourierEllipse_Galaxy):
    """fourier mode perturbations to ellipse galaxy model with a gaussian
    profile for the radial light profile. The gaussian radial profile
    is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    model_type = f"gaussian {FourierEllipse_Galaxy.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0, None)},
        "flux": {"units": "log10(flux)"},
    }
    _parameter_order = FourierEllipse_Galaxy._parameter_order + ("sigma", "flux")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_gauss, ("sigma", "flux"), _x0_func
        )

    from ._shared_methods import gaussian_radial_model as radial_model


class Gaussian_FourierEllipse_Warp(FourierEllipse_Warp):
    """fourier mode perturbations to ellipse galaxy model with a gaussian
    profile for the radial light profile. The gaussian radial profile
    is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    model_type = f"gaussian {FourierEllipse_Warp.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0, None)},
        "flux": {"units": "log10(flux)"},
    }
    _parameter_order = FourierEllipse_Warp._parameter_order + ("sigma", "flux")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_gauss, ("sigma", "flux"), _x0_func
        )

    from ._shared_methods import gaussian_radial_model as radial_model


class Gaussian_Warp(Warp_Galaxy):
    """Coordinate warped galaxy model with Gaussian as the radial light
    profile. The gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    model_type = f"gaussian {Warp_Galaxy.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0, None)},
        "flux": {"units": "log10(flux)"},
    }
    _parameter_order = Warp_Galaxy._parameter_order + ("sigma", "flux")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_gauss, ("sigma", "flux"), _x0_func
        )

    from ._shared_methods import gaussian_radial_model as radial_model


class Gaussian_Star(Star_Model):
    """Basica star model with a Gaussian as the radial light profile. The
    gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    model_type = f"gaussian {Star_Model.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0, None)},
        "flux": {"units": "log10(flux)"},
    }
    _parameter_order = Star_Model._parameter_order + ("sigma", "flux")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_gauss, ("sigma", "flux"), _x0_func
        )

    from ._shared_methods import gaussian_radial_model as radial_model

    @default_internal
    def evaluate_model(self, X=None, Y=None, image=None, parameters=None):
        if X is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]
        return self.radial_model(torch.sqrt(X ** 2 + Y ** 2), image, parameters)


class Gaussian_Ray(Ray_Galaxy):
    """ray galaxy model with a gaussian profile for the radial light
    model. The gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    model_type = f"gaussian {Ray_Galaxy.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0, None)},
        "flux": {"units": "log10(flux)"},
    }
    _parameter_order = Ray_Galaxy._parameter_order + ("sigma", "flux")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_segment_initialize(
            model=self,
            parameters=parameters,
            target=target,
            prof_func=_wrap_gauss,
            params=("sigma", "flux"),
            x0_func=_x0_func,
            segments=self.rays,
        )

    from ._shared_methods import gaussian_iradial_model as iradial_model


class Gaussian_Wedge(Wedge_Galaxy):
    """wedge galaxy model with a gaussian profile for the radial light
    model. The gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    model_type = f"gaussian {Wedge_Galaxy.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0, None)},
        "flux": {"units": "log10(flux)"},
    }
    _parameter_order = Wedge_Galaxy._parameter_order + ("sigma", "flux")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_segment_initialize(
            self,
            parameters,
            target,
            _wrap_gauss,
            ("sigma", "flux"),
            _x0_func,
            self.wedges,
        )

    from ._shared_methods import gaussian_iradial_model as iradial_model

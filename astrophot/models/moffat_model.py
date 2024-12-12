import torch
import numpy as np

from .galaxy_model_object import Galaxy_Model
from .psf_model_object import PSF_Model
from ._shared_methods import parametric_initialize, select_target
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.parametric_profiles import moffat_np
from ..utils.conversions.functions import moffat_I0_to_flux, general_uncertainty_prop
from ..param import Param_Unlock, Param_SoftLimits

__all__ = ["Moffat_Galaxy", "Moffat_PSF"]


def _x0_func(model_params, R, F):
    return 2.0, R[4], F[0]


def _wrap_moffat(R, n, rd, i0):
    return moffat_np(R, n, rd, 10 ** (i0))


class Moffat_Galaxy(Galaxy_Model):
    """basic galaxy model with a Moffat profile for the radial light
    profile. The functional form of the Moffat profile is defined as:

    I(R) = I0 / (1 + (R/Rd)^2)^n

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, I0 is the central flux
    density, Rd is the scale length for the profile, and n is the
    concentration index which controls the shape of the profile.

    Parameters:
        n: Concentration index which controls the shape of the brightness profile
        I0: brightness at the center of the profile, represented as the log of the brightness divided by pixel scale squared.
        Rd: scale length radius

    """

    model_type = f"moffat {Galaxy_Model.model_type}"
    parameter_specs = {
        "n": {"units": "none", "limits": (0.1, 10), "uncertainty": 0.05},
        "Rd": {"units": "arcsec", "limits": (0, None)},
        "I0": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("n", "Rd", "I0")
    usable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(self, parameters, target, _wrap_moffat, ("n", "Rd", "I0"), _x0_func)

    @default_internal
    def total_flux(self, parameters=None):
        return moffat_I0_to_flux(
            10 ** parameters["I0"].value,
            parameters["n"].value,
            parameters["Rd"].value,
            parameters["q"].value,
        )

    @default_internal
    def total_flux_uncertainty(self, parameters=None):
        return general_uncertainty_prop(
            (
                10 ** parameters["I0"].value,
                parameters["n"].value,
                parameters["Rd"].value,
                parameters["q"].value,
            ),
            (
                (10 ** parameters["I0"].value)
                * parameters["I0"].uncertainty
                * torch.log(10 * torch.ones_like(parameters["I0"].value)),
                parameters["n"].uncertainty,
                parameters["Rd"].uncertainty,
                parameters["q"].uncertainty,
            ),
            moffat_I0_to_flux,
        )

    from ._shared_methods import moffat_radial_model as radial_model


class Moffat_PSF(PSF_Model):
    """basic point source model with a Moffat profile for the radial light
    profile. The functional form of the Moffat profile is defined as:

    I(R) = I0 / (1 + (R/Rd)^2)^n

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, I0 is the central flux
    density, Rd is the scale length for the profile, and n is the
    concentration index which controls the shape of the profile.

    Parameters:
        n: Concentration index which controls the shape of the brightness profile
        I0: brightness at the center of the profile, represented as the log of the brightness divided by pixel scale squared.
        Rd: scale length radius

    """

    model_type = f"moffat {PSF_Model.model_type}"
    parameter_specs = {
        "n": {"units": "none", "limits": (0.1, 10), "uncertainty": 0.05},
        "Rd": {"units": "arcsec", "limits": (0, None)},
        "I0": {"units": "log10(flux/arcsec^2)", "value": 0.0, "locked": True},
    }
    _parameter_order = PSF_Model._parameter_order + ("n", "Rd", "I0")
    usable = True
    model_integrated = False

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(self, parameters, target, _wrap_moffat, ("n", "Rd", "I0"), _x0_func)

    from ._shared_methods import moffat_radial_model as radial_model

    @default_internal
    def total_flux(self, parameters=None):
        return moffat_I0_to_flux(
            10 ** parameters["I0"].value,
            parameters["n"].value,
            parameters["Rd"].value,
            torch.ones_like(parameters["n"].value),
        )

    @default_internal
    def total_flux_uncertainty(self, parameters=None):
        return general_uncertainty_prop(
            (
                10 ** parameters["I0"].value,
                parameters["n"].value,
                parameters["Rd"].value,
                torch.ones_like(parameters["n"].value),
            ),
            (
                (10 ** parameters["I0"].value)
                * parameters["I0"].uncertainty
                * torch.log(10 * torch.ones_like(parameters["I0"].value)),
                parameters["n"].uncertainty,
                parameters["Rd"].uncertainty,
                torch.zeros_like(parameters["n"].value),
            ),
            moffat_I0_to_flux,
        )

    from ._shared_methods import radial_evaluate_model as evaluate_model


class Moffat2D_PSF(Moffat_PSF):

    model_type = f"moffat2d {PSF_Model.model_type}"
    parameter_specs = {
        "q": {"units": "b/a", "limits": (0, 1), "uncertainty": 0.03},
        "PA": {
            "units": "radians",
            "limits": (0, np.pi),
            "cyclic": True,
            "uncertainty": 0.06,
        },
    }
    _parameter_order = Moffat_PSF._parameter_order + ("q", "PA")
    usable = True
    model_integrated = False

    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        with Param_Unlock(parameters["q"]), Param_SoftLimits(parameters["q"]):
            if parameters["q"].value is None:
                parameters["q"].value = 0.9

        with Param_Unlock(parameters["PA"]), Param_SoftLimits(parameters["PA"]):
            if parameters["PA"].value is None:
                parameters["PA"].value = 0.1
        super().initialize(target=target, parameters=parameters)

    from ._shared_methods import inclined_transform_coordinates as transform_coordinates
    from ._shared_methods import transformed_evaluate_model as evaluate_model

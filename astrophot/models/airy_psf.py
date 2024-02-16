import torch
import numpy as np

from ..utils.decorators import ignore_numpy_warnings, default_internal
from ._shared_methods import select_target
from .psf_model_object import PSF_Model
from ..param import Param_Unlock, Param_SoftLimits
from .. import AP_config

__all__ = ("Airy_PSF",)

class Airy_PSF(PSF_Model):
    """The Airy disk is an analytic description of the diffraction pattern
    for a circular aperture.

    The diffraction pattern is described exactly by the configuration
    of the lens system under the assumption that all elements are
    perfect. This expression goes as:

    .. math::

      I(\\theta) = I_0\\left[\\frac{2J_1(x)}{x}\\right]^2

      x = ka\\sin(\\theta) = \\frac{2\\pi a r}{\\lambda R}

    where :math:`I(\\theta)` is the intensity as a function of the
    angular poisition within the diffraction system along its main
    axis, :math:`I_0` is the central intensity of the airy disk,
    :math:`J_1` is the Bessel function of the first kind of order one,
    :math:`k = \\frac{2\\pi}{\\lambda}` is the wavenumber of the
    light, :math:`a` is the aperture radius, :math:`r` is the radial
    position from the center of the pattern, :math:`R` is the distance
    from the circular aperture to the observation plane.

    In the `Airy_PSF` class we combine the parameters
    :math:`a,R,\\lambda` into a single ratio to be optimized (or fixed
    by the optical configuration).

    """
    model_type = f"airy {PSF_Model.model_type}"
    parameter_specs = {
        "I0": {"units": "log10(flux/arcsec^2)", "value": 0., "locked": True},
        "aRL": {"units": "a/(R lambda)"}
    }
    _parameter_order = PSF_Model._parameter_order + ("I0", "aRL")
    useable = True
    model_integrated = False
    
    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        if (parameters["I0"].value is not None) and (
            parameters["aRL"].value is not None
        ):
            return
        target_area = target[self.window]
        icenter = target_area.plane_to_pixel(parameters["center"].value)

        if parameters["I0"].value is None:
            with Param_Unlock(parameters["I0"]), Param_SoftLimits(parameters["I0"]):
                parameters["I0"].value = torch.log10(
                    torch.mean(
                        target_area.data[
                            int(icenter[0]) - 2 : int(icenter[0]) + 2,
                            int(icenter[1]) - 2 : int(icenter[1]) + 2,
                        ]
                    )
                    / target.pixel_area.item()
                )
                parameters["I0"].uncertainty = torch.std(
                    target_area.data[
                        int(icenter[0]) - 2 : int(icenter[0]) + 2,
                        int(icenter[1]) - 2 : int(icenter[1]) + 2,
                    ]
                ) / (torch.abs(parameters["I0"].value) * target.pixel_area)
        if parameters["aRL"].value is None:
            with Param_Unlock(parameters["aRL"]), Param_SoftLimits(parameters["aRL"]):
                parameters["aRL"].value = (5./8.) * 2 * target.pixel_length
                parameters["aRL"].uncertainty = parameters["aRL"].value * self.default_uncertainty
        

    @default_internal
    def radial_model(self, R, image=None, parameters=None):
        x = 2 * torch.pi * parameters["aRL"].value * R

        return (image.pixel_area * 10**parameters["I0"].value) * (2 * torch.special.bessel_j1(x) / x)**2
    
    from ._shared_methods import radial_evaluate_model as evaluate_model
    

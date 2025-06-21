import torch

from ..utils.decorators import ignore_numpy_warnings
from .psf_model_object import PSFModel
from .mixins import RadialMixin

__all__ = ("AiryPSF",)


class AiryPSF(RadialMixin, PSFModel):
    """The Airy disk is an analytic description of the diffraction pattern
    for a circular aperture.

    The diffraction pattern is described exactly by the configuration
    of the lens system under the assumption that all elements are
    perfect. This expression goes as:

    .. math::

      I(\\theta) = I_0\\left[\\frac{2J_1(x)}{x}\\right]^2

      x = ka\\sin(\\theta) = \\frac{2\\pi a r}{\\lambda R}

    where :math:`I(\\theta)` is the intensity as a function of the
    angular position within the diffraction system along its main
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

    _model_type = "airy"
    _parameter_specs = {
        "I0": {"units": "flux/arcsec^2", "value": 1.0, "shape": ()},
        "aRL": {"units": "a/(R lambda)", "shape": ()},
    }
    usable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if (self.I0.value is not None) and (self.aRL.value is not None):
            return
        icenter = self.target.plane_to_pixel(*self.center.value)

        if self.I0.value is None:
            mid_chunk = self.target.data.value[
                int(icenter[0]) - 2 : int(icenter[0]) + 2,
                int(icenter[1]) - 2 : int(icenter[1]) + 2,
            ]
            self.I0.dynamic_value = torch.mean(mid_chunk) / self.target.pixel_area
            self.I0.uncertainty = torch.std(mid_chunk) / self.target.pixel_area
        if self.aRL.value is None:
            self.aRL.value = (5.0 / 8.0) * 2 * self.target.pixel_length
            self.aRL.uncertainty = self.aRL.value * self.default_uncertainty

    def radial_model(self, R, I0, aRL):
        x = 2 * torch.pi * aRL * R
        return I0 * (2 * torch.special.bessel_j1(x) / x) ** 2

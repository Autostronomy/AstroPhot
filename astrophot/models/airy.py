import torch
import numpy as np

from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from .psf_model_object import PSFModel
from .mixins import RadialMixin
from ..param import forward
from ..backend_obj import backend, ArrayLike

__all__ = ("AiryPSF",)


@combine_docstrings
class AiryPSF(RadialMixin, PSFModel):
    """The Airy disk is an analytic description of the diffraction pattern
    for a circular aperture.

    The diffraction pattern is described exactly by the configuration
    of the lens system under the assumption that all elements are
    perfect. This expression goes as:

    $$I(\\theta) = I_0\\left[\\frac{2J_1(x)}{x}\\right]^2$$
    $$x = ka\\sin(\\theta) = \\frac{2\\pi a r}{\\lambda R}$$

    where $I(\\theta)$ is the intensity as a function of the
    angular position within the diffraction system along its main
    axis, $I_0$ is the central intensity of the airy disk,
    $J_1$ is the Bessel function of the first kind of order one,
    $k = \\frac{2\\pi}{\\lambda}$ is the wavenumber of the
    light, $a$ is the aperture radius, $r$ is the radial
    position from the center of the pattern, $R$ is the distance
    from the circular aperture to the observation plane.

    In the `Airy_PSF` class we combine the parameters
    $a,R,\\lambda$ into a single ratio to be optimized (or fixed
    by the optical configuration).

    **Parameters:**
    -    `I0`: The central intensity of the airy disk in flux/arcsec^2.
    -    `aRL`: The ratio of the aperture radius to the
        product of the wavelength and the distance from the aperture to the
        observation plane, $\\frac{a}{R \\lambda}$.

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

        if self.I0.initialized and self.aRL.initialized:
            return
        icenter = self.target.plane_to_pixel(*self.center.value)

        if not self.I0.initialized:
            mid_chunk = self.target.data[
                int(icenter[0]) - 2 : int(icenter[0]) + 2,
                int(icenter[1]) - 2 : int(icenter[1]) + 2,
            ]
            self.I0.dynamic_value = backend.mean(mid_chunk) / self.target.pixel_area
        if not self.aRL.initialized:
            self.aRL.dynamic_value = (5.0 / 8.0) * 2 * self.target.pixelscale

    @forward
    def radial_model(self, R: ArrayLike, I0: ArrayLike, aRL: ArrayLike) -> ArrayLike:
        x = 2 * np.pi * aRL * R
        return I0 * (2 * backend.bessel_j1(x) / x) ** 2

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

    WARNING: This model does not work in JAX (it doesn't have the required Bessel function implemented)

    WARNING: PyTorch appears to have an issue with gradients wrt the R1 parameter. Optimization doesn't seem to work for this model (maybe try scipy optimize?).

    The diffraction pattern is described exactly by the configuration of the
    lens system under the assumption that all elements are perfect. This
    expression goes as:

    .. math::

       I(\\theta) = I_0\\left[\\frac{2J_1(x)}{x}\\right]^2

    .. math::

       x = ka\\sin(\\theta) = \\frac{2\\pi a r}{\\lambda R}

    where :math:`I(\\theta)` is the intensity as a function of the angular
    position within the diffraction system along its main axis, :math:`I_0` is
    the central intensity of the airy disk, :math:`J_1` is the Bessel function
    of the first kind of order one, :math:`k = \\frac{2\\pi}{\\lambda}` is the
    wavenumber of the light, :math:`a` is the aperture radius, :math:`r` is the
    radial position from the center of the pattern, :math:`R` is the distance
    from the circular aperture to the observation plane.

    In the ``AiryPSF`` class we combine the parameters :math:`a,R,\\lambda` and
    scale based on the first zero of the :math:`J_1` function. This way you can
    work with the more intuitive radius parameter.

    :param I0: The central intensity of the airy disk in flux/pix^2.
    :param R1: The radius of the first zero of the airy disk in pix.

    """

    _model_type = "airy"
    _parameter_specs = {
        "I0": {
            "units": "flux/pix^2",
            "value": 1.0,
            "shape": (),
            "dynamic": False,
            "description": "The central intensity of the airy disk in flux/pix^2.",
        },
        "R1": {
            "units": "pix",
            "shape": (),
            "dynamic": True,
            "description": "The radius of the first zero of the airy disk in pix.",
        },
    }
    usable = True

    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if self.I0.initialized and self.R1.initialized:
            return
        icenter = self.target.targpixel_to_mypixel(*self.center.value)

        if not self.I0.initialized:
            mid_chunk = self.target._data[
                int(icenter[0]) - 2 : int(icenter[0]) + 2,
                int(icenter[1]) - 2 : int(icenter[1]) + 2,
            ]
            self.I0.value = backend.mean(mid_chunk) / self.target.upsample**2
        if not self.R1.initialized:
            self.R1.value = 2.0

    @forward
    def radial_model(self, R: ArrayLike, I0: ArrayLike, R1: ArrayLike) -> ArrayLike:
        x = 3.8317 * R / R1
        return I0 * (2 * backend.bessel_j1(x) / x) ** 2

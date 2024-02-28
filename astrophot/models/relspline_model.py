from .galaxy_model_object import Galaxy_Model
from .psf_model_object import PSF_Model
from ..utils.decorators import default_internal

__all__ = [
    "RelSpline_Galaxy",
    "RelSpline_PSF",
]


# First Order
######################################################################
class RelSpline_Galaxy(Galaxy_Model):
    """Basic galaxy model with a spline radial light profile. The
    light profile is defined as a cubic spline interpolation of the
    stored brightness values:

    I(R) = interp(R, profR, I)

    where I(R) is the brightness along the semi-major axis, interp is
    a cubic spline function, R is the semi-major axis length, profR is
    a list of radii for the spline, I is a corresponding list of
    brightnesses at each profR value.

    Parameters:
        I0: Central brightness
        dI(R): Tensor of brighntess values relative to central brightness, represented as the log of the brightness divided by pixelscale squared

    """

    model_type = f"relspline {Galaxy_Model.model_type}"
    parameter_specs = {
        "I0": {"units": "log10(flux/arcsec^2)"},
        "dI(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("I0", "dI(R)")
    usable = True
    extend_profile = True

    from ._shared_methods import relspline_initialize as initialize
    from ._shared_methods import relspline_radial_model as radial_model


class RelSpline_PSF(PSF_Model):
    """point source model with a spline radial light profile. The light
    profile is defined as a cubic spline interpolation of the stored
    brightness values:

    I(R) = interp(R, profR, I)

    where I(R) is the brightness along the semi-major axis, interp is
    a cubic spline function, R is the semi-major axis length, profR is
    a list of radii for the spline, I is a corresponding list of
    brightnesses at each profR value.

    Parameters:
        I0: Central brightness
        dI(R): Tensor of brighntess values relative to central brightness, represented as the log of the brightness divided by pixelscale squared

    """

    model_type = f"relspline {PSF_Model.model_type}"
    parameter_specs = {
        "I0": {"units": "log10(flux/arcsec^2)", "value": 0.0, "locked": True},
        "dI(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = PSF_Model._parameter_order + ("I0", "dI(R)")
    usable = True
    extend_profile = True
    model_integrated = False

    @default_internal
    def transform_coordinates(self, X=None, Y=None, image=None, parameters=None):
        return X, Y

    from ._shared_methods import relspline_initialize as initialize
    from ._shared_methods import relspline_radial_model as radial_model
    from ._shared_methods import radial_evaluate_model as evaluate_model

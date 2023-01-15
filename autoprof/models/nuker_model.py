from .galaxy_model_object import Galaxy_Model
from .star_model_object import Star_Model
import torch
import numpy as np

__all__ = ["Nuker_Galaxy", "Nuker_Star"]

class Nuker_Galaxy(Galaxy_Model):
    """basic galaxy model with a Nuker profile for the radial light
    profile. The functional form of the Nuker profile is defined as:

    I(R) = Ib * 2^((beta-gamma)/alpha) * (R / Rb)^(-gamma) * (1 + (R/Rb)^alpha)^((gamma - beta)/alpha)

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ib is the flux density at
    the scale radius Rb, Rb is the scale length for the profile, beta
    is the outer power law slope, gamma is the iner power law slope,
    and alpha is the sharpness of the transition.

    Parameters:
        Ib: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
        Rb: scale length radius
        alpha: sharpness of transition between power law slopes
        beta: outer power law slope
        gamma: inner power law slope

    """
    model_type = f"nuker {Galaxy_Model.model_type}"
    parameter_specs = {
        "Rb": {"units": "arcsec", "limits": (0,None)},
        "Ib": {"units": "log10(flux/arcsec^2)"},
        "alpha": {"units": "none", "limits": (0, None)},
        "beta": {"units": "none", "limits": (0, None)},
        "gamma": {"units": "none"},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("Rb", "Ib", "alpha", "beta", "gamma")

    from ._shared_methods import nuker_radial_model as radial_model
    from ._shared_methods import nuker_initialize as initialize

class Nuker_Star(Star_Model):
    """basic star model with a Nuker profile for the radial light
    profile. The functional form of the Nuker profile is defined as:

    I(R) = Ib * 2^((beta-gamma)/alpha) * (R / Rb)^(-gamma) * (1 + (R/Rb)^alpha)^((gamma - beta)/alpha)

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ib is the flux density at
    the scale radius Rb, Rb is the scale length for the profile, beta
    is the outer power law slope, gamma is the iner power law slope,
    and alpha is the sharpness of the transition.

    Parameters:
        Ib: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
        Rb: scale length radius
        alpha: sharpness of transition between power law slopes
        beta: outer power law slope
        gamma: inner power law slope
    
    """
    model_type = f"nuker {Star_Model.model_type}"
    parameter_specs = {
        "Rb": {"units": "arcsec", "limits": (0,None)},
        "Ib": {"units": "log10(flux/arcsec^2)"},
        "alpha": {"units": "none", "limits": (0, None)},
        "beta": {"units": "none", "limits": (0, None)},
        "gamma": {"units": "none"},
    }
    _parameter_order = Star_Model._parameter_order + ("Rb", "Ib", "alpha", "beta", "gamma")

    from ._shared_methods import nuker_radial_model as radial_model
    from ._shared_methods import nuker_initialize as initialize
    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        return self.radial_model(self.radius_metric(X, Y), image)
    

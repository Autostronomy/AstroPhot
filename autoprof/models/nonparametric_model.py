from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from .star_model_object import Star_Model
from .ray_model import Ray_Galaxy
import numpy as np
import torch

__all__ = [
    "NonParametric_Galaxy", "NonParametric_Star", "NonParametric_Warp",
    "NonParametric_SuperEllipse", "NonParametric_FourierEllipse", "NonParametric_Ray",
    "NonParametric_SuperEllipse_Warp", "NonParametric_FourierEllipse_Warp"
]

# First Order
######################################################################
class NonParametric_Galaxy(Galaxy_Model):
    """basic galaxy model with a nonparametric radial light profile.

    """
    model_type = f"nonparametric {Galaxy_Model.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    parameter_order = Galaxy_Model.parameter_order + ("I(R)",)

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)    
            
    from ._shared_methods import nonparametric_set_fit_window as set_fit_window
    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

class NonParametric_Star(Star_Model):
    """star model with a nonparametric radial light profile.

    """
    model_type = f"nonparametric {Star_Model.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    parameter_order = Star_Model.parameter_order + ("I(R)",)
    # fixme don't have q anymore
    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)    
            
    from ._shared_methods import nonparametric_set_fit_window as set_fit_window
    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

class NonParametric_Warp(Warp_Galaxy):
    """warped coordinate galaxy model with a nonparametric light profile.

    """
    model_type = f"nonparametric {Warp_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    parameter_order = Warp_Galaxy.parameter_order + ("I(R)",)

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)
    
    from ._shared_methods import nonparametric_set_fit_window as set_fit_window
    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

# Second Order
######################################################################
class NonParametric_SuperEllipse(SuperEllipse_Galaxy):
    model_type = f"nonparametric {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    parameter_order = SuperEllipse_Galaxy.parameter_order + ("I(R)",)

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)
            
    from ._shared_methods import nonparametric_set_fit_window as set_fit_window
    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

class NonParametric_FourierEllipse(FourierEllipse_Galaxy):
    model_type = f"nonparametric {FourierEllipse_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    parameter_order = FourierEllipse_Galaxy.parameter_order + ("I(R)",)

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)
            
    from ._shared_methods import nonparametric_set_fit_window as set_fit_window
    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

class NonParametric_Ray(Ray_Galaxy):
    """ray galaxy model with a nonparametric light profile.

    """
    model_type = f"nonparametric {Ray_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    parameter_order = Ray_Galaxy.parameter_order + ("I(R)",)

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)
    
    from ._shared_methods import nonparametric_set_fit_window as set_fit_window
    from ._shared_methods import nonparametric_initialize as initialize # fixme specialized initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

# Third Order
######################################################################
class NonParametric_SuperEllipse_Warp(SuperEllipse_Warp):
    model_type = f"nonparametric {SuperEllipse_Warp.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    parameter_order = SuperEllipse_Warp.parameter_order + ("I(R)",)

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)
            
    from ._shared_methods import nonparametric_set_fit_window as set_fit_window
    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

class NonParametric_FourierEllipse_Warp(FourierEllipse_Warp):
    model_type = f"nonparametric {FourierEllipse_Warp.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    parameter_order = FourierEllipse_Warp.parameter_order + ("I(R)",)

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)
            
    from ._shared_methods import nonparametric_set_fit_window as set_fit_window
    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model




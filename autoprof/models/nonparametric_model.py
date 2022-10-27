from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .superellipse_model import SuperEllipse_Galaxy
from autoprof.utils.interpolate import cubic_spline_torch
import numpy as np
from scipy.stats import binned_statistic, iqr
import torch

__all__ = ["NonParametric_Galaxy", "NonParametric_Warp", "NonParametric_SuperEllipse"]

class NonParametric_Galaxy(Galaxy_Model):
    """basic galaxy model with a nonparametric radial light profile.

    """
    model_type = f"nonparametric {Galaxy_Model.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }

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

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)
    
    from ._shared_methods import nonparametric_set_fit_window as set_fit_window
    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

    
class NonParametric_SuperEllipse(SuperEllipse_Galaxy):
    model_type = f"nonparametric {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)
    
            
    from ._shared_methods import nonparametric_set_fit_window as set_fit_window
    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model
    

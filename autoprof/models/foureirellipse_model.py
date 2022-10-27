from .galaxy_model_object import Galaxy_Model
import torch
import numpy as np

__all__ = ["FourierEllipse_Galaxy"]

class FourierEllipse_Galaxy(Galaxy_Model):
    """Expanded galaxy model which includes a Fourier transformation in
    its radius metric. This allows for the expression of arbitrarily
    complex isophotes instead of pure ellipses. This is a common
    extension of the standard elliptical representation.

    """
    model_type = f"fourierellipse {Galaxy_Model.model_type}"
    parameter_specs = {
        "am": {"units": "C-2", "value": 0.},
    }
    
    def radius_metric(self, X, Y):
        R = super().radius_metric(X, Y)
        return R * 

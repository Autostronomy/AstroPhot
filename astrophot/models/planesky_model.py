import numpy as np
from scipy.stats import iqr
import torch

from .sky_model_object import Sky_Model
from ._shared_methods import select_target
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..param import Param_Unlock, Param_SoftLimits

__all__ = ["Plane_Sky"]


class Plane_Sky(Sky_Model):
    """Sky background model using a tilted plane for the sky flux. The brightness for each pixel is defined as:

    I(X, Y) = S + X*dx + Y*dy

    where I(X,Y) is the brightness as a funcion of image position X Y,
    S is the central sky brightness value, and dx dy are the slopes of
    the sky brightness plane.

    Parameters:
        sky: central sky brightness value
        delta: Tensor for slope of the sky brightness in each image dimension

    """

    model_type = f"plane {Sky_Model.model_type}"
    parameter_specs = {
        "F": {"units": "flux/arcsec^2"},
        "delta": {"units": "flux/arcsec"},
    }
    _parameter_order = Sky_Model._parameter_order + ("F", "delta")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        with Param_Unlock(parameters["F"]), Param_SoftLimits(parameters["F"]):
            if parameters["F"].value is None:
                parameters["F"].value = np.median(target[self.window].data.detach().cpu().numpy()) / target.pixel_area.item()
            if parameters["F"].uncertainty is None:
                parameters["F"].uncertainty = (
                    iqr(
                        target[self.window].data.detach().cpu().numpy(),
                        rng=(31.731 / 2, 100 - 31.731 / 2),
                    )
                    / (2.0)
                ) / np.sqrt(np.prod(self.window.shape.detach().cpu().numpy()))
        with Param_Unlock(parameters["delta"]), Param_SoftLimits(parameters["delta"]):
            if parameters["delta"].value is None:
                parameters["delta"].value = [0.0, 0.0]
                parameters["delta"].uncertainty = [self.default_uncertainty, self.default_uncertainty]

    @default_internal
    def evaluate_model(self, X=None, Y=None, image=None, parameters=None, **kwargs):
        if X is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]
        return (
            image.pixel_area * parameters["F"].value
            + X * parameters["delta"].value[0]
            + Y * parameters["delta"].value[1]
        )

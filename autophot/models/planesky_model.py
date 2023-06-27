import numpy as np
from scipy.stats import iqr
import torch

from .sky_model_object import Sky_Model
from ._shared_methods import select_target
from ..utils.decorators import ignore_numpy_warnings, default_internal

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
        "sky": {"units": "flux/arcsec^2"},
        "delta": {"units": "flux/arcsec"},
    }
    _parameter_order = Sky_Model._parameter_order + ("sky", "delta")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        if parameters["sky"].value is None:
            parameters["sky"].set_value(
                np.median(target[self.window].data.detach().cpu().numpy())
                / target.pixel_area.item(),
                override_locked=True,
            )
        if parameters["sky"].uncertainty is None:
            parameters["sky"].set_uncertainty(
                (
                    iqr(
                        target[self.window].data.detach().cpu().numpy(),
                        rng=(31.731 / 2, 100 - 31.731 / 2),
                    )
                    / (2.0)
                )
                / np.sqrt(np.prod(self.window.shape.detach().cpu().numpy())),
                override_locked=True,
            )
        if parameters["delta"].value is None:
            parameters["delta"].set_value([0.0, 0.0], override_locked=True)
            parameters["delta"].set_uncertainty([0.1, 0.1], override_locked=True)

    @default_internal
    def evaluate_model(self, X=None, Y=None, image=None, parameters=None, **kwargs):
        if X is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]
        return (
            image.pixel_area * parameters["sky"].value
            + X * parameters["delta"].value[0]
            + Y * parameters["delta"].value[1]
        )

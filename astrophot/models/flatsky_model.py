import numpy as np
from scipy.stats import iqr
import torch

from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..param import Param_Unlock, Param_SoftLimits
from .sky_model_object import Sky_Model
from ._shared_methods import select_target

__all__ = ["Flat_Sky"]


class Flat_Sky(Sky_Model):
    """Model for the sky background in which all values across the image
    are the same.

    Parameters:
        sky: brightness for the sky, represented as the log of the brightness over pixel scale squared, this is proportional to a surface brightness

    """

    model_type = f"flat {Sky_Model.model_type}"
    parameter_specs = {
        "F": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Sky_Model._parameter_order + ("F",)
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        with Param_Unlock(parameters["F"]), Param_SoftLimits(parameters["F"]):
            if parameters["F"].value is None:
                parameters["F"].value = torch.log10(torch.median(target[self.window].data) / target.pixel_area)
            if parameters["F"].uncertainty is None:
                parameters["F"].uncertainty = (
                    (
                        iqr(
                            target[self.window].data.detach().cpu().numpy(),
                            rng=(31.731 / 2, 100 - 31.731 / 2),
                        )
                        / (2.0 * target.pixel_area.item())
                    )
                    / np.sqrt(np.prod(self.window.shape.detach().cpu().numpy()))
                ) / (10 ** parameters["F"].value.item() * np.log(10))

    def evaluate_model(self, X=None, Y=None, image=None, parameters=None, **kwargs):
        ref = image.data if X is None else X
        return torch.ones_like(ref) * (image.pixel_area * 10 ** parameters["F"].value)

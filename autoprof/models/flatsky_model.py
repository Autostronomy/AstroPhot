import numpy as np
from scipy.stats import iqr
import torch

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
        "sky": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Sky_Model._parameter_order + ("sky",)
    useable = True

    @torch.no_grad()
    @select_target
    def initialize(self, target=None):
        super().initialize(target)

        if self["sky"].value is None:
            self["sky"].set_representation(
                np.log10(
                    torch.median(target[self.window].data) / target.pixelscale ** 2
                ),
                override_locked=True,
            )
        if self["sky"].uncertainty is None:
            self["sky"].set_uncertainty(
                (
                    (
                        iqr(
                            target[self.window].data.detach().cpu().numpy(),
                            rng=(31.731 / 2, 100 - 31.731 / 2),
                        )
                        / (2.0 * target.pixelscale.item() ** 2)
                    )
                    / np.sqrt(np.prod(self.window.shape.detach().cpu().numpy()))
                )
                / (10 ** self["sky"].value * np.log(10)),
                override_locked=True,
            )

    def evaluate_model(self, image, X = None, Y = None, **kwargs):
        if X is None:
            ref = image.data
        else:
            ref = X
        return torch.ones_like(ref) * (
            (10 ** self["sky"].value) * image.pixelscale ** 2
        )
            

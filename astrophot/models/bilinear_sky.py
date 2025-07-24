import numpy as np
import torch

from .sky_model_object import SkyModel
from ..utils.decorators import ignore_numpy_warnings
from ..utils.interpolate import interp2d
from ..param import forward
from . import func
from ..utils.initialize import polar_decomposition

__all__ = ["BilinearSky"]


class BilinearSky(SkyModel):
    """Sky background model using a coarse bilinear grid for the sky flux.

    Parameters:
        I: sky brightness grid

    """

    _model_type = "bilinear"
    _parameter_specs = {
        "I": {"units": "flux/arcsec^2"},
        "PA": {"units": "radians", "shape": ()},
        "scale": {"units": "arcsec/grid-unit", "shape": ()},
    }
    sampling_mode = "midpoint"
    usable = True

    def __init__(self, *args, nodes=(3, 3), **kwargs):
        """Initialize the BilinearSky model with a grid of nodes."""
        super().__init__(*args, **kwargs)
        self.nodes = nodes

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if self.I.initialized:
            self.nodes = tuple(self.I.value.shape)

        if not self.PA.initialized:
            R, _ = polar_decomposition(self.target.CD.value.detach().cpu().numpy())
            self.PA.value = np.arccos(np.abs(R[0, 0]))
        if not self.scale.initialized:
            self.scale.value = (
                self.target.pixelscale.item() * self.target.data.shape[0] / self.nodes[0]
            )

        if self.I.initialized:
            return

        target_dat = self.target[self.window]
        dat = target_dat.data.detach().cpu().numpy().copy()
        if self.target.has_mask:
            mask = target_dat.mask.detach().cpu().numpy().copy()
            dat[mask] = np.nanmedian(dat)
        iS = dat.shape[0] // self.nodes[0]
        jS = dat.shape[1] // self.nodes[1]

        self.I.dynamic_value = (
            np.median(
                dat[: iS * self.nodes[0], : jS * self.nodes[1]].reshape(
                    iS, self.nodes[0], jS, self.nodes[1]
                ),
                axis=(0, 2),
            )
            / self.target.pixel_area.item()
        )

    @forward
    def transform_coordinates(self, x, y, I, PA, scale):
        x, y = super().transform_coordinates(x, y)
        i, j = func.rotate(-PA, x, y)
        pixel_center = (I.shape[0] - 1) / 2, (I.shape[1] - 1) / 2
        return i / scale + pixel_center[0], j / scale + pixel_center[1]

    @forward
    def brightness(self, x, y, I):
        x, y = self.transform_coordinates(x, y)
        return interp2d(I, y, x)

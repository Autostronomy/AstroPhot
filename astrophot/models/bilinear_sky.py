import numpy as np
import torch

from .sky_model_object import SkyModel
from ..utils.decorators import ignore_numpy_warnings
from ..utils.interpolate import interp2d
from ..param import forward
from .. import AP_config

__all__ = ["BilinearSky"]


class BilinearSky(SkyModel):
    """Sky background model using a coarse bilinear grid for the sky flux.

    Parameters:
        I: sky brightness grid

    """

    _model_type = "bilinear"
    _parameter_specs = {
        "I": {"units": "flux/arcsec^2"},
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
            self.update_transform()
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
        self.update_transform()

    def update_transform(self):
        target_dat = self.target[self.window]
        P = torch.stack(list(torch.stack(c) for c in target_dat.corners()))
        centroid = P.mean(dim=0)
        dP = P - centroid
        evec = torch.linalg.eig(dP.T @ dP / 4)[1].real.to(
            dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        if torch.dot(evec[0], P[3] - P[0]).abs() < torch.dot(evec[1], P[3] - P[0]).abs():
            evec = evec.flip(0)
        evec[0] = evec[0] * self.nodes[0] / torch.linalg.norm(P[3] - P[0])
        evec[1] = evec[1] * self.nodes[1] / torch.linalg.norm(P[1] - P[0])
        self.evec = evec
        self.shift = torch.tensor(
            [(self.nodes[0] - 1) / 2, (self.nodes[1] - 1) / 2],
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )

    @forward
    def transform_coordinates(self, x, y):
        x, y = super().transform_coordinates(x, y)
        xy = torch.stack((x, y), dim=-1)
        xy = xy @ self.evec
        xy = xy + self.shift
        return xy[..., 0], xy[..., 1]

    @forward
    def brightness(self, x, y, I):
        x, y = self.transform_coordinates(x, y)
        return interp2d(I, x, y)

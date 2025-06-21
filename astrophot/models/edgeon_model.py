import torch
import numpy as np

from .model_object import ComponentModel
from ..utils.decorators import ignore_numpy_warnings
from . import func

__all__ = ["EdgeonModel", "EdgeonSech", "EdgeonIsothermal"]


class EdgeonModel(ComponentModel):
    """General Edge-On galaxy model to be subclassed for any specific
    representation such as radial light profile or the structure of
    the galaxy on the sky. Defines an edgeon galaxy as an object with
    a position angle, no inclination information is included.

    """

    _model_type = "edgeon"
    _parameter_specs = {
        "PA": {
            "units": "radians",
            "limits": (0, np.pi),
            "cyclic": True,
            "uncertainty": 0.06,
        },
    }
    usable = False

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        if self.PA.value is not None:
            return
        target_area = self.target[self.window]
        dat = target_area.data.npvalue
        edge = np.concatenate((dat[:, 0], dat[:, -1], dat[0, :], dat[-1, :]))
        edge_average = np.median(edge)
        dat = dat - edge_average

        x, y = target_area.coordinate_center_meshgrid()
        x = (x - self.center.value[0]).detach().cpu().numpy()
        y = (y - self.center.value[1]).detach().cpu().numpy()
        mu20 = np.median(dat * np.abs(x))
        mu02 = np.median(dat * np.abs(y))
        mu11 = np.median(dat * x * y / np.sqrt(np.abs(x * y)))
        M = np.array([[mu20, mu11], [mu11, mu02]])
        if np.any(np.iscomplex(M)) or np.any(~np.isfinite(M)):
            self.PA.dynamic_value = np.pi / 2
        else:
            self.PA.dynamic_value = (0.5 * np.arctan2(2 * mu11, mu20 - mu02) - np.pi / 2) % np.pi
        self.PA.uncertainty = self.PA.value * self.default_uncertainty

    def transform_coordinates(self, x, y, PA):
        x, y = super().transform_coordinates(x, y)
        return func.rotate(PA - np.pi / 2, x, y)


class EdgeonSech(EdgeonModel):
    """An edgeon profile where the vertical distribution is a sech^2
    profile, subclasses define the radial profile.

    """

    _model_type = "sech2"
    _parameter_specs = {
        "I0": {"units": "flux/arcsec^2"},
        "hs": {"units": "arcsec", "valid": (0, None)},
    }
    usable = False

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        if (self.I0.value is not None) and (self.hs.value is not None):
            return
        target_area = self.target[self.window]
        icenter = target_area.plane_to_pixel(*self.center.value)

        if self.I0.value is None:
            chunk = target_area.data.value[
                int(icenter[0]) - 2 : int(icenter[0]) + 2,
                int(icenter[1]) - 2 : int(icenter[1]) + 2,
            ]
            self.I0.dynamic_value = torch.mean(chunk) / self.target.pixel_area
            self.I0.uncertainty = torch.std(chunk) / self.target.pixel_area
        if self.hs.value is None:
            self.hs.value = torch.max(self.window.shape) * target_area.pixel_length * 0.1
            self.hs.uncertainty = self.hs.value / 2

    def brightness(self, x, y, I0, hs):
        x, y = self.transform_coordinates(x, y)
        return I0 * self.radial_model(x) / (torch.cosh((y + self.softening) / hs) ** 2)


class EdgeonIsothermal(EdgeonSech):
    """A self-gravitating locally-isothermal edgeon disk. This comes from
    van der Kruit & Searle 1981.

    """

    _model_type = "isothermal"
    _parameter_specs = {"rs": {"units": "arcsec", "valid": (0, None)}}
    usable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        if self.rs.value is not None:
            return
        self.rs.value = torch.max(self.window.shape) * self.target.pixel_length * 0.4
        self.rs.uncertainty = self.rs.value / 2

    def radial_model(self, R, rs):
        Rscaled = torch.abs(R / rs)
        return (
            Rscaled
            * torch.exp(-Rscaled)
            * torch.special.scaled_modified_bessel_k1(Rscaled + self.softening / rs)
        )

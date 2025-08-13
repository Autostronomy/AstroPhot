from typing import Tuple
import torch
import numpy as np

from .model_object import ComponentModel
from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from . import func
from ..backend_obj import backend, ArrayLike
from ..param import forward

__all__ = ["EdgeonModel", "EdgeonSech", "EdgeonIsothermal"]


class EdgeonModel(ComponentModel):
    """General Edge-On galaxy model to be subclassed for any specific
    representation such as radial light profile or the structure of
    the galaxy on the sky. Defines an edgeon galaxy as an object with
    a position angle, no inclination information is included.

    **Parameters:**
    -    `PA`: Position angle of the edgeon disk in radians.

    """

    _model_type = "edgeon"
    _parameter_specs = {
        "PA": {"units": "radians", "valid": (0, np.pi), "cyclic": True, "shape": ()},
    }
    usable = False

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        if self.PA.initialized:
            return
        target_area = self.target[self.window]
        dat = backend.to_numpy(target_area.data).copy()
        edge = np.concatenate((dat[:, 0], dat[:, -1], dat[0, :], dat[-1, :]))
        edge_average = np.median(edge)
        dat = dat - edge_average

        x, y = target_area.coordinate_center_meshgrid()
        x = backend.to_numpy(x - self.center.value[0])
        y = backend.to_numpy(y - self.center.value[1])
        mu20 = np.median(dat * np.abs(x))
        mu02 = np.median(dat * np.abs(y))
        mu11 = np.median(dat * x * y / np.sqrt(np.abs(x * y)))
        M = np.array([[mu20, mu11], [mu11, mu02]])
        if np.any(np.iscomplex(M)) or np.any(~np.isfinite(M)):
            self.PA.dynamic_value = np.pi / 2
        else:
            self.PA.dynamic_value = (0.5 * np.arctan2(2 * mu11, mu20 - mu02)) % np.pi

    @forward
    def transform_coordinates(
        self, x: ArrayLike, y: ArrayLike, PA: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        x, y = super().transform_coordinates(x, y)
        return func.rotate(-(PA + np.pi / 2), x, y)


class EdgeonSech(EdgeonModel):
    """An edgeon profile where the vertical distribution is a sech^2
    profile, subclasses define the radial profile.

    **Parameters:**
    -    `I0`: The central intensity of the sech^2 profile in flux/arcsec^2.
    -    `hs`: The scale height of the sech^2 profile in arcseconds.
    """

    _model_type = "sech2"
    _parameter_specs = {
        "I0": {"units": "flux/arcsec^2", "shape": ()},
        "hs": {"units": "arcsec", "valid": (0, None), "shape": ()},
    }
    usable = False

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        if self.I0.initialized and self.hs.initialized:
            return
        target_area = self.target[self.window]
        icenter = target_area.plane_to_pixel(*self.center.value)

        if not self.I0.initialized:
            chunk = target_area.data[
                int(icenter[0]) - 2 : int(icenter[0]) + 2,
                int(icenter[1]) - 2 : int(icenter[1]) + 2,
            ]
            self.I0.dynamic_value = backend.mean(chunk) / self.target.pixel_area
        if not self.hs.initialized:
            self.hs.value = max(self.window.shape) * target_area.pixelscale * 0.1

    @forward
    def brightness(self, x: ArrayLike, y: ArrayLike, I0: ArrayLike, hs: ArrayLike) -> ArrayLike:
        x, y = self.transform_coordinates(x, y)
        return I0 * self.radial_model(x) / (backend.cosh((y + self.softening) / hs) ** 2)


@combine_docstrings
class EdgeonIsothermal(EdgeonSech):
    """A self-gravitating locally-isothermal edgeon disk. This comes from
    van der Kruit & Searle 1981.

    **Parameters:**
    -    `rs`: Scale radius of the isothermal disk in arcseconds.
    """

    _model_type = "isothermal"
    _parameter_specs = {"rs": {"units": "arcsec", "valid": (0, None), "shape": ()}}
    usable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        if self.rs.initialized:
            return
        self.rs.value = max(self.window.shape) * self.target.pixelscale * 0.4

    @forward
    def radial_model(self, R: ArrayLike, rs: ArrayLike) -> ArrayLike:
        Rscaled = backend.abs(R / rs)
        return Rscaled * backend.exp(-Rscaled) * backend.bessel_k1(Rscaled + self.softening / rs)

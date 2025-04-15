from typing import Optional

import torch
import numpy as np
from scipy.stats import iqr
from caskade import Param, forward

from ..utils.initialize import isophotes
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.angle_operations import Angle_COM_PA
from ..utils.conversions.coordinates import (
    Rotate_Cartesian,
)
from .model_object import Component_Model
from ._shared_methods import select_target


__all__ = ["Galaxy_Model"]


class Galaxy_Model(Component_Model):
    """General galaxy model to be subclassed for any specific
    representation. Defines a galaxy as an object with a position
    angle and axis ratio, or effectively a tilted disk. Most
    subclassing models should simply define a radial model or update
    to the coordinate transform. The definition of the position angle and axis ratio used here is simply a scaling along the minor axis. The transformation can be written as:

    X, Y = meshgrid(image)
    X', Y' = Rot(theta, X, Y)
    Y'' = Y' / q

    where X Y are the coordinates of an image, X' Y' are the rotated
    coordinates, Rot is a rotation matrix by angle theta applied to the
    initial X Y coordinates, Y'' is the scaled semi-minor axis, and q
    is the axis ratio.

    Parameters:
        q: axis ratio to scale minor axis from the ratio of the minor/major axis b/a, this parameter is unitless, it is restricted to the range (0,1)
        PA: position angle of the smei-major axis relative to the image positive x-axis in radians, it is a cyclic parameter in the range [0,pi)

    """

    model_type = f"galaxy {Component_Model.model_type}"
    parameter_specs = {
        "q": {"units": "b/a", "limits": (0, 1), "uncertainty": 0.03},
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
    @select_target
    @default_internal
    def initialize(self, target=None, **kwargs):
        super().initialize(target=target)

        if not (self.PA.value is None or self.q.value is None):
            return
        target_area = target[self.window]
        target_dat = target_area.data.detach().cpu().numpy()
        if target_area.has_mask:
            mask = target_area.mask.detach().cpu().numpy()
            target_dat[mask] = np.median(target_dat[np.logical_not(mask)])
        edge = np.concatenate(
            (
                target_dat[:, 0],
                target_dat[:, -1],
                target_dat[0, :],
                target_dat[-1, :],
            )
        )
        edge_average = np.nanmedian(edge)
        edge_scatter = iqr(edge[np.isfinite(edge)], rng=(16, 84)) / 2
        icenter = target_area.plane_to_pixel(self.center.value)

        if self.PA.value is None:
            weights = target_dat - edge_average
            Coords = target_area.get_coordinate_meshgrid()
            X, Y = Coords - self.center.value[..., None, None]
            X, Y = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
            if target_area.has_mask:
                seg = np.logical_not(target_area.mask.detach().cpu().numpy())
                PA = Angle_COM_PA(weights[seg], X[seg], Y[seg])
            else:
                PA = Angle_COM_PA(weights, X, Y)

            self.PA.value = (PA + target_area.north) % np.pi
            if self.PA.uncertainty is None:
                self.PA.uncertainty = (5 * np.pi / 180) * torch.ones_like(
                    self.PA.value
                )  # default uncertainty of 5 degrees is assumed
        if self.q.value is None:
            q_samples = np.linspace(0.2, 0.9, 15)
            iso_info = isophotes(
                target_area.data.detach().cpu().numpy() - edge_average,
                (icenter[1].detach().cpu().item(), icenter[0].detach().cpu().item()),
                threshold=3 * edge_scatter,
                pa=(self.PA.value - target.north).detach().cpu().item(),
                q=q_samples,
            )
            self.q.value = q_samples[np.argmin(list(iso["amplitude2"] for iso in iso_info))]
            if self.q.uncertainty is None:
                self.q.uncertainty = self.q.value * self.default_uncertainty

    from ._shared_methods import inclined_transform_coordinates as transform_coordinates
    from ._shared_methods import transformed_evaluate_model as evaluate_model

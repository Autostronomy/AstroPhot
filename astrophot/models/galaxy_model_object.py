from typing import Optional

import torch
import numpy as np
from scipy.stats import iqr

from ..utils.initialize import isophotes
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.angle_operations import Angle_Average
from ..utils.conversions.coordinates import (
    Rotate_Cartesian,
    Axis_Ratio_Cartesian,
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
    _parameter_order = Component_Model._parameter_order + ("q", "PA")
    useable = False

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(
        self, target=None, parameters: Optional["Parameter_Group"] = None, **kwargs
    ):
        super().initialize(target=target, parameters=parameters)

        if not (parameters["PA"].value is None or parameters["q"].value is None):
            return
        target_area = target[self.window]
        target_area_data = target_area.data.detach().cpu().numpy()
        edge = np.concatenate(
            (
                target_area_data[:, 0],
                target_area_data[:, -1],
                target_area_data[0, :],
                target_area_data[-1, :],
            )
        )
        edge_average = np.nanmedian(edge)
        edge_scatter = iqr(edge[np.isfinite(edge)], rng=(16, 84)) / 2
        icenter = target_area.world_to_pixel(parameters["center"].value)

        if parameters["PA"].value is None:
            iso_info = isophotes(
                target_area.data.detach().cpu().numpy() - edge_average,
                (icenter[1].detach().cpu().item(), icenter[0].detach().cpu().item()),
                threshold=3 * edge_scatter,
                pa=0.0,
                q=1.0,
                n_isophotes=15,
            )
            parameters["PA"].set_value(
                (
                    -(
                        (
                            Angle_Average(
                                list(
                                    iso["phase2"]
                                    for iso in iso_info[-int(len(iso_info) / 3) :]
                                )
                            )
                            / 2
                        )
                        + target.north
                    )
                )
                % np.pi,
                override_locked=True,
            )
        if parameters["q"].value is None:
            q_samples = np.linspace(0.1, 0.9, 15)
            iso_info = isophotes(
                target_area.data.detach().cpu().numpy() - edge_average,
                (icenter[1].detach().cpu().item(), icenter[0].detach().cpu().item()),
                threshold=3 * edge_scatter,
                pa=(parameters["PA"].value - target.north).detach().cpu().item(),
                q=q_samples,
            )
            parameters["q"].set_value(
                q_samples[np.argmin(list(iso["amplitude2"] for iso in iso_info))],
                override_locked=True,
            )

    @default_internal
    def transform_coordinates(self, X, Y, image=None, parameters=None):
        X, Y = Rotate_Cartesian(-(parameters["PA"].value - image.north), X, Y)
        return (
            X,
            Y / parameters["q"].value,
        )

    @default_internal
    def evaluate_model(
        self, X=None, Y=None, image=None, parameters: "Parameter_Group" = None, **kwargs
    ):
        if X is None or Y is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]
        XX, YY = self.transform_coordinates(X, Y, image, parameters)
        return self.radial_model(
            self.radius_metric(XX, YY, image, parameters),
            image=image,
            parameters=parameters,
        )

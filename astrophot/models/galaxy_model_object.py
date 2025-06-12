from typing import Optional

import torch
import numpy as np
from scipy.stats import iqr
from caskade import Param, forward

from . import func
from ..utils.initialize import isophotes
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.angle_operations import Angle_COM_PA
from ..utils.conversions.coordinates import (
    Rotate_Cartesian,
)
from .model_object import Component_Model
from ._shared_methods import select_target
from .mixins import InclinedMixin


__all__ = ["Galaxy_Model"]


class Galaxy_Model(InclinedMixin, Component_Model):
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

    _model_type = "galaxy"
    usable = False

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self, **kwargs):
        super().initialize()

        if not (self.PA.value is None or self.q.value is None):
            return
        target_area = self.target[self.window]
        target_dat = target_area.data.npvalue
        if target_area.has_mask:
            mask = target_area.mask.detach().cpu().numpy()
            target_dat[mask] = np.median(target_dat[~mask])
        edge = np.concatenate(
            (
                target_dat[:, 0],
                target_dat[:, -1],
                target_dat[0, :],
                target_dat[-1, :],
            )
        )
        edge_average = np.nanmedian(edge)
        target_dat -= edge_average
        icenter = target_area.plane_to_pixel(self.center.value)

        i, j = func.pixel_center_meshgrid(
            target_area.shape, dtype=target_area.data.dtype, device=target_area.data.device
        )
        i, j = (i - icenter[0]).detach().cpu().item(), (j - icenter[1]).detach().cpu().item()
        mu20 = np.sum(target_dat * i**2)
        mu02 = np.sum(target_dat * j**2)
        mu11 = np.sum(target_dat * i * j)
        M = np.array([[mu20, mu11], [mu11, mu02]])
        if self.PA.value is None:
            self.PA.value = (0.5 * np.arctan2(2 * mu11, mu20 - mu02)) % np.pi
        if self.q.value is None:
            l = np.sorted(np.linalg.eigvals(M))
            self.q.value = np.sqrt(l[1] / l[0])

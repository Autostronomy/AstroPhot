import numpy as np
import torch

from ...utils.decorators import ignore_numpy_warnings
from ...param import forward


def rotate(theta, x, y):
    """
    Applies a rotation matrix to the X,Y coordinates
    """
    s = theta.sin()
    c = theta.cos()
    return c * x - s * y, s * x + c * y


class InclinedMixin:

    _parameter_specs = {
        "q": {"units": "b/a", "valid": (0, 1), "uncertainty": 0.03, "shape": ()},
        "PA": {
            "units": "radians",
            "valid": (0, np.pi),
            "cyclic": True,
            "uncertainty": 0.06,
            "shape": (),
        },
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
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
        x, y = target_area.coordinate_center_meshgrid()
        x = (x - self.center.value[0]).detach().cpu().numpy()
        y = (y - self.center.value[1]).detach().cpu().numpy()
        mu20 = np.median(target_dat * np.abs(x))
        mu02 = np.median(target_dat * np.abs(y))
        mu11 = np.median(target_dat * x * y / np.sqrt(np.abs(x * y)))
        # mu20 = np.median(target_dat * x**2)
        # mu02 = np.median(target_dat * y**2)
        # mu11 = np.median(target_dat * x * y)
        M = np.array([[mu20, mu11], [mu11, mu02]])
        if self.PA.value is None:
            if np.any(np.iscomplex(M)) or np.any(~np.isfinite(M)):
                self.PA.dynamic_value = np.pi / 2
            else:
                self.PA.dynamic_value = (
                    0.5 * np.arctan2(2 * mu11, mu20 - mu02) - np.pi / 2
                ) % np.pi
        if self.q.value is None:
            l = np.sort(np.linalg.eigvals(M))
            if np.any(np.iscomplex(l)) or np.any(~np.isfinite(l)):
                l = (0.7, 1.0)
            self.q.dynamic_value = np.clip(np.sqrt(l[0] / l[1]), 0.1, 0.9)

    @forward
    def transform_coordinates(self, x, y, PA, q):
        """
        Transform coordinates based on the position angle and axis ratio.
        """
        x, y = super().transform_coordinates(x, y)
        x, y = rotate(-(PA + np.pi / 2), x, y)
        return x, y / q

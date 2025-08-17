import torch
import numpy as np
from torch import Tensor

from .model_object import ComponentModel
from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from . import func
from ..param import forward
from ..backend_obj import backend, ArrayLike

__all__ = ["GaussianEllipsoid"]


@combine_docstrings
class GaussianEllipsoid(ComponentModel):
    """Model that represents a galaxy as a 3D Gaussian ellipsoid.

    The model is triaxial, meaning it has three different standard deviations
    along the three axes. The orientation of the ellipsoid is defined by Euler
    angles.

    If all three Euler angles are set to zero, the ellipsoid is aligned with the
    image axes meaning sigma_a gives the std along the x axis of the tangent
    plane, sigma_b gives the std along the y axis of the tangent plane, and
    sigma_z gives the std into the tangent plane. We use the ZXZ convention for
    the Euler angles. This means that for a disk galaxy, one can naturally
    consider sigma_c as the disk thickness and sigma_a=sigma_b as the disk
    radius; setting the Euler angles to zero would leave the disk face-on in the
    x-y tangent plane.

    Note:
        the model is highly degenerate, meaning that it is not possible to
        uniquely determine the parameters from the data. The model is useful if
        one already has a 3D model of the galaxy in mind and wants to produce
        mock data. Alternately, if one applies some constraints on the
        parameters, such as sigma_a = sigma_b and alpha=0, then the model will
        be better determined. In that case, beta is related to the inclination
        of the disk and gamma is related to the position angle of the disk. The
        initialization for this model assumes exactly this interpretation with a
        disk thickness of sigma_c = 0.2 *sigma_a.

    **Parameters:**
    -    `sigma_a`: Standard deviation of the Gaussian along the alpha axis in arcseconds.
    -    `sigma_b`: Standard deviation of the Gaussian along the beta axis in arcseconds.
    -    `sigma_c`: Standard deviation of the Gaussian along the gamma axis in arcseconds.
    -    `alpha`: Euler angle representing the rotation around the alpha axis in radians.
    -    `beta`: Euler angle representing the rotation around the beta axis in radians.
    -    `gamma`: Euler angle representing the rotation around the gamma axis in radians.
    -    `flux`: Total flux of the galaxy in arbitrary units.

    """

    _model_type = "gaussianellipsoid"
    _parameter_specs = {
        "sigma_a": {"units": "arcsec", "valid": (0, None), "shape": ()},
        "sigma_b": {"units": "arcsec", "valid": (0, None), "shape": ()},
        "sigma_c": {"units": "arcsec", "valid": (0, None), "shape": ()},
        "alpha": {"units": "radians", "valid": (0, 2 * np.pi), "cyclic": True, "shape": ()},
        "beta": {"units": "radians", "valid": (0, 2 * np.pi), "cyclic": True, "shape": ()},
        "gamma": {"units": "radians", "valid": (0, 2 * np.pi), "cyclic": True, "shape": ()},
        "flux": {"units": "flux", "shape": ()},
    }
    usable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if any(self[key].initialized for key in GaussianEllipsoid._parameter_specs):
            return

        self.sigma_b = self.sigma_a
        self.sigma_c = lambda p: 0.2 * p.sigma_a.value
        self.sigma_c.link(self.sigma_a)
        self.alpha = 0.0

        target_area = self.target[self.window]
        dat = backend.to_numpy(target_area.data).copy()
        if target_area.has_mask:
            mask = backend.to_numpy(target_area.mask).copy()
            dat[mask] = np.median(dat[~mask])
        edge = np.concatenate((dat[:, 0], dat[:, -1], dat[0, :], dat[-1, :]))
        edge_average = np.nanmedian(edge)
        dat -= edge_average
        x, y = target_area.coordinate_center_meshgrid()
        center = self.center.value
        x = x - center[0]
        y = y - center[1]
        r = backend.to_numpy(self.radius_metric(x, y, params=()))
        self.sigma_a.dynamic_value = np.sqrt(np.sum((r * dat) ** 2) / np.sum(r**2))

        x = backend.to_numpy(x)
        y = backend.to_numpy(y)

        mu20 = np.median(dat * np.abs(x))
        mu02 = np.median(dat * np.abs(y))
        mu11 = np.median(dat * x * y / np.sqrt(np.abs(x * y) + self.softening**2))
        M = np.array([[mu20, mu11], [mu11, mu02]])
        if np.any(np.iscomplex(M)) or np.any(~np.isfinite(M)):
            PA = np.pi / 2
            l = (0.7, 1.0)
        else:
            PA = (0.5 * np.arctan2(2 * mu11, mu20 - mu02) - np.pi / 2) % np.pi
            l = np.sort(np.linalg.eigvals(M))
        q = np.clip(np.sqrt(l[0] / l[1]), 0.1, 0.9)
        self.beta.dynamic_value = np.arccos(q)
        self.gamma.dynamic_value = PA
        self.flux.dynamic_value = np.sum(dat)

    @forward
    def brightness(
        self,
        x: ArrayLike,
        y: ArrayLike,
        sigma_a: ArrayLike,
        sigma_b: ArrayLike,
        sigma_c: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        gamma: ArrayLike,
        flux: ArrayLike,
    ) -> ArrayLike:
        """Brightness of the Gaussian ellipsoid."""
        D = backend.diag(backend.stack((sigma_a, sigma_b, sigma_c)) ** 2)
        R = func.euler_rotation_matrix(alpha, beta, gamma)
        Sigma = R @ D @ R.T
        Sigma2D = Sigma[:2, :2]
        inv_Sigma = backend.linalg.inv(Sigma2D)
        v = backend.stack(self.transform_coordinates(x, y), dim=0).reshape(2, -1)
        return (
            flux
            * backend.sum(backend.exp(-0.5 * (v * (inv_Sigma @ v))), dim=0)
            / (2 * np.pi * backend.sqrt(backend.linalg.det(Sigma2D)))
        ).reshape(x.shape)

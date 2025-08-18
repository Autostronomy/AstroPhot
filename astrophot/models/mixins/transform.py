from typing import Tuple
import numpy as np
import torch

from ...utils.decorators import ignore_numpy_warnings
from ...utils.interpolate import default_prof
from ...backend_obj import backend, ArrayLike
from ...param import forward
from .. import func
from ... import config


class InclinedMixin:
    """A model which defines a position angle and axis ratio.

    PA and q operate on the coordinates to transform the model. Given some x,y
    the updated values are:

    $$x', y' = {\\rm rotate}(-PA + \\pi/2, x, y)$$
    $$y'' = y' / q$$

    where x' and y'' are the final transformed coordinates. The $\pi/2$ is included
    such that the position angle is defined with 0 at north. The -PA is such
    that the position angle increases to the East. Thus, the position angle is a
    standard East of North definition assuming the WCS of the image is correct.

    Note that this means radii are defined with $R = \\sqrt{x^2 +
    \\left(\\frac{y}{q}\\right)^2}$ rather than the common alternative which is $R =
    \\sqrt{qx^2 + \\frac{y^2}{q}}$

    **Parameters:**
    -    `q`: Axis ratio of the model, defined as the ratio of the
        semi-minor axis to the semi-major axis. A value of 1.0 is
        circular.
    -    `PA`: Position angle of the model, defined as the angle
        between the semi-major axis and North, measured East of North.
        A value of 0.0 is North, a value of pi/2 is East.
    """

    _parameter_specs = {
        "q": {"units": "b/a", "valid": (0.01, 1), "shape": ()},
        "PA": {"units": "radians", "valid": (0, np.pi), "cyclic": True, "shape": ()},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if self.PA.initialized and self.q.initialized:
            return
        target_area = self.target[self.window]
        dat = backend.to_numpy(backend.copy(target_area.data))
        if target_area.has_mask:
            mask = backend.to_numpy(backend.copy(target_area.mask))
            dat[mask] = np.median(dat[~mask])
        edge = np.concatenate((dat[:, 0], dat[:, -1], dat[0, :], dat[-1, :]))
        edge_average = np.nanmedian(edge)
        dat -= edge_average
        x, y = target_area.coordinate_center_meshgrid()
        x = backend.to_numpy(x - self.center.value[0])
        y = backend.to_numpy(y - self.center.value[1])
        mu20 = np.mean(dat * np.abs(x))
        mu02 = np.mean(dat * np.abs(y))
        mu11 = np.mean(dat * x * y / np.sqrt(np.abs(x * y) + self.softening**2))
        M = np.array([[mu20, mu11], [mu11, mu02]])
        if not self.PA.initialized:
            if np.any(np.iscomplex(M)) or np.any(~np.isfinite(M)):
                self.PA.dynamic_value = np.pi / 2
            else:
                self.PA.dynamic_value = (
                    0.5 * np.arctan2(2 * mu11, mu20 - mu02) - np.pi / 2
                ) % np.pi
        if not self.q.initialized:
            if np.any(np.iscomplex(M)) or np.any(~np.isfinite(M)):
                l = (0.7, 1.0)
            else:
                l = np.sort(np.linalg.eigvals(M))
            self.q.dynamic_value = np.clip(np.sqrt(np.abs(l[0] / l[1])), 0.1, 0.9)

    @forward
    def transform_coordinates(
        self, x: ArrayLike, y: ArrayLike, PA: ArrayLike, q: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        x, y = super().transform_coordinates(x, y)
        x, y = func.rotate(-PA + np.pi / 2, x, y)
        return x, y / q


class SuperEllipseMixin:
    """Generalizes the definition of radius and so modifies the evaluation of radial models.

    A superellipse transformation allows for the expression of "boxy" and
    "disky" modifications to traditional elliptical isophotes. This is a common
    extension of the standard elliptical representation, especially for
    early-type galaxies. The functional form for this is:

    $$R = (|x|^C + |y|^C)^{1/C}$$

    where $R$ is the new distance metric, $X$ and $Y$ are the coordinates, and $C$ is the
    coefficient for the superellipse. $C$ can take on any value greater than zero
    where $C = 2$ is the standard distance metric, $0 < C < 2$ creates disky or
    pointed perturbations to an ellipse, and $C > 2$ transforms an ellipse to be
    more boxy.

    **Parameters:**
    -   `C`: Superellipse distance metric parameter, controls the shape of the isophotes.
        A value of 2.0 is a standard elliptical distance metric, values
        less than 2.0 create disky or pointed perturbations to an ellipse,
        and values greater than 2.0 create boxy perturbations to an ellipse.

    """

    _model_type = "superellipse"
    _parameter_specs = {
        "C": {"units": "none", "dynamic_value": 2.0, "valid": (0, 10)},
    }

    @forward
    def radius_metric(self, x: ArrayLike, y: ArrayLike, C: ArrayLike) -> ArrayLike:
        return (backend.abs(x) ** C + backend.abs(y) ** C + self.softening**C) ** (1.0 / C)


class FourierEllipseMixin:
    """Sine wave perturbation of the elliptical radius metric.

    This allows for the expression of arbitrarily complex isophotes instead of
    pure ellipses. This is a common extension of the standard elliptical
    representation. The form of the Fourier perturbations is:

    $$R' = R * \\exp\\left(\\sum_m(a_m * \\cos(m * \\theta + \\phi_m))\\right)$$

    where R' is the new radius value, R is the original radius (typically
    computed as $\\sqrt{x^2+y^2}$), m is the index of the Fourier mode, a_m is
    the amplitude of the m'th Fourier mode, theta is the angle around the
    ellipse (typically $\\arctan(y/x)$), and phi_m is the phase of the m'th
    fourier mode.

    One can create extremely complex shapes using different Fourier modes,
    however usually it is only low order modes that are of interest. For
    intuition, the first Fourier mode is roughly equivalent to a lopsided
    galaxy, one side will be compressed and the opposite side will be expanded.
    The second mode is almost never used as it is nearly degenerate with
    ellipticity. The third mode is an alternate kind of lopsidedness for a
    galaxy which makes it somewhat triangular, meaning that it is wider on one
    side than the other. The fourth mode is similar to a boxyness/diskyness
    parameter of a superelllipse which tends to make more pronounced peanut
    shapes since it is more rounded than a superellipse representation. Modes
    higher than 4 are only useful in very specialized situations. In general one
    should consider carefully why the Fourier modes are being used for the
    science case at hand.

    **Parameters:**
    -    `am`: Tensor of amplitudes for the Fourier modes, indicates the strength
            of each mode.
    -    `phim`: Tensor of phases for the Fourier modes, adjusts the
            orientation of the mode perturbation relative to the major axis. It
            is cyclically defined in the range [0,2pi)

    **Options:**
    -    `modes`: Tuple of integers indicating which Fourier modes to use.
    """

    _model_type = "fourier"
    _parameter_specs = {
        "am": {"units": "none"},
        "phim": {"units": "radians", "valid": (0, 2 * np.pi), "cyclic": True},
    }
    _options = ("modes",)

    def __init__(self, *args, modes: Tuple[int] = (3, 4), **kwargs):
        super().__init__(*args, **kwargs)
        self.modes = backend.as_array(modes, dtype=config.DTYPE, device=config.DEVICE)

    @forward
    def radius_metric(
        self, x: ArrayLike, y: ArrayLike, am: ArrayLike, phim: ArrayLike
    ) -> ArrayLike:
        R = super().radius_metric(x, y)
        theta = self.angular_metric(x, y)
        return R * backend.exp(
            backend.sum(
                am[..., None]
                * backend.cos(self.modes[..., None] * theta.flatten() + phim[..., None]),
                0,
            ).reshape(x.shape)
        )

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if not self.am.initialized:
            self.am.dynamic_value = np.zeros(len(self.modes))
        if not self.phim.initialized:
            self.phim.value = np.zeros(len(self.modes))


class WarpMixin:
    """Warped model with varying PA and q as a function of radius.

    This works by warping the coordinates using the same transform for a global
    PA, q except applied to each pixel individually based on its unwarped radius
    value. In the limit that PA and q are a constant, this recovers a basic
    model with global PA, q. However, a linear PA profile will give a spiral
    appearance, variations of PA, q profiles can create complex galaxy models.
    The form of the coordinate transformation for each pixel looks like:

    $$R = \\sqrt{x^2 + y^2}$$
    $$x', y' = \\rm{rotate}(-PA(R) + \\pi/2, x, y)$$
    $$y'' = y' / q(R)$$

    Note that now PA and q are functions of radius R, which is computed from the
    original coordinates X, Y. This is achieved by making PA and q a spline
    profile.

    **Parameters:**
    -  `q_R`: Tensor of axis ratio values for axis ratio spline
    -  `PA_R`: Tensor of position angle values as input to the spline

    """

    _model_type = "warp"
    _parameter_specs = {
        "q_R": {"units": "b/a", "valid": (0, 1)},
        "PA_R": {"units": "radians", "valid": (0, np.pi), "cyclic": True},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if not self.PA_R.initialized:
            if self.PA_R.prof is None:
                self.PA_R.prof = default_prof(self.window.shape, self.target.pixelscale, 2, 0.2)
            self.PA_R.dynamic_value = np.zeros(len(self.PA_R.prof)) + np.pi / 2
        if not self.q_R.initialized:
            if self.q_R.prof is None:
                self.q_R.prof = default_prof(self.window.shape, self.target.pixelscale, 2, 0.2)
            self.q_R.dynamic_value = np.ones(len(self.q_R.prof)) * 0.8

    @forward
    def transform_coordinates(
        self, x: ArrayLike, y: ArrayLike, q_R: ArrayLike, PA_R: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        x, y = super().transform_coordinates(x, y)
        R = self.radius_metric(x, y)
        PA = func.spline(R, self.PA_R.prof, PA_R, extend="const")
        q = func.spline(R, self.q_R.prof, q_R, extend="const")
        x, y = func.rotate(-PA + np.pi / 2, x, y)
        return x, y / q


class TruncationMixin:
    """Truncated model with radial brightness profile.

    This model will smoothly truncate the radial brightness profile at Rt. The
    truncation is centered on Rt and thus two identical models with the same Rt
    (and St) where one is inner truncated and the other is outer truncated will
    reproduce nearly the same as a single un-truncated model.

    By default the St parameter is set fixed to 1.0, giving a relatively smooth
    truncation. This can be set to a smaller value for sharper truncations or a
    larger value for even more gradual truncation. It can be set dynamic to be
    optimized in a model, though it is possible for this parameter to be
    unstable if there isn't a clear truncation signal in the data.

    **Parameters:**
    -  `Rt`: The truncation radius in arcseconds.
    -  `St`: The steepness of the truncation profile, controlling how quickly
             the brightness drops to zero at the truncation radius.

    **Options:**
    -   `outer_truncation`: If True, the model will truncate the brightness beyond
         the truncation radius. If False, the model will truncate the
         brightness within the truncation radius.
    """

    _model_type = "truncated"
    _parameter_specs = {
        "Rt": {"units": "arcsec", "valid": (0, None), "shape": ()},
        "St": {"units": "none", "valid": (0, None), "shape": (), "value": 1.0},
    }
    _options = ("outer_truncation",)

    def __init__(self, *args, outer_truncation: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.outer_truncation = outer_truncation

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        if not self.Rt.initialized:
            prof = default_prof(self.window.shape, self.target.pixelscale, 2, 0.2)
            self.Rt.dynamic_value = prof[len(prof) // 2]

    @forward
    def radial_model(self, R: ArrayLike, Rt: ArrayLike, St: ArrayLike) -> ArrayLike:
        I = super().radial_model(R)
        if self.outer_truncation:
            return I * (1 - backend.tanh(St * (R - Rt))) / 2
        return I * (backend.tanh(St * (R - Rt)) + 1) / 2

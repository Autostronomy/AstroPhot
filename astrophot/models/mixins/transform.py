import numpy as np
import torch

from ...utils.decorators import ignore_numpy_warnings
from ...utils.interpolate import default_prof
from ...param import forward
from .. import func
from ... import AP_config


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

        if self.PA.initialized and self.q.initialized:
            return
        target_area = self.target[self.window]
        dat = target_area.data.npvalue.copy()
        if target_area.has_mask:
            mask = target_area.mask.detach().cpu().numpy()
            dat[mask] = np.median(dat[~mask])
        edge = np.concatenate((dat[:, 0], dat[:, -1], dat[0, :], dat[-1, :]))
        edge_average = np.nanmedian(edge)
        dat -= edge_average
        x, y = target_area.coordinate_center_meshgrid()
        x = (x - self.center.value[0]).detach().cpu().numpy()
        y = (y - self.center.value[1]).detach().cpu().numpy()
        mu20 = np.median(dat * np.abs(x))
        mu02 = np.median(dat * np.abs(y))
        mu11 = np.median(dat * x * y / np.sqrt(np.abs(x * y) + self.softening**2))
        # mu20 = np.median(dat * x**2)
        # mu02 = np.median(dat * y**2)
        # mu11 = np.median(dat * x * y)
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
            self.q.dynamic_value = np.clip(np.sqrt(l[0] / l[1]), 0.1, 0.9)

    @forward
    def transform_coordinates(self, x, y, PA, q):
        """
        Transform coordinates based on the position angle and axis ratio.
        """
        x, y = super().transform_coordinates(x, y)
        x, y = func.rotate(-PA + np.pi / 2, x, y)
        return x, y / q


class SuperEllipseMixin:
    """Expanded galaxy model which includes a superellipse transformation
    in its radius metric. This allows for the expression of "boxy" and
    "disky" isophotes instead of pure ellipses. This is a common
    extension of the standard elliptical representation, especially
    for early-type galaxies. The functional form for this is:

    R = (|X|^C + |Y|^C)^(1/C)

    where R is the new distance metric, X Y are the coordinates, and C
    is the coefficient for the superellipse. C can take on any value
    greater than zero where C = 2 is the standard distance metric, 0 <
    C < 2 creates disky or pointed perturbations to an ellipse, and C
    > 2 transforms an ellipse to be more boxy.

    Parameters:
        C: superellipse distance metric parameter.

    """

    _model_type = "superellipse"
    _parameter_specs = {
        "C": {"units": "none", "value": 2.0, "uncertainty": 1e-2, "valid": (0, None)},
    }

    @forward
    def radius_metric(self, x, y, C):
        return torch.pow(x.abs().pow(C) + y.abs().pow(C), 1.0 / C)


class FourierEllipseMixin:
    """Expanded galaxy model which includes a Fourier transformation in
    its radius metric. This allows for the expression of arbitrarily
    complex isophotes instead of pure ellipses. This is a common
    extension of the standard elliptical representation. The form of
    the Fourier perturbations is:

    R' = R * exp(sum_m(a_m * cos(m * theta + phi_m)))

    where R' is the new radius value, R is the original ellipse
    radius, a_m is the amplitude of the m'th Fourier mode, m is the
    index of the Fourier mode, theta is the angle around the ellipse,
    and phi_m is the phase of the m'th fourier mode. This
    representation is somewhat different from other Fourier mode
    implementations where instead of an expoenntial it is just 1 +
    sum_m(...), we opt for this formulation as it is more numerically
    stable. It cannot ever produce negative radii, but to first order
    the two representation are the same as can be seen by a Taylor
    expansion of exp(x) = 1 + x + O(x^2).

    One can create extremely complex shapes using different Fourier
    modes, however usually it is only low order modes that are of
    interest. For intuition, the first Fourier mode is roughly
    equivalent to a lopsided galaxy, one side will be compressed and
    the opposite side will be expanded. The second mode is almost
    never used as it is nearly degenerate with ellipticity. The third
    mode is an alternate kind of lopsidedness for a galaxy which makes
    it somewhat triangular, meaning that it is wider on one side than
    the other. The fourth mode is similar to a boxyness/diskyness
    parameter which tends to make more pronounced peanut shapes since
    it is more rounded than a superellipse representation. Modes
    higher than 4 are only useful in very specialized situations. In
    general one should consider carefully why the Fourier modes are
    being used for the science case at hand.

    Parameters:
        am: Tensor of amplitudes for the Fourier modes, indicates the strength of each mode.
        phi_m: Tensor of phases for the Fourier modes, adjusts the orientation of the mode perturbation relative to the major axis. It is cyclically defined in the range [0,2pi)

    """

    _model_type = "fourier"
    _parameter_specs = {
        "am": {"units": "none"},
        "phim": {"units": "radians", "valid": (0, 2 * np.pi), "cyclic": True},
    }
    _options = ("modes",)

    def __init__(self, *args, modes=(3, 4), **kwargs):
        super().__init__(*args, **kwargs)
        self.modes = torch.tensor(modes, dtype=AP_config.ap_dtype, device=AP_config.ap_device)

    @forward
    def radius_metric(self, x, y, am, phim):
        R = super().radius_metric(x, y)
        theta = self.angular_metric(x, y)
        return R * torch.exp(
            torch.sum(
                am.unsqueeze(-1)
                * torch.cos(self.modes.unsqueeze(-1) * theta.flatten() + phim.unsqueeze(-1)),
                0,
            ).reshape(x.shape)
        )

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if not self.am.initialized:
            self.am.dynamic_value = np.zeros(len(self.modes))
            self.am.uncertainty = self.default_uncertainty * np.ones(len(self.modes))
        if not self.phim.initialized:
            self.phim.value = np.zeros(len(self.modes))
            self.phim.uncertainty = (10 * np.pi / 180) * np.ones(len(self.modes))


class WarpMixin:
    """Galaxy model which includes radially varrying PA and q
    profiles. This works by warping the coordinates using the same
    transform for a global PA/q except applied to each pixel
    individually. In the limit that PA and q are a constant, this
    recovers a basic galaxy model with global PA/q. However, a linear
    PA profile will give a spiral appearance, variations of PA/q
    profiles can create complex galaxy models. The form of the
    coordinate transformation looks like:

    X, Y = meshgrid(image)
    R = sqrt(X^2 + Y^2)
    X', Y' = Rot(theta(R), X, Y)
    Y'' = Y' / q(R)

    where the definitions are the same as for a regular galaxy model,
    except now the theta is a function of radius R (before
    transformation) and the axis ratio q is also a function of radius
    (before the transformation).

    Parameters:
        q(R): Tensor of axis ratio values for axis ratio spline
        PA(R): Tensor of position angle values as input to the spline

    """

    _model_type = "warp"
    _parameter_specs = {
        "q_R": {"units": "b/a", "valid": (0.0, 1), "uncertainty": 0.04},
        "PA_R": {
            "units": "radians",
            "valid": (0, np.pi),
            "cyclic": True,
            "uncertainty": 0.08,
        },
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if not self.PA_R.initialized:
            if self.PA_R.prof is None:
                self.PA_R.prof = default_prof(self.window.shape, self.target.pixel_length, 2, 0.2)
            self.PA_R.dynamic_value = np.zeros(len(self.PA_R.prof)) + np.pi / 2
            self.PA_R.uncertainty = (10 * np.pi / 180) * torch.ones_like(self.PA_R.value)
        if not self.q_R.initialized:
            if self.q_R.prof is None:
                self.q_R.prof = default_prof(self.window.shape, self.target.pixel_length, 2, 0.2)
            self.q_R.dynamic_value = np.ones(len(self.q_R.prof)) * 0.8
            self.q_R.uncertainty = self.default_uncertainty * self.q_R.value

    @forward
    def transform_coordinates(self, x, y, q_R, PA_R):
        x, y = super().transform_coordinates(x, y)
        R = self.radius_metric(x, y)
        PA = func.spline(R, self.PA_R.prof, PA_R, extend="const")
        q = func.spline(R, self.q_R.prof, q_R, extend="const")
        x, y = func.rotate(PA, x, y)
        return x, y / q


class TruncationMixin:
    """Mixin for models that include a truncation radius. This is used to
    limit the radial extent of the model, effectively setting a maximum
    radius beyond which the model's brightness is zero.

    Parameters:
        R_trunc: The truncation radius in arcseconds.
    """

    _model_type = "truncated"
    _parameter_specs = {
        "Rt": {"units": "arcsec", "valid": (0, None), "shape": ()},
        "sharpness": {"units": "none", "valid": (0, None), "shape": ()},
    }
    _options = ("outer_truncation",)

    def __init__(self, *args, outer_truncation=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.outer_truncation = outer_truncation

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        if not self.Rt.initialize:
            prof = default_prof(self.window.shape, self.target.pixel_length, 2, 0.2)
            self.Rt.dynamic_value = prof[len(prof) // 2]
            self.Rt.uncertainty = 0.1
        if not self.sharpness.initialized:
            self.sharpness.dynamic_value = 1.0
            self.sharpness.uncertainty = 0.1

    @forward
    def radial_model(self, R, Rt, sharpness):
        I = super().radial_model(R)
        if self.outer_truncation:
            return I * (1 - torch.tanh(sharpness * (R - Rt))) / 2
        return I * (torch.tanh(sharpness * (R - Rt)) + 1) / 2

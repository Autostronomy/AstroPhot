import torch
import numpy as np

from ...param import forward
from .. import func
from ...utils.decorators import ignore_numpy_warnings
from ...utils.interpolate import default_prof
from ... import AP_config


class RadialMixin:

    @forward
    def brightness(self, x, y):
        """
        Calculate the brightness at a given point (x, y) based on radial distance from the center.
        """
        x, y = self.transform_coordinates(x, y)
        return self.radial_model(self.radius_metric(x, y))


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

        if self.am.value is None:
            self.am.dynamic_value = np.zeros(len(self.modes))
            self.am.uncertainty = self.default_uncertainty * np.ones(len(self.modes))
        if self.phim.value is None:
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

        if self.PA_R.value is None:
            if self.PA_R.prof is None:
                self.PA_R.prof = default_prof(self.window.shape, self.target.pixel_length, 2, 0.2)
            self.PA_R.dynamic_value = np.zeros(len(self.PA_R.prof)) + np.pi / 2
            self.PA_R.uncertainty = (10 * np.pi / 180) * torch.ones_like(self.PA_R.value)
        if self.q_R.value is None:
            if self.q_R.prof is None:
                self.q_R.prof = default_prof(self.window.shape, self.target.pixel_length, 2, 0.2)
            self.q_R.dynamic_value = np.ones(len(self.q_R.prof)) * 0.8
            self.q_R.uncertainty = self.default_uncertainty * self.q_R.value

    @forward
    def transform_coordinates(self, x, y, q_R, PA_R):
        x, y = super().transform_coordinates(x, y)
        R = self.radius_metric(x, y)
        PA = func.spline(R, self.PA_R.prof, PA_R)
        q = func.spline(R, self.q_R.prof, q_R)
        x, y = func.rotate(PA, x, y)
        return x, y / q


class WedgeMixin:
    """Variant of the ray model where no smooth transition is performed
    between regions as a function of theta, instead there is a sharp
    trnasition boundary. This may be desirable as it cleanly
    separates where the pixel information is going. Due to the sharp
    transition though, it may cause unusual behaviour when fitting. If
    problems occur, try fitting a ray model first then fix the center,
    PA, and q and then fit the wedge model. Essentially this breaks
    down the structure fitting and the light profile fitting into two
    steps. The wedge model, like the ray model, defines no extra
    parameters, however a new option can be supplied on instantiation
    of the wedge model which is "wedges" or the number of wedges in
    the model.

    """

    _model_type = "wedge"
    _options = ("segments", "symmetric")

    def __init__(self, *args, symmetric=True, segments=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetric = symmetric
        self.segments = segments

    def polar_model(self, R, T):
        model = torch.zeros_like(R)
        cycle = np.pi if self.symmetric else 2 * np.pi
        w = cycle / self.segments
        angles = (T + w / 2) % cycle
        v = w * np.arange(self.segments)
        for s in range(self.segments):
            indices = (angles >= v[s]) & (angles < (v[s] + w))
            model[indices] += self.iradial_model(s, R[indices])
        return model

    def brightness(self, x, y):
        x, y = self.transform_coordinates(x, y)
        return self.polar_model(self.radius_metric(x, y), self.angular_metric(x, y))


class RayMixin:
    """Variant of a galaxy model which defines multiple radial models
    seprarately along some number of rays projected from the galaxy
    center. These rays smoothly transition from one to another along
    angles theta. The ray transition uses a cosine smoothing function
    which depends on the number of rays, for example with two rays the
    brightness would be:

    I(R,theta) = I1(R)*cos(theta % pi) + I2(R)*cos((theta + pi/2) % pi)

    Where I(R,theta) is the brightness function in polar coordinates,
    R is the semi-major axis, theta is the polar angle (defined after
    galaxy axis ratio is applied), I1(R) is the first brightness
    profile, % is the modulo operator, and I2 is the second brightness
    profile. The ray model defines no extra parameters, though now
    every model parameter related to the brightness profile gains an
    extra dimension for the ray number. Also a new input can be given
    when instantiating the ray model: "rays" which is an integer for
    the number of rays.

    """

    _model_type = "ray"
    _options = ("symmetric", "segments")

    def __init__(self, *args, symmetric=True, segments=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetric = symmetric
        self.segments = segments

    def polar_model(self, R, T):
        model = torch.zeros_like(R)
        weight = torch.zeros_like(R)
        cycle = np.pi if self.symmetric else 2 * np.pi
        w = cycle / self.segments
        v = w * np.arange(self.segments)
        for s in range(self.segments):
            angles = (T + cycle / 2 - v[s]) % cycle - cycle / 2
            indices = (angles >= -w) & (angles < w)
            weights = (torch.cos(angles[indices] * self.segments) + 1) / 2
            model[indices] += self.iradial_model(s, R[indices])
            weight[indices] += weights
        return model / weight

    def brightness(self, x, y):
        x, y = self.transform_coordinates(x, y)
        return self.polar_model(self.radius_metric(x, y), self.angular_metric(x, y))

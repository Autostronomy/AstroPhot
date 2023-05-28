import torch
import numpy as np

from ..utils.decorators import ignore_numpy_warnings, default_internal
from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from ._shared_methods import select_target
from .. import AP_config

__all__ = ["FourierEllipse_Galaxy", "FourierEllipse_Warp"]


class FourierEllipse_Galaxy(Galaxy_Model):
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
    implimentations where instead of an expoenntial it is just 1 +
    sum_m(...), we opt for this formulation as it is more numerically
    stable. It cannot ever produce negative radii, but to first order
    the two representation are the same as can be seen by a Taylor
    expansion of exp(x) = 1 + x + O(x^2).

    One can create extrememly complex shapes using different Fourier
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

    model_type = f"fourier {Galaxy_Model.model_type}"
    parameter_specs = {
        "am": {"units": "none"},
        "phim": {"units": "radians", "limits": (0, 2 * np.pi), "cyclic": True},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("am", "phim")
    modes = (1, 3, 4)
    track_attrs = Galaxy_Model.track_attrs + ["modes"]
    useable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modes = torch.tensor(
            kwargs.get("modes", FourierEllipse_Galaxy.modes),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )

    @default_internal
    def angular_metric(self, X, Y, image=None, parameters=None):
        return torch.atan2(Y, X)

    @default_internal
    def radius_metric(self, X, Y, image=None, parameters=None):
        R = super().radius_metric(X, Y, image, parameters)
        theta = self.angular_metric(X, Y, image, parameters)
        return R * torch.exp(
            torch.sum(
                parameters["am"].value.view(len(self.modes), -1)
                * torch.cos(
                    self.modes.view(len(self.modes), -1) * theta.view(-1)
                    + parameters["phim"].value.view(len(self.modes), -1)
                ),
                0,
            ).view(theta.shape)
        )

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        if parameters["am"].value is None:
            parameters["am"].set_value(
                torch.zeros(
                    len(self.modes),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                ),
                override_locked=True,
            )
        if parameters["am"].uncertainty is None:
            parameters["am"].set_uncertainty(
                torch.tensor(
                    0.05 * np.ones(len(self.modes)),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                ),
                override_locked=True,
            )
        if parameters["phim"].value is None:
            parameters["phim"].set_value(
                torch.zeros(
                    len(self.modes),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                ),
                override_locked=True,
            )
        if parameters["phim"].uncertainty is None:
            parameters["phim"].set_uncertainty(
                torch.tensor(
                    (5 * np.pi / 180) * np.ones(len(self.modes)),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                ),
                override_locked=True,
            )


class FourierEllipse_Warp(Warp_Galaxy):
    """Expanded warp galaxy model which includes a Fourier transformation
    in its radius metric. This allows for the expression of
    arbitrarily complex isophotes instead of pure ellipses. This is a
    common extension of the standard elliptical representation. The
    form of the Fourier perturbations is:

    R' = R * exp(sum_m(a_m * cos(m * theta + phi_m)))

    where R' is the new radius value, R is the original ellipse
    radius, a_m is the amplitude of the m'th Fourier mode, m is the
    index of the Fourier mode, theta is the angle around the ellipse,
    and phi_m is the phase of the m'th fourier mode. This
    representation is somewhat different from other Fourier mode
    implimentations where instead of an expoenntial it is just 1 +
    sum_m(...), we opt for this formulation as it is more numerically
    stable. It cannot ever produce negative radii, but to first order
    the two representation are the same as can be seen by a Taylor
    expansion of exp(x) = 1 + x + O(x^2).

    One can create extrememly complex shapes using different Fourier
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

    model_type = f"fourier {Warp_Galaxy.model_type}"
    parameter_specs = {
        "am": {"units": "none"},
        "phim": {"units": "radians", "limits": (0, 2 * np.pi), "cyclic": True},
    }
    _parameter_order = Warp_Galaxy._parameter_order + ("am", "phim")
    modes = (1, 3, 4)
    track_attrs = Galaxy_Model.track_attrs + ["modes"]
    useable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modes = torch.tensor(
            kwargs.get("modes", FourierEllipse_Warp.modes),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )

    @default_internal
    def angular_metric(self, X, Y, image=None, parameters=None):
        return torch.atan2(Y, X)

    @default_internal
    def radius_metric(self, X, Y, image=None, parameters=None):
        R = super().radius_metric(X, Y, image, parameters)
        theta = self.angular_metric(X, Y, image, parameters)
        return R * torch.exp(
            torch.sum(
                parameters["am"].value.view(len(self.modes), -1)
                * torch.cos(
                    self.modes.view(len(self.modes), -1) * theta.view(-1)
                    + parameters["phim"].value.view(len(self.modes), -1)
                ),
                0,
            ).view(theta.shape)
        )

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        if parameters["am"].value is None:
            parameters["am"].set_value(
                torch.zeros(
                    len(self.modes),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                ),
                override_locked=True,
            )
        if parameters["am"].uncertainty is None:
            parameters["am"].set_uncertainty(
                torch.tensor(
                    0.05 * np.ones(len(self.modes)),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                ),
                override_locked=True,
            )
        if parameters["phim"].value is None:
            parameters["phim"].set_value(
                torch.zeros(
                    len(self.modes),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                ),
                override_locked=True,
            )
        if parameters["phim"].uncertainty is None:
            parameters["phim"].set_uncertainty(
                torch.tensor(
                    (5 * np.pi / 180) * np.ones(len(self.modes)),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                ),
                override_locked=True,
            )

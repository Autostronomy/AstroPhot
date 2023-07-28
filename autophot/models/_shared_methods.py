import functools

from scipy.special import gamma
from scipy.stats import binned_statistic, iqr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import torch
from scipy.optimize import minimize

from ..utils.initialize import isophotes
from ..utils.parametric_profiles import (
    sersic_torch,
    sersic_np,
    gaussian_torch,
    gaussian_np,
    exponential_torch,
    exponential_np,
    spline_torch,
    moffat_torch,
    moffat_np,
    nuker_torch,
    nuker_np,
)
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.conversions.coordinates import Rotate_Cartesian
from ..utils.conversions.functions import sersic_I0_to_flux_np, sersic_flux_to_I0_torch
from ..image import (
    Image_List,
    Target_Image,
    Model_Image_List,
    Target_Image_List,
    Window_List,
)
from .. import AP_config

# Target Selector Decorator
######################################################################
def select_target(func):
    @functools.wraps(func)
    def targeted(self, target=None, **kwargs):
        if target is None:
            send_target = self.target
        elif isinstance(target, Target_Image_List) and not isinstance(
            self.target, Image_List
        ):
            for sub_target in target:
                if sub_target.identity == self.target.identity:
                    send_target = sub_target
                    break
            else:
                raise RuntimeError(
                    "{self.name} could not find matching target to initialize with"
                )
        else:
            send_target = target
        return func(self, target=send_target, **kwargs)

    return targeted


def select_sample(func):
    @functools.wraps(func)
    def targeted(self, image=None, **kwargs):
        if isinstance(image, Model_Image_List) and not isinstance(
            self.target, Image_List
        ):
            for i, sub_image in enumerate(image):
                if sub_image.target_identity == self.target.identity:
                    send_image = sub_image
                    if "window" in kwargs and isinstance(kwargs["window"], Window_List):
                        kwargs["window"] = kwargs["window"].window_list[i]
                    break
            else:
                raise RuntimeError(
                    "{self.name} could not find matching image to sample with"
                )
        else:
            send_image = image
        return func(self, image=send_image, **kwargs)

    return targeted


# General parametric
######################################################################
@torch.no_grad()
@ignore_numpy_warnings
def parametric_initialize(
    model, parameters, target, prof_func, params, x0_func, force_uncertainty=None
):
    if all(list(parameters[param].value is not None for param in params)):
        return
    # Get the sub-image area corresponding to the model image
    target_area = target[model.window]
    edge = np.concatenate(
        (
            target_area.data.detach().cpu().numpy()[:, 0],
            target_area.data.detach().cpu().numpy()[:, -1],
            target_area.data.detach().cpu().numpy()[0, :],
            target_area.data.detach().cpu().numpy()[-1, :],
        )
    )
    edge_average = np.median(edge)
    edge_scatter = iqr(edge, rng=(16, 84)) / 2
    # Convert center coordinates to target area array indices
    icenter = target_area.world_to_pixel(parameters["center"].value)

    # Collect isophotes for 1D fit
    iso_info = isophotes(
        target_area.data.detach().cpu().numpy() - edge_average,
        (icenter[1].item(), icenter[0].item()),
        threshold=3 * edge_scatter,
        pa=(parameters["PA"].value - target.north).detach().cpu().item()
        if "PA" in parameters
        else 0.0,
        q=parameters["q"].value.detach().cpu().item() if "q" in parameters else 1.0,
        n_isophotes=15,
    )
    R = np.array(list(iso["R"] for iso in iso_info)) * target.pixel_length.item()
    flux = np.array(list(iso["flux"] for iso in iso_info)) / target.pixel_area.item()
    # Correct the flux if values are negative, so fit can be done in log space
    if np.sum(flux < 0) > 0:
        AP_config.ap_logger.debug("fixing flux")
        flux -= np.min(flux) - np.abs(np.min(flux) * 0.1)
    flux = np.log10(flux)

    x0 = list(x0_func(model, R, flux))
    for i, param in enumerate(params):
        x0[i] = (
            x0[i] if parameters[param].value is None else parameters[param].value.item()
        )

    def optim(x, r, f):
        residual = (f - np.log10(prof_func(r, *x))) ** 2
        N = np.argsort(residual)
        return np.mean(residual[:-3])

    res = minimize(optim, x0=x0, args=(R, flux), method="Nelder-Mead")
    if not res.success and AP_config.ap_verbose >= 2:
        AP_config.ap_logger.warn(
            f"initialization fit not successful for {model.name}, falling back to defaults"
        )

    if force_uncertainty is None:
        reses = []
        for i in range(10):
            N = np.random.randint(0, len(R), len(R))
            reses.append(
                minimize(optim, x0=x0, args=(R[N], flux[N]), method="Nelder-Mead")
            )
    for param, resx, x0x in zip(params, res.x, x0):
        if parameters[param].value is None:
            parameters[param].set_value(
                resx if res.success else x0x, override_locked=True
            )
        if force_uncertainty is None and parameters[param].uncertainty is None:
            parameters[param].set_uncertainty(
                np.std(list(subres.x[params.index(param)] for subres in reses)),
                override_locked=True,
            )
        elif force_uncertainty is not None:
            parameters[param].set_uncertainty(
                force_uncertainty[params.index(param)], override_locked=True
            )


@torch.no_grad()
@ignore_numpy_warnings
def parametric_segment_initialize(
    model=None,
    parameters=None,
    target=None,
    prof_func=None,
    params=None,
    x0_func=None,
    segments=None,
    force_uncertainty=None,
):
    if all(list(model[param].value is not None for param in params)):
        return
    # Get the sub-image area corresponding to the model image
    target_area = target[model.window]
    edge = np.concatenate(
        (
            target_area.data[:, 0].detach().cpu().numpy(),
            target_area.data[:, -1].detach().cpu().numpy(),
            target_area.data[0, :].detach().cpu().numpy(),
            target_area.data[-1, :].detach().cpu().numpy(),
        )
    )
    edge_average = np.median(edge)
    edge_scatter = iqr(edge, rng=(16, 84)) / 2
    # Convert center coordinates to target area array indices
    icenter = target_area.world_to_pixel(model["center"].value)

    iso_info = isophotes(
        target_area.data.detach().cpu().numpy() - edge_average,
        (icenter[1].item(), icenter[0].item()),
        threshold=3 * edge_scatter,
        pa=(model["PA"].value - target.north).item() if "PA" in model else 0.0,
        q=model["q"].value.item() if "q" in model else 1.0,
        n_isophotes=15,
        more=True,
    )
    R = np.array(list(iso["R"] for iso in iso_info)) * target.pixel_length.item()
    was_none = list(False for i in range(len(params)))
    for i, p in enumerate(params):
        if model[p].value is None:
            was_none[i] = True
            model[p].set_value(np.zeros(segments), override_locked=True)
            model[p].set_uncertainty(np.zeros(segments), override_locked=True)
    for r in range(segments):
        flux = []
        for iso in iso_info:
            modangles = (
                iso["angles"]
                - (
                    (model["PA"].value - target.north).detach().cpu().item()
                    + r * np.pi / segments
                )
            ) % np.pi
            flux.append(
                np.median(
                    iso["isovals"][
                        np.logical_or(
                            modangles < (0.5 * np.pi / segments),
                            modangles >= (np.pi * (1 - 0.5 / segments)),
                        )
                    ]
                )
            )
        flux = np.array(flux) / target.pixel_area.item()
        if np.sum(flux < 0) >= 1:
            flux -= np.min(flux) - np.abs(np.min(flux) * 0.1)
        flux = np.log10(flux)

        x0 = list(x0_func(model, R, flux))
        for i, param in enumerate(params):
            x0[i] = (
                x0[i] if was_none[i] else model[param].value.detach().cpu().numpy()[r]
            )
        res = minimize(
            lambda x: np.mean((flux - np.log10(prof_func(R, *x))) ** 2),
            x0=x0,
            method="Nelder-Mead",
        )
        if force_uncertainty is None:
            reses = []
            for i in range(10):
                N = np.random.randint(0, len(R), len(R))
                reses.append(
                    minimize(
                        lambda x: np.mean((flux - np.log10(prof_func(R, *x))) ** 2),
                        x0=x0,
                        method="Nelder-Mead",
                    )
                )
        for i, param in enumerate(params):
            if was_none[i]:
                model[param].set_value(
                    res.x[i] if res.success else x0[i], override_locked=True, index=r
                )
                if force_uncertainty is None and model[param].uncertainty is None:
                    model[param].set_uncertainty(
                        np.std(list(subres.x[params.index(param)] for subres in reses)),
                        override_locked=True,
                        index=r,
                    )
                elif force_uncertainty is not None:
                    model[param].set_uncertainty(
                        force_uncertainty[params.index(param)][r],
                        override_locked=True,
                        index=r,
                    )


# Exponential
######################################################################
@default_internal
def exponential_radial_model(self, R, image=None, parameters=None):
    return exponential_torch(
        R,
        parameters["Re"].value,
        image.pixel_area * 10 ** parameters["Ie"].value,
    )


@default_internal
def exponential_iradial_model(self, i, R, image=None, parameters=None):
    return exponential_torch(
        R,
        parameters["Re"].value[i],
        image.pixel_area * 10 ** parameters["Ie"].value[i],
    )


# Sersic
######################################################################
@default_internal
def sersic_radial_model(self, R, image=None, parameters=None):
    return sersic_torch(
        R,
        parameters["n"].value,
        parameters["Re"].value,
        image.pixel_area * 10 ** parameters["Ie"].value,
    )


@default_internal
def sersic_iradial_model(self, i, R, image=None, parameters=None):
    return sersic_torch(
        R,
        parameters["n"].value[i],
        parameters["Re"].value[i],
        image.pixel_area * 10 ** parameters["Ie"].value[i],
    )


# Moffat
######################################################################
@default_internal
def moffat_radial_model(self, R, image=None, parameters=None):
    return moffat_torch(
        R,
        parameters["n"].value,
        parameters["Rd"].value,
        image.pixel_area * 10 ** parameters["I0"].value,
    )


@default_internal
def moffat_iradial_model(self, i, R, image=None, parameters=None):
    return moffat_torch(
        R,
        parameters["n"].value[i],
        parameters["Rd"].value[i],
        image.pixel_area * 10 ** parameters["I0"].value[i],
    )


# Nuker Profile
######################################################################
@default_internal
def nuker_radial_model(self, R, image=None, parameters=None):
    return nuker_torch(
        R,
        parameters["Rb"].value,
        image.pixel_area * 10 ** parameters["Ib"].value,
        parameters["alpha"].value,
        parameters["beta"].value,
        parameters["gamma"].value,
    )


@default_internal
def nuker_iradial_model(self, i, R, image=None, parameters=None):
    return nuker_torch(
        R,
        parameters["Rb"].value[i],
        image.pixel_area * 10 ** parameters["Ib"].value[i],
        parameters["alpha"].value[i],
        parameters["beta"].value[i],
        parameters["gamma"].value[i],
    )


# Gaussian
######################################################################
@default_internal
def gaussian_radial_model(self, R, image=None, parameters=None):
    return gaussian_torch(
        R,
        parameters["sigma"].value,
        image.pixel_area * 10 ** parameters["flux"].value,
    )


@default_internal
def gaussian_iradial_model(self, i, R, image=None, parameters=None):
    return gaussian_torch(
        R,
        parameters["sigma"].value[i],
        image.pixel_area * 10 ** parameters["flux"].value[i],
    )


# Spline
######################################################################
@torch.no_grad()
@ignore_numpy_warnings
@select_target
@default_internal
def spline_initialize(self, target=None, parameters=None, **kwargs):
    super(self.__class__, self).initialize(target=target, parameters=parameters)

    if parameters["I(R)"].value is not None and parameters["I(R)"].prof is not None:
        return

    # Create the I(R) profile radii if needed
    if parameters["I(R)"].prof is None:
        new_prof = [0, 2 * target.pixel_length]
        while new_prof[-1] < torch.max(self.window.shape / 2):
            new_prof.append(
                new_prof[-1] + torch.max(2 * target.pixel_length, new_prof[-1] * 0.2)
            )
        new_prof.pop()
        new_prof.pop()
        new_prof.append(torch.sqrt(torch.sum((self.window.shape / 2) ** 2)))
        parameters["I(R)"].set_profile(new_prof)

    profR = parameters["I(R)"].prof.detach().cpu().numpy()
    target_area = target[self.window]
    Coords = target_area.get_coordinate_meshgrid()
    X, Y = Coords - parameters["center"].value[..., None, None]
    X, Y = self.transform_coordinates(X, Y, target, parameters)
    R = self.radius_metric(X, Y, target, parameters).detach().cpu().numpy()
    rad_bins = [profR[0]] + list((profR[:-1] + profR[1:]) / 2) + [profR[-1] * 100]
    raveldat = target_area.data.detach().cpu().numpy().ravel()

    I = (
        binned_statistic(R.ravel(), raveldat, statistic="median", bins=rad_bins)[0]
    ) / target.pixel_area.item()
    N = np.isfinite(I)
    if not np.all(N):
        I[np.logical_not(N)] = np.interp(profR[np.logical_not(N)], profR[N], I[N])
    if I[-1] >= I[-2]:
        I[-1] = I[-2] / 2
    S = binned_statistic(
        R.ravel(), raveldat, statistic=lambda d: iqr(d, rng=[16, 84]) / 2, bins=rad_bins
    )[0]
    N = np.isfinite(S)
    if not np.all(N):
        S[np.logical_not(N)] = np.interp(profR[np.logical_not(N)], profR[N], S[N])
    parameters["I(R)"].set_value(np.log10(np.abs(I)), override_locked=True)
    parameters["I(R)"].set_uncertainty(
        S / (np.abs(I) * np.log(10)), override_locked=True
    )


@torch.no_grad()
@ignore_numpy_warnings
@select_target
@default_internal
def spline_segment_initialize(
    self, target=None, parameters=None, segments=1, symmetric=True, **kwargs
):
    super(self.__class__, self).initialize(target=target, parameters=parameters)

    if parameters["I(R)"].value is not None and parameters["I(R)"].prof is not None:
        return

    # Create the I(R) profile radii if needed
    if parameters["I(R)"].prof is None:
        new_prof = [0, 2 * target.pixel_length]
        while new_prof[-1] < torch.max(self.window.shape / 2):
            new_prof.append(
                new_prof[-1] + torch.max(2 * target.pixel_length, new_prof[-1] * 0.2)
            )
        new_prof.pop()
        new_prof.pop()
        new_prof.append(torch.sqrt(torch.sum((self.window.shape / 2) ** 2)))
        parameters["I(R)"].set_profile(new_prof)

    parameters["I(R)"].set_value(
        np.zeros((segments, len(parameters["I(R)"].prof))), override_locked=True
    )
    parameters["I(R)"].set_uncertainty(
        np.zeros((segments, len(parameters["I(R)"].prof))), override_locked=True
    )
    profR = parameters["I(R)"].prof.detach().cpu().numpy()
    target_area = target[self.window]
    Coords = target_area.get_coordinate_meshgrid()
    X, Y = Coords - parameters["center"].value[..., None, None]
    X, Y = self.transform_coordinates(X, Y, target, parameters)
    R = self.radius_metric(X, Y, target, parameters).detach().cpu().numpy()
    T = self.angular_metric(X, Y, target, parameters).detach().cpu().numpy()
    rad_bins = [profR[0]] + list((profR[:-1] + profR[1:]) / 2) + [profR[-1] * 100]
    raveldat = target_area.data.detach().cpu().numpy().ravel()
    for s in range(segments):
        if segments % 2 == 0 and symmetric:
            angles = (T - (s * np.pi / segments)) % np.pi
            TCHOOSE = np.logical_or(
                angles < (np.pi / segments), angles >= (np.pi * (1 - 1 / segments))
            )
        elif segments % 2 == 1 and symmetric:
            angles = (T - (s * np.pi / segments)) % (2 * np.pi)
            TCHOOSE = np.logical_or(
                angles < (np.pi / segments), angles >= (np.pi * (2 - 1 / segments))
            )
            angles = (T - (np.pi + r * np.pi / segments)) % (2 * np.pi)
            TCHOOSE = np.logical_or(
                TCHOOSE,
                np.logical_or(
                    angles < (np.pi / segments), angles >= (np.pi * (2 - 1 / segments))
                ),
            )
        elif segments % 2 == 0 and not symmetric:
            angles = (T - (s * 2 * np.pi / segments)) % (2 * np.pi)
            TCHOOSE = torch.logical_or(
                angles < (2 * np.pi / segments),
                angles >= (2 * np.pi * (1 - 1 / segments)),
            )
        else:
            angles = (T - (s * 2 * np.pi / segments)) % (2 * np.pi)
            TCHOOSE = torch.logical_or(
                angles < (2 * np.pi / segments), angles >= (np.pi * (2 - 1 / segments))
            )
        TCHOOSE = TCHOOSE.ravel()
        I = (
            binned_statistic(
                R.ravel()[TCHOOSE], raveldat[TCHOOSE], statistic="median", bins=rad_bins
            )[0]
        ) / target.pixel_area.item()
        N = np.isfinite(I)
        if not np.all(N):
            I[np.logical_not(N)] = np.interp(profR[np.logical_not(N)], profR[N], I[N])
        S = binned_statistic(
            R.ravel(),
            raveldat,
            statistic=lambda d: iqr(d, rng=[16, 84]) / 2,
            bins=rad_bins,
        )[0]
        N = np.isfinite(S)
        if not np.all(N):
            S[np.logical_not(N)] = np.interp(profR[np.logical_not(N)], profR[N], S[N])
        parameters["I(R)"].set_value(np.log10(np.abs(I)), override_locked=True, index=s)
        parameters["I(R)"].set_uncertainty(
            S / (np.abs(I) * np.log(10)), override_locked=True, index=s
        )


@default_internal
def spline_radial_model(self, R, image=None, parameters=None):
    return (
        spline_torch(
            R,
            parameters["I(R)"].prof,
            parameters["I(R)"].value,
            extend=self.extend_profile,
        )
        * image.pixel_area
    )


@default_internal
def spline_iradial_model(self, i, R, image=None, parameters=None):
    return (
        spline_torch(
            R,
            parameters["I(R)"].prof,
            parameters["I(R)"].value[i],
            extend=self.extend_profile,
        )
        * image.pixel_area
    )

# RelSpline
######################################################################
@torch.no_grad()
@ignore_numpy_warnings
@select_target
@default_internal
def relspline_initialize(self, target=None, parameters=None, **kwargs):
    super(self.__class__, self).initialize(target=target, parameters=parameters)

    target_area = target[self.window]
    if parameters["I0"].value is None:
        center = target_area.world_to_pixel(parameters["center"].value)
        flux = target_area.data[center[1].int().item(), center[0].int().item()]
        parameters["I0"].set_value(torch.log10(torch.abs(flux) / target_area.pixel_area), override_locked = True)
        parameters["I0"].set_uncertainty(0.01, override_locked = True)
        
    if parameters["dI(R)"].value is not None and parameters["dI(R)"].prof is not None:
        return

    # Create the I(R) profile radii if needed
    if parameters["dI(R)"].prof is None:
        new_prof = [2 * target.pixel_length]
        while new_prof[-1] < torch.max(self.window.shape / 2):
            new_prof.append(
                new_prof[-1] + torch.max(2 * target.pixel_length, new_prof[-1] * 0.2)
            )
        new_prof.pop()
        new_prof.pop()
        new_prof.append(torch.sqrt(torch.sum((self.window.shape / 2) ** 2)))
        parameters["dI(R)"].set_profile(new_prof)

    profR = parameters["dI(R)"].prof.detach().cpu().numpy()
        
    Coords = target_area.get_coordinate_meshgrid()
    X, Y = Coords - parameters["center"].value[..., None, None]
    X, Y = self.transform_coordinates(X, Y, target, parameters)
    R = self.radius_metric(X, Y, target, parameters).detach().cpu().numpy()
    rad_bins = [profR[0]] + list((profR[:-1] + profR[1:]) / 2) + [profR[-1] * 100]
    raveldat = target_area.data.detach().cpu().numpy().ravel()

    I = (
        binned_statistic(R.ravel(), raveldat, statistic="median", bins=rad_bins)[0]
    ) / target.pixel_area.item()
    N = np.isfinite(I)
    if not np.all(N):
        I[np.logical_not(N)] = np.interp(profR[np.logical_not(N)], profR[N], I[N])
    if I[-1] >= I[-2]:
        I[-1] = I[-2] / 2
    S = binned_statistic(
        R.ravel(), raveldat, statistic=lambda d: iqr(d, rng=[16, 84]) / 2, bins=rad_bins
    )[0]
    N = np.isfinite(S)
    if not np.all(N):
        S[np.logical_not(N)] = np.interp(profR[np.logical_not(N)], profR[N], S[N])
    parameters["dI(R)"].set_value(np.log10(np.abs(I)) - parameters["I0"].value.item(), override_locked=True)
    parameters["dI(R)"].set_uncertainty(
        S / (np.abs(I) * np.log(10)), override_locked=True
    )

@default_internal
def relspline_radial_model(self, R, image=None, parameters=None):
    return (
        spline_torch(
            R,
            torch.cat((torch.zeros_like(parameters["I0"].value).unsqueeze(-1),parameters["dI(R)"].prof)),
            torch.cat((parameters["I0"].value.unsqueeze(-1), parameters["I0"].value + parameters["dI(R)"].value)),
            extend=self.extend_profile,
        )
        * image.pixel_area
    )

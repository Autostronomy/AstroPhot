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
from ..utils.conversions.coordinates import (
    Rotate_Cartesian,
    coord_to_index,
    index_to_coord,
)
from ..utils.conversions.functions import sersic_I0_to_flux_np, sersic_flux_to_I0_torch
from ..image import Image_List, Target_Image, Model_Image_List, Target_Image_List
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
            for sub_image in image:
                if sub_image.target_identity == self.target.identity:
                    send_image = sub_image
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
    icenter = coord_to_index(
        parameters["center"].value[0], parameters["center"].value[1], target_area
    )
    # Collect isophotes for 1D fit
    iso_info = isophotes(
        target_area.data.detach().cpu().numpy() - edge_average,
        (icenter[1].item(), icenter[0].item()),
        threshold=3 * edge_scatter,
        pa=parameters["PA"].value.detach().cpu().item() if "PA" in parameters else 0.0,
        q=parameters["q"].value.detach().cpu().item() if "q" in parameters else 1.0,
        n_isophotes=15,
    )
    R = np.array(list(iso["R"] for iso in iso_info)) * target.pixelscale.item()
    flux = (
        np.array(list(iso["flux"] for iso in iso_info)) / target.pixelscale.item() ** 2
    )
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
            target_area.data[:, 0],
            target_area.data[:, -1],
            target_area.data[0, :],
            target_area.data[-1, :],
        )
    )
    edge_average = np.median(edge)
    edge_scatter = iqr(edge, rng=(16, 84)) / 2
    # Convert center coordinates to target area array indices
    icenter = coord_to_index(
        model["center"].value[0], model["center"].value[1], target_area
    )
    iso_info = isophotes(
        target_area.data.detach().cpu().numpy() - edge_average,
        (icenter[1].item(), icenter[0].item()),
        threshold=3 * edge_scatter,
        pa=model["PA"].value.detach().cpu().item() if "PA" in model else 0.0,
        q=model["q"].value.detach().cpu().item() if "q" in model else 1.0,
        n_isophotes=15,
        more=True,
    )
    R = np.array(list(iso["R"] for iso in iso_info)) * target.pixelscale.item()
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
                - (model["PA"].value.detach().cpu().item() + r * np.pi / segments)
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
                / target.pixelscale.item() ** 2
            )
        flux = np.array(flux)
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
        (10 ** parameters["Ie"].value) * image.pixelscale ** 2,
    )


@default_internal
def exponential_iradial_model(self, i, R, image=None, parameters=None):
    return exponential_torch(
        R,
        parameters["Re"].value[i],
        (10 ** parameters["Ie"].value[i]) * image.pixelscale ** 2,
    )


# Sersic
######################################################################
@default_internal
def sersic_radial_model(self, R, image=None, parameters=None):
    return sersic_torch(
        R,
        parameters["n"].value,
        parameters["Re"].value,
        (10 ** parameters["Ie"].value) * image.pixelscale ** 2,
    )


@default_internal
def sersic_iradial_model(self, i, R, image=None, parameters=None):
    return sersic_torch(
        R,
        parameters["n"].value[i],
        parameters["Re"].value[i],
        (10 ** parameters["Ie"].value[i]) * image.pixelscale ** 2,
    )


# Moffat
######################################################################
@default_internal
def moffat_radial_model(self, R, image=None, parameters=None):
    return moffat_torch(
        R,
        parameters["n"].value,
        parameters["Rd"].value,
        (10 ** parameters["I0"].value) * image.pixelscale ** 2,
    )


@default_internal
def moffat_iradial_model(self, i, R, image=None, parameters=None):
    return moffat_torch(
        R,
        parameters["n"].value[i],
        parameters["Rd"].value[i],
        (10 ** parameters["I0"].value[i]) * image.pixelscale ** 2,
    )


# Nuker Profile
######################################################################
@default_internal
def nuker_radial_model(self, R, image=None, parameters=None):
    return nuker_torch(
        R,
        parameters["Rb"].value,
        (10 ** parameters["Ib"].value) * image.pixelscale ** 2,
        parameters["alpha"].value,
        parameters["beta"].value,
        parameters["gamma"].value,
    )


@default_internal
def nuker_iradial_model(self, i, R, image=None, parameters=None):
    return nuker_torch(
        R,
        parameters["Rb"].value[i],
        (10 ** parameters["Ib"].value[i]) * image.pixelscale ** 2,
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
        (10 ** parameters["flux"].value) * image.pixelscale ** 2,
    )


@default_internal
def gaussian_iradial_model(self, i, R, image=None, parameters=None):
    return gaussian_torch(
        R,
        parameters["sigma"].value[i],
        (10 ** parameters["flux"].value[i]) * image.pixelscale ** 2,
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
        new_prof = [0, 2 * target.pixelscale]
        while new_prof[-1] < torch.max(self.window.shape / 2):
            new_prof.append(
                new_prof[-1] + torch.max(2 * target.pixelscale, new_prof[-1] * 0.2)
            )
        new_prof.pop()
        new_prof.pop()
        new_prof.append(torch.sqrt(torch.sum((self.window.shape / 2) ** 2)))
        parameters["I(R)"].set_profile(new_prof)

    profR = parameters["I(R)"].prof.detach().cpu().numpy()
    target_area = target[self.window]
    X, Y = target_area.get_coordinate_meshgrid_torch(
        parameters["center"].value[0], parameters["center"].value[1]
    )
    X, Y = self.transform_coordinates(X, Y, target, parameters)
    R = self.radius_metric(X, Y, target, parameters).detach().cpu().numpy()
    rad_bins = [profR[0]] + list((profR[:-1] + profR[1:]) / 2) + [profR[-1] * 100]
    raveldat = target_area.data.detach().cpu().numpy().ravel()
    I = (
        binned_statistic(R.ravel(), raveldat, statistic="median", bins=rad_bins)[0]
        / target_area.pixelscale.item() ** 2
    )
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
        new_prof = [0, 2 * target.pixelscale]
        while new_prof[-1] < torch.max(self.window.shape / 2):
            new_prof.append(
                new_prof[-1] + torch.max(2 * target.pixelscale, new_prof[-1] * 0.2)
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
    X, Y = target_area.get_coordinate_meshgrid_torch(
        parameters["center"].value[0], parameters["center"].value[1]
    )
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
            / target_area.pixelscale.item() ** 2
        )
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
    return spline_torch(
        R,
        parameters["I(R)"].prof,
        parameters["I(R)"].value,
        image.pixelscale ** 2,
        extend=self.extend_profile,
    )


@default_internal
def spline_iradial_model(self, i, R, image=None, parameters=None):
    return spline_torch(
        R,
        parameters["I(R)"].prof,
        parameters["I(R)"].value[i],
        image.pixelscale ** 2,
        extend=self.extend_profile,
    )

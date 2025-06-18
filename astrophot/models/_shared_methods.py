from scipy.stats import binned_statistic, iqr
import numpy as np
import torch
from scipy.optimize import minimize

from ..utils.initialize import isophotes
from ..utils.decorators import ignore_numpy_warnings, default_internal
from . import func
from .. import AP_config


def _sample_image(image, transform):
    dat = image.data.npvalue.copy()
    # Fill masked pixels
    if image.has_mask:
        mask = image.mask.detach().cpu().numpy()
        dat[mask] = np.median(dat[~mask])
    # Subtract median of edge pixels to avoid effect of nearby sources
    edge = np.concatenate((dat[:, 0], dat[:, -1], dat[0, :], dat[-1, :]))
    dat -= np.median(edge)
    # Get the radius of each pixel relative to object center
    x, y = transform(*image.coordinate_center_meshgrid())

    R = torch.sqrt(x**2 + y**2).detach().cpu().numpy()

    # Bin fluxes by radius
    if rad_bins is None:
        rad_bins = np.logspace(np.log10(R.min() * 0.9), np.log10(R.max() * 1.1), 11)
    else:
        rad_bins = np.array(rad_bins)
    raveldat = dat.ravel()
    I = (
        binned_statistic(R, raveldat, statistic="median", bins=rad_bins)[0]
    ) / image.pixel_area.item()
    sigma = lambda d: iqr(d, rng=[16, 84]) / 2
    S = binned_statistic(R, raveldat, statistic=sigma, bins=rad_bins)[0] / image.pixel_area.item()
    R = (rad_bins[:-1] + rad_bins[1:]) / 2

    # Ensure enough values are positive
    I[~np.isfinite(I)] = np.median(I[np.isfinite(I)])
    if np.sum(I > 0) <= 3:
        I = I - np.min(I)
    I[I <= 0] = np.min(I[I > 0])
    # Ensure decreasing brightness with radius in outer regions
    for i in range(5, len(I)):
        if I[i] >= I[i - 1]:
            I[i] = I[i - 1] * 0.9
    # Convert to log scale
    S = S / (I * np.log(10))
    I = np.log10(I)
    # Ensure finite
    N = np.isfinite(I)
    if not np.all(N):
        I[~N] = np.interp(R[~N], R[N], I[N])
    N = np.isfinite(S)
    if not np.all(N):
        S[~N] = np.abs(np.interp(R[~N], R[N], S[N]))

    return R, I, S


# General parametric
######################################################################
@torch.no_grad()
@ignore_numpy_warnings
def parametric_initialize(model, target, prof_func, params, x0_func):
    if all(list(model[param].value is not None for param in params)):
        return

    # Get the sub-image area corresponding to the model image
    R, I, S = _sample_image(target, model.transform_coordinates)

    x0 = list(x0_func(model, R, I))
    for i, param in enumerate(params):
        x0[i] = x0[i] if model[param].value is None else model[param].npvalue

    def optim(x, r, f, u):
        residual = ((f - np.log10(prof_func(r, *x))) / u) ** 2
        N = np.argsort(residual)
        return np.mean(residual[N][:-2])

    res = minimize(optim, x0=x0, args=(R, I, S), method="Nelder-Mead")
    if not res.success:
        if AP_config.ap_verbose >= 2:
            AP_config.ap_logger.warning(
                f"initialization fit not successful for {model.name}, falling back to defaults"
            )
    else:
        x0 = res.x

    reses = []
    for i in range(10):
        N = np.random.randint(0, len(R), len(R))
        reses.append(minimize(optim, x0=x0, args=(R[N], I[N], S[N]), method="Nelder-Mead"))
    for param, x0x in zip(params, x0):
        if model[param].value is None:
            model[param].value = x0x
        if model[param].uncertainty is None:
            model[param].uncertainty = np.std(
                list(subres.x[params.index(param)] for subres in reses)
            )


@torch.no_grad()
@ignore_numpy_warnings
def parametric_segment_initialize(
    model=None,
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
    target_dat = target_area.data.detach().cpu().numpy()
    if target_area.has_mask:
        mask = target_area.mask.detach().cpu().numpy()
        target_dat[mask] = np.median(target_dat[np.logical_not(mask)])
    edge = np.concatenate(
        (
            target_dat[:, 0],
            target_dat[:, -1],
            target_dat[0, :],
            target_dat[-1, :],
        )
    )
    edge_average = np.median(edge)
    edge_scatter = iqr(edge, rng=(16, 84)) / 2
    # Convert center coordinates to target area array indices
    icenter = target_area.plane_to_pixel(model["center"].value)

    iso_info = isophotes(
        target_dat - edge_average,
        (icenter[1].item(), icenter[0].item()),
        threshold=3 * edge_scatter,
        pa=(model["PA"].value - target.north).item() if "PA" in model else 0.0,
        q=model["q"].value.item() if "q" in model else 1.0,
        n_isophotes=15,
        more=True,
    )
    R = np.array(list(iso["R"] for iso in iso_info)) * target.pixel_length.item()
    was_none = list(False for i in range(len(params)))
    val = {}
    unc = {}
    for i, p in enumerate(params):
        if model[p].value is None:
            was_none[i] = True
            val[p] = np.zeros(segments)
            unc[p] = np.zeros(segments)
    for r in range(segments):
        flux = []
        for iso in iso_info:
            modangles = (
                iso["angles"]
                - ((model["PA"].value - target.north).detach().cpu().item() + r * np.pi / segments)
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
            x0[i] = x0[i] if was_none[i] else model[param].value.detach().cpu().numpy()[r]
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
                val[param][r] = res.x[i] if res.success else x0[i]
                if force_uncertainty is None and model[param].uncertainty is None:
                    unc[r] = np.std(list(subres.x[params.index(param)] for subres in reses))
                elif force_uncertainty is not None:
                    unc[r] = force_uncertainty[params.index(param)][r]

            with Param_Unlock(model[param]), Param_SoftLimits(model[param]):
                model[param].value = val[param]
                model[param].uncertainty = unc[param]


# # Spline
# ######################################################################
# @torch.no_grad()
# @ignore_numpy_warnings
# @select_target
# @default_internal
# def spline_initialize(self, target=None, parameters=None, **kwargs):
#     super(self.__class__, self).initialize(target=target, parameters=parameters)

#     if parameters["I(R)"].value is not None and parameters["I(R)"].prof is not None:
#         return

#     # Create the I(R) profile radii if needed
#     if parameters["I(R)"].prof is None:
#         new_prof = [0, 2 * target.pixel_length]
#         while new_prof[-1] < torch.max(self.window.shape / 2):
#             new_prof.append(new_prof[-1] + torch.max(2 * target.pixel_length, new_prof[-1] * 0.2))
#         new_prof.pop()
#         new_prof.pop()
#         new_prof.append(torch.sqrt(torch.sum((self.window.shape / 2) ** 2)))
#         parameters["I(R)"].prof = new_prof

#     profR = parameters["I(R)"].prof.detach().cpu().numpy()
#     target_area = target[self.window]
#     R, I, S = _sample_image(
#         target_area,
#         self.transform_coordinates,
#         self.radius_metric,
#         parameters,
#         rad_bins=[profR[0]] + list((profR[:-1] + profR[1:]) / 2) + [profR[-1] * 100],
#     )
#     with Param_Unlock(parameters["I(R)"]), Param_SoftLimits(parameters["I(R)"]):
#         parameters["I(R)"].value = I
#         parameters["I(R)"].uncertainty = S


# @torch.no_grad()
# @ignore_numpy_warnings
# @select_target
# @default_internal
# def spline_segment_initialize(
#     self, target=None, parameters=None, segments=1, symmetric=True, **kwargs
# ):
#     super(self.__class__, self).initialize(target=target, parameters=parameters)

#     if parameters["I(R)"].value is not None and parameters["I(R)"].prof is not None:
#         return

#     # Create the I(R) profile radii if needed
#     if parameters["I(R)"].prof is None:
#         new_prof = [0, 2 * target.pixel_length]
#         while new_prof[-1] < torch.max(self.window.shape / 2):
#             new_prof.append(new_prof[-1] + torch.max(2 * target.pixel_length, new_prof[-1] * 0.2))
#         new_prof.pop()
#         new_prof.pop()
#         new_prof.append(torch.sqrt(torch.sum((self.window.shape / 2) ** 2)))
#         parameters["I(R)"].prof = new_prof

#     profR = parameters["I(R)"].prof.detach().cpu().numpy()
#     target_area = target[self.window]
#     target_dat = target_area.data.detach().cpu().numpy()
#     if target_area.has_mask:
#         mask = target_area.mask.detach().cpu().numpy()
#         target_dat[mask] = np.median(target_dat[np.logical_not(mask)])
#     Coords = target_area.get_coordinate_meshgrid()
#     X, Y = Coords - parameters["center"].value[..., None, None]
#     X, Y = self.transform_coordinates(X, Y, target, parameters)
#     R = self.radius_metric(X, Y, target, parameters).detach().cpu().numpy()
#     T = self.angular_metric(X, Y, target, parameters).detach().cpu().numpy()
#     rad_bins = [profR[0]] + list((profR[:-1] + profR[1:]) / 2) + [profR[-1] * 100]
#     raveldat = target_dat.ravel()
#     val = np.zeros((segments, len(parameters["I(R)"].prof)))
#     unc = np.zeros((segments, len(parameters["I(R)"].prof)))
#     for s in range(segments):
#         if segments % 2 == 0 and symmetric:
#             angles = (T - (s * np.pi / segments)) % np.pi
#             TCHOOSE = np.logical_or(
#                 angles < (np.pi / segments), angles >= (np.pi * (1 - 1 / segments))
#             )
#         elif segments % 2 == 1 and symmetric:
#             angles = (T - (s * np.pi / segments)) % (2 * np.pi)
#             TCHOOSE = np.logical_or(
#                 angles < (np.pi / segments), angles >= (np.pi * (2 - 1 / segments))
#             )
#             angles = (T - (np.pi + s * np.pi / segments)) % (2 * np.pi)
#             TCHOOSE = np.logical_or(
#                 TCHOOSE,
#                 np.logical_or(angles < (np.pi / segments), angles >= (np.pi * (2 - 1 / segments))),
#             )
#         elif segments % 2 == 0 and not symmetric:
#             angles = (T - (s * 2 * np.pi / segments)) % (2 * np.pi)
#             TCHOOSE = torch.logical_or(
#                 angles < (2 * np.pi / segments),
#                 angles >= (2 * np.pi * (1 - 1 / segments)),
#             )
#         else:
#             angles = (T - (s * 2 * np.pi / segments)) % (2 * np.pi)
#             TCHOOSE = torch.logical_or(
#                 angles < (2 * np.pi / segments), angles >= (np.pi * (2 - 1 / segments))
#             )
#         TCHOOSE = TCHOOSE.ravel()
#         I = (
#             binned_statistic(
#                 R.ravel()[TCHOOSE], raveldat[TCHOOSE], statistic="median", bins=rad_bins
#             )[0]
#         ) / target.pixel_area.item()
#         N = np.isfinite(I)
#         if not np.all(N):
#             I[np.logical_not(N)] = np.interp(profR[np.logical_not(N)], profR[N], I[N])
#         S = binned_statistic(
#             R.ravel(),
#             raveldat,
#             statistic=lambda d: iqr(d, rng=[16, 84]) / 2,
#             bins=rad_bins,
#         )[0]
#         N = np.isfinite(S)
#         if not np.all(N):
#             S[np.logical_not(N)] = np.interp(profR[np.logical_not(N)], profR[N], S[N])
#         val[s] = np.log10(np.abs(I))
#         unc[s] = S / (np.abs(I) * np.log(10))
#     with Param_Unlock(parameters["I(R)"]), Param_SoftLimits(parameters["I(R)"]):
#         parameters["I(R)"].value = val
#         parameters["I(R)"].uncertainty = unc

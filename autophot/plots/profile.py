from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, iqr

from .. import AP_config
from ..models import Warp_Galaxy
from ..utils.conversions.units import flux_to_sb
from .visuals import *

__all__ = [
    "galaxy_light_profile",
    "radial_median_profile",
    "ray_light_profile",
    "wedge_light_profile",
    "warp_phase_profile",
]


def galaxy_light_profile(
    fig,
    ax,
    model,
    rad_unit="arcsec",
    extend_profile=1.0,
    R0=0.0,
    resolution=1000,
    doassert=True,
    plot_kwargs={},
):
    xx = torch.linspace(
        R0,
        torch.max(model.window.shape / 2) * extend_profile,
        int(resolution),
        dtype=AP_config.ap_dtype,
        device=AP_config.ap_device,
    )
    flux = model.radial_model(xx).detach().cpu().numpy()
    if model.target.zeropoint is not None:
        yy = flux_to_sb(
            flux, model.target.pixel_area.item(), model.target.zeropoint.item()
        )
    else:
        yy = np.log10(flux)

    kwargs = {
        "linewidth": 2,
        "color": main_pallet["primary1"],
        "label": f"{model.name} profile",
    }
    kwargs.update(plot_kwargs)
    with torch.no_grad():
        ax.plot(
            xx.detach().cpu().numpy(),
            yy,
            **kwargs,
        )

    if model.target.zeropoint is not None:
        ax.set_ylabel("Surface Brightness [mag/arcsec$^2$]")
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
    else:
        ax.set_ylabel("log$_{10}$(flux/arcsec$^2$)")
    ax.set_xlabel(f"Radius [{rad_unit}]")
    ax.set_xlim([R0, None])
    return fig, ax


def radial_median_profile(
    fig,
    ax,
    model: "AutoPhot_Model",
    count_limit: int = 10,
    return_profile: bool = False,
    rad_unit: str = "arcsec",
    doassert: bool = True,
    plot_kwargs: dict = {},
):
    """Plot an SB profile by taking flux median at each radius.

    Using the coordinate transforms defined by the model object,
    assigns a radius to each pixel then bins the pixel-radii and
    computes the median in each bin. This gives a simple
    representation of the image data if one were to simply average the
    pixels along isophotes.

    Args:
      fig: matplotlib figure object
      ax: matplotlib axis object
      model (AutoPhot_Model): Model object from which to determine the radial binning. Also provides the target image to extract the data
      count_limit (int): The limit of pixels in a bin, below which uncertainties are not computed. Default: 10
      return_profile (bool): Instead of just returning the fig and ax object, will return the extracted profile formatted as: Rbins (the radial bin edges), medians (the median in each bin), scatter (the 16-84 quartile range / 2), count (the number of pixels in each bin). Default: False
      rad_unit (str): The name of the physical radius units. Default: "arcsec"
      doassert (bool): If any requirements are imposed on which kind of profile can be plotted, this activates them. Default: True

    """

    Rlast_phys = torch.max(model.window.shape / 2).item()
    Rlast_pix = Rlast_phys / model.target.pixel_length.item()

    Rbins = [0.0]
    while Rbins[-1] < Rlast_pix:
        Rbins.append(Rbins[-1] + max(2, Rbins[-1] * 0.1))
    Rbins = np.array(Rbins)

    with torch.no_grad():
        image = model.target[model.window]
        X, Y = image.get_coordinate_meshgrid() - model["center"].value[..., None, None]
        X, Y = model.transform_coordinates(X, Y)
        R = model.radius_metric(X, Y)
        R = R.detach().cpu().numpy()

    count, bins, binnum = binned_statistic(
        R.ravel(),
        image.data.detach().cpu().numpy().ravel(),
        statistic="count",
        bins=Rbins,
    )

    stat, bins, binnum = binned_statistic(
        R.ravel(),
        image.data.detach().cpu().numpy().ravel(),
        statistic="median",
        bins=Rbins,
    )
    stat[count == 0] = np.nan

    scat, bins, binnum = binned_statistic(
        R.ravel(),
        image.data.detach().cpu().numpy().ravel(),
        statistic=partial(iqr, rng=(16, 84)),
        bins=Rbins,
    )
    scat[count > count_limit] /= 2 * np.sqrt(count[count > count_limit])
    scat[count <= count_limit] = 0

    if model.target.zeropoint is not None:
        stat = flux_to_sb(
            stat, model.target.pixel_area.item(), model.target.zeropoint.item()
        )
        ax.set_ylabel("Surface Brightness [mag/arcsec$^2$]")
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
    else:
        stat = np.log10(stat)
        ax.set_ylabel("log$_{10}$(flux/arcsec^2)")

    kwargs = {
        "linewidth": 0,
        "elinewidth": 1,
        "color": main_pallet["primary2"],
        "label": f"data profile",
    }
    kwargs.update(plot_kwargs)
    ax.errorbar(
        (Rbins[:-1] + Rbins[1:]) / 2,
        stat,
        yerr=scat,
        fmt=".",
        **kwargs,
    )
    ax.set_xlabel(f"Radius [{rad_unit}]")

    if return_profile:
        return Rbins, stat, scat, count
    return fig, ax


def ray_light_profile(
    fig,
    ax,
    model,
    rad_unit="arcsec",
    extend_profile=1.0,
    resolution=1000,
    doassert=True,
):
    xx = torch.linspace(
        0,
        torch.max(model.window.shape / 2) * extend_profile,
        int(resolution),
        dtype=AP_config.ap_dtype,
        device=AP_config.ap_device,
    )
    for r in range(model.rays):
        if model.rays <= 5:
            col = main_pallet[f"primary{r+1}"]
        else:
            col = cmap_grad(r / model.rays)
        with torch.no_grad():
            ax.plot(
                xx.detach().cpu().numpy(),
                np.log10(model.iradial_model(r, xx).detach().cpu().numpy()),
                linewidth=2,
                color=col,
                label=f"{model.name} profile {r}",
            )
    ax.set_ylabel("log$_{10}$(flux)")
    ax.set_xlabel(f"Radius [{rad_unit}]")

    return fig, ax


def wedge_light_profile(
    fig,
    ax,
    model,
    rad_unit="arcsec",
    extend_profile=1.0,
    resolution=1000,
    doassert=True,
):
    xx = torch.linspace(
        0,
        torch.max(model.window.shape / 2) * extend_profile,
        int(resolution),
        dtype=AP_config.ap_dtype,
        device=AP_config.ap_device,
    )
    for r in range(model.wedges):
        if model.wedges <= 5:
            col = main_pallet[f"primary{r+1}"]
        else:
            col = cmap_grad(r / model.wedges)
        with torch.no_grad():
            ax.plot(
                xx.detach().cpu().numpy(),
                np.log10(model.iradial_model(r, xx).detach().cpu().numpy()),
                linewidth=2,
                color=col,
                label=f"{model.name} profile {r}",
            )
    ax.set_ylabel("log$_{10}$(flux)")
    ax.set_xlabel(f"Radius [{rad_unit}]")

    return fig, ax


def warp_phase_profile(fig, ax, model, rad_unit="arcsec", doassert=True):
    if doassert:
        assert isinstance(model, Warp_Galaxy)

    ax.plot(
        model.profR,
        model["q(R)"].value.detach().cpu().numpy(),
        linewidth=2,
        color=main_pallet["primary1"],
        label=f"{model.name} axis ratio",
    )
    ax.plot(
        model.profR,
        model["PA(R)"].detach().cpu().numpy() / np.pi,
        linewidth=2,
        color=main_pallet["secondary1"],
        label=f"{model.name} position angle",
    )
    ax.set_ylim([0, 1])
    ax.set_ylabel("q [b/a], PA [rad/$\\pi$]")
    ax.set_xlabel(f"Radius [{rad_unit}]")

    return fig, ax

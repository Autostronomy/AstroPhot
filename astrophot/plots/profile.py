from functools import partial
from typing import Literal

import numpy as np
import torch
from scipy.stats import binned_statistic, iqr

from .. import AP_config
from ..models import Model

# from ..models import Warp_Galaxy
from ..utils.conversions.units import flux_to_sb
from .visuals import *

__all__ = [
    "radial_light_profile",
    "radial_median_profile",
    "ray_light_profile",
    "wedge_light_profile",
    "warp_phase_profile",
]


def radial_light_profile(
    fig,
    ax,
    model: Model,
    rad_unit="arcsec",
    extend_profile=1.0,
    R0=0.0,
    resolution=1000,
    plot_kwargs={},
):
    xx = torch.linspace(
        R0,
        max(model.window.shape)
        * model.target.pixelscale.detach().cpu().numpy()
        * extend_profile
        / 2,
        int(resolution),
        dtype=AP_config.ap_dtype,
        device=AP_config.ap_device,
    )
    flux = model.radial_model(xx, params=()).detach().cpu().numpy()
    if model.target.zeropoint is not None:
        yy = flux_to_sb(flux, 1.0, model.target.zeropoint.item())
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
    model: Model,
    count_limit: int = 10,
    return_profile: bool = False,
    rad_unit: str = "arcsec",
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
      model (AstroPhot_Model): Model object from which to determine the radial binning. Also provides the target image to extract the data
      count_limit (int): The limit of pixels in a bin, below which uncertainties are not computed. Default: 10
      return_profile (bool): Instead of just returning the fig and ax object, will return the extracted profile formatted as: Rbins (the radial bin edges), medians (the median in each bin), scatter (the 16-84 quartile range / 2), count (the number of pixels in each bin). Default: False
      rad_unit (str): The name of the radius units to plot. If you select "pixel" then the plot will work in pixel units (physical radii divided by pixelscale) if you choose any other string then it will remain in the physical units of the image and the axis label will be whatever you set the value to. Default: "arcsec". Options: "arcsec", "pixel"
      bin_scale (float): The geometric scaling factor for the binning, each bin will be this much larger than the previous. Default: 0.1
      min_bin_width (float): The minimum width of a bin in pixel units, default is 2 so that each bin will have some data to compute the median with. Default: 2
      doassert (bool): If any requirements are imposed on which kind of profile can be plotted, this activates them. Default: True

    """

    Rlast_pix = max(model.window.shape) / 2
    Rlast_phys = Rlast_pix * model.target.pixelscale.item()

    Rbins = [0.0]
    while Rbins[-1] < Rlast_phys:
        Rbins.append(Rbins[-1] + max(2 * model.target.pixelscale.item(), Rbins[-1] * 0.1))
    Rbins = np.array(Rbins)
    Rbins = Rbins * model.target.pixel_length.item()  # back to physical units

    with torch.no_grad():
        image = model.target[model.window]
        x, y = image.coordinate_center_meshgrid()
        x, y = model.transform_coordinates(x, y, params=())
        R = (x**2 + y**2).sqrt()
        R = R.detach().cpu().numpy()

    dat = image.data.detach().cpu().numpy()
    count, bins, binnum = binned_statistic(
        R.ravel(),
        dat.ravel(),
        statistic="count",
        bins=Rbins,
    )

    stat, bins, binnum = binned_statistic(
        R.ravel(),
        dat.ravel(),
        statistic="median",
        bins=Rbins,
    )
    stat[count == 0] = np.nan

    scat, bins, binnum = binned_statistic(
        R.ravel(),
        dat.ravel(),
        statistic=partial(iqr, rng=(16, 84)),
        bins=Rbins,
    )
    scat[count > count_limit] /= 2 * np.sqrt(count[count > count_limit])
    scat[count <= count_limit] = 0

    if model.target.zeropoint is not None:
        stat = flux_to_sb(stat, model.target.pixel_area.item(), model.target.zeropoint.item())
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
        "label": "data profile",
        **plot_kwargs,
    }
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
    model: Model,
    rad_unit="arcsec",
    extend_profile=1.0,
    resolution=1000,
):
    xx = torch.linspace(
        0,
        max(model.window.shape) * model.target.pixelscale * extend_profile / 2,
        int(resolution),
        dtype=AP_config.ap_dtype,
        device=AP_config.ap_device,
    )
    for r in range(model.segments):
        if model.segments <= 3:
            col = main_pallet[f"primary{r+1}"]
        else:
            col = cmap_grad(r / model.segments)
        with torch.no_grad():
            ax.plot(
                xx.detach().cpu().numpy(),
                np.log10(model.iradial_model(r, xx, params=()).detach().cpu().numpy()),
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
    model: Model,
    rad_unit="arcsec",
    extend_profile=1.0,
    resolution=1000,
):
    xx = torch.linspace(
        0,
        max(model.window.shape) * model.target.pixelscale * extend_profile / 2,
        int(resolution),
        dtype=AP_config.ap_dtype,
        device=AP_config.ap_device,
    )
    for r in range(model.segments):
        if model.segments <= 3:
            col = main_pallet[f"primary{r+1}"]
        else:
            col = cmap_grad(r / model.segments)
        with torch.no_grad():
            ax.plot(
                xx.detach().cpu().numpy(),
                np.log10(model.iradial_model(r, xx, params=()).detach().cpu().numpy()),
                linewidth=2,
                color=col,
                label=f"{model.name} profile {r}",
            )
    ax.set_ylabel("log$_{10}$(flux)")
    ax.set_xlabel(f"Radius [{rad_unit}]")

    return fig, ax


def warp_phase_profile(fig, ax, model: Model, rad_unit="arcsec"):

    ax.plot(
        model.q_R.prof.detach().cpu().numpy(),
        model.q_R.npvalue,
        linewidth=2,
        color=main_pallet["primary1"],
        label=f"{model.name} axis ratio",
    )
    ax.plot(
        model.PA_R.prof.detach().cpu().numpy(),
        model.PA_R.npvalue / np.pi,
        linewidth=2,
        color=main_pallet["primary2"],
        label=f"{model.name} position angle/$\\pi$",
    )
    ax.set_ylim([0, 1])
    ax.set_ylabel("q [b/a], PA [rad/$\\pi$]")
    ax.set_xlabel(f"Radius [{rad_unit}]")

    return fig, ax

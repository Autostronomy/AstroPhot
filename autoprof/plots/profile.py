import matplotlib.pyplot as plt
from autoprof import models
from autoprof.utils.conversions.units import flux_to_sb
import numpy as np
from .visuals import *
import torch

__all__ = ["galaxy_light_profile", "ray_light_profile", "wedge_light_profile", "warp_phase_profile"]


def galaxy_light_profile(
        fig,
        ax,
        model,
        rad_unit="arcsec",
        extend_profile=1.0,
        R0 = 0.,    
        resolution=1000,
        doassert=True,
):

    xx = np.linspace(
        R0,
        np.max(model.fit_window.shape/2) * extend_profile,
        int(resolution),
    )
    flux = model.radial_model(torch.tensor(xx, dtype = model.dtype, device = model.device)).detach().cpu().numpy()
    if model.target.zeropoint is not None:
        yy = flux_to_sb(flux, model.target.pixelscale, model.target.zeropoint)
    else:
        yy = np.log10(flux)
    with torch.no_grad():
        ax.plot(
            xx,
            yy,
            linewidth=2,
            color=main_pallet["primary1"],
            label=f"{model.name} profile",
        )

    if model.target.zeropoint is not None:
        ax.set_ylabel("Surface Brightness")
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
    else:
        ax.set_ylabel("log$_{10}$(flux/arcsec^2)")
    ax.set_xlabel(f"Radius [{rad_unit}]")
    ax.set_xlim([R0,None])
    return fig, ax

def ray_light_profile(
        fig,
        ax,
        model,
        rad_unit="arcsec",
        extend_profile=1.0,
        resolution=1000,
        doassert=True
):
        
    xx = np.linspace(
        0,
        np.max(model.fit_window.shape/2) * extend_profile,
        int(resolution),
    )
    for r in range(model.rays):
        if model.rays <= 5:
            col = main_pallet[f"primary{r+1}"]
        else:
            col = cmap_grad(r / model.rays)
        with torch.no_grad():
            ax.plot(
                xx,
                np.log10(model.iradial_model(r, torch.tensor(xx,dtype = model.dtype, device = model.device)).detach().cpu().numpy()),
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
        doassert=True
):
        
    xx = np.linspace(
        0,
        np.max(model.fit_window.shape/2) * extend_profile,
        int(resolution),
    )
    for r in range(model.wedges):
        if model.wedges <= 5:
            col = main_pallet[f"primary{r+1}"]
        else:
            col = cmap_grad(r / model.wedges)
        with torch.no_grad():
            ax.plot(
                xx,
                np.log10(model.iradial_model(r, torch.tensor(xx,dtype = model.dtype, device = model.device)).detach().cpu().numpy()),
                linewidth=2,
                color=col,
                label=f"{model.name} profile {r}",
            )
    ax.set_ylabel("log$_{10}$(flux)")
    ax.set_xlabel(f"Radius [{rad_unit}]")

    return fig, ax

def warp_phase_profile(
    fig,
    ax,
    model,
    rad_unit="arcsec",
    doassert=True
):

    if doassert:
        assert isinstance(model, models.Warp_Model)

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

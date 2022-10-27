import matplotlib.pyplot as plt
from autoprof import models
import numpy as np
from .visuals import *
import torch

__all__ = ["galaxy_light_profile", "ray_light_profile", "warp_phase_profile"]


def galaxy_light_profile(
    fig,
    ax,
    model,
    rad_unit="arcsec",
    extend_profile=1.0,
    resolution=1000,
    doassert=True,
):

    if doassert:
        assert isinstance(model, models.Galaxy_Model)

    xx = np.linspace(
        0,
        np.sqrt(np.sum(model.fit_window.shape ** 2)) * extend_profile,
        int(resolution),
    )
    with torch.no_grad():
        ax.plot(
            xx,
            np.log10(model.radial_model(torch.tensor(xx)).detach().numpy()),
            linewidth=2,
            color=main_pallet["primary1"],
            label=f"{model.name} profile",
        )
    ax.set_ylabel("log$_{10}$(flux)")
    ax.set_xlabel(f"Radius [{rad_unit}]")

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
    if doassert:
        assert isinstance(model, models.Ray_Galaxy)

        
    xx = np.linspace(
        0,
        np.sqrt(np.sum(model.fit_window.shape ** 2)) * extend_profile,
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
                np.log10(model.iradial_model(r, torch.tensor(xx)).detach().numpy()),
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
        model["q(R)"].value.detach().numpy(),
        linewidth=2,
        color=main_pallet["primary1"],
        label=f"{model.name} axis ratio",
    )
    ax.plot(
        model.profR,
        model["PA(R)"].detach().numpy() / np.pi,
        linewidth=2,
        color=main_pallet["secondary1"],
        label=f"{model.name} position angle",
    )
    ax.set_ylim([0, 1])
    ax.set_ylabel("q [b/a], PA [rad/$\\pi$]")
    ax.set_xlabel(f"Radius [{rad_unit}]")

    return fig, ax

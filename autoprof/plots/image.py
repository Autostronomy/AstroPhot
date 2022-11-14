import matplotlib.pyplot as plt
import numpy as np
from .visuals import *
from scipy.stats import iqr
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch, LogStretch, HistEqStretch
import torch
from matplotlib.patches import Rectangle
from autoprof import models

__all__ = ["target_image", "model_image", "residual_image", "supermodel_boxes"]

def target_image(fig, ax, target, window = None, **kwargs):
    if window is None:
        window = target.window
    if target.masked:
        dat = np.ma.masked_array(target[window].data.detach().numpy(), mask = target[window].mask)
    else:
        dat = target[window].data.detach().numpy()
        
    sky = np.median(dat)
    noise = iqr(dat)/2
    vmin = sky - 5*noise
    vmax = sky + 5*noise
    im = ax.imshow(
        dat,
        origin="lower",
        cmap="Greys",
        extent = window.plt_extent,
        norm=ImageNormalize(stretch=HistEqStretch(dat[dat <= (sky + 3*noise)]), clip = False, vmax = sky + 3*noise, vmin = np.min(dat)),
    )
    ax.imshow(
        np.ma.masked_where(dat < (sky + 3*noise), dat), 
        origin="lower",
        cmap=cmap_grad,
        extent = window.plt_extent,
        norm=ImageNormalize(stretch=LogStretch(),clip = False),
        clim=[sky + 3 * noise, None],
        interpolation = 'none',
    )

    return fig, ax
    
def model_image(fig, ax, model, image = None, showcbar = True, **kwargs):

    if image is None:
        with torch.no_grad():
            model.sample(model.model_image)
        image = model.model_image.data.detach().numpy()

    imshow_kwargs = {
        "extent": model.model_image.window.plt_extent,
        "cmap": cmap_grad,
        "origin": "lower",
    }
    imshow_kwargs.update(kwargs)
    sky_level = 0.
    # if isinstance(model, models.Super_Model):
    #     for M in model.model_list:
    #         if isinstance(M,models.Sky_Model):
    #             try:
    #                 sky_level = M["sky"].value.detach().item()*(1. + 1e-6)*model.target.pixelscale**2
    #                 print("subtracting sky level: ", sky_level)
    #                 break
    #             except Exception as e:
    #                 print(e)
    im = ax.imshow(
        np.log10(image - sky_level),
        **imshow_kwargs,
    )
    if showcbar:
        clb = fig.colorbar(im, ax=ax, label = f"log$_{{10}}$(flux)")

    return fig, ax


def residual_image(fig, ax, model, showcbar = True, window = None, **kwargs):

    if window is None:
        window = model.fit_window
    with torch.no_grad():
        model.sample(model.model_image)
    residuals = (
        model.target[window].data.detach().numpy()
        - model.model_image[window].data.detach().numpy()
    )    
    if model.target.masked:
        residuals[model.target[window].mask] = np.nan
    residuals = np.arctan(residuals/(iqr(residuals[np.isfinite(residuals)], rng = [10,90])*2))
    extreme = np.max(np.abs(residuals[np.isfinite(residuals)]))
    imshow_kwargs = {
        "extent": window.plt_extent,
        "cmap": cmap_div,
        "vmin": -extreme,
        "vmax": extreme,
        "origin": "lower",
    }
    imshow_kwargs.update(kwargs)
    im = ax.imshow(
        residuals,
        **imshow_kwargs,
    )
    if showcbar:
        clb = fig.colorbar(im, ax=ax, label = f"Target - {model.name} [arb.]")
        clb.ax.set_yticks([])
        clb.ax.set_yticklabels([])
    return fig, ax

def supermodel_boxes(fig, ax, model, **kwargs):
    assert isinstance(model, models.Super_Model)
    target_image(fig, ax, model.target[model.fit_window])

    for m in model.model_list:
        ax.add_patch(Rectangle(xy = (m.fit_window.origin[0], m.fit_window.origin[1]), width = m.fit_window.shape[0], height = m.fit_window.shape[1], fill = False, linewidth = 2, edgecolor = main_pallet["secondary1"]))

    return fig, ax

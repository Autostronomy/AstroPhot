import matplotlib.pyplot as plt
import numpy as np
from .visuals import *
from scipy.stats import iqr
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch, LogStretch, HistEqStretch
import torch

__all__ = ["target_image", "model_image", "residual_image"]

def target_image(fig, ax, target, **kwargs):
    dat = target.data.detach().numpy()
    sky = np.median(dat)
    noise = iqr(dat)/2
    vmin = sky - 5*noise
    vmax = sky + 5*noise
    im = ax.imshow(
        dat,
        origin="lower",
        cmap=cmap_div,
        norm=ImageNormalize(stretch=HistEqStretch(dat[dat <= (sky + 3*noise)]), clip = False, vmax = sky + 3*noise, vmin = np.min(dat)),
    )
    ax.imshow(
        np.ma.masked_where(dat < (sky + 3*noise), dat), 
        origin="lower",
        cmap=cmap_grad,
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
        
    im = ax.imshow(
        np.log10(image),
        **imshow_kwargs,
    )
    if showcbar:
        clb = fig.colorbar(im, ax=ax, label = f"log$_{{10}}$(flux)")

    return fig, ax


def residual_image(fig, ax, model, showcbar = True, **kwargs):

    with torch.no_grad():
        model.sample(model.model_image)
    residuals = (
        model.target[model.model_image.window].data.detach().numpy()
        - model.model_image.data.detach().numpy()
    )    
    if model.target.masked:
        residuals[model.target[model.model_image.window].mask] = np.nan
            
    vlim = np.nanmax(np.abs(residuals))
    if vlim > (3 * abs(np.nanmin(residuals))):
        vlim = abs(np.nanmin(residuals))

    imshow_kwargs = {
        "extent": model.model_image.window.plt_extent,
        "cmap": cmap_div,
        "vmin": -vlim,
        "vmax": vlim,
        "origin": "lower",
    }
    imshow_kwargs.update(kwargs)
    im = ax.imshow(
        residuals,
        **imshow_kwargs,
    )
    if showcbar:
        clb = fig.colorbar(im, ax=ax, label = f"Target - {model.name} [flux]")

    return fig, ax

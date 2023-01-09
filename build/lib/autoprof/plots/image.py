import matplotlib.pyplot as plt
import numpy as np
from .visuals import *
from scipy.stats import iqr
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch, LogStretch, HistEqStretch
import torch
from matplotlib.patches import Rectangle
from ..models import Group_Model, Sky_Model
from ..image import Image_List, Window_List

__all__ = ["target_image", "model_image", "residual_image", "model_window"]

def target_image(fig, ax, target, window = None, **kwargs):

    # recursive call for target image list
    if isinstance(target, Image_List):
        for i in range(len(target.image_list)):
            target_image(fig, ax[i], target.image_list[i], window = window, **kwargs)
        return fig, ax
    if window is None:
        window = target.window
    dat = np.copy(target[window].data.detach().cpu().numpy())
    if target.has_mask:
        dat[target[window].mask] = np.nan

    sky = np.nanmedian(dat)
    noise = iqr(dat[np.isfinite(dat)])/2
    vmin = sky - 5*noise
    vmax = sky + 5*noise
    
    im = ax.imshow(
        dat,
        origin="lower",
        cmap="Greys",
        extent = window.plt_extent,
        norm=ImageNormalize(stretch=HistEqStretch(dat[np.logical_and(dat <= (sky + 3*noise), np.isfinite(dat))]), clip = False, vmax = sky + 3*noise, vmin = np.nanmin(dat)),
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

@torch.no_grad()
def model_image(fig, ax, model, sample_image = None, window = None, showcbar = True, **kwargs):

    if sample_image is None:
        sample_image = model.make_model_image()
        sample_image = model.sample(sample_image)

    if isinstance(sample_image, Image_List):
        for i, image in enumerate(sample_image): 
            model_image(fig, ax[i], model, sample_image = image, window = window, showcbar = showcbar, **kwargs)
        return fig, ax
    
    sample_image = sample_image.data.detach().cpu().numpy()
    imshow_kwargs = {
        "extent": model.window.plt_extent,
        "cmap": cmap_grad,
        "origin": "lower",
    }
    imshow_kwargs.update(kwargs)
    sky_level = 0.
    if isinstance(model, Group_Model):
        for M in model.model_list:
            if isinstance(M,Sky_Model):
                try:
                    sky_level = (10**(M["sky"].value)*(1 - 1e-6)*model.target.pixelscale**2).detach().cpu().item()
                    break
                except Exception as e:
                    print(e)
    im = ax.imshow(
        np.log10(sample_image - sky_level),
        **imshow_kwargs,
    )
    if showcbar:
        clb = fig.colorbar(im, ax=ax, label = f"log$_{{10}}$(flux)")

    return fig, ax

@torch.no_grad()
def residual_image(fig, ax, model, target = None, sample_image = None, showcbar = True, window = None, center_residuals = False, **kwargs):

    if window is None:
        window = model.window
    if target is None:
        target = model.target
    if sample_image is None:
        sample_image = model.make_model_image()
        sample_image = model.sample(sample_image)
    if isinstance(window, Window_List):
        for i_ax, win, tar, sam in zip(ax, window, target, sample_image):
            residual_image(fig, i_ax, model, target = tar, sample_image = sam, window = win, showcbar = showcbar, center_residuals = center_residuals, **kwargs)
        return fig, ax
    
    residuals = (
        target[window]
        - sample_image[window]
    ).data.detach().cpu().numpy()
            
    if target.has_mask:
        residuals[target[window].mask] = np.nan
    if center_residuals:
        residuals -= np.nanmedian(residuals)
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

def model_window(fig, ax, model, **kwargs):
    if isinstance(ax, np.ndarray):
        for axitem in ax:
            model_window(fig, axitem, model, **kwargs)
        return fig, ax
    
    if isinstance(model, Group_Model):
        for m in model.model_list:
            ax.add_patch(Rectangle(xy = (m.window.origin[0], m.window.origin[1]), width = m.window.shape[0], height = m.window.shape[1], fill = False, linewidth = 2, edgecolor = main_pallet["secondary1"]))
    else:
        ax.add_patch(Rectangle(xy = (model.window.origin[0], model.window.origin[1]), width = model.window.shape[0], height = model.window.shape[1], fill = False, linewidth = 2, edgecolor = main_pallet["secondary1"]))

    return fig, ax

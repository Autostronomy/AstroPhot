from typing import Literal, Optional, Union
import numpy as np
import torch

from astropy.visualization import HistEqStretch, ImageNormalize
from matplotlib.patches import Polygon
import matplotlib
from scipy.stats import iqr

from ..models import GroupModel, PSFModel, PSFGroupModel
from ..image import ImageList, WindowList, PSFImage
from .. import config
from ..backend_obj import backend
from ..utils.conversions.units import flux_to_sb
from ..utils.decorators import ignore_numpy_warnings
from .visuals import *


__all__ = ("target_image", "psf_image", "model_image", "residual_image", "model_window")


@ignore_numpy_warnings
def target_image(fig, ax, target, window=None, **kwargs):
    """
    This function is used to display a target image using the provided figure
    and axes. The target is plotted using histogram equalization for better
    visibility of the image data for the faint areas of the image, while it uses
    log scale normalization for the bright areas.

    **Args:**
    - `fig` (matplotlib.figure.Figure): The figure object in which the target image will be displayed.
    - `ax` (matplotlib.axes.Axes): The axes object on which the target image will be plotted.
    - `target` (Image or Image_List): The image or list of images to be displayed.
    - `window` (Window, optional): The window through which the image is viewed. If `None`, the window of the
        provided `target` is used. Defaults to `None`.
    - **kwargs: Arbitrary keyword arguments.

    Note:
        If the `target` is an `Image_List`, this function will recursively call itself for each image in the list.
        The `window` parameter and `kwargs` are passed unchanged to each recursive call.
    """

    # recursive call for target image list
    if isinstance(target, ImageList):
        for i in range(len(target.images)):
            target_image(fig, ax[i], target.images[i], window=window, **kwargs)
        return fig, ax
    if window is None:
        window = target.window
    target_area = target[window]
    dat = np.copy(backend.to_numpy(target_area.data))
    if target_area.has_mask:
        dat[backend.to_numpy(target_area.mask)] = np.nan
    X, Y = target_area.coordinate_corner_meshgrid()
    X = backend.to_numpy(X)
    Y = backend.to_numpy(Y)
    sky = np.nanmedian(dat)
    noise = iqr(dat[np.isfinite(dat)], rng=(16, 84)) / 2
    if noise == 0:
        noise = np.nanstd(dat)

    if kwargs.get("linear", False):
        im = ax.pcolormesh(
            X,
            Y,
            dat,
            cmap=cmap_grad,
        )
    else:
        im = ax.pcolormesh(
            X,
            Y,
            dat,
            cmap="gray_r",
            norm=ImageNormalize(
                stretch=HistEqStretch(
                    dat[np.logical_and(dat <= (sky + 3 * noise), np.isfinite(dat))]
                ),
                clip=False,
                vmax=sky + 3 * noise,
                vmin=np.nanmin(dat),
            ),
        )
        pickhist = dat < (sky + 3 * noise)
        if np.sum(~pickhist) > 5:  # only draw log if multiple pixels above noise
            im = ax.pcolormesh(
                X,
                Y,
                np.ma.masked_where(pickhist, dat),
                cmap=cmap_grad,
                norm=matplotlib.colors.LogNorm(),
                clim=[sky + 3 * noise, None],
            )

    if np.linalg.det(target.CD.npvalue) < 0:
        ax.invert_xaxis()
    ax.axis("equal")
    ax.set_xlabel("Tangent Plane X [arcsec]")
    ax.set_ylabel("Tangent Plane Y [arcsec]")

    return fig, ax


@torch.no_grad()
@ignore_numpy_warnings
def psf_image(
    fig,
    ax,
    psf: Union[PSFImage, PSFModel, PSFGroupModel],
    cmap_levels: Optional[int] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs,
):
    """For plotting PSF images, or the output of a PSF model.

    **Args:**
    - `fig` (matplotlib.figure.Figure): The figure object in which the PSF image will be displayed.
    - `ax` (matplotlib.axes.Axes): The axes object on which the PSF image will be plotted.
    - `psf` (PSFImage or PSFModel or PSFGroupModel): The PSF model or group model to be displayed.
    - `cmap_levels` (int, optional): The number of discrete levels to convert the continuous color map to. If not `None`, the color map is converted to a ListedColormap with the specified number of levels. Defaults to `None`.
    - `vmin` (float, optional): The minimum value for the color scale. Defaults to `None`.
    - `vmax` (float, optional): The maximum value for the color scale. Defaults to `None`.
    """
    if isinstance(psf, (PSFModel, PSFGroupModel)):
        psf = psf()
    # recursive call for target image list
    if isinstance(psf, ImageList):
        for i in range(len(psf.images)):
            psf_image(fig, ax[i], psf.images[i], **kwargs)
        return fig, ax

    # Evaluate the model image
    x, y = psf.coordinate_corner_meshgrid()
    x = backend.to_numpy(x)
    y = backend.to_numpy(y)
    psf = backend.to_numpy(psf.data)

    # Default kwargs for image
    kwargs = {
        "cmap": cmap_grad,
        "norm": matplotlib.colors.LogNorm(
            vmin=vmin, vmax=vmax
        ),  # "norm": ImageNormalize(stretch=LogStretch(), clip=False),
        **kwargs,
    }

    # if requested, convert the continuous colourmap into discrete levels
    if cmap_levels is not None:
        kwargs["cmap"] = matplotlib.colors.ListedColormap(
            list(kwargs["cmap"](c) for c in np.linspace(0.0, 1.0, cmap_levels))
        )

    # Plot the image
    ax.pcolormesh(x, y, psf, **kwargs)

    # Enforce equal spacing on x y
    ax.axis("equal")
    ax.set_xlabel("PSF X [arcsec]")
    ax.set_ylabel("PSF Y [arcsec]")

    return fig, ax


@torch.no_grad()
@ignore_numpy_warnings
def model_image(
    fig,
    ax,
    model,
    sample_image=None,
    window=None,
    target=None,
    showcbar: bool = True,
    target_mask: bool = False,
    cmap_levels: Optional[int] = None,
    magunits: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs,
):
    """
    This function is used to generate a model image and display it using the provided figure and axes.

    **Args:**
    -  `fig` (matplotlib.figure.Figure): The figure object in which the image will be displayed.
    -  `ax` (matplotlib.axes.Axes): The axes object on which the image will be plotted.
    -  `model` (Model): The model object used to generate a model image if `sample_image` is not provided.
    -  `sample_image` (Image or Image_List, optional): The image or list of images to be displayed. If `None`, a model image is generated using the provided `model`. Defaults to `None`.
    -  `window` (Window, optional): The window through which the image is viewed. If `None`, the window of the provided `model` is used. Defaults to `None`.
    -  `target` (Target, optional): The target or list of targets for the image or image list. If `None`, the target of the `model` is used. Defaults to `None`.
    -  `showcbar` (bool, optional): Whether to show the color bar. Defaults to `True`.
    -  `target_mask` (bool, optional): Whether to apply the mask of the target. If `True` and if the target has a mask, the mask is applied to the image. Defaults to `False`.
    -  `cmap_levels` (int, optional): The number of discrete levels to convert the continuous color map to. If not `None`, the color map is converted to a ListedColormap with the specified number of levels. Defaults to `None`.
    -  `magunits` (bool, optional): Whether to convert the image to surface brightness units. If `True`, the zeropoint of the target is used to convert the image to surface brightness units. Defaults to `True`.
    -  `vmin` (float, optional): The minimum value for the color scale. Defaults to `None`.
    -  `vmax` (float, optional): The maximum value for the color scale. Defaults to `None`.
    -  **kwargs: Arbitrary keyword arguments. These are used to override the default imshow_kwargs.

    Note:
        If the `sample_image` is an `Image_List`, this function will recursively call itself for each image in the list,
        with the corresponding target and window. The `showcbar` parameter and `kwargs` are passed unchanged to each recursive call.
    """

    if sample_image is None:
        sample_image = model()

    # Use model target if not given
    if target is None:
        target = model.target

    # Use model window if not given
    if window is None:
        window = model.window

    # Handle image lists
    if isinstance(sample_image, ImageList):
        for i, (images, targets, windows) in enumerate(zip(sample_image, target, window)):
            model_image(
                fig,
                ax[i],
                model,
                sample_image=images,
                window=windows,
                target=targets,
                showcbar=showcbar,
                target_mask=target_mask,
                cmap_levels=cmap_levels,
                magunits=magunits,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
        return fig, ax

    # cut out the requested window
    sample_image = sample_image[window]

    # Evaluate the model image
    X, Y = sample_image.coordinate_corner_meshgrid()
    X = backend.to_numpy(X)
    Y = backend.to_numpy(Y)
    sample_image = backend.to_numpy(sample_image.data)

    # Default kwargs for image
    kwargs = {
        "cmap": cmap_grad,
        **kwargs,
    }

    # if requested, convert the continuous colourmap into discrete levels
    if cmap_levels is not None:
        kwargs["cmap"] = matplotlib.colors.ListedColormap(
            list(kwargs["cmap"](c) for c in np.linspace(0.0, 1.0, cmap_levels))
        )

    # If zeropoint is available, convert to surface brightness units
    if target.zeropoint is not None and magunits:
        sample_image = flux_to_sb(sample_image, target.pixel_area.item(), target.zeropoint.item())
        kwargs["cmap"] = kwargs["cmap"].reversed()
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
    else:
        kwargs = {
            "norm": matplotlib.colors.LogNorm(
                vmin=vmin, vmax=vmax
            ),  # "norm": ImageNormalize(stretch=LogStretch(), clip=False),
            **kwargs,
        }

    # Apply the mask if available
    if target_mask and target.has_mask:
        sample_image[backend.to_numpy(target.mask)] = np.nan

    # Plot the image
    im = ax.pcolormesh(X, Y, sample_image, **kwargs)

    if np.linalg.det(target.CD.npvalue) < 0:
        ax.invert_xaxis()

    # Enforce equal spacing on x y
    ax.axis("equal")
    ax.set_xlabel("Tangent Plane X [arcsec]")
    ax.set_ylabel("Tangent Plane Y [arcsec]")

    # Add a colourbar
    if showcbar:
        if target.zeropoint is not None and magunits:
            clb = fig.colorbar(im, ax=ax, label="Surface Brightness [mag/arcsec$^2$]")
            clb.ax.invert_yaxis()
        else:
            clb = fig.colorbar(im, ax=ax, label="log$_{10}$(flux)")

    return fig, ax


@torch.no_grad()
@ignore_numpy_warnings
def residual_image(
    fig,
    ax,
    model,
    target=None,
    sample_image=None,
    showcbar=True,
    window=None,
    clb_label=None,
    normalize_residuals=False,
    scaling: Literal["arctan", "clip", "none"] = "arctan",
    **kwargs,
):
    """
    This function is used to calculate and display the residuals of a model image with respect to a target image.
    The residuals are calculated as the difference between the target image and the sample image and may be normalized by the standard deviation.

    **Args:**
    - `fig` (matplotlib.figure.Figure): The figure object in which the residuals will be displayed.
    - `ax` (matplotlib.axes.Axes): The axes object on which the residuals will be plotted.
    - `model` (Model): The model object used to generate a model image if `sample_image` is not provided.
    - `target` (Target or Image_List, optional): The target or list of targets for the image or image list. If `None`, the target of the `model` is used. Defaults to `None`.
    - `sample_image` (Image or Image_List, optional): The image or list of images from which residuals will be calculated. If `None`, a model image is generated using the provided `model`. Defaults to `None`.
    - `showcbar` (bool, optional): Whether to show the color bar. Defaults to `True`.
    - `window` (Window or Window_List, optional): The window through which the image is viewed. If `None`, the window of the provided `model` is used. Defaults to `None`.
    - `clb_label` (str, optional): The label for the colorbar. If `None`, a default label is used based on the normalization of the residuals. Defaults to `None`.
    - `normalize_residuals` (bool, optional): Whether to normalize the residuals. If `True`, residuals are divided by the square root of the variance of the target. Defaults to `False`.
    - `scaling` (str, optional): The scaling method for the residuals. Options are "arctan", "clip", or "none". arctan will show all residuals, though squish high values to make the fainter residuals more visible, clip will show the residuals in linear space but remove any values above/below 5 sigma, none does no scaling and simply shows the residuals in linear space. Defaults to "arctan".
    - `**kwargs`: Arbitrary keyword arguments. These are used to override the default imshow_kwargs.

    Note:
        If the `window`, `target`, or `sample_image` are lists, this function will recursively call itself for each element in the list,
        with the corresponding window, target, and sample image. The `showcbar`, `center_residuals`, and `kwargs` are passed unchanged to
        each recursive call.
    """

    if window is None:
        window = model.window
    if target is None:
        target = model.target
    if sample_image is None:
        sample_image = model()
    if isinstance(window, WindowList) or isinstance(target, ImageList):
        for i_ax, win, tar, sam in zip(ax, window, target, sample_image):
            residual_image(
                fig,
                i_ax,
                model,
                target=tar,
                sample_image=sam,
                window=win,
                showcbar=showcbar,
                clb_label=clb_label,
                normalize_residuals=normalize_residuals,
                **kwargs,
            )
        return fig, ax

    sample_image = sample_image[window]
    target = target[window]
    X, Y = sample_image.coordinate_corner_meshgrid()
    X = backend.to_numpy(X)
    Y = backend.to_numpy(Y)
    residuals = (target - sample_image).data

    if normalize_residuals is True:
        residuals = residuals / backend.sqrt(target.variance)
    elif isinstance(normalize_residuals, backend.array_type):
        residuals = residuals / backend.sqrt(normalize_residuals)
        normalize_residuals = True
    residuals = backend.to_numpy(residuals)
    if target.has_mask:
        residuals[backend.to_numpy(target.mask)] = np.nan

    if scaling == "clip":
        if normalize_residuals is not True:
            config.logger.warning(
                "Using clipping scaling without normalizing residuals. This may lead to confusing results."
            )
        residuals = np.clip(residuals, -5, 5)
        vmax = 5
        default_label = (
            f"(Target - {model.name}) / $\\sigma$"
            if normalize_residuals
            else f"(Target - {model.name})"
        )
    elif scaling == "arctan":
        residuals = np.arctan(
            residuals / (iqr(residuals[np.isfinite(residuals)], rng=[10, 90]) * 2)
        )
        vmax = np.pi / 2
        if normalize_residuals:
            default_label = f"tan$^{{-1}}$((Target - {model.name}) / $\\sigma$)"
        else:
            default_label = f"tan$^{{-1}}$(Target - {model.name})"
    elif scaling == "none":
        vmax = np.max(np.abs(residuals[np.isfinite(residuals)]))
        default_label = (
            f"(Target - {model.name}) / $\\sigma$"
            if normalize_residuals
            else f"(Target - {model.name})"
        )
    else:
        raise ValueError(f"Unknown scaling type {scaling}. Use 'clip', 'arctan', or 'none'.")
    imshow_kwargs = {
        "cmap": cmap_div,
        "vmin": -vmax,
        "vmax": vmax,
    }
    imshow_kwargs.update(kwargs)
    im = ax.pcolormesh(X, Y, residuals, **imshow_kwargs)
    if np.linalg.det(target.CD.npvalue) < 0:
        ax.invert_xaxis()
    ax.axis("equal")
    ax.set_xlabel("Tangent Plane X [arcsec]")
    ax.set_ylabel("Tangent Plane Y [arcsec]")

    if showcbar:
        clb = fig.colorbar(im, ax=ax, label=default_label if clb_label is None else clb_label)
        clb.ax.set_yticks([])
        clb.ax.set_yticklabels([])
    return fig, ax


@ignore_numpy_warnings
def model_window(fig, ax, model, target=None, rectangle_linewidth=2, **kwargs):
    """Used for plotting the window(s) of a model on a target image. These
    windows bound the region that a model will be evaluated/fit to.

    **Args:**
    - `fig` (matplotlib.figure.Figure): The figure object in which the model window will be displayed.
    - `ax` (matplotlib.axes.Axes): The axes object on which the model window will be plotted.
    - `model` (Model): The model object whose window will be displayed.
    - `target` (Target or Image_List, optional): The target or list of targets for the image or image list. If `None`, the target of the `model` is used. Defaults to `None`.
    - `rectangle_linewidth` (int, optional): The linewidth of the rectangle drawn around the model window. Defaults to 2.
    - **kwargs: Arbitrary keyword arguments. These are used to override the default rectangle properties.
    """
    if target is None:
        target = model.target
    if isinstance(ax, np.ndarray):
        for i, axitem in enumerate(ax):
            model_window(fig, axitem, model, target=target.images[i], **kwargs)
        return fig, ax

    if isinstance(model, GroupModel):
        for m in model.models:
            if isinstance(m.window, WindowList):
                use_window = m.window.windows[m.target.index(target)]
            else:
                use_window = m.window

            corners = target[use_window].corners()
            x = [
                corners[0][0].item(),
                corners[1][0].item(),
                corners[2][0].item(),
                corners[3][0].item(),
            ]
            y = [
                corners[0][1].item(),
                corners[1][1].item(),
                corners[2][1].item(),
                corners[3][1].item(),
            ]
            ax.add_patch(
                Polygon(
                    xy=list(zip(x, y)),
                    fill=False,
                    linewidth=rectangle_linewidth,
                    edgecolor=main_pallet["secondary1"],
                    **kwargs,
                )
            )
    else:
        use_window = model.window
        corners = target[use_window].corners()
        x = [
            corners[0][0].item(),
            corners[1][0].item(),
            corners[2][0].item(),
            corners[3][0].item(),
        ]
        y = [
            corners[0][1].item(),
            corners[1][1].item(),
            corners[2][1].item(),
            corners[3][1].item(),
        ]
        ax.add_patch(
            Polygon(
                xy=list(zip(x, y)),
                fill=False,
                linewidth=rectangle_linewidth,
                edgecolor=main_pallet["secondary1"],
                **kwargs,
            )
        )

    return fig, ax

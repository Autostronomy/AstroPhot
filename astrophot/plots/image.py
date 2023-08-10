import numpy as np
import torch

from astropy.visualization import HistEqStretch, ImageNormalize, LogStretch, SqrtStretch
from matplotlib.patches import Rectangle, Polygon
from matplotlib import pyplot as plt
import matplotlib
from scipy.stats import iqr

from ..models import Group_Model, Sky_Model
from ..image import Image_List, Window_List
from ..utils.conversions.units import flux_to_sb
from .visuals import *


__all__ = ["target_image", "model_image", "residual_image", "model_window"]


def target_image(fig, ax, target, window=None, **kwargs):
    """
    This function is used to display a target image using the provided figure and axes.

    Args:
        fig (matplotlib.figure.Figure): The figure object in which the target image will be displayed.
        ax (matplotlib.axes.Axes): The axes object on which the target image will be plotted.
        target (Image or Image_List): The image or list of images to be displayed.
        window (Window, optional): The window through which the image is viewed. If `None`, the window of the
            provided `target` is used. Defaults to `None`.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the displayed target image.
        ax (matplotlib.axes.Axes): The axes object containing the displayed target image.

    Note:
        If the `target` is an `Image_List`, this function will recursively call itself for each image in the list.
        The `window` parameter and `kwargs` are passed unchanged to each recursive call.
    """

    # recursive call for target image list
    if isinstance(target, Image_List):
        for i in range(len(target.image_list)):
            target_image(fig, ax[i], target.image_list[i], window=window, **kwargs)
        return fig, ax
    if window is None:
        window = target.window
    target_area = target[window]
    dat = np.copy(target_area.data.detach().cpu().numpy())
    if target_area.has_mask:
        dat[target_area.mask.detach().cpu().numpy()] = np.nan
    X, Y = target_area.get_coordinate_corner_meshgrid()
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    sky = np.nanmedian(dat)
    noise = iqr(dat[np.isfinite(dat)]) / 2
    vmin = sky - 5 * noise
    vmax = sky + 5 * noise

    im = ax.pcolormesh(
        X,
        Y,
        dat,
        cmap="Greys",
        norm=ImageNormalize(
            stretch=HistEqStretch(
                dat[np.logical_and(dat <= (sky + 3 * noise), np.isfinite(dat))]
            ),
            clip=False,
            vmax=sky + 3 * noise,
            vmin=np.nanmin(dat),
        ),
    )

    im = ax.pcolormesh(
        X,
        Y,
        np.ma.masked_where(dat < (sky + 3 * noise), dat),
        cmap=cmap_grad,
        norm=matplotlib.colors.LogNorm(),
        clim=[sky + 3 * noise, None],
    )

    ax.axis("equal")

    return fig, ax


@torch.no_grad()
def model_image(
    fig,
    ax,
    model,
    sample_image=None,
    window=None,
    target=None,
    showcbar=True,
    target_mask=False,
    cmap_levels=None,
    **kwargs,
):
    """
    This function is used to generate a model image and display it using the provided figure and axes.

    Args:
        fig (matplotlib.figure.Figure): The figure object in which the image will be displayed.
        ax (matplotlib.axes.Axes): The axes object on which the image will be plotted.
        model (Model): The model object used to generate a model image if `sample_image` is not provided.
        sample_image (Image or Image_List, optional): The image or list of images to be displayed.
            If `None`, a model image is generated using the provided `model`. Defaults to `None`.
        window (Window, optional): The window through which the image is viewed. If `None`, the window of the
            provided `model` is used. Defaults to `None`.
        target (Target, optional): The target or list of targets for the image or image list.
            If `None`, the target of the `model` is used. Defaults to `None`.
        showcbar (bool, optional): Whether to show the color bar. Defaults to `True`.
        target_mask (bool, optional): Whether to apply the mask of the target. If `True` and if the target has a mask,
            the mask is applied to the image. Defaults to `False`.
        cmap_levels (int, optional): The number of discrete levels to convert the continuous color map to.
            If not `None`, the color map is converted to a ListedColormap with the specified number of levels.
            Defaults to `None`.
        **kwargs: Arbitrary keyword arguments. These are used to override the default imshow_kwargs.

    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the displayed image.
        ax (matplotlib.axes.Axes): The axes object containing the displayed image.

    Note:
        If the `sample_image` is an `Image_List`, this function will recursively call itself for each image in the list,
        with the corresponding target and window. The `showcbar` parameter and `kwargs` are passed unchanged to each recursive call.
    """

    if sample_image is None:
        sample_image = model.make_model_image()
        sample_image = model(sample_image)

    # Use model target if not given
    if target is None:
        target = model.target

    # Use model window if not given
    if window is None:
        window = model.window

    # Handle image lists
    if isinstance(sample_image, Image_List):
        for i, images in enumerate(zip(sample_image, target, window)):
            model_image(
                fig,
                ax[i],
                model,
                sample_image=images[0],
                window=images[2],
                target=images[1],
                showcbar=showcbar,
                **kwargs,
            )
        return fig, ax

    # Evaluate the model image
    X, Y = sample_image.get_coordinate_corner_meshgrid()
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    sample_image = sample_image.data.detach().cpu().numpy()

    # Default kwargs for image
    imshow_kwargs = {
        "cmap": cmap_grad,
        "norm": matplotlib.colors.LogNorm(),  # "norm": ImageNormalize(stretch=LogStretch(), clip=False),
    }

    # Update with user provided kwargs
    imshow_kwargs.update(kwargs)

    # if requested, convert the continuous colourmap into discrete levels
    if cmap_levels is not None:
        imshow_kwargs["cmap"] = matplotlib.colors.ListedColormap(
            list(imshow_kwargs["cmap"](c) for c in np.linspace(0.0, 1.0, cmap_levels))
        )

    # If zeropoint is available, convert to surface brightness units
    if target.zeropoint is not None:
        sample_image = flux_to_sb(
            sample_image, target.pixel_area.item(), target.zeropoint.item()
        )
        del imshow_kwargs["norm"]
        imshow_kwargs["cmap"] = imshow_kwargs["cmap"].reversed()

    # Apply the mask if available
    if target_mask and target.has_mask:
        sample_image[target.mask.detach().cpu().numpy()] = np.nan

    # Plot the image
    im = ax.pcolormesh(X, Y, sample_image, **imshow_kwargs)

    # Enforce equal spacing on x y
    ax.axis("equal")

    # Add a colourbar
    if showcbar:
        if target.zeropoint is not None:
            clb = fig.colorbar(im, ax=ax, label="Surface Brightness [mag/arcsec$^2$]")
            clb.ax.invert_yaxis()
        else:
            clb = fig.colorbar(im, ax=ax, label=f"log$_{{10}}$(flux)")

    return fig, ax


@torch.no_grad()
def residual_image(
    fig,
    ax,
    model,
    target=None,
    sample_image=None,
    showcbar=True,
    window=None,
    center_residuals=False,
    clb_label=None,
    normalize_residuals=False,
    **kwargs,
):
    """
    This function is used to calculate and display the residuals of a model image with respect to a target image.
    The residuals are calculated as the difference between the target image and the sample image.

    Args:
        fig (matplotlib.figure.Figure): The figure object in which the residuals will be displayed.
        ax (matplotlib.axes.Axes): The axes object on which the residuals will be plotted.
        model (Model): The model object used to generate a model image if `sample_image` is not provided.
        target (Target or Image_List, optional): The target or list of targets for the image or image list.
            If `None`, the target of the `model` is used. Defaults to `None`.
        sample_image (Image or Image_List, optional): The image or list of images from which residuals will be calculated.
            If `None`, a model image is generated using the provided `model`. Defaults to `None`.
        showcbar (bool, optional): Whether to show the color bar. Defaults to `True`.
        window (Window or Window_List, optional): The window through which the image is viewed. If `None`, the window of the
            provided `model` is used. Defaults to `None`.
        center_residuals (bool, optional): Whether to subtract the median of the residuals. If `True`, the median is subtracted
            from the residuals. Defaults to `False`.
        clb_label (str, optional): The label for the colorbar. If `None`, a default label is used based on the normalization of the
            residuals. Defaults to `None`.
        normalize_residuals (bool, optional): Whether to normalize the residuals. If `True`, residuals are divided by the square root
            of the variance of the target. Defaults to `False`.
        **kwargs: Arbitrary keyword arguments. These are used to override the default imshow_kwargs.

    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the displayed residuals.
        ax (matplotlib.axes.Axes): The axes object containing the displayed residuals.

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
        sample_image = model.make_model_image()
        sample_image = model(sample_image)
    if isinstance(window, Window_List) or isinstance(target, Image_List):
        for i_ax, win, tar, sam in zip(ax, window, target, sample_image):
            residual_image(
                fig,
                i_ax,
                model,
                target=tar,
                sample_image=sam,
                window=win,
                showcbar=showcbar,
                center_residuals=center_residuals,
                **kwargs,
            )
        return fig, ax

    X, Y = sample_image[window].get_coordinate_corner_meshgrid()
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    residuals = (target[window] - sample_image[window]).data
    if normalize_residuals:
        residuals = residuals / torch.sqrt(target[window].variance)
    residuals = residuals.detach().cpu().numpy()

    if target.has_mask:
        residuals[target[window].mask.detach().cpu().numpy()] = np.nan
    if center_residuals:
        residuals -= np.nanmedian(residuals)
    residuals = np.arctan(
        residuals / (iqr(residuals[np.isfinite(residuals)], rng=[10, 90]) * 2)
    )
    extreme = np.max(np.abs(residuals[np.isfinite(residuals)]))
    imshow_kwargs = {
        "cmap": cmap_div,
        "vmin": -extreme,
        "vmax": extreme,
    }
    imshow_kwargs.update(kwargs)
    im = ax.pcolormesh(X, Y, residuals, **imshow_kwargs)
    ax.axis("equal")

    if showcbar:
        if normalize_residuals:
            default_label = f"tan$^{{-1}}$((Target - {model.name}) / $\\sigma$)"
        else:
            default_label = f"tan$^{{-1}}$(Target - {model.name})"
        clb = fig.colorbar(
            im, ax=ax, label=default_label if clb_label is None else clb_label
        )
        clb.ax.set_yticks([])
        clb.ax.set_yticklabels([])
    return fig, ax


def model_window(fig, ax, model, target=None, rectangle_linewidth=2, **kwargs):
    if isinstance(ax, np.ndarray):
        for i, axitem in enumerate(ax):
            model_window(
                fig, axitem, model, target=model.target.image_list[i], **kwargs
            )
        return fig, ax

    if isinstance(model, Group_Model):
        for m in model.models.values():
            if isinstance(m.window, Window_List):
                use_window = m.window.window_list[m.target.index(target)]
            else:
                use_window = m.window

            lowright = use_window.shape.clone()
            lowright[1] = 0.0
            lowright = use_window.origin + use_window.cartesian_to_world(lowright)
            lowright = lowright.detach().cpu().numpy()
            upleft = use_window.shape.clone()
            upleft[0] = 0.0
            upleft = use_window.origin + use_window.cartesian_to_world(upleft)
            upleft = upleft.detach().cpu().numpy()
            end = use_window.origin + use_window.end
            end = end.detach().cpu().numpy()
            x = [
                use_window.origin[0].detach().cpu().numpy(),
                lowright[0],
                end[0],
                upleft[0],
            ]
            y = [
                use_window.origin[1].detach().cpu().numpy(),
                lowright[1],
                end[1],
                upleft[1],
            ]
            ax.add_patch(
                Polygon(
                    xy=list(zip(x, y)),
                    fill=False,
                    linewidth=rectangle_linewidth,
                    edgecolor=main_pallet["secondary1"],
                )
            )
    else:
        if isinstance(model.window, Window_List):
            use_window = model.window.window_list[model.target.index(target)]
        else:
            use_window = model.window
        lowright = use_window.shape.clone()
        lowright[1] = 0.0
        lowright = use_window.origin + use_window.cartesian_to_world(lowright)
        lowright = lowright.detach().cpu().numpy()
        upleft = use_window.shape.clone()
        upleft[0] = 0.0
        upleft = use_window.origin + use_window.cartesian_to_world(upleft)
        upleft = upleft.detach().cpu().numpy()
        end = use_window.origin + use_window.end
        end = end.detach().cpu().numpy()
        x = [
            use_window.origin[0].detach().cpu().numpy(),
            lowright[0],
            end[0],
            upleft[0],
        ]
        y = [
            use_window.origin[1].detach().cpu().numpy(),
            lowright[1],
            end[1],
            upleft[1],
        ]
        ax.add_patch(
            Polygon(
                xy=list(zip(x, y)),
                fill=False,
                linewidth=rectangle_linewidth,
                edgecolor=main_pallet["secondary1"],
            )
        )

    return fig, ax

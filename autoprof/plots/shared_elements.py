import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch, LogStretch, HistEqStretch


def LSBImage(dat, noise):
    plt.figure(figsize=(6, 6))
    plt.imshow(
        dat,
        origin="lower",
        cmap="Greys",
        norm=ImageNormalize(
            stretch=HistEqStretch(dat[dat <= 3 * noise]),
            clip=False,
            vmax=3 * noise,
            vmin=np.min(dat),
        ),
    )
    my_cmap = copy(cm.Greys_r)
    my_cmap.set_under("k", alpha=0)

    plt.imshow(
        np.ma.masked_where(dat < 3 * noise, dat),
        origin="lower",
        cmap=my_cmap,
        norm=ImageNormalize(stretch=LogStretch(), clip=False),
        clim=[3 * noise, None],
        interpolation="none",
    )
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05)
    plt.xlim([0, dat.shape[1]])
    plt.ylim([0, dat.shape[0]])


def _display_time(seconds):
    intervals = (
        ("hours", 3600),  # 60 * 60
        ("arcminutes", 60),
        ("arcseconds", 1),
    )
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip("s")
            result.append("{} {}".format(value, name))
    return ", ".join(result)


def AddScale(ax, img_width, loc="lower right"):
    """
    ax: figure axis object
    img_width: image width in arcseconds
    loc: location to put hte scale bar
    """
    scale_width = int(img_width / 6)

    if scale_width > 60 and scale_width % 60 <= 15:
        scale_width -= scale_width % 60
    if scale_width > 45 and scale_width % 60 >= 45:
        scale_width += 60 - (scale_width % 60)
    if 15 < scale_width % 60 < 45:
        scale_width += 30 - (scale_width % 60)

    label = _display_time(scale_width)

    xloc = 0.05 if "left" in loc else 0.95
    yloc = 0.95 if "upper" in loc else 0.05

    ax.text(
        xloc - 0.5 * scale_width / img_width,
        yloc + 0.005,
        label,
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize="x-small" if len(label) < 20 else "xx-small",
        weight="bold",
        color=autocolours["red1"],
    )
    ax.plot(
        [xloc - scale_width / img_width, xloc],
        [yloc, yloc],
        transform=ax.transAxes,
        color=autocolours["red1"],
    )


def AddLogo(fig, loc=[0.8, 0.01, 0.844 / 5, 0.185 / 5], white=False):
    im = plt.imread(
        get_sample_data(
            os.path.join(
                os.environ["AUTOPROF"],
                "_static/",
                ("AP_logo_white.png" if white else "AP_logo.png"),
            )
        )
    )
    newax = fig.add_axes(loc, zorder=1000)
    if white:
        newax.imshow(np.zeros(im.shape) + np.array([0, 0, 0, 1]))
    else:
        newax.imshow(np.ones(im.shape))
    newax.imshow(im)
    newax.axis("off")

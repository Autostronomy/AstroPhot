import numpy as np

from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, norm
from .visuals import main_pallet

__all__ = ("covariance_matrix", "corner_plot")


def covariance_matrix(
    covariance_matrix,
    mean,
    labels=None,
    figsize=(10, 10),
    reference_values=None,
    ellipse_colors=main_pallet["primary1"],
    showticks=True,
    **kwargs,
):
    """
    Create a covariance matrix plot. Creates a corner plot with ellipses representing the covariance between parameters.

    :param covariance_matrix: Covariance matrix of shape (n_params, n_params).
    :type covariance_matrix: np.ndarray
    :param mean: Mean values of the parameters, shape (n_params,).
    :type mean: np.ndarray
    :param labels: Labels for the parameters.
    :type labels: list, optional
    :param figsize: Size of the figure. Default is (10, 10).
    :type figsize: tuple, optional
    :param reference_values: Reference values for the parameters, used to draw vertical and horizontal lines. Typically these are the true values of the parameters.
    :type reference_values: np.ndarray, optional
    :param ellipse_colors: Color for the ellipses. Default is `main_pallet["primary1"]`.
    :type ellipse_colors: str or list, optional
    :param showticks: Whether to show ticks on the axes. Default is True.
    :type showticks: bool, optional

    returns the figure and the (n_params, n_params) array of axes, to allow further customization by the user.
    """
    num_params = covariance_matrix.shape[0]
    fig, axes = plt.subplots(num_params, num_params, figsize=figsize)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    for i in range(num_params):
        for j in range(num_params):
            ax = axes[i, j]

            if i == j:
                x = np.linspace(
                    mean[i] - 3 * np.sqrt(covariance_matrix[i, i]),
                    mean[i] + 3 * np.sqrt(covariance_matrix[i, i]),
                    100,
                )
                y = norm.pdf(x, mean[i], np.sqrt(covariance_matrix[i, i]))
                ax.plot(x, y, color=ellipse_colors, lw=1.5)
                ax.set_xlim(
                    mean[i] - 3 * np.sqrt(covariance_matrix[i, i]),
                    mean[i] + 3 * np.sqrt(covariance_matrix[i, i]),
                )
                if reference_values is not None:
                    ax.axvline(reference_values[i], color=main_pallet["pop"], linestyle="-", lw=1)
            elif j < i:
                cov = covariance_matrix[np.ix_([j, i], [j, i])]
                lambda_, v = np.linalg.eigh(cov)
                lambda_ = np.sqrt(lambda_)
                angle = np.rad2deg(np.arctan2(v[1, 1], v[0, 1]))
                for k in [1, 2]:
                    ellipse = Ellipse(
                        xy=(mean[j], mean[i]),
                        width=lambda_[1] * k * 2,
                        height=lambda_[0] * k * 2,
                        angle=angle,
                        edgecolor=ellipse_colors,
                        facecolor="none",
                        lw=1.5,
                    )
                    ax.add_artist(ellipse)

                # Set axis limits
                margin = 3
                ax.set_xlim(
                    mean[j] - margin * np.sqrt(covariance_matrix[j, j]),
                    mean[j] + margin * np.sqrt(covariance_matrix[j, j]),
                )
                ax.set_ylim(
                    mean[i] - margin * np.sqrt(covariance_matrix[i, i]),
                    mean[i] + margin * np.sqrt(covariance_matrix[i, i]),
                )

                if reference_values is not None:
                    ax.axvline(reference_values[j], color=main_pallet["pop"], linestyle="-", lw=1)
                    ax.axhline(reference_values[i], color=main_pallet["pop"], linestyle="-", lw=1)

            _corner_axis(ax, i, j, num_params, labels, showticks)

    return fig, axes


def corner_plot(
    chain,
    labels=None,
    figsize=(10, 10),
    reference_values=None,
    contour_colors=main_pallet["primary1"],
    showticks=True,
    sigmas=(1, 2),
    resolution=50,
):
    """
    Create a corner plot of the samples from an MCMC chain. The diagonal shows the density of each parameter and the panels below it show contours of the density for each pair of parameters, both from a Gaussian kernel density estimate of the samples.

    The contours enclose the same fraction of the samples as the 1 and 2 sigma ellipses of a 2D Gaussian (39% and 86%), so they can be compared directly with the ellipses drawn by `covariance_matrix`.

    :param chain: Samples of the parameters with shape (..., n_params), for example (n_samples, n_params) or (n_steps, n_chains, n_params). Every sample is used, so remove any burn-in first. The time to make the plot grows with the number of samples, thin a chain of more than a few thousand.
    :type chain: np.ndarray
    :param labels: Labels for the parameters.
    :type labels: list, optional
    :param figsize: Size of the figure. Default is (10, 10).
    :type figsize: tuple, optional
    :param reference_values: Reference values for the parameters, used to draw vertical and horizontal lines. Typically these are the true values of the parameters.
    :type reference_values: np.ndarray, optional
    :param contour_colors: Color for the contours and density curves. Default is `main_pallet["primary1"]`.
    :type contour_colors: str, optional
    :param showticks: Whether to show ticks on the axes. Default is True.
    :type showticks: bool, optional
    :param sigmas: The contours to draw, as the number of sigma of a 2D Gaussian which encloses the same fraction of the samples. Default is (1, 2).
    :type sigmas: tuple, optional
    :param resolution: Number of points along each axis at which the density is evaluated. Default is 50.
    :type resolution: int, optional

    returns the figure and the (n_params, n_params) array of axes, to allow further customization by the user.
    """
    chain = np.asarray(chain)
    chain = chain.reshape(-1, chain.shape[-1])
    num_params = chain.shape[1]
    fig, axes = plt.subplots(num_params, num_params, figsize=figsize, squeeze=False)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    # fraction of a 2D Gaussian inside each number of sigma, largest first
    enclosed = 1 - np.exp(-0.5 * np.sort(np.atleast_1d(sigmas))[::-1] ** 2)
    limits = np.stack((np.min(chain, axis=0), np.max(chain, axis=0)), axis=1)

    for i in range(num_params):
        for j in range(num_params):
            ax = axes[i, j]

            if i == j:
                x = np.linspace(limits[i][0], limits[i][1], 2 * resolution)
                ax.plot(x, gaussian_kde(chain[:, i])(x), color=contour_colors, lw=1.5)
                ax.set_xlim(limits[i])
                if reference_values is not None:
                    ax.axvline(reference_values[i], color=main_pallet["pop"], linestyle="-", lw=1)
            elif j < i:
                x, y = np.meshgrid(
                    np.linspace(limits[j][0], limits[j][1], resolution),
                    np.linspace(limits[i][0], limits[i][1], resolution),
                )
                density = gaussian_kde(chain[:, [j, i]].T)(np.stack((x.ravel(), y.ravel())))
                # the density above which each fraction of the samples is found
                ordered = np.sort(density)[::-1]
                total = np.cumsum(ordered) / np.sum(ordered)
                levels = np.unique(
                    ordered[np.searchsorted(total, enclosed).clip(0, len(total) - 1)]
                )
                ax.contour(
                    x,
                    y,
                    density.reshape(x.shape),
                    levels=levels,
                    colors=contour_colors,
                    linewidths=1.5,
                )
                ax.set_xlim(limits[j])
                ax.set_ylim(limits[i])

                if reference_values is not None:
                    ax.axvline(reference_values[j], color=main_pallet["pop"], linestyle="-", lw=1)
                    ax.axhline(reference_values[i], color=main_pallet["pop"], linestyle="-", lw=1)

            _corner_axis(ax, i, j, num_params, labels, showticks)

    return fig, axes


def _corner_axis(ax, i, j, num_params, labels, showticks):
    """Labels and ticks for the panel in row i and column j of a corner plot."""
    if j > i:
        ax.axis("off")

    if i < num_params - 1:
        ax.set_xticklabels([])
    else:
        if labels is not None:
            ax.set_xlabel(labels[j])
    if not showticks:
        ax.yaxis.set_major_locator(plt.NullLocator())

    if j > 0:
        ax.set_yticklabels([])
    else:
        if labels is not None:
            ax.set_ylabel(labels[i])
    if not showticks:
        ax.xaxis.set_major_locator(plt.NullLocator())


if __name__ == "__main__":

    fig, ax = covariance_matrix(np.array([[4, -2], [-2, 4]]), np.array([0, 0]))
    plt.show()

"""The plotting functions of astrophot/plots.

Visuals can't be checked, so these only test that the code runs and draws
something on the axis.
"""

import astrophot as ap
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pytest

######################################################################
# Images (astrophot/plots/image.py)
######################################################################


def test_target_image(fig_ax, sersic_target):
    fig, ax = fig_ax
    ap.plots.target_image(fig, ax, sersic_target)
    assert ax.has_data()


def test_psf_image(fig_ax, psf_target):
    fig, ax = fig_ax
    ap.plots.psf_image(fig, ax, psf_target)
    assert ax.has_data()


def test_target_image_list(sersic_target_factory):
    target = ap.TargetImageList(
        [sersic_target_factory(name="target1"), sersic_target_factory(name="target2")]
    )
    fig, ax = plt.subplots(2)
    ap.plots.target_image(fig, ax, target)
    assert all(a.has_data() for a in ax)


@pytest.mark.parametrize("plot", [ap.plots.model_image, ap.plots.residual_image])
def test_model_image(plot, fig_ax, sersic_model):
    fig, ax = fig_ax
    plot(fig, ax, sersic_model)
    assert ax.has_data()


def test_model_windows(fig_ax, sersic_target):
    model = ap.Model(
        name="windowed_sersic",
        model_type="sersic galaxy model",
        window=(10, 30, 10, 30),
        target=sersic_target,
    )
    fig, ax = fig_ax
    ap.plots.model_window(fig, ax, model)
    assert len(ax.patches) + len(ax.lines) > 0, "the window should be drawn"


######################################################################
# Profiles (astrophot/plots/profile.py)
######################################################################


@pytest.mark.parametrize("plot", [ap.plots.radial_light_profile, ap.plots.radial_median_profile])
def test_radial_profile(plot, fig_ax, sersic_model):
    fig, ax = fig_ax
    plot(fig, ax, sersic_model)
    assert ax.has_data()


######################################################################
# Diagnostics (astrophot/plots/diagnostic.py)
######################################################################


def test_covariance_matrix():
    covariance_matrix = np.array([[1, 0.5], [0.5, 1]])
    mean = np.array([0, 0])
    fig, ax = ap.plots.covariance_matrix(covariance_matrix, mean, labels=["x", "y"])
    assert ax.shape == (2, 2), "one panel per pair of parameters"
    assert ax[1, 0].get_xlabel() == "x" and ax[1, 0].get_ylabel() == "y"


def test_corner_plot(monkeypatch):
    rng = np.random.default_rng(1)
    chain = rng.multivariate_normal([0, 5, -2], np.diag([1, 4, 0.25]), size=(100, 2))
    # Keep what Axes.contour returns. Looking the contours up in the axes afterwards
    # depends on the version of matplotlib
    contours = []
    contour = Axes.contour
    monkeypatch.setattr(
        Axes, "contour", lambda *args, **kwargs: contours.append(contour(*args, **kwargs))
    )
    fig, ax = ap.plots.corner_plot(chain, labels=["x", "y", "z"], reference_values=[0, 5, -2])
    assert ax.shape == (3, 3), "one panel per pair of parameters, chains are joined"
    assert ax[2, 0].get_xlabel() == "x" and ax[2, 0].get_ylabel() == "z"
    assert ax[2, 1].get_xlim() == (chain[..., 1].min(), chain[..., 1].max())
    assert len(contours) == 3, "a panel for each pair of parameters"
    assert all(len(c.levels) == 2 for c in contours), "1 and 2 sigma contours"
    assert not ax[0, 1].axison, "nothing above the diagonal"
    plt.close(fig)

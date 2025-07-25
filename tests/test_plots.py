import numpy as np
import matplotlib.pyplot as plt

import astrophot as ap
from utils import make_basic_sersic, make_basic_gaussian_psf
import pytest


"""
Can't test visuals, so this only tests that the code runs
"""


def test_target_image():
    target = make_basic_sersic()
    try:
        fig, ax = plt.subplots()
    except Exception:
        pytest.skip("skipping test_target_image because matplotlib is not installed properly")
    ap.plots.target_image(fig, ax, target)
    plt.close()


def test_psf_image():
    target = make_basic_gaussian_psf()
    try:
        fig, ax = plt.subplots()
    except Exception:
        pytest.skip("skipping test_target_image because matplotlib is not installed properly")
    ap.plots.psf_image(fig, ax, target)
    plt.close()


def test_target_image_list():
    target1 = make_basic_sersic(name="target1")
    target2 = make_basic_sersic(name="target2")
    target = ap.TargetImageList([target1, target2])
    try:
        fig, ax = plt.subplots(2)
    except Exception:
        pytest.skip("skipping test_target_image_list because matplotlib is not installed properly")
    ap.plots.target_image(fig, ax, target)
    plt.close()


def test_model_image():
    target = make_basic_sersic()
    new_model = ap.Model(
        name="constrained sersic",
        model_type="sersic galaxy model",
        center=[20, 20],
        PA=60 * np.pi / 180,
        q=0.5,
        n=2,
        Re=5,
        Ie=1,
        target=target,
    )
    new_model.initialize()
    try:
        fig, ax = plt.subplots()
    except Exception:
        pytest.skip("skipping test because matplotlib is not installed properly")
    ap.plots.model_image(fig, ax, new_model)
    plt.close()


def test_residual_image():
    target = make_basic_sersic()
    new_model = ap.Model(
        name="constrained sersic",
        model_type="sersic galaxy model",
        center=[20, 20],
        PA=60 * np.pi / 180,
        q=0.5,
        n=2,
        Re=5,
        Ie=1,
        target=target,
    )
    new_model.initialize()
    try:
        fig, ax = plt.subplots()
    except Exception:
        pytest.skip("skipping test because matplotlib is not installed properly")
    ap.plots.residual_image(fig, ax, new_model)
    plt.close()


def test_model_windows():
    target = make_basic_sersic()
    new_model = ap.Model(
        name="constrained sersic",
        model_type="sersic galaxy model",
        center=[20, 20],
        PA=60 * np.pi / 180,
        q=0.5,
        n=2,
        Re=5,
        Ie=1,
        window=(10, 10, 30, 30),
        target=target,
    )
    new_model.initialize()
    try:
        fig, ax = plt.subplots()
    except Exception:
        pytest.skip("skipping test because matplotlib is not installed properly")
    ap.plots.model_window(fig, ax, new_model)
    plt.close()


def test_covariance_matrix():
    covariance_matrix = np.array([[1, 0.5], [0.5, 1]])
    mean = np.array([0, 0])
    try:
        fig, ax = plt.subplots()
    except Exception:
        pytest.skip("skipping test because matplotlib is not installed properly")
    fig, ax = ap.plots.covariance_matrix(covariance_matrix, mean, labels=["x", "y"])
    plt.close()


def test_radial_profile():
    target = make_basic_sersic()
    new_model = ap.Model(
        name="constrained sersic",
        model_type="sersic galaxy model",
        center=[20, 20],
        PA=60 * np.pi / 180,
        q=0.5,
        n=2,
        Re=5,
        Ie=1,
        target=target,
    )
    new_model.initialize()
    try:
        fig, ax = plt.subplots()
    except Exception:
        pytest.skip("skipping test because matplotlib is not installed properly")
    ap.plots.radial_light_profile(fig, ax, new_model)
    plt.close()


def test_radial_median_profile():
    target = make_basic_sersic()
    new_model = ap.Model(
        name="constrained sersic",
        model_type="sersic galaxy model",
        center=[20, 20],
        PA=60 * np.pi / 180,
        q=0.5,
        n=2,
        Re=5,
        Ie=1,
        target=target,
    )
    new_model.initialize()
    try:
        fig, ax = plt.subplots()
    except Exception:
        pytest.skip("skipping test because matplotlib is not installed properly")
    ap.plots.radial_median_profile(fig, ax, new_model)
    plt.close()

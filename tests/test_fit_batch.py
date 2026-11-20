"""BatchLM, the Levenberg-Marquardt fitter for many images at once (astrophot/fit/batch_lm.py)."""

import astrophot as ap
import numpy as np
import pytest


def make_batch_lm_setup(
    integrate_mode="none", sampling_mode="quad:3", n_images=3, pixelscale=1.0, cd_angles=None
):
    """Three images of the same Sersic at different positions, and a model that
    starts near the truth with one set of parameters per image."""
    np.random.seed(42)
    if cd_angles is None:
        cd_angles = [np.pi / 3, np.pi / 3, np.pi / 3]

    true_centers = [(32, 40), (15, 15), (45, 20)]
    base_target = ap.TargetImage(data=np.zeros((64, 64)), pixelscale=pixelscale)
    model_gen = ap.Model(
        model_type="sersic galaxy model",
        name="batch_gen",
        center=true_centers[0],
        q=0.6,
        PA=np.pi / 3,
        n=1,
        Re=10,
        Ie=1,
        target=base_target,
        integrate_mode="none",
        sampling_mode="quad:3",
    )
    model_gen.initialize()

    images = []
    for k in range(n_images):
        model_gen.center = true_centers[k % len(true_centers)]
        kwargs = dict(
            name=f"target{k}",
            data=ap.backend.to_numpy(model_gen().data) + np.random.normal(scale=0.5, size=(64, 64)),
            pixelscale=pixelscale,
        )
        if cd_angles is not None:
            kwargs["CD"] = ap.utils.initialize.R(cd_angles[k % len(cd_angles)])
        images.append(ap.TargetImage(**kwargs))

    batch_target = ap.TargetImageBatch(images)

    model = ap.Model(
        model_type="sersic galaxy model",
        name="batch_fit",
        center=true_centers[0],
        q=0.6,
        PA=np.pi / 3,
        n=1,
        Re=10,
        Ie=1,
        target=batch_target.images[0],
        integrate_mode=integrate_mode,
        sampling_mode=sampling_mode,
    )
    model.initialize()

    init_centers = [(30, 42), (16, 16), (46, 21)]
    model.center = tuple(init_centers[k % len(init_centers)] for k in range(n_images))
    model.q = tuple(0.6 for _ in range(n_images))
    model.PA = tuple(np.pi / 3 for _ in range(n_images))
    model.n = tuple(1.1 if k % 2 == 0 else 0.9 for k in range(n_images))
    model.Re = tuple(11 for _ in range(n_images))
    model.Ie = tuple(1.0 for _ in range(n_images))

    return model, batch_target


@pytest.mark.parametrize("integrate_mode", ["none", "bright", "curvature"])
def test_batch_lm_integrate_modes(integrate_mode):
    """BatchLM must converge without error for all integrate_mode values."""
    model, batch_target = make_batch_lm_setup(integrate_mode=integrate_mode)
    res = ap.fit.BatchLM(model, batch_target, batch_target.window, max_iter=5).fit()
    assert len(res.loss_history) >= 2, f"BatchLM ({integrate_mode}) must take at least one step"
    assert np.all(
        np.isfinite(res.loss_history[-1])
    ), f"BatchLM ({integrate_mode}) final loss should be finite"


@pytest.mark.parametrize("sampling_mode", ["midpoint", "simpsons", "quad:3"])
def test_batch_lm_sampling_modes(sampling_mode):
    """BatchLM must converge without error for common sampling_mode values."""
    model, batch_target = make_batch_lm_setup(sampling_mode=sampling_mode)
    res = ap.fit.BatchLM(model, batch_target, batch_target.window, max_iter=5).fit()
    assert len(res.loss_history) >= 2, f"BatchLM ({sampling_mode}) must take at least one step"
    assert np.all(
        np.isfinite(res.loss_history[-1])
    ), f"BatchLM ({sampling_mode}) final loss should be finite"


def test_batch_lm_rotated_images():
    """BatchLM should work on images with non-trivial (rotated) CD matrices."""
    model, batch_target = make_batch_lm_setup(
        cd_angles=[np.pi / 3, np.pi / 6, np.pi / 16],
    )
    res = ap.fit.BatchLM(model, batch_target, batch_target.window, max_iter=5).fit()
    assert len(res.loss_history) >= 2, "BatchLM should take at least one step on rotated images"
    assert np.all(
        np.isfinite(res.loss_history[-1])
    ), "BatchLM final loss should be finite for rotated images"


def test_batch_lm_poisson_likelihood():
    """BatchLM should work with Poisson likelihood."""
    model, batch_target = make_batch_lm_setup()
    res = ap.fit.BatchLM(
        model, batch_target, batch_target.window, max_iter=5, likelihood="poisson"
    ).fit()
    assert len(res.loss_history) >= 2, "BatchLM (poisson) must take at least one step"
    assert np.all(
        np.isfinite(res.loss_history[-1])
    ), "BatchLM (poisson) final loss should be finite"


def test_batch_lm_bright_integrate_improves():
    """BatchLM with integrate_mode='bright' should improve the loss (chi^2/ndf).

    Ten iterations, so this is also the test of the convergence checks."""
    model, batch_target = make_batch_lm_setup(integrate_mode="bright")
    res = ap.fit.BatchLM(model, batch_target, batch_target.window, max_iter=10).fit()
    assert np.all(
        res.loss_history[-1] <= res.loss_history[0]
    ), "BatchLM with integrate_mode='bright' should not worsen the loss"

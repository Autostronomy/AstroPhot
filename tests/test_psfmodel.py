import astrophot as ap
import numpy as np
from utils import make_basic_gaussian_psf
import pytest

# torch.autograd.set_detect_anomaly(True)
######################################################################
# PSF Model Objects
######################################################################


@pytest.mark.parametrize("model_type", ap.models.PSFModel.List_Models(usable=True, types=True))
def test_all_psfmodel_sample(model_type):
    if model_type == "airy psf model" and ap.backend.backend == "jax":
        pytest.skip(
            "Skipping airy psf model, JAX does not support bessel_j1 with finite derivatives it seems"
        )
    if any(t in model_type for t in ["warp", "fourier"]):
        pytest.skip("Skipping warp and fourier psf models, which are slow")

    target = make_basic_gaussian_psf(pixelscale=0.8)
    MODEL = ap.Model(
        name="test_model",
        model_type=model_type,
        target=target,
        normalize_psf=False,
    )
    for p in MODEL.all_params:
        if p.units in ["flux", "flux/pix^2"]:
            p.to_dynamic(None)
    MODEL.initialize()
    for p in MODEL.all_params:
        if p.units in ["flux", "flux/pix^2"]:
            p.to_dynamic(p.value * 1.5)
        if p.units == "pix" and not p.name == "center":
            p.to_dynamic(p.value + 0.5)
    print(MODEL)
    for P in MODEL.dynamic_params:
        assert P.value is not None, (
            f"Model type {model_type} parameter {P} should not be None after initialization",
        )
    img = MODEL()

    assert ap.backend.all(
        ap.backend.isfinite(img.data)
    ), "Model should evaluate a real number for the full image"

    if model_type == "pixelated psf model":
        psf = ap.utils.initialize.gaussian_psf(3 * 0.8, 25, 0.8)
        MODEL.pixels.value = psf / np.sum(psf)

    assert ap.backend.all(
        ap.backend.isfinite(MODEL.jacobian().data)
    ), "Model should evaluate a real number for the jacobian"

    res = ap.fit.LM(MODEL, max_iter=10).fit()

    assert len(res.loss_history) >= 2, "Optimizer must be able to find steps to improve the model"

    if res.message == "success":
        # Be less strict if fit succeeded quickly
        assert res.loss_history[-1] < res.loss_history[0], (
            f"Model {model_type} should fit to the target image, but did not. "
            f"Initial loss: {res.loss_history[0]}, Final loss: {res.loss_history[-1]}"
        )
    else:
        assert ((res.loss_history[0] - 1) > (2 * (res.loss_history[-1] - 1))) or (
            res.loss_history[-1] < 1.0
        ), (
            f"Model {model_type} should fit to the target image, but did not. "
            f"Initial loss: {res.loss_history[0]}, Final loss: {res.loss_history[-1]}"
        )

import astrophot as ap
import torch
import numpy as np
from utils import make_basic_gaussian_psf
import pytest

# torch.autograd.set_detect_anomaly(True)
######################################################################
# PSF Model Objects
######################################################################


@pytest.mark.parametrize("model_type", ap.models.PSFModel.List_Models(usable=True, types=True))
def test_all_psfmodel_sample(model_type):

    target = make_basic_gaussian_psf(pixelscale=0.8)
    if "eigen" in model_type:
        kwargs = {
            "eigen_basis": np.stack(
                list(
                    ap.utils.initialize.gaussian_psf(sigma / 0.8, 25, 0.8)
                    for sigma in np.linspace(1, 10, 5)
                )
            )
        }
    else:
        kwargs = {}
    MODEL = ap.Model(
        name="test model",
        model_type=model_type,
        target=target,
        **kwargs,
    )
    MODEL.initialize()
    print(MODEL)
    for P in MODEL.dynamic_params:
        assert P.value is not None, (
            f"Model type {model_type} parameter {P} should not be None after initialization",
        )
    img = MODEL()
    import matplotlib.pyplot as plt

    plt.imshow(img.data.detach().cpu().numpy())
    plt.colorbar()
    plt.title(f"Model type: {model_type}")
    plt.savefig(f"test_psfmodel_{model_type}.png")
    assert torch.all(
        torch.isfinite(img.data)
    ), "Model should evaluate a real number for the full image"

    if model_type == "pixelated psf model":
        MODEL.pixels = ap.utils.initialize.gaussian_psf(3 / 0.8, 25, 0.8)
    res = ap.fit.LM(MODEL, max_iter=10).fit()
    print(res.message)
    print(res.loss_history)
    assert res.loss_history[0] > (2 * res.loss_history[-1]), (
        f"Model {model_type} should fit to the target image, but did not. "
        f"Initial loss: {res.loss_history[0]}, Final loss: {res.loss_history[-1]}"
    )

import unittest
import astrophot as ap
import torch
import numpy as np
from utils import make_basic_sersic, make_basic_gaussian_psf
import pytest

# torch.autograd.set_detect_anomaly(True)
######################################################################
# Model Objects
######################################################################


def test_model_sampling_modes():

    target = make_basic_sersic(90, 100)
    model = ap.Model(
        name="test sersic",
        model_type="sersic galaxy model",
        center=[20, 20],
        PA=60 * np.pi / 180,
        q=0.5,
        n=2,
        Re=5,
        logIe=1,
        target=target,
    )
    model()
    model.sampling_mode = "midpoint"
    model()
    model.sampling_mode = "simpsons"
    model()
    model.sampling_mode = "quad:3"
    model()
    model.integrate_mode = "none"
    model()
    model.integrate_mode = "should raise"
    with pytest.raises(ap.errors.SpecificationConflict):
        model()
    model.integrate_mode = "none"
    model.sampling_mode = "should raise"
    with pytest.raises(ap.errors.SpecificationConflict):
        model()
    model.sampling_mode = "midpoint"
    model.integrate_mode = "none"

    # test PSF modes
    model.psf = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
    model.psf_convolve = True
    model()


def test_model_errors():

    # Target that is not a target image
    arr = torch.zeros((10, 15))
    target = ap.image.Image(data=arr, pixelscale=1.0, zeropoint=1.0)

    with pytest.raises(ap.errors.InvalidTarget):
        ap.Model(
            name="test model",
            model_type="sersic galaxy model",
            target=target,
        )

    # model that doesn't exist
    target = make_basic_sersic()
    with pytest.raises(ap.errors.UnrecognizedModel):
        ap.Model(
            name="test model",
            model_type="sersic gaaxy model",
            target=target,
        )


@pytest.mark.parametrize(
    "model_type", ap.models.ComponentModel.List_Models(usable=True, types=True)
)
def test_all_model_sample(model_type):

    target = make_basic_sersic()
    MODEL = ap.Model(
        name="test model",
        model_type=model_type,
        target=target,
    )
    MODEL.initialize()
    for P in MODEL.dynamic_params:
        assert (
            P.value is not None
        ), f"Model type {model_type} parameter {P.name} should not be None after initialization"
    img = MODEL()
    import matplotlib.pyplot as plt

    print(MODEL)
    fig, ax = plt.subplots(1, 2)
    ap.plots.model_image(fig, ax[0], MODEL)
    ap.plots.residual_image(fig, ax[1], MODEL)
    plt.savefig(f"test_{model_type}_sample.png")
    plt.close()
    assert torch.all(
        torch.isfinite(img.data)
    ), "Model should evaluate a real number for the full image"
    res = ap.fit.LM(MODEL, max_iter=10).fit()
    print(res.message)
    assert res.loss_history[0] > res.loss_history[-1], (
        f"Model {model_type} should fit to the target image, but did not. "
        f"Initial loss: {res.loss_history[0]}, Final loss: {res.loss_history[-1]}"
    )


def test_sersic_save_load():

    target = make_basic_sersic()
    model = ap.Model(
        name="test sersic",
        model_type="sersic galaxy model",
        center=[20, 20],
        PA=60 * np.pi / 180,
        q=0.5,
        n=2,
        Re=5,
        logIe=1,
        target=target,
    )

    model.initialize()
    model.save_state("test_AstroPhot_sersic.hdf5", appendable=True)
    model.center = [30, 30]
    model.PA = 30 * np.pi / 180
    model.q = 0.8
    model.n = 3
    model.Re = 10
    model.logIe = 2
    target.crtan = [1.0, 2.0]
    model.append_state("test_AstroPhot_sersic.hdf5")
    model.load_state("test_AstroPhot_sersic.hdf5", index=0)

    assert model.center.value[0].item() == 20, "Model center should be loaded correctly"
    assert model.center.value[1].item() == 20, "Model center should be loaded correctly"
    assert model.PA.value.item() == 60 * np.pi / 180, "Model PA should be loaded correctly"
    assert model.q.value.item() == 0.5, "Model q should be loaded correctly"
    assert model.n.value.item() == 2, "Model n should be loaded correctly"
    assert model.Re.value.item() == 5, "Model Re should be loaded correctly"
    assert model.logIe.value.item() == 1, "Model logIe should be loaded correctly"
    assert model.target.crtan.value[0] == 0.0, "Model target crtan should be loaded correctly"
    assert model.target.crtan.value[1] == 0.0, "Model target crtan should be loaded correctly"

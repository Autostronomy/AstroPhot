import astrophot as ap
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
        center=[40, 41.9],
        PA=60 * np.pi / 180,
        q=0.8,
        n=0.5,
        Re=20,
        Ie=1,
        target=target,
    )

    # With subpixel integration
    model.integrate_mode = "bright"
    auto = ap.backend.to_numpy(model().data)
    model.sampling_mode = "midpoint"
    midpoint = ap.backend.to_numpy(model().data)
    midpoint_bright = midpoint.copy()
    model.sampling_mode = "simpsons"
    simpsons = ap.backend.to_numpy(model().data)
    model.sampling_mode = "quad:5"
    quad5 = ap.backend.to_numpy(model().data)
    assert np.allclose(midpoint, auto, rtol=1e-2), "Midpoint sampling should match auto sampling"
    assert np.allclose(midpoint, simpsons, rtol=1e-2), "Simpsons sampling should match midpoint"
    assert np.allclose(midpoint, quad5, rtol=1e-2), "Quad5 sampling should match midpoint sampling"
    assert np.allclose(simpsons, quad5, rtol=1e-6), "Quad5 sampling should match Simpsons sampling"

    # Without subpixel integration
    model.integrate_mode = "none"
    auto = ap.backend.to_numpy(model().data)
    model.sampling_mode = "midpoint"
    midpoint = ap.backend.to_numpy(model().data)
    model.sampling_mode = "simpsons"
    simpsons = ap.backend.to_numpy(model().data)
    model.sampling_mode = "quad:5"
    quad5 = ap.backend.to_numpy(model().data)
    assert np.allclose(
        midpoint, midpoint_bright, rtol=1e-2
    ), "no integrate sampling should match bright sampling"
    assert np.allclose(midpoint, auto, rtol=1e-2), "Midpoint sampling should match auto sampling"
    assert np.allclose(midpoint, simpsons, rtol=1e-2), "Simpsons sampling should match midpoint"
    assert np.allclose(midpoint, quad5, rtol=1e-2), "Quad5 sampling should match midpoint sampling"
    assert np.allclose(simpsons, quad5, rtol=1e-6), "Quad5 sampling should match Simpsons sampling"

    # curvature based subpixel integration
    model.integrate_mode = "curvature"
    auto = ap.backend.to_numpy(model().data)
    model.sampling_mode = "midpoint"
    midpoint = ap.backend.to_numpy(model().data)
    model.sampling_mode = "simpsons"
    simpsons = ap.backend.to_numpy(model().data)
    model.sampling_mode = "quad:5"
    quad5 = ap.backend.to_numpy(model().data)
    assert np.allclose(
        midpoint, midpoint_bright, rtol=1e-2
    ), "curvature integrate sampling should match bright sampling"
    assert np.allclose(midpoint, auto, rtol=1e-2), "Midpoint sampling should match auto sampling"
    assert np.allclose(midpoint, simpsons, rtol=1e-2), "Simpsons sampling should match midpoint"
    assert np.allclose(midpoint, quad5, rtol=1e-2), "Quad5 sampling should match midpoint sampling"
    assert np.allclose(simpsons, quad5, rtol=1e-6), "Quad5 sampling should match Simpsons sampling"

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
    arr = np.zeros((10, 15))
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

    if ap.backend.backend == "jax" and np.random.randint(0, 3) > 0:
        pytest.skip("JAX is very slow, randomly reducing the number of tests")
    if model_type == "isothermal sech2 edgeon model" and ap.backend.backend == "jax":
        pytest.skip("JAX doesnt have bessel function k1 yet")

    if (
        model_type in ["ferrer warp galaxy model", "king warp galaxy model"]
        and ap.backend.backend == "jax"
    ):
        pytest.skip("JAX version doesnt support these models yet, difficulty with gradients")

    target = make_basic_sersic()
    target.zeropoint = 22.5
    MODEL = ap.Model(
        name="test model",
        model_type=model_type,
        target=target,
        integrate_mode=(
            "none" if ap.backend.backend == "jax" else "bright"
        ),  # JAX JIT is reallly slow for any integration
    )
    MODEL.initialize()
    MODEL.to()
    for P in MODEL.dynamic_params:
        assert (
            P.value is not None
        ), f"Model type {model_type} parameter {P.name} should not be None after initialization"
    img = MODEL()
    assert ap.backend.all(
        ap.backend.isfinite(img.data)
    ), "Model should evaluate a real number for the full image"

    res = ap.fit.LM(MODEL, max_iter=10, verbose=1).fit()
    print(res.loss_history)

    print(MODEL)  # test printing

    # sky has little freedom to fit, some more complex models need extra
    # attention to get a good fit so here we just check that they can improve
    if (
        "sky" in model_type
        or "king" in model_type
        or "spline" in model_type
        or model_type
        in [
            "exponential warp galaxy model",
            "ferrer warp galaxy model",
            "ferrer ray galaxy model",
            "isothermal sech2 edgeon model",
        ]
    ):
        assert res.loss_history[0] > res.loss_history[-1], (
            f"Model {model_type} should fit to the target image, but did not. "
            f"Initial loss: {res.loss_history[0]}, Final loss: {res.loss_history[-1]}"
        )
    else:  # Most models should get significantly better after just a few iterations
        assert res.loss_history[0] > (1.5 * res.loss_history[-1]), (
            f"Model {model_type} should fit to the target image, but did not. "
            f"Initial loss: {res.loss_history[0]}, Final loss: {res.loss_history[-1]}"
        )

    F = MODEL.total_flux()
    assert ap.backend.isfinite(F), "Model total flux should be finite after fitting"
    assert F > 0, "Model total flux should be positive after fitting"
    U = MODEL.total_flux_uncertainty()
    assert ap.backend.isfinite(U), "Model total flux uncertainty should be finite after fitting"
    assert U >= 0, "Model total flux uncertainty should be non-negative after fitting"
    M = MODEL.total_magnitude()
    assert ap.backend.isfinite(M), "Model total magnitude should be finite after fitting"
    U_M = MODEL.total_magnitude_uncertainty()
    assert ap.backend.isfinite(
        U_M
    ), "Model total magnitude uncertainty should be finite after fitting"
    assert U_M >= 0, "Model total magnitude uncertainty should be non-negative after fitting"

    allnames = set()
    for name in MODEL.build_params_array_names():
        assert name not in allnames, f"Duplicate parameter name found: {name}"
        allnames.add(name)


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
        Ie=1,
        target=target,
    )

    model.initialize()
    model.save_state("test_AstroPhot_sersic.hdf5", appendable=True)
    model.center = [30, 30]
    model.PA = 30 * np.pi / 180
    model.q = 0.8
    model.n = 3
    model.Re = 10
    model.Ie = 2
    target.crtan = [1.0, 2.0]
    model.append_state("test_AstroPhot_sersic.hdf5")
    model.load_state("test_AstroPhot_sersic.hdf5", index=0)

    assert model.center.value[0].item() == 20, "Model center should be loaded correctly"
    assert model.center.value[1].item() == 20, "Model center should be loaded correctly"
    assert model.PA.value.item() == 60 * np.pi / 180, "Model PA should be loaded correctly"
    assert model.q.value.item() == 0.5, "Model q should be loaded correctly"
    assert model.n.value.item() == 2, "Model n should be loaded correctly"
    assert model.Re.value.item() == 5, "Model Re should be loaded correctly"
    assert model.Ie.value.item() == 1, "Model Ie should be loaded correctly"
    assert model.target.crtan.value[0] == 0.0, "Model target crtan should be loaded correctly"
    assert model.target.crtan.value[1] == 0.0, "Model target crtan should be loaded correctly"


@pytest.mark.parametrize("center", [[20, 20], [25.1, 17.324567]])
@pytest.mark.parametrize("PA", [0, 60 * np.pi / 180])
@pytest.mark.parametrize("q", [0.2, 0.8])
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("Re", [10, 25.1])
def test_chunk_sample(center, PA, q, n, Re):
    target = make_basic_sersic()
    model = ap.Model(
        name="test sersic",
        model_type="sersic galaxy model",
        center=center,
        PA=PA,
        q=q,
        n=n,
        Re=Re,
        Ie=10.0,
        target=target,
        integrate_mode="none",
    )

    full_img = model.sample()

    chunk_img = target.model_image()

    for chunk in model.window.chunk(20**2):
        sample = model.sample(window=chunk)
        chunk_img += sample

    assert ap.backend.allclose(
        full_img.data, chunk_img.data
    ), "Chunked sample should match full sample within tolerance"

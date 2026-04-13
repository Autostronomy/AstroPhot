import numpy as np

import astrophot as ap
from utils import make_basic_sersic
import pytest

######################################################################
# Fit Objects
######################################################################


def _make_batch_lm_setup(
    integrate_mode="none", sampling_mode="quad:3", n_images=3, pixelscale=1.0, cd_angles=None
):
    """Helper to create a BatchLM fitting setup with the given parameters."""
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
    model, batch_target = _make_batch_lm_setup(integrate_mode=integrate_mode)
    res = ap.fit.BatchLM(model, batch_target, batch_target.window, max_iter=5).fit()
    assert len(res.loss_history) >= 2, f"BatchLM ({integrate_mode}) must take at least one step"
    assert np.all(
        np.isfinite(res.loss_history[-1])
    ), f"BatchLM ({integrate_mode}) final loss should be finite"


@pytest.mark.parametrize("sampling_mode", ["midpoint", "simpsons", "quad:3"])
def test_batch_lm_sampling_modes(sampling_mode):
    """BatchLM must converge without error for common sampling_mode values."""
    model, batch_target = _make_batch_lm_setup(sampling_mode=sampling_mode)
    res = ap.fit.BatchLM(model, batch_target, batch_target.window, max_iter=5).fit()
    assert len(res.loss_history) >= 2, f"BatchLM ({sampling_mode}) must take at least one step"
    assert np.all(
        np.isfinite(res.loss_history[-1])
    ), f"BatchLM ({sampling_mode}) final loss should be finite"


def test_batch_lm_rotated_images():
    """BatchLM should work on images with non-trivial (rotated) CD matrices."""
    model, batch_target = _make_batch_lm_setup(
        cd_angles=[np.pi / 3, np.pi / 6, np.pi / 16],
    )
    res = ap.fit.BatchLM(model, batch_target, batch_target.window, max_iter=5).fit()
    assert len(res.loss_history) >= 2, "BatchLM should take at least one step on rotated images"
    assert np.all(
        np.isfinite(res.loss_history[-1])
    ), "BatchLM final loss should be finite for rotated images"


def test_batch_lm_poisson_likelihood():
    """BatchLM should work with Poisson likelihood."""
    model, batch_target = _make_batch_lm_setup()
    res = ap.fit.BatchLM(
        model, batch_target, batch_target.window, max_iter=5, likelihood="poisson"
    ).fit()
    assert len(res.loss_history) >= 2, "BatchLM (poisson) must take at least one step"
    assert np.all(
        np.isfinite(res.loss_history[-1])
    ), "BatchLM (poisson) final loss should be finite"


def test_batch_lm_bright_integrate_improves():
    """BatchLM with integrate_mode='bright' should improve the loss (chi^2/ndf)."""
    model, batch_target = _make_batch_lm_setup(integrate_mode="bright")
    res = ap.fit.BatchLM(model, batch_target, batch_target.window, max_iter=10).fit()
    assert np.all(
        res.loss_history[-1] <= res.loss_history[0]
    ), "BatchLM with integrate_mode='bright' should not worsen the loss"


@pytest.mark.parametrize("center", [[20.01, 20.02], [25.1, 17.324567]])
@pytest.mark.parametrize("PA", [0, 60 * np.pi / 180])
@pytest.mark.parametrize("q", [0.4, 0.8])
@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("Re", [15, 25.1])
def test_chunk_jacobian(center, PA, q, n, Re):
    target = make_basic_sersic()
    model = ap.Model(
        name="test_sersic",
        model_type="sersic galaxy model",
        center=center,
        PA=PA,
        q=q,
        n=n,
        Re=Re,
        Ie=10.0,
        target=target,
        integrate_mode="none",
        psf_convolve=False,
    )

    Jtrue = model.jacobian()

    model.jacobian_maxparams = 3

    Jchunked = model.jacobian()
    assert ap.backend.allclose(
        Jtrue.data, Jchunked.data
    ), "Param chunked Jacobian should match full Jacobian"


@pytest.fixture
def sersic_model():
    target = make_basic_sersic()
    model = ap.Model(
        name="test_sersic",
        model_type="sersic galaxy model",
        center=[20, 20],
        PA=np.pi,
        q=0.7,
        n=2,
        Re=15,
        Ie=10.0,
        target=target,
    )
    model.initialize()
    return model


@pytest.mark.parametrize(
    "fitter,extra",
    [
        (ap.fit.LM, {}),
        (ap.fit.LM, {"likelihood": "poisson"}),
        (ap.fit.IterParam, {"chunks": 3, "chunk_order": "sequential", "verbose": 2}),
        (
            ap.fit.IterParam,
            {"chunks": 3, "chunk_order": "random", "verbose": 2, "likelihood": "poisson"},
        ),
        (ap.fit.Grad, {}),
        (ap.fit.ScipyFit, {}),
        (ap.fit.MHMCMC, {}),
        (ap.fit.HMC, {}),
        (ap.fit.MALA, {"epsilon": 1e-3}),
        (
            ap.fit.MALA,
            {
                "epsilon": 1e-3,
                "likelihood": "poisson",
                "initial_state": [[20, 20, 0.7, np.pi, 2, 15, 10]],
            },
        ),
        (ap.fit.Slalom, {}),
    ],
)
@pytest.mark.parametrize("fit_valid", [True, False])
def test_fitters(fitter, extra, sersic_model, fit_valid):
    if ap.backend.backend == "jax" and fitter in [ap.fit.Grad, ap.fit.HMC]:
        pytest.skip("Grad and HMC not implemented for JAX backend")
    model = sersic_model
    model.initialize()
    ll_init = model.gaussian_log_likelihood()
    pll_init = model.poisson_log_likelihood()
    result = fitter(model, max_iter=100, fit_valid=fit_valid, **extra).fit()
    ll_final = model.gaussian_log_likelihood()
    pll_final = model.poisson_log_likelihood()
    assert ll_final > ll_init, f"{fitter.__name__} should improve the log likelihood"
    assert pll_final > pll_init, f"{fitter.__name__} should improve the poisson log likelihood"


def test_fitters_iter():
    target = make_basic_sersic()
    model1 = ap.Model(
        name="test1",
        model_type="sersic galaxy model",
        center=[20, 20],
        PA=np.pi,
        q=0.7,
        n=2,
        Re=15,
        Ie=10.0,
        target=target,
    )
    model2 = ap.Model(
        name="test2",
        model_type="sersic galaxy model",
        center=[20.5, 21],
        PA=1.5 * np.pi,
        q=0.9,
        n=1,
        Re=10,
        Ie=8.0,
        target=target,
    )
    model = ap.Model(
        name="test_group",
        model_type="group model",
        models=[model1, model2],
        target=target,
    )
    model.initialize()
    ll_init = model.gaussian_log_likelihood()
    pll_init = model.poisson_log_likelihood()
    result = ap.fit.Iter(model, max_iter=10).fit()
    ll_final = model.gaussian_log_likelihood()
    pll_final = model.poisson_log_likelihood()
    assert ll_final > ll_init, f"Iter should improve the log likelihood"
    assert pll_final > pll_init, f"Iter should improve the poisson log likelihood"

    # test hessian
    Hgauss = model.hessian(likelihood="gaussian")
    assert ap.backend.all(
        ap.backend.isfinite(Hgauss)
    ), "Hessian should be finite for Gaussian likelihood"
    Hpoisson = model.hessian(likelihood="poisson")
    assert ap.backend.all(
        ap.backend.isfinite(Hpoisson)
    ), "Hessian should be finite for Poisson likelihood"


def test_hessian(sersic_model):
    model = sersic_model
    model.initialize()
    Hgauss = model.hessian(likelihood="gaussian")
    assert ap.backend.all(
        ap.backend.isfinite(Hgauss)
    ), "Hessian should be finite for Gaussian likelihood"
    Hpoisson = model.hessian(likelihood="poisson")
    assert ap.backend.all(
        ap.backend.isfinite(Hpoisson)
    ), "Hessian should be finite for Poisson likelihood"
    assert Hgauss is not None, "Hessian should be computed for Gaussian likelihood"
    assert Hpoisson is not None, "Hessian should be computed for Poisson likelihood"
    with pytest.raises(ValueError):
        model.hessian(likelihood="unknown")


def test_gradient(sersic_model):
    if ap.backend.backend == "jax":
        pytest.skip("JAX backend does not support backward function")
    model = sersic_model
    target = model.target
    target.weight = 1 / (10 + target.variance)
    model.initialize()
    x = model.get_values()
    grad = model.gradient()
    assert ap.backend.all(ap.backend.isfinite(grad)), "Gradient should be finite"
    assert grad.shape == x.shape, "Gradient shape should match parameters shape"
    x.requires_grad = True
    ll = model.gaussian_log_likelihood(x)
    ll.backward()
    autograd = x.grad
    assert ap.backend.allclose(grad, autograd, rtol=1e-4), "Gradient should match autograd gradient"

    funcgrad = ap.backend.grad(model.gaussian_log_likelihood)(x)
    assert ap.backend.allclose(
        grad, funcgrad, rtol=1e-4
    ), "Gradient should match functional gradient"


def test_options(sersic_model):
    model = sersic_model
    model.initialize()

    with pytest.raises(ValueError):
        ap.fit.LM(model, likelihood="unknown")
    with pytest.raises(ValueError):
        ap.fit.IterParam(model, likelihood="unknown")
    with pytest.raises(ap.errors.OptimizeStopSuccess):
        model.target.mask = ap.backend.ones_like(model.target.mask, dtype=bool)
        ap.fit.IterParam(model)
    model.target.mask = ap.backend.zeros_like(model.target.mask, dtype=bool)

    fitter = ap.fit.IterParam(
        model=model,
        W=model.target.weight,
        ndf=np.prod(model.target.data.shape),
        chunk_order="invalid",
    )
    with pytest.raises(ValueError):
        fitter.fit()

    model.to_static(False)
    res = ap.fit.IterParam(model).fit()
    assert "No parameters to optimize" in res.message, "Should exit if no dynamic parameters"


# class TestHMC(unittest.TestCase):
#     def test_hmc_sample(self):
#         np.random.seed(12345)
#         N = 50
#         pixelscale = 0.8
#         true_params = {
#             "n": 2,
#             "Re": 10,
#             "Ie": 1,
#             "center": [-3.3, 5.3],
#             "q": 0.7,
#             "PA": np.pi / 4,
#         }
#         target = ap.image.Target_Image(
#             data=np.zeros((N, N)),
#             pixelscale=pixelscale,
#         )

#         MODEL = ap.models.Sersic_Galaxy(
#             name="sersic model",
#             target=target,
#             parameters=true_params,
#         )
#         img = MODEL().data.detach().cpu().numpy()
#         target.data = torch.Tensor(
#             img
#             + np.random.normal(scale=0.1, size=img.shape)
#             + np.random.normal(scale=np.sqrt(img) / 10)
#         )
#         target.variance = torch.Tensor(0.1**2 + img / 100)

#         HMC = ap.fit.HMC(MODEL, epsilon=1e-5, max_iter=5, warmup=2)
#         HMC.fit()


# class TestNUTS(unittest.TestCase):
#     def test_nuts_sample(self):
#         np.random.seed(12345)
#         N = 50
#         pixelscale = 0.8
#         true_params = {
#             "n": 2,
#             "Re": 10,
#             "Ie": 1,
#             "center": [-3.3, 5.3],
#             "q": 0.7,
#             "PA": np.pi / 4,
#         }
#         target = ap.image.Target_Image(
#             data=np.zeros((N, N)),
#             pixelscale=pixelscale,
#         )

#         MODEL = ap.models.Sersic_Galaxy(
#             name="sersic model",
#             target=target,
#             parameters=true_params,
#         )
#         img = MODEL().data.detach().cpu().numpy()
#         target.data = torch.Tensor(
#             img
#             + np.random.normal(scale=0.1, size=img.shape)
#             + np.random.normal(scale=np.sqrt(img) / 10)
#         )
#         target.variance = torch.Tensor(0.1**2 + img / 100)

#         NUTS = ap.fit.NUTS(MODEL, max_iter=5, warmup=2)
#         NUTS.fit()

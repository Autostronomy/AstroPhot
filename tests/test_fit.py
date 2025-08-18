import torch
import numpy as np

import astrophot as ap
from utils import make_basic_sersic
import pytest

######################################################################
# Fit Objects
######################################################################


@pytest.mark.parametrize("center", [[20, 20], [25.1, 17.324567]])
@pytest.mark.parametrize("PA", [0, 60 * np.pi / 180])
@pytest.mark.parametrize("q", [0.2, 0.8])
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("Re", [10, 25.1])
def test_chunk_jacobian(center, PA, q, n, Re):
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

    Jtrue = model.jacobian()

    model.jacobian_maxparams = 3

    Jchunked = model.jacobian()
    assert ap.backend.allclose(
        Jtrue.data, Jchunked.data
    ), "Param chunked Jacobian should match full Jacobian"

    model.jacobian_maxparams = 10
    model.jacobian_maxpixels = 20**2

    Jchunked = model.jacobian()

    assert ap.backend.allclose(
        Jtrue.data, Jchunked.data
    ), "Pixel chunked Jacobian should match full Jacobian"


@pytest.fixture
def sersic_model():
    target = make_basic_sersic()
    model = ap.Model(
        name="test sersic",
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
    "fitter",
    [
        ap.fit.LM,
        ap.fit.LMfast,
        ap.fit.Grad,
        ap.fit.ScipyFit,
        ap.fit.MHMCMC,
        ap.fit.HMC,
        ap.fit.MiniFit,
        ap.fit.Slalom,
    ],
)
def test_fitters(fitter, sersic_model):
    if ap.backend.backend == "jax" and fitter in [ap.fit.Grad, ap.fit.HMC]:
        pytest.skip("Grad and HMC not implemented for JAX backend")
    model = sersic_model
    model.initialize()
    ll_init = model.gaussian_log_likelihood()
    pll_init = model.poisson_log_likelihood()
    result = fitter(model, max_iter=100).fit()
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
        name="test group",
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
    target.weight = 1 / (10 + target.variance.T)
    model.initialize()
    x = model.build_params_array()
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

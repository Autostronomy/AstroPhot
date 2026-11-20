"""Behaviour shared by all models (astrophot/models/base.py, model_object.py, mixins/sample.py).

Sampling and integration modes, errors on construction, saving the state, and
the derivatives (jacobian, gradient, hessian). Each family of models has its
own file: test_models_component.py, test_models_psf.py, test_models_group.py
and test_models_batch.py.
"""

import astrophot as ap
import numpy as np
import pytest

######################################################################
# Sampling
######################################################################


def test_model_sampling_modes(sersic_target_factory):
    """Every sampling mode gives the same image, for every integration mode."""
    model = ap.Model(
        name="test_sersic",
        model_type="sersic galaxy model",
        center=[40, 41.9],
        PA=60 * np.pi / 180,
        q=0.8,
        n=0.5,
        Re=20,
        Ie=1,
        target=sersic_target_factory(90, 100),
    )

    def sample(sampling_mode):
        model.sampling_mode = sampling_mode
        return ap.backend.to_numpy(model().data)

    midpoint_bright = None
    for integrate_mode in ("bright", "none", "curvature"):
        model.integrate_mode = integrate_mode
        auto = sample("auto")
        midpoint = sample("midpoint")
        simpsons = sample("simpsons")
        upsample5 = sample("upsample:5")
        quad5 = sample("quad:5")
        message = f"sampling should match midpoint sampling with integrate_mode {integrate_mode}"
        assert np.allclose(midpoint, auto, rtol=1e-2), "auto " + message
        assert np.allclose(midpoint, simpsons, rtol=1e-2), "simpsons " + message
        assert np.allclose(midpoint, quad5, rtol=1e-2), "quad:5 " + message
        assert np.allclose(midpoint, upsample5, rtol=1e-2), "upsample:5 " + message
        assert np.allclose(simpsons, quad5, rtol=1e-6), "quad:5 sampling should match simpsons"
        if midpoint_bright is None:
            midpoint_bright = midpoint
        assert np.allclose(
            midpoint, midpoint_bright, rtol=1e-2
        ), f"integrate_mode {integrate_mode} should match bright"

    with pytest.raises(ap.errors.SpecificationConflict):
        model.integrate_mode = "should raise"
    with pytest.raises(ap.errors.SpecificationConflict):
        model.sampling_mode = "should raise"
    with pytest.raises(ap.errors.SpecificationConflict):
        model.sampling_mode = "upsample:4"  # must be odd


def test_model_psf_convolve(sersic_model):
    """A PSF given to the model is used instead of the one of the target."""
    sersic_model.psf_convolve = False
    sharp = sersic_model().data
    sersic_model.psf = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
    sersic_model.psf_convolve = True
    blurred = sersic_model().data
    assert ap.backend.all(ap.backend.isfinite(blurred)), "convolved model should be finite"
    assert ap.backend.max(blurred) < ap.backend.max(sharp), "convolution should lower the peak"


######################################################################
# Construction and state
######################################################################


def test_model_errors(sersic_target):
    # Target that is not a target image
    target = ap.image.Image(data=np.zeros((10, 15)), pixelscale=1.0, zeropoint=1.0)
    with pytest.raises(ap.errors.InvalidTarget):
        ap.Model(name="test_model", model_type="sersic galaxy model", target=target)

    # model that doesn't exist
    with pytest.raises(ap.errors.UnrecognizedModel):
        ap.Model(name="test_model", model_type="sersic gaaxy model", target=sersic_target)


def test_model_save_load_state(sersic_model, tmp_path):
    saved = {
        "center": [20.0, 20.0],
        "PA": 60 * np.pi / 180,
        "q": 0.5,
        "n": 2.0,
        "Re": 5.0,
        "Ie": 1.0,
    }
    changed = {"center": [30, 30], "PA": 30 * np.pi / 180, "q": 0.8, "n": 3, "Re": 10, "Ie": 2}
    model, target = sersic_model, sersic_model.target
    for name, value in saved.items():
        setattr(model, name, value)

    state_file = str(tmp_path / "sersic_state.hdf5")
    model.save_state(state_file, appendable=True)
    for name, value in changed.items():
        setattr(model, name, value)
    target.crtan = [1.0, 2.0]
    model.append_state(state_file)
    model.load_state(state_file, index=0)

    for name, value in saved.items():
        assert np.all(model[name].npvalue == value), f"Model {name} should be loaded correctly"
    assert np.all(target.crtan.npvalue == 0.0), "Model target crtan should be loaded correctly"

    model.load_state(state_file, index=1)
    for name, value in changed.items():
        assert np.all(model[name].npvalue == value), f"appended {name} should be loaded correctly"


######################################################################
# Derivatives
######################################################################


@pytest.mark.parametrize(
    "params",
    [
        {"center": [20.01, 20.02], "PA": 0.0, "q": 0.4, "n": 1, "Re": 15},
        {"center": [25.1, 17.324567], "PA": np.pi / 3, "q": 0.8, "n": 3, "Re": 25.1},
    ],
)
@pytest.mark.parametrize("maxparams", [1, 3])
def test_chunk_jacobian(sersic_target, params, maxparams, monkeypatch):
    """A jacobian computed a few parameters at a time is the same as all at once."""
    model = ap.Model(
        name="test_sersic",
        model_type="sersic galaxy model",
        Ie=10.0,
        target=sersic_target,
        integrate_mode="none",
        psf_convolve=False,
        **params,
    )

    Jtrue = model.jacobian()

    # count the passes, each is at most `maxparams` of the 7 parameters
    chunks = []
    one_pass = model._jacobian
    monkeypatch.setattr(
        model, "_jacobian", lambda pre, x, post: chunks.append(len(x)) or one_pass(pre, x, post)
    )
    model.jacobian_maxparams = maxparams
    Jchunked = model.jacobian()
    assert chunks == [maxparams] * (7 // maxparams) + [7 % maxparams] * (7 % maxparams > 0)
    assert ap.backend.allclose(
        Jtrue.data, Jchunked.data
    ), "Param chunked Jacobian should match full Jacobian"


def test_hessian(sersic_model):
    for likelihood in ("gaussian", "poisson"):
        H = sersic_model.hessian(likelihood=likelihood)
        assert H.shape == (7, 7), "Hessian should have a row and column per parameter"
        assert ap.backend.all(
            ap.backend.isfinite(H)
        ), f"Hessian should be finite for {likelihood} likelihood"
    with pytest.raises(ValueError):
        sersic_model.hessian(likelihood="unknown")


def test_gradient(sersic_model):
    if ap.backend.backend == "jax":
        pytest.skip("JAX backend does not support backward function")
    model = sersic_model
    target = model.target
    target.weight = 1 / (10 + target.variance)
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

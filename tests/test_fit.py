"""The fitters and samplers of astrophot/fit, each asked to improve the same Sersic model.

BatchLM is in test_fit_batch.py. The derivatives the fitters rely on
(jacobian, gradient, hessian) are tested in test_models.py.
"""

import logging

import astrophot as ap
import numpy as np
import pytest

# Each fitter with its options. The iterations are kept to the few needed to
# see the likelihood improve, the tests do not check for convergence.
FITTERS = {
    "LM": (ap.fit.LM, {"max_iter": 10}),
    "LM poisson": (ap.fit.LM, {"max_iter": 10, "likelihood": "poisson"}),
    "IterParam sequential": (
        ap.fit.IterParam,
        {"max_iter": 3, "chunks": 3, "chunk_order": "sequential", "verbose": 2},
    ),
    "IterParam random poisson": (
        ap.fit.IterParam,
        {
            "max_iter": 3,
            "chunks": 3,
            "chunk_order": "random",
            "verbose": 2,
            "likelihood": "poisson",
        },
    ),
    "ScipyFit": (ap.fit.ScipyFit, {"max_iter": 20}),
    "MHMCMC": (ap.fit.MHMCMC, {"max_iter": 20}),
    "HMC": (ap.fit.HMC, {"max_iter": 5, "warmup": 5}),
    "MALA": (ap.fit.MALA, {"max_iter": 20, "epsilon": 1e-3}),
    "MALA poisson": (
        ap.fit.MALA,
        {
            "max_iter": 40,
            "epsilon": 1e-3,
            "likelihood": "poisson",
            "initial_state": [[20, 20, 0.7, np.pi, 2, 15, 10]],
        },
    ),
    "Slalom": (ap.fit.Slalom, {"max_iter": 20}),
}


@pytest.mark.parametrize("name", list(FITTERS))
@pytest.mark.parametrize("fit_valid", [True, False])
def test_fitters(name, sersic_model, fit_valid):
    fitter, options = FITTERS[name]
    if ap.backend.backend == "jax" and fitter in [ap.fit.HMC]:
        pytest.skip("HMC not implemented for JAX backend")
    model = sersic_model
    ll_init = model.gaussian_log_likelihood()
    pll_init = model.poisson_log_likelihood()
    fitter(model, fit_valid=fit_valid, **options).fit()
    ll_final = model.gaussian_log_likelihood()
    pll_final = model.poisson_log_likelihood()
    assert ll_final > ll_init, f"{name} should improve the log likelihood"
    assert pll_final > pll_init, f"{name} should improve the poisson log likelihood"


def test_fitters_iter(sersic_target):
    """Iter fits the models of a group one at a time."""
    model1 = ap.Model(
        name="test1",
        model_type="sersic galaxy model",
        center=[20, 20],
        PA=np.pi,
        q=0.7,
        n=2,
        Re=15,
        Ie=10.0,
        target=sersic_target,
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
        target=sersic_target,
    )
    model = ap.Model(
        name="test_group",
        model_type="group model",
        models=[model1, model2],
        target=sersic_target,
    )
    model.initialize()
    ll_init = model.gaussian_log_likelihood()
    pll_init = model.poisson_log_likelihood()
    ap.fit.Iter(model, max_iter=2).fit()
    ll_final = model.gaussian_log_likelihood()
    pll_final = model.poisson_log_likelihood()
    assert ll_final > ll_init, "Iter should improve the log likelihood"
    assert pll_final > pll_init, "Iter should improve the poisson log likelihood"

    # hessian of a group model
    for likelihood in ("gaussian", "poisson"):
        assert ap.backend.all(
            ap.backend.isfinite(model.hessian(likelihood=likelihood))
        ), f"Hessian should be finite for {likelihood} likelihood"


def test_lm_singular_hessian():
    """A parameter that the data do not constrain makes the hessian singular.

    torch raises an error for a singular matrix and jax returns nan, the fit
    has to carry on in the same way with both."""
    from astrophot.fit.func import lm

    def array(values):
        return ap.backend.as_array(values, dtype=ap.config.DTYPE, device=ap.config.DEVICE)

    hess = array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]])
    grad = array([[1.0], [-0.5], [0.0]])

    hessD, h = lm.solve(hess, grad, L=1.0)
    h = ap.backend.to_numpy(h).ravel()
    assert np.all(np.isfinite(h)), "the step should be finite"
    assert np.all(h[:2] != 0) and h[2] == 0, "only the constrained parameters should move"
    assert np.allclose(ap.backend.to_numpy(hessD) @ h, ap.backend.to_numpy(grad).ravel())

    covariance, singular = lm.invert_hessian(hess)
    assert singular, "the pseudo-inverse should be used"
    covariance = ap.backend.to_numpy(covariance)
    assert np.all(np.isfinite(covariance))
    assert np.allclose(covariance[:2, :2], np.linalg.inv(ap.backend.to_numpy(hess)[:2, :2]))

    covariance, singular = lm.invert_hessian(hess[:2, :2])
    assert not singular
    assert np.allclose(
        ap.backend.to_numpy(covariance) @ ap.backend.to_numpy(hess)[:2, :2], np.eye(2)
    )


def linear_problem(slope, start, data):
    """Model `slope * x` with one parameter, as the arguments of `lm.lm_step`."""

    def array(values):
        return ap.backend.as_array(values, dtype=ap.config.DTYPE, device=ap.config.DEVICE)

    slope, data = array(slope), array(data)
    return {
        "x": array([start]),
        "data": data,
        "model": lambda x: slope * x[0],
        "weight": ap.backend.ones_like(data),
        "jacobian": lambda x: slope.reshape(-1, 1),
    }


def test_lm_step_stops_trying():
    """Ways for one LM step to stop looking for a better damping and keep what it has."""
    from astrophot.fit.func import lm

    # The best fit is x = 2, the start is x = 1 and the residual which no value of
    # x removes keeps each improvement below the 10% that ends the search early.
    problem = linear_problem([1.0, 1.0], 1.0, [5.0, -1.0])
    inside = problem["model"]
    # With L = 1 the first trial goes half way, to x = 1.5. It is accepted and L
    # is lowered. The next trial would go to x = 1.92, where this model is nan.
    problem["model"] = lambda x: ap.backend.module.where(x[0] > 1.75, np.nan, 1.0) * inside(x)
    step = lm.lm_step(**problem, L=1.0)
    assert np.isclose(ap.backend.to_numpy(step["x"])[0], 1.5), "the step before the nan is kept"
    assert np.isclose(step["nll"], 9.25) and step["L"] == 1.0, "down from 10 at the start"

    # Same problem with a slope of 1e6 and a start 1e-9 from the best fit. The first
    # trial improves the fit, the second is too short to matter so the search ends.
    problem = linear_problem([1e6, 1e6], 1.0, [1e6 + 1e-3 + 1.0, 1e6 + 1e-3 - 1.0])
    step = lm.lm_step(**problem, L=1.0)
    assert np.isclose(ap.backend.to_numpy(step["x"])[0] - 1.0, 5e-10, rtol=1e-3)

    with pytest.raises(ValueError, match="Unsupported likelihood"):
        lm.lm_step(**problem, likelihood="cauchy")
    batch = {
        name: value[None] for name, value in problem.items() if name in ("x", "data", "weight")
    }
    with pytest.raises(ValueError, match="Unsupported likelihood"):
        lm.batch_lm_step(
            **batch,
            model=lambda x: batch["data"],
            jacobian=lambda x: ap.backend.ones_like(batch["data"])[..., None],
            mask=ap.backend.ones_like(batch["data"]),
            likelihood="cauchy",
        )


def test_fitter_verbose(sersic_model, capsys):
    """The level of the logger chooses what fitters report, unless they are given `verbose`."""

    def shown(verbose, level):
        ap.set_logger(filename=None, level=level)
        fitter = ap.fit.LM(sersic_model, verbose=verbose)
        fitter.report("progress")
        fitter.report("detail", detail=2)
        fitter.report("problem", level=logging.WARNING)
        return capsys.readouterr().out.splitlines()

    assert shown(None, "INFO") == ["progress", "WARNING: problem"]
    assert shown(None, "DEBUG") == ["progress", "detail", "WARNING: problem"]
    assert shown(None, "WARNING") == ["WARNING: problem"]
    assert shown(0, "DEBUG") == []
    assert shown(1, "DEBUG") == ["progress", "WARNING: problem"]
    assert shown(2, "INFO") == ["progress", "detail", "WARNING: problem"]
    assert shown(2, "WARNING") == ["WARNING: problem"], "the logger has the last word"

    # a fit says how it went, by default
    ap.set_logger(filename=None)
    ap.fit.LM(sersic_model, max_iter=1).fit()
    assert "Final Chi^2/DoF" in capsys.readouterr().out
    ap.fit.LM(sersic_model, max_iter=1, verbose=0).fit()
    assert capsys.readouterr().out == ""


def test_fitter_options(sersic_model):
    model = sersic_model

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

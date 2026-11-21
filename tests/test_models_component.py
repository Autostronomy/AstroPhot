"""Every galaxy, sky and point model type: initialize, sample, fit and total flux.

One parametrized test builds each model type listed by
`ap.models.ComponentModel.List_Models` (astrophot/models/*.py and
astrophot/models/mixins) on the same mock galaxy. PSF models have their own
list in test_models_psf.py.
"""

import astrophot as ap
import numpy as np
import pytest

# Options for model types that need them to stay quick
MODEL_KWARGS = {
    # one parameter per pixel in its window, so give it a small window
    "pixelated model": {"window": (20, 32, 22, 32)},
}

# The sky has little freedom to fit and some of the more complex models need
# extra attention to get a good fit, so these only have to improve at all
MODEST_IMPROVEMENT = (
    "exponential warp galaxy model",
    "ferrer warp galaxy model",
    "ferrer ray galaxy model",
    "isothermal sech2 edgeon model",
)


def skip_reason(model_type):
    jax = ap.backend.backend == "jax"
    if jax and np.random.randint(0, 3) > 0:
        return "JAX is very slow, randomly reducing the number of tests"
    if jax and model_type == "isothermal sech2 edgeon model":
        return "JAX doesnt have bessel function k1 yet"
    if jax and model_type in ["ferrer warp galaxy model", "king warp galaxy model"]:
        return "JAX version doesnt support these models yet, difficulty with gradients"
    if any(t in model_type for t in ["warp", "fourier"]) and "gaussian" not in model_type:
        return "Warp and Fourier models are complex and slow to fit, skipping for now"
    if model_type.startswith("truncated") and "gaussian" not in model_type:
        return "Testing truncated models is redundant"
    return None


@pytest.mark.parametrize(
    "model_type", ap.models.ComponentModel.List_Models(usable=True, types=True)
)
def test_all_model_sample(model_type, sersic_target):
    reason = skip_reason(model_type)
    if reason:
        pytest.skip(reason)

    sersic_target.zeropoint = 22.5
    MODEL = ap.Model(
        name="test_model",
        model_type=model_type,
        target=sersic_target,
        integrate_mode=(
            "none" if ap.backend.backend == "jax" else "bright"
        ),  # JAX JIT is reallly slow for any integration
        **MODEL_KWARGS.get(model_type, {}),
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

    # A few iterations are enough to see whether the model can follow the data
    res = ap.fit.LM(MODEL, max_iter=3, verbose=1).fit()

    print(MODEL)  # test printing

    modest = (
        any(t in model_type for t in ("sky", "king", "spline")) or model_type in MODEST_IMPROVEMENT
    )
    assert res.loss_history[0] > (1.0 if modest else 1.5) * res.loss_history[-1], (
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

    names = MODEL.build_params_array_names()
    assert len(names) == len(set(names)), f"Duplicate parameter name found in {names}"

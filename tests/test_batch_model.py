import astrophot as ap
import numpy as np


def test_batch_model(sersic):

    M = ap.Model(model_type="batch model", model=sersic)
    assert (
        M.target is sersic.target
    ), "BatchModel should share the same target as its component model"
    assert (
        M.window.extent == sersic.window.extent
    ), "BatchModel should share the same window as its component model"
    assert M.mask is sersic.mask, "BatchModel should share the same mask as its component model"

    sersic.center = [[5, 5], [30, 10], [20, 35]]
    sersic.q = [0.7, 0.4, 0.3]

    gll0 = M.gaussian_log_likelihood()
    pll0 = M.poisson_log_likelihood()
    assert ap.backend.isfinite(gll0), "Gaussian log likelihood should be finite"
    assert ap.backend.isfinite(pll0), "Poisson log likelihood should be finite"
    grad = M.gradient()
    assert ap.backend.all(ap.backend.isfinite(grad)), "Gradient should be finite"
    jac = M.jacobian()
    assert ap.backend.all(ap.backend.isfinite(jac.data)), "Jacobian should be finite"

    res = ap.fit.LM(M, max_iter=5).fit()

    assert len(res.loss_history) >= 2, "Optimizer must be able to find steps to improve the model"
    gll1 = M.gaussian_log_likelihood()
    pll1 = M.poisson_log_likelihood()
    assert ap.backend.isfinite(gll1), "Gaussian log likelihood should be finite"
    assert ap.backend.isfinite(pll1), "Poisson log likelihood should be finite"
    assert gll1 > gll0 and pll1 > pll0, "Model should improve the likelihood after fitting"
    assert np.all(
        np.abs(sersic.q.npvalue - np.array([0.7, 0.4, 0.3])) > 0.1
    ), "Model parameters should change after fitting"

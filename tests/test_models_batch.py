"""Batch models: one model evaluated for many sets of parameters
(astrophot/models/batch_model_object.py).

BatchModel adds the copies together on one image, BatchSceneModel makes one
image per copy. The fitter for batches is tested in test_fit_batch.py.
"""

import astrophot as ap
import numpy as np


def test_batch_model_sample(true_sersic_model):
    sersic = true_sersic_model
    M = ap.Model(model_type="batch model", model=sersic)
    assert (
        M.target is sersic.target
    ), "BatchModel should share the same target as its component model"
    assert (
        M.window.extent == sersic.window.extent
    ), "BatchModel should share the same window as its component model"
    assert M.mask is sersic.mask, "BatchModel should share the same mask as its component model"
    M.initialize()

    # Now any parameters may be given an extra dimension for the batch
    sersic.center = [[10, 10], [20, 70], [50, 50], [70, 30], [80, 80]]
    sersic.q = [0.7, 0.6, 0.5, 0.4, 0.3]
    sersic.PA = np.array([30, 60, 90, 120, 150]) * np.pi / 180

    img = M()
    assert isinstance(img, ap.image.ModelImage), "Output of batch model should be a ModelImage"
    assert img.data.shape == sersic.target.data.shape, "batched model should produce regular output"

    M.set_values(ap.backend.ones_like(M.get_values()), attribute="uncertainty")

    assert M.total_flux() > 0, "Total flux of batch model should be positive"
    assert (
        M.total_flux_uncertainty() > 0
    ), "Total flux uncertainty of batch model should be positive"


def test_batch_model_fit(true_sersic_model):
    sersic = true_sersic_model
    sersic.sampling_mode = "midpoint"  # quick, the sampling modes are tested in test_models.py
    M = ap.Model(model_type="batch model", model=sersic)

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

    res = ap.fit.LM(M, max_iter=3).fit()

    assert len(res.loss_history) >= 2, "Optimizer must be able to find steps to improve the model"
    gll1 = M.gaussian_log_likelihood()
    pll1 = M.poisson_log_likelihood()
    assert ap.backend.isfinite(gll1), "Gaussian log likelihood should be finite"
    assert ap.backend.isfinite(pll1), "Poisson log likelihood should be finite"
    assert gll1 > gll0 and pll1 > pll0, "Model should improve the likelihood after fitting"
    assert np.all(
        np.abs(sersic.q.npvalue - np.array([0.7, 0.4, 0.3])) > 0.1
    ), "Model parameters should change after fitting"


def test_batch_scene_model(true_sersic_model):
    sersic = true_sersic_model
    target_batch = ap.TargetImageBatch(
        images=[
            ap.TargetImage(
                data=np.zeros((52, 50)),
                CD=ap.utils.initialize.R(np.pi / 3),
                crtan=(10, 0),
                name="image1",
            ),
            ap.TargetImage(
                data=np.zeros((52, 50)), CD=ap.utils.initialize.R(np.pi / 6), name="image2"
            ),
            ap.TargetImage(
                data=np.zeros((52, 50)),
                CD=ap.utils.initialize.R(np.pi / 16),
                crtan=(-15, 0),
                name="image3",
            ),
        ]
    )
    model = ap.Model(model_type="batch scene model", model=sersic, target=target_batch)
    model.initialize()
    sersic.Ie = [10, 20, 30]
    img = model()
    assert isinstance(
        img, ap.image.ModelImageBatch
    ), "Output of batch scene model should be a ModelImageBatch"
    assert img.data.shape[0] == 3, "Batch scene model should produce output for each target image"
    jac = model.jacobian()
    assert isinstance(
        jac, ap.image.JacobianImageBatch
    ), "Jacobian of batch scene model should be a JacobianImageBatch"
    assert (
        jac.data.shape[0] == 3
    ), "Batch scene model jacobian should produce output for each target image"

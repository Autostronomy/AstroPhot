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
    assert torch.allclose(
        Jtrue.data, Jchunked.data
    ), "Param chunked Jacobian should match full Jacobian"

    model.jacobian_maxparams = 10
    model.jacobian_maxpixels = 20**2

    Jchunked = model.jacobian()

    assert torch.allclose(
        Jtrue.data, Jchunked.data
    ), "Pixel chunked Jacobian should match full Jacobian"


# def test_lm():
#     target = make_basic_sersic()
#     new_model = ap.Model(
#         name="test sersic",
#         model_type="sersic galaxy model",
#         center=[20, 20],
#         PA=60 * np.pi / 180,
#         q=0.5,
#         n=2,
#         Re=5,
#         Ie=10,
#         target=target,
#     )

#     res = ap.fit.LM(new_model).fit()
#     print(res.loss_history)
#     raise Exception()

#     assert res.message == "success", "LM should converge successfully"


# def test_chunk_parameter_jacobian():
#     target = make_basic_sersic()
#     new_model = ap.Model(
#         name="test sersic",
#         model_type="sersic galaxy model",
#         center=[20, 20],
#         PA=60 * np.pi / 180,
#         q=0.5,
#         n=2,
#         Re=5,
#         Ie=10,
#         target=target,
#         jacobian_maxparams=3,
#     )

#     res = ap.fit.LM(new_model).fit()
#     print(res.loss_history)
#     raise Exception()
#     assert res.message == "success", "LM should converge successfully"


# def test_chunk_image_jacobian():
#     target = make_basic_sersic()
#     new_model = ap.Model(
#         name="test sersic",
#         model_type="sersic galaxy model",
#         center=[20, 20],
#         PA=60 * np.pi / 180,
#         q=0.5,
#         n=2,
#         Re=5,
#         Ie=1,
#         target=target,
#         jacobian_maxpixels=20**2,
#     )

#     res = ap.fit.LM(new_model).fit()
#     print(res.loss_history)
#     raise Exception()
#     assert res.message == "success", "LM should converge successfully"


# class TestIter(unittest.TestCase):
#     def test_iter_basic(self):
#         target = make_basic_sersic()
#         model_list = []
#         model_list.append(
#             ap.models.AstroPhot_Model(
#                 name="basic sersic",
#                 model_type="sersic galaxy model",
#                 parameters={
#                     "center": [20, 20],
#                     "PA": 60 * np.pi / 180,
#                     "q": 0.5,
#                     "n": 2,
#                     "Re": 5,
#                     "Ie": 1,
#                 },
#                 target=target,
#             )
#         )
#         model_list.append(
#             ap.models.AstroPhot_Model(
#                 name="basic sky",
#                 model_type="flat sky model",
#                 parameters={"F": -1},
#                 target=target,
#             )
#         )

#         MODEL = ap.models.AstroPhot_Model(
#             name="model",
#             model_type="group model",
#             target=target,
#             models=model_list,
#         )

#         MODEL.initialize()

#         res = ap.fit.Iter(MODEL, method=ap.fit.LM)

#         res.fit()


# class TestIterLM(unittest.TestCase):
#     def test_iter_basic(self):
#         target = make_basic_sersic()
#         model_list = []
#         model_list.append(
#             ap.models.AstroPhot_Model(
#                 name="basic sersic",
#                 model_type="sersic galaxy model",
#                 parameters={
#                     "center": [20, 20],
#                     "PA": 60 * np.pi / 180,
#                     "q": 0.5,
#                     "n": 2,
#                     "Re": 5,
#                     "Ie": 1,
#                 },
#                 target=target,
#             )
#         )
#         model_list.append(
#             ap.models.AstroPhot_Model(
#                 name="basic sky",
#                 model_type="flat sky model",
#                 parameters={"F": -1},
#                 target=target,
#             )
#         )

#         MODEL = ap.models.AstroPhot_Model(
#             name="model",
#             model_type="group model",
#             target=target,
#             models=model_list,
#         )

#         MODEL.initialize()

#         res = ap.fit.Iter_LM(MODEL)

#         res.fit()


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


# class TestMHMCMC(unittest.TestCase):
#     def test_singlesersic(self):
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

#         MHMCMC = ap.fit.MHMCMC(MODEL, epsilon=1e-4, max_iter=100)
#         MHMCMC.fit()

#         self.assertGreater(
#             MHMCMC.acceptance,
#             0.1,
#             "MHMCMC should have nonzero acceptance for simple fits",
#         )

import unittest

import torch
import numpy as np
import matplotlib.pyplot as plt

import astrophot as ap
from utils import make_basic_sersic, make_basic_gaussian

######################################################################
# Fit Objects
######################################################################


class TestComponentModelFits(unittest.TestCase):
    def test_sersic_fit_grad(self):
        """
        Simply test that the gradient optimizer changes the parameters
        """
        np.random.seed(12345)
        N = 50
        Width = 20
        shape = (N + 10, N)
        true_params = [2, 5, 10, -3, 5, 0.7, np.pi / 4]
        IXX, IYY = np.meshgrid(
            np.linspace(-Width, Width, shape[1]), np.linspace(-Width, Width, shape[0])
        )
        QPAXX, QPAYY = ap.utils.conversions.coordinates.Axis_Ratio_Cartesian_np(
            true_params[5], IXX - true_params[3], IYY - true_params[4], true_params[6]
        )
        Z0 = ap.utils.parametric_profiles.sersic_np(
            np.sqrt(QPAXX**2 + QPAYY**2),
            true_params[0],
            true_params[1],
            true_params[2],
        ) + np.random.normal(loc=0, scale=0.1, size=shape)
        tar = ap.image.Target_Image(
            data=Z0,
            pixelscale=0.8,
            variance=np.ones(Z0.shape) * (0.1**2),
        )

        mod = ap.models.Sersic_Galaxy(
            name="sersic model",
            target=tar,
            parameters={
                "center": [-3.2 + N / 2, 5.1 + (N + 10) / 2],
                "q": 0.6,
                "PA": np.pi / 4,
                "n": 2,
                "Re": 5,
                "Ie": 10,
            },
        )

        self.assertFalse(mod.locked, "default model should not be locked")

        mod.initialize()

        mod_initparams = {}
        for p in mod.parameters:
            mod_initparams[p.name] = np.copy(p.vector_representation().detach().cpu().numpy())

        res = ap.fit.Grad(model=mod, max_iter=10).fit()

        for p in mod.parameters:
            self.assertFalse(
                np.any(p.vector_representation().detach().cpu().numpy() == mod_initparams[p.name]),
                f"parameter {p.name} should update with optimization",
            )

    def test_sersic_fit_lm(self):
        """
        Test sersic fitting with entirely independent sersic sampling at 10x resolution.
        """
        N = 50
        pixelscale = 0.8
        shape = (N + 10, N)
        true_params = {
            "center": [shape[0] * pixelscale / 2 - 3.3, shape[1] * pixelscale / 2 + 5.3],
            "n": 2,
            "Re": 10,
            "Ie": 1.0,
            "q": 0.7,
            "PA": np.pi / 4,
        }
        expected_uncertainty = {
            "center": [0.0047, 0.0049],
            "n": 0.0013,
            "Re": 0.0026,
            "Ie": 0.0072,
            "q": 0.0277,
            "PA": 0.0022,
        }
        tar = make_basic_sersic(
            N=shape[0],
            M=shape[1],
            pixelscale=pixelscale,
            x=true_params["center"][0],
            y=true_params["center"][1],
            n=true_params["n"],
            Re=true_params["Re"],
            Ie=true_params["Ie"],
            q=true_params["q"],
            PA=true_params["PA"],
        )

        mod = ap.models.Sersic_Galaxy(
            name="sersic model",
            target=tar,
        )

        mod.initialize()
        ap.AP_config.set_logging_output(stdout=True, filename="AstroPhot.log")
        res = ap.fit.LM(model=mod, verbose=2).fit()

        res.update_uncertainty()

        self.assertAlmostEqual(
            mod["center"].value[0].item() / true_params["center"][0],
            1,
            2,
            "LM should accurately recover parameters in simple cases",
        )
        self.assertAlmostEqual(
            mod["center"].value[1].item() / true_params["center"][1],
            1,
            2,
            "LM should accurately recover parameters in simple cases",
        )
        self.assertAlmostEqual(
            mod["n"].value.item(),
            true_params["n"],
            1,
            msg="LM should accurately recover parameters in simple cases",
        )
        self.assertAlmostEqual(
            (mod["Re"].value.item()) / true_params["Re"],
            1,
            delta=1,
            msg="LM should accurately recover parameters in simple cases",
        )
        self.assertAlmostEqual(
            mod["Ie"].value.item(),
            true_params["Ie"],
            1,
            "LM should accurately recover parameters in simple cases",
        )
        self.assertAlmostEqual(
            mod["PA"].value.item() / true_params["PA"],
            1,
            delta=0.5,
            msg="LM should accurately recover parameters in simple cases",
        )
        self.assertAlmostEqual(
            mod["q"].value.item(),
            true_params["q"],
            1,
            "LM should accurately recover parameters in simple cases",
        )
        cov = res.covariance_matrix
        self.assertAlmostEqual(
            mod["center"].uncertainty[0].item(),
            expected_uncertainty["center"][0],
            1,
            "LM should accurately recover parameter uncertainty in simple cases",
        )
        self.assertAlmostEqual(
            mod["center"].uncertainty[1].item(),
            expected_uncertainty["center"][1],
            1,
            "LM should accurately recover parameter uncertainty in simple cases",
        )
        self.assertAlmostEqual(
            mod["n"].uncertainty.item(),
            expected_uncertainty["n"],
            1,
            "LM should accurately recover parameter uncertainty in simple cases",
        )
        self.assertAlmostEqual(
            mod["Re"].uncertainty.item(),
            expected_uncertainty["Re"],
            1,
            "LM should accurately recover parameter uncertainty in simple cases",
        )
        self.assertAlmostEqual(
            mod["Ie"].uncertainty.item(),
            expected_uncertainty["Ie"],
            1,
            "LM should accurately recover parameter uncertainty in simple cases",
        )
        self.assertAlmostEqual(
            mod["q"].uncertainty.item(),
            expected_uncertainty["q"],
            1,
            "LM should accurately recover parameter uncertainty in simple cases",
        )
        self.assertAlmostEqual(
            mod["q"].uncertainty.item(),
            expected_uncertainty["q"],
            1,
            "LM should accurately recover parameter uncertainty in simple cases",
        )
        self.assertAlmostEqual(
            mod["q"].uncertainty.item(),
            expected_uncertainty["q"],
            1,
            "LM should accurately recover parameter uncertainty in simple cases",
        )
        self.assertAlmostEqual(
            mod["PA"].uncertainty.item(),
            expected_uncertainty["PA"],
            1,
            "LM should accurately recover parameter uncertainty in simple cases",
        )


class TestGroupModelFits(unittest.TestCase):
    def test_groupmodel_fit(self):
        """
        Simply test that fitting a group model changes the parameter values
        """
        np.random.seed(12345)
        N = 50
        Width = 20
        shape = (N + 10, N)
        true_params1 = [2, 4, 10, -3, 5, 0.7, np.pi / 4]
        true_params2 = [1.2, 6, 8, 2, -3, 0.5, -np.pi / 4]
        IXX, IYY = np.meshgrid(
            np.linspace(-Width, Width, shape[1]), np.linspace(-Width, Width, shape[0])
        )
        QPAXX, QPAYY = ap.utils.conversions.coordinates.Axis_Ratio_Cartesian_np(
            true_params1[5],
            IXX - true_params1[3],
            IYY - true_params1[4],
            true_params1[6],
        )
        Z0 = ap.utils.parametric_profiles.sersic_np(
            np.sqrt(QPAXX**2 + QPAYY**2),
            true_params1[0],
            true_params1[1],
            true_params1[2],
        )
        QPAXX, QPAYY = ap.utils.conversions.coordinates.Axis_Ratio_Cartesian_np(
            true_params2[5],
            IXX - true_params2[3],
            IYY - true_params2[4],
            true_params2[6],
        )
        Z0 += ap.utils.parametric_profiles.sersic_np(
            np.sqrt(QPAXX**2 + QPAYY**2),
            true_params2[0],
            true_params2[1],
            true_params2[2],
        )
        Z0 += np.random.normal(loc=0, scale=0.1, size=shape)
        tar = ap.image.Target_Image(
            data=Z0,
            pixelscale=0.8,
            variance=np.ones(Z0.shape) * (0.1**2),
        )

        mod1 = ap.models.Sersic_Galaxy(
            name="sersic model 1",
            target=tar,
            parameters={"center": {"value": [-3.2 + N / 2, 5.1 + (N + 10) / 2]}},
        )
        mod2 = ap.models.Sersic_Galaxy(
            name="sersic model 2",
            target=tar,
            parameters={"center": {"value": [2.1 + N / 2, -3.1 + (N + 10) / 2]}},
        )

        smod = ap.models.Group_Model(name="group model", models=[mod1, mod2], target=tar)

        self.assertFalse(smod.locked, "default model should not be locked")

        smod.initialize()

        mod1_initparams = {}
        for p in mod1.parameters:
            mod1_initparams[p.name] = np.copy(p.vector_representation().detach().cpu().numpy())
        mod2_initparams = {}
        for p in mod2.parameters:
            mod2_initparams[p.name] = np.copy(p.vector_representation().detach().cpu().numpy())

        res = ap.fit.Grad(model=smod, max_iter=10).fit()

        for p in mod1.parameters:
            self.assertFalse(
                np.any(p.vector_representation().detach().cpu().numpy() == mod1_initparams[p.name]),
                f"mod1 parameter {p.name} should update with optimization",
            )
        for p in mod2.parameters:
            self.assertFalse(
                np.any(p.vector_representation().detach().cpu().numpy() == mod2_initparams[p.name]),
                f"mod2 parameter {p.name} should update with optimization",
            )


class TestLM(unittest.TestCase):
    def test_lm_creation(self):
        target = make_basic_sersic()
        new_model = ap.models.AstroPhot_Model(
            name="test sersic",
            model_type="sersic galaxy model",
            parameters={
                "center": [20, 20],
                "PA": 60 * np.pi / 180,
                "q": 0.5,
                "n": 2,
                "Re": 5,
                "Ie": 1,
            },
            target=target,
        )

        LM = ap.fit.LM(new_model, max_iter=10)

        LM.fit()

    def test_chunk_parameter_jacobian(self):
        target = make_basic_sersic()
        new_model = ap.models.AstroPhot_Model(
            name="test sersic",
            model_type="sersic galaxy model",
            parameters={
                "center": [20, 20],
                "PA": 60 * np.pi / 180,
                "q": 0.5,
                "n": 2,
                "Re": 5,
                "Ie": 1,
            },
            target=target,
            jacobian_chunksize=3,
        )

        LM = ap.fit.LM(new_model, max_iter=10)

        LM.fit()

    def test_chunk_image_jacobian(self):
        target = make_basic_sersic()
        new_model = ap.models.AstroPhot_Model(
            name="test sersic",
            model_type="sersic galaxy model",
            parameters={
                "center": [20, 20],
                "PA": 60 * np.pi / 180,
                "q": 0.5,
                "n": 2,
                "Re": 5,
                "Ie": 1,
            },
            target=target,
            image_chunksize=15,
        )

        LM = ap.fit.LM(new_model, max_iter=10)

        LM.fit()


class TestMiniFit(unittest.TestCase):
    def test_minifit(self):
        target = make_basic_sersic()
        new_model = ap.models.AstroPhot_Model(
            name="test sersic",
            model_type="sersic galaxy model",
            parameters={
                "center": [20, 20],
                "PA": 60 * np.pi / 180,
                "q": 0.5,
                "n": 2,
                "Re": 5,
                "Ie": 1,
            },
            target=target,
        )

        MF = ap.fit.MiniFit(
            new_model, downsample_factor=2, method_quargs={"max_iter": 10}, verbose=1
        )

        MF.fit()


class TestIter(unittest.TestCase):
    def test_iter_basic(self):
        target = make_basic_sersic()
        model_list = []
        model_list.append(
            ap.models.AstroPhot_Model(
                name="basic sersic",
                model_type="sersic galaxy model",
                parameters={
                    "center": [20, 20],
                    "PA": 60 * np.pi / 180,
                    "q": 0.5,
                    "n": 2,
                    "Re": 5,
                    "Ie": 1,
                },
                target=target,
            )
        )
        model_list.append(
            ap.models.AstroPhot_Model(
                name="basic sky",
                model_type="flat sky model",
                parameters={"F": -1},
                target=target,
            )
        )

        MODEL = ap.models.AstroPhot_Model(
            name="model",
            model_type="group model",
            target=target,
            models=model_list,
        )

        MODEL.initialize()

        res = ap.fit.Iter(MODEL, method=ap.fit.LM)

        res.fit()


class TestIterLM(unittest.TestCase):
    def test_iter_basic(self):
        target = make_basic_sersic()
        model_list = []
        model_list.append(
            ap.models.AstroPhot_Model(
                name="basic sersic",
                model_type="sersic galaxy model",
                parameters={
                    "center": [20, 20],
                    "PA": 60 * np.pi / 180,
                    "q": 0.5,
                    "n": 2,
                    "Re": 5,
                    "Ie": 1,
                },
                target=target,
            )
        )
        model_list.append(
            ap.models.AstroPhot_Model(
                name="basic sky",
                model_type="flat sky model",
                parameters={"F": -1},
                target=target,
            )
        )

        MODEL = ap.models.AstroPhot_Model(
            name="model",
            model_type="group model",
            target=target,
            models=model_list,
        )

        MODEL.initialize()

        res = ap.fit.Iter_LM(MODEL)

        res.fit()


class TestHMC(unittest.TestCase):
    def test_hmc_sample(self):
        np.random.seed(12345)
        N = 50
        pixelscale = 0.8
        true_params = {
            "n": 2,
            "Re": 10,
            "Ie": 1,
            "center": [-3.3, 5.3],
            "q": 0.7,
            "PA": np.pi / 4,
        }
        target = ap.image.Target_Image(
            data=np.zeros((N, N)),
            pixelscale=pixelscale,
        )

        MODEL = ap.models.Sersic_Galaxy(
            name="sersic model",
            target=target,
            parameters=true_params,
        )
        img = MODEL().data.detach().cpu().numpy()
        target.data = torch.Tensor(
            img
            + np.random.normal(scale=0.1, size=img.shape)
            + np.random.normal(scale=np.sqrt(img) / 10)
        )
        target.variance = torch.Tensor(0.1**2 + img / 100)

        HMC = ap.fit.HMC(MODEL, epsilon=1e-5, max_iter=5, warmup=2)
        HMC.fit()


class TestNUTS(unittest.TestCase):
    def test_nuts_sample(self):
        np.random.seed(12345)
        N = 50
        pixelscale = 0.8
        true_params = {
            "n": 2,
            "Re": 10,
            "Ie": 1,
            "center": [-3.3, 5.3],
            "q": 0.7,
            "PA": np.pi / 4,
        }
        target = ap.image.Target_Image(
            data=np.zeros((N, N)),
            pixelscale=pixelscale,
        )

        MODEL = ap.models.Sersic_Galaxy(
            name="sersic model",
            target=target,
            parameters=true_params,
        )
        img = MODEL().data.detach().cpu().numpy()
        target.data = torch.Tensor(
            img
            + np.random.normal(scale=0.1, size=img.shape)
            + np.random.normal(scale=np.sqrt(img) / 10)
        )
        target.variance = torch.Tensor(0.1**2 + img / 100)

        NUTS = ap.fit.NUTS(MODEL, max_iter=5, warmup=2)
        NUTS.fit()


class TestMHMCMC(unittest.TestCase):
    def test_singlesersic(self):
        np.random.seed(12345)
        N = 50
        pixelscale = 0.8
        true_params = {
            "n": 2,
            "Re": 10,
            "Ie": 1,
            "center": [-3.3, 5.3],
            "q": 0.7,
            "PA": np.pi / 4,
        }
        target = ap.image.Target_Image(
            data=np.zeros((N, N)),
            pixelscale=pixelscale,
        )

        MODEL = ap.models.Sersic_Galaxy(
            name="sersic model",
            target=target,
            parameters=true_params,
        )
        img = MODEL().data.detach().cpu().numpy()
        target.data = torch.Tensor(
            img
            + np.random.normal(scale=0.1, size=img.shape)
            + np.random.normal(scale=np.sqrt(img) / 10)
        )
        target.variance = torch.Tensor(0.1**2 + img / 100)

        MHMCMC = ap.fit.MHMCMC(MODEL, epsilon=1e-4, max_iter=100)
        MHMCMC.fit()

        self.assertGreater(
            MHMCMC.acceptance,
            0.1,
            "MHMCMC should have nonzero acceptance for simple fits",
        )


if __name__ == "__main__":
    unittest.main()

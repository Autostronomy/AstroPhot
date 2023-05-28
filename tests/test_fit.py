import unittest

import torch
import numpy as np

import autoprof as ap
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
            np.sqrt(QPAXX ** 2 + QPAYY ** 2),
            true_params[0],
            true_params[1],
            true_params[2],
        ) + np.random.normal(loc=0, scale=0.1, size=shape)
        tar = ap.image.Target_Image(
            data=Z0,
            pixelscale=0.8,
            variance=np.ones(Z0.shape) * (0.1 ** 2),
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
            mod_initparams[p.name] = np.copy(p.representation.detach().numpy())

        res = ap.fit.Grad(model=mod, max_iter=10).fit()

        for p in mod.parameters:
            self.assertFalse(
                np.any(p.representation.detach().numpy() == mod_initparams[p.name]),
                f"parameter {p.name} should update with optimization",
            )

    def test_sersic_fit_lm(self):
        """
        Test sersic fitting with entirely independent sersic sampling at 10x resolution.
        """
        np.random.seed(12345)
        N = 50
        pixelscale = 0.8
        upsample = 10
        shape = (N + 10, N)
        # true_params = [2,5,10,-3, 5, 0.7, np.pi/4]
        true_params = {
            "n": 2,
            "Re": 10,
            "Ie": 1,
            "center": [shape[0]*pixelscale/2 - 3.3, shape[1]*pixelscale/2 + 5.3],
            "q": 0.7,
            "pa": np.pi / 4,
        }
        IXX, IYY = np.meshgrid(
            np.linspace(
                pixelscale / (2 * upsample),
                shape[1]*pixelscale - (pixelscale / (2 * upsample)),
                shape[1] * upsample,
            ),
            np.linspace(
                pixelscale / (2 * upsample),
                shape[0]*pixelscale - (pixelscale / (2 * upsample)),
                shape[0] * upsample,
            ),
        )
        QPAXX, QPAYY = ap.utils.conversions.coordinates.Axis_Ratio_Cartesian_np(
            true_params["q"],
            IXX - true_params["center"][0],
            IYY - true_params["center"][1],
            -true_params["pa"],
        )
        Z0 = ap.utils.parametric_profiles.sersic_np(
            np.sqrt(QPAXX ** 2 + QPAYY ** 2),
            true_params["n"],
            true_params["Re"],
            true_params["Ie"],
        ) + np.random.normal(
            loc=0, scale=0.1, size=(shape[0] * upsample, shape[1] * upsample)
        )
        Z0 = (
            Z0.reshape(shape[0], upsample, shape[1], upsample).sum(axis=(1, 3))
            / upsample ** 2
        )
        tar = ap.image.Target_Image(
            data=Z0,
            pixelscale=pixelscale,
            variance=np.ones(Z0.shape) * (0.1 ** 2),
        )

        mod = ap.models.Sersic_Galaxy(
            name="sersic model",
            target=tar,
        )

        mod.initialize()

        ap.AP_config.set_logging_output(stdout=True, filename="AutoProf.log")
        res = ap.fit.LM(model=mod, verbose=1).fit()
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
            np.log10(true_params["Ie"] / pixelscale ** 2),
            2,
            "LM should accurately recover parameters in simple cases",
        )
        self.assertAlmostEqual(
            mod["PA"].value.item() / true_params["pa"],
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
            np.sqrt(QPAXX ** 2 + QPAYY ** 2),
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
            np.sqrt(QPAXX ** 2 + QPAYY ** 2),
            true_params2[0],
            true_params2[1],
            true_params2[2],
        )
        Z0 += np.random.normal(loc=0, scale=0.1, size=shape)
        tar = ap.image.Target_Image(
            data=Z0,
            pixelscale=0.8,
            variance=np.ones(Z0.shape) * (0.1 ** 2),
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

        smod = ap.models.Group_Model(
            name="group model", models=[mod1, mod2], target=tar
        )

        self.assertFalse(smod.locked, "default model should not be locked")

        smod.initialize()

        mod1_initparams = {}
        for p in mod1.parameters:
            mod1_initparams[p.name] = np.copy(p.representation.detach().numpy())
        mod2_initparams = {}
        for p in mod2.parameters:
            mod2_initparams[p.name] = np.copy(p.representation.detach().numpy())

        res = ap.fit.Grad(model=smod, max_iter=10).fit()

        for p in mod1.parameters:
            self.assertFalse(
                np.any(p.representation.detach().numpy() == mod1_initparams[p.name]),
                f"mod1 parameter {p.name} should update with optimization",
            )
        for p in mod2.parameters:
            self.assertFalse(
                np.any(p.representation.detach().numpy() == mod2_initparams[p.name]),
                f"mod2 parameter {p.name} should update with optimization",
            )


class TestLM(unittest.TestCase):
    def test_lm_creation(self):
        target = make_basic_sersic()
        new_model = ap.models.AutoProf_Model(
            name="constrained sersic",
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

        LM = ap.fit.LM(new_model)

        LM.fit()

        LM.grad_step()

        LM.undo_step()

        LM.take_low_rho_step()
        for i in reversed(range(len(LM.decision_history))):
            if "accept" in LM.decision_history[i]:
                LM.decision_history[i] = "reject"
                break
        LM.take_low_rho_step()

    def test_lm_constraint(self):
        def dummy_constraint(p):
            return torch.sum(torch.abs(p))

        new_constraint = ap.fit.LM_Constraint(dummy_constraint)
        target = make_basic_sersic()
        new_model = ap.models.AutoProf_Model(
            name="constrained sersic",
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

        jac = new_constraint.jacobian(new_model)
        samp = new_constraint(new_model)

        self.assertTrue(
            torch.all(torch.isfinite(jac)),
            "Constraint Jacobian should produce real numbers",
        )
        self.assertTrue(
            torch.all(torch.isclose(jac, torch.ones_like(jac))),
            "Constraint Jacobian should produce correct derivatives",
        )
        self.assertTrue(
            torch.all(torch.isfinite(samp)),
            "Constraint sample should produce real numbers",
        )


class TestIter(unittest.TestCase):
    def test_iter_basic(self):
        target = make_basic_sersic()
        model_list = []
        model_list.append(
            ap.models.AutoProf_Model(
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
            ap.models.AutoProf_Model(
                name="basic sky",
                model_type="flat sky model",
                parameters={"sky": -1},
                target=target,
            )
        )

        MODEL = ap.models.AutoProf_Model(
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
            ap.models.AutoProf_Model(
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
            ap.models.AutoProf_Model(
                name="basic sky",
                model_type="flat sky model",
                parameters={"sky": -1},
                target=target,
            )
        )

        MODEL = ap.models.AutoProf_Model(
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
        target.variance = torch.Tensor(0.1 ** 2 + img / 100)

        HMC = ap.fit.HMC(MODEL, epsilon=1e-5, max_iter=10, warmup = 5)
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
        target.variance = torch.Tensor(0.1 ** 2 + img / 100)

        NUTS = ap.fit.NUTS(MODEL, max_iter=10, warmup = 5)
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
        target.variance = torch.Tensor(0.1 ** 2 + img / 100)

        MHMCMC = ap.fit.MHMCMC(MODEL, epsilon=1e-4, max_iter=100)
        MHMCMC.fit()

        self.assertGreater(
            MHMCMC.acceptance,
            0.1,
            "MHMCMC should have nonzero acceptance for simple fits",
        )

if __name__ == "__main__":
    unittest.main()

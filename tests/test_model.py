import unittest
import astrophot as ap
import torch
import numpy as np
from utils import make_basic_sersic

######################################################################
# Model Objects
######################################################################


class TestModel(unittest.TestCase):
    def test_AstroPhot_Model(self):

        self.assertRaises(AssertionError, ap.models.AstroPhot_Model, name = "my|model")

        model = ap.models.AstroPhot_Model(name = "test model")

        self.assertIsNone(model.target, "model should not have a target at this point")

        target = ap.image.Target_Image(data=torch.zeros((16, 32)), pixelscale=1.0)

        model.target = target

        model.window = target.window

        model.locked = True
        model.locked = False

        state = model.get_state()

    # def test_basic_model_methods(self):

    #     target = make_basic_sersic()
    #     model = ap.models.AstroPhot_Model(
    #         name="test sersic",
    #         model_type="sersic galaxy model",
    #         parameters={
    #             "center": [20, 20],
    #             "PA": 60 * np.pi / 180,
    #             "q": 0.5,
    #             "n": 2,
    #             "Re": 5,
    #             "Ie": 1,
    #         },
    #         target=target,
    #     )
    #     rep = model.parameters.get_vector(as_representation = True)
    #     nat = model.parameters.get_vector(as_representation = False)
    #     self.assertTrue(torch.all(torch.isclose(rep, model.parameters.transform(nat, to_representation = True))), "transform should map between parameter natural and representation")
    #     self.assertTrue(torch.all(torch.isclose(nat, model.parameters.transform(rep, to_representation = False))), "transform should map between parameter representation and natural")

    def test_model_sampling_modes(self):

        target = make_basic_sersic(100,100)
        model = ap.models.AstroPhot_Model(
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
        res = model()
        model.sampling_mode = "trapezoid"
        res = model()
        model.sampling_mode = "simpson"
        res = model()
        model.integrate_mode = "none"
        res = model()
        model.integrate_mode = "should raise"
        self.assertRaises(ValueError, model)

        # test PSF modes
        target.psf = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
        model.integrate_mode = "none"
        model.psf_mode = "full"
        model.psf_convolve_mode = "direct"
        res = model()
        model.psf_convolve_mode = "fft"
        res = model()

    def test_model_creation(self):
        np.random.seed(12345)
        shape = (10, 15)
        tar = ap.image.Target_Image(
            data=np.random.normal(loc=0, scale=1.4, size=shape),
            pixelscale=0.8,
            variance=np.ones(shape) * (1.4 ** 2),
            psf=np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]]),
        )

        mod = ap.models.Component_Model(
            name="base model",
            target=tar,
            parameters={"center": {"value": [5, 5], "locked": True}},
        )

        mod.initialize()

        self.assertFalse(mod.locked, "default model should not be locked")

        self.assertTrue(
            torch.all(mod().data == 0), "Component_Model model_image should be zeros"
        )
    
    def test_mask(self):

        target = make_basic_sersic()
        mask = torch.zeros_like(target.data)
        mask[10,13] = 1
        model = ap.models.AstroPhot_Model(
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
            mask = mask
        )

        sample = model()
        self.assertEqual(sample.data[10,13].item(), 0., "masked values should be zero")
        self.assertNotEqual(sample.data[11,12].item(), 0., "unmasked values should NOT be zero")


class TestAllModelBasics(unittest.TestCase):
    def test_all_model_init(self):

        target = make_basic_sersic()
        for model_type in ap.models.Component_Model.List_Model_Names(useable=True):
            MODEL = ap.models.AstroPhot_Model(
                name="test model",
                model_type=model_type,
                target=target,
            )
            MODEL.initialize()
            for P in MODEL.parameter_order:
                self.assertIsNotNone(
                    MODEL[P].value,
                    f"Model type {model_type} parameter {P} should not be None after initialization",
                )
                # perhaps add check that uncertainty is not none

    def test_all_model_sample(self):

        target = make_basic_sersic()
        for model_type in ap.models.Component_Model.List_Model_Names(useable=True):
            MODEL = ap.models.AstroPhot_Model(
                name="test model",
                model_type=model_type,
                target=target,
            )
            MODEL.initialize()
            img = MODEL()
            self.assertTrue(
                torch.all(torch.isfinite(img.data)),
                "Model should evaluate a real number for the full image",
            )


class TestSersic(unittest.TestCase):
    def test_sersic_creation(self):
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
            parameters={"center": [-3.2 + N / 2, 5.1 + (N + 10) / 2]},
        )

        self.assertFalse(mod.locked, "default model should not be locked")

        mod.initialize()

    def test_sersic_save_load(self):

        target = make_basic_sersic()
        model = ap.models.AstroPhot_Model(
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

        model.initialize()

        model.save("test_AstroPhot_sersic.yaml")
        model2 = ap.models.AstroPhot_Model(
            name="load model",
            filename="test_AstroPhot_sersic.yaml",
        )

        # fixme
        for P in model.parameter_order:
            self.assertAlmostEqual(
                model[P].value.detach().cpu().tolist(),
                model2[P].value.detach().cpu().tolist(),
                msg="loaded model should have same parameters",
            )


class TestGroup(unittest.TestCase):
    def test_groupmodel_creation(self):
        np.random.seed(12345)
        shape = (10, 15)
        tar = ap.image.Target_Image(
            data=np.random.normal(loc=0, scale=1.4, size=shape),
            pixelscale=0.8,
            variance=np.ones(shape) * (1.4 ** 2),
        )

        mod1 = ap.models.Component_Model(
            name="base model 1",
            target=tar,
            parameters={"center": {"value": [5, 5], "locked": True}},
        )
        mod2 = ap.models.Component_Model(
            name="base model 2",
            target=tar,
            parameters={"center": {"value": [5, 5], "locked": True}},
        )

        smod = ap.models.AstroPhot_Model(
            name="group model",
            model_type="group model",
            model_list=[mod1, mod2],
            target=tar,
        )

        self.assertFalse(smod.locked, "default model state should not be locked")

        smod.initialize()

        self.assertTrue(torch.all(smod().data == 0), "model_image should be zeros")


if __name__ == "__main__":
    unittest.main()

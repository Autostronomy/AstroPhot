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

        model = ap.models.AstroPhot_Model(name = "test model")

        self.assertIsNone(model.target, "model should not have a target at this point")

        target = ap.image.Target_Image(data=torch.zeros((16, 32)), pixelscale=1.0)

        model.target = target

        model.window = target.window

        model.locked = True
        model.locked = False

        state = model.get_state()

    def test_initialize_does_not_recurse(self):
        "Test case for error where missing parameter name triggered print that triggered missing parameter name ..."
        target = make_basic_sersic()
        model = ap.models.AstroPhot_Model(
            name="test model",
            model_type="sersic galaxy model",
            target=target,
        )
        # Define a function that accesses a parameter that doesn't exist
        def calc(params):
            return params["A"].value

        model["center"].value = calc

        with self.assertRaises(KeyError) as context:
            model.initialize()

    def test_basic_model_methods(self):

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
        rep = model.parameters.vector_representation()
        nat = model.parameters.vector_values()
        self.assertTrue(torch.all(torch.isclose(rep, model.parameters.vector_transform_val_to_rep(nat))), "transform should map between parameter natural and representation")
        self.assertTrue(torch.all(torch.isclose(nat, model.parameters.vector_transform_rep_to_val(rep))), "transform should map between parameter representation and natural")

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
        model.sampling_mode = "simpsons"
        res = model()
        model.sampling_mode = "quad:3"
        res = model()
        model.integrate_mode = "none"
        res = model()
        model.integrate_mode = "should raise"
        self.assertRaises(ap.errors.SpecificationConflict, model)
        model.integrate_mode = "none"
        model.sampling_mode = "should raise"
        self.assertRaises(ap.errors.SpecificationConflict, model)     
        model.sampling_mode = "midpoint"   

        # test PSF modes
        model.psf = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
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

    def test_model_errors(self):

        # Invalid name
        self.assertRaises(ap.errors.NameNotAllowed, ap.models.AstroPhot_Model, name = "my|model")

        # Target that is not a target image
        arr = torch.zeros((10, 15))
        target = ap.image.Image(
            data = arr, pixelscale=1.0, zeropoint=1.0, origin=torch.zeros(2), note="test image"
        )

        with self.assertRaises(ap.errors.InvalidTarget):
            model = ap.models.AstroPhot_Model(
                name="test model",
                model_type="sersic galaxy model",
                target=target,
            )

        # model that doesnt exist
        target = make_basic_sersic()
        with self.assertRaises(ap.errors.UnrecognizedModel):
            model = ap.models.AstroPhot_Model(
                name="test model",
                model_type="sersic gaaxy model",
                target=target,
            )

        # invalid window
        with self.assertRaises(ap.errors.InvalidWindow):
            model = ap.models.AstroPhot_Model(
                name="test model",
                model_type="sersic galaxy model",
                target=target,
                window = (1,2,3),
            )
            
        


class TestAllModelBasics(unittest.TestCase):
    def test_all_model_sample(self):

        target = make_basic_sersic()
        for model_type in ap.models.Component_Model.List_Model_Names(useable=True):
            print(model_type)
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
            img = MODEL()
            self.assertTrue(
                torch.all(torch.isfinite(img.data)),
                "Model should evaluate a real number for the full image",
            )
            self.assertIsInstance(str(MODEL), str, "String representation should return string")
            self.assertIsInstance(repr(MODEL), str, "Repr should return string")

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
            models=[mod1, mod2],
            target=tar,
        )

        self.assertFalse(smod.locked, "default model state should not be locked")

        smod.initialize()

        self.assertTrue(torch.all(smod().data == 0), "model_image should be zeros")

    def test_jointmodel_creation(self):
        np.random.seed(12345)
        shape = (10, 15)
        tar = ap.image.Target_Image(
            data=np.random.normal(loc=0, scale=1.4, size=shape),
            pixelscale=0.8,
            variance=np.ones(shape) * (1.4 ** 2),
        )
        shape2 = (33, 42)
        tar2 = ap.image.Target_Image(
            data=np.random.normal(loc=0, scale=1.4, size=shape2),
            pixelscale=0.3,
            origin = (43.2, 78.01),
            variance=np.ones(shape2) * (1.4 ** 2),
        )

        mod1 = ap.models.Flat_Sky(
            name="base model 1",
            target=tar,
        )
        mod2 = ap.models.Flat_Sky(
            name="base model 2",
            target=tar2,
        )

        smod = ap.models.AstroPhot_Model(
            name="group model",
            model_type="group model",
            models=[mod1, mod2],
            target=tar,
        )

        self.assertFalse(smod.locked, "default model state should not be locked")

        smod.initialize()

        self.assertTrue(torch.all(torch.isfinite(smod().data)), "model_image should be real")

    def test_groupmodel_saveload(self):
        np.random.seed(12345)
        tar = make_basic_sersic(N = 51, M=51)

        psf = ap.models.Moffat_Star(
            name="psf model 1",
            target=tar,
            parameters={
                "center": {"value": [5, 5], "locked": True},
                "n": 2.,
                "Rd": 3.,
                "I0": {"value": 0., "locked": True},
            },
        )
        
        mod1 = ap.models.Sersic_Galaxy(
            name="base model 1",
            target=tar,
            parameters={"center": {"value": [5, 5], "locked": False}},
            psf = psf,
            psf_mode = "full",
        )
        mod2 = ap.models.Sersic_Galaxy(
            name="base model 2",
            target=tar,
            parameters={"center": {"value": [5, 5], "locked": False}},
        )

        smod = ap.models.AstroPhot_Model(
            name="group model",
            model_type="group model",
            models=[mod1, mod2],
            target=tar,
        )

        self.assertFalse(smod.locked, "default model state should not be locked")

        smod.initialize()

        self.assertTrue(torch.all(torch.isfinite(smod().data)), "model_image should be real values")

        smod.save("test_save_group_model.yaml")

        newmod = ap.models.AstroPhot_Model(
            name = "group model",
            filename = "test_save_group_model.yaml",
        )
        self.assertEqual(len(smod.models), len(newmod.models), "Group model should load sub models")

        self.assertEqual(newmod.parameters.size, 16, "Group model size should sum all parameters")

        self.assertTrue(torch.all(newmod.parameters.vector_values() == smod.parameters.vector_values()), "Save/load should extract all parameters")
        
if __name__ == "__main__":
    unittest.main()

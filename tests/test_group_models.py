import unittest
import astrophot as ap
import torch
import numpy as np
from utils import make_basic_sersic, make_basic_gaussian, make_basic_gaussian_psf


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

        psf = ap.models.Moffat_PSF(
            name="psf model 1",
            target=make_basic_gaussian_psf(N = 11),
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
        

class TestPSFGroup(unittest.TestCase):
    def test_psfgroupmodel_creation(self):
        tar = make_basic_gaussian_psf()

        mod1 = ap.models.AstroPhot_Model(
            name="base model 1",
            model_type="moffat psf model",
            target=tar,
        )
        
        mod2 = ap.models.AstroPhot_Model(
            name="base model 2",
            model_type="moffat psf model",
            target=tar,
        )

        smod = ap.models.AstroPhot_Model(
            name="group model",
            model_type="psf group model",
            models=[mod1, mod2],
            target=tar,
        )

        smod.initialize()

        self.assertTrue(torch.all(smod().data >= 0), "PSF group sample should be greater than or equal to zero")

    def test_psfgroupmodel_saveload(self):
        np.random.seed(12345)
        tar = make_basic_gaussian_psf()

        psf1 = ap.models.Moffat_PSF(
            name="psf model 1",
            target=tar,
            parameters={
                "n": 2.,
                "Rd": 3.,
            },
        )
        
        psf2 = ap.models.Sersic_PSF(
            name="psf model 2",
            target=tar,
            parameters={
                "n": 2.,
                "Re": 3.,
            },
        )

        smod = ap.models.AstroPhot_Model(
            name="group model",
            model_type="psf group model",
            models=[psf1, psf2],
            target=tar,
        )

        smod.initialize()

        self.assertTrue(torch.all(torch.isfinite(smod().data)), "psf_image should be real values")

        smod.save("test_save_psfgroup_model.yaml")

        newmod = ap.models.AstroPhot_Model(
            name = "group model",
            filename = "test_save_psfgroup_model.yaml",
        )
        self.assertEqual(len(smod.models), len(newmod.models), "Group model should load sub models")

        self.assertEqual(newmod.parameters.size, 4, "Group model size should sum all parameters")

        self.assertTrue(torch.all(newmod.parameters.vector_values() == smod.parameters.vector_values()), "Save/load should extract all parameters")
        
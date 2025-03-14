import unittest

import numpy as np
import torch

import astrophot as ap
from utils import make_basic_sersic, make_basic_gaussian_psf


class TestGroup(unittest.TestCase):
    def test_groupmodel_creation(self):
        np.random.seed(12345)
        shape = (10, 15)
        tar = ap.image.Target_Image(
            data=np.random.normal(loc=0, scale=1.4, size=shape),
            pixelscale=0.8,
            variance=np.ones(shape) * (1.4**2),
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
            psf_mode="none",
            target=tar,
        )

        self.assertFalse(smod.locked, "default model state should not be locked")

        smod.initialize()

        self.assertTrue(torch.all(smod().data == 0), "model_image should be zeros")

        # add existing model does nothing
        smod.add_model(mod1)
        self.assertEqual(len(smod.models), 2, "Adding existing model should not change model count")

        # error for adding mdoels with the same name
        mod3 = ap.models.Component_Model(
            name="base model 2",
            target=tar,
            parameters={"center": {"value": [5, 5], "locked": True}},
        )
        with self.assertRaises(KeyError):
            smod.add_model(mod3)

        # Warning for wrong kwarg name
        with self.assertLogs(ap.AP_config.ap_logger.name, level="WARNING"):
            ap.models.AstroPhot_Model(
                name="group model",
                model_type="group model",
                model=[mod1, mod2],
                psf_mode="none",
                target=tar,
            )

    def test_jointmodel_creation(self):
        np.random.seed(12345)
        shape = (10, 15)
        tar1 = ap.image.Target_Image(
            data=np.random.normal(loc=0, scale=1.4, size=shape),
            pixelscale=0.8,
            variance=np.ones(shape) * (1.4**2),
        )
        shape2 = (33, 42)
        tar2 = ap.image.Target_Image(
            data=np.random.normal(loc=0, scale=1.4, size=shape2),
            pixelscale=0.3,
            origin=(43.2, 78.01),
            variance=np.ones(shape2) * (1.4**2),
        )

        tar = ap.image.Target_Image_List([tar1, tar2])

        mod1 = ap.models.Flat_Sky(
            name="base model 1",
            target=tar1,
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
        self.assertTrue(
            torch.all(torch.isfinite(smod().flatten("data"))).item(), "model_image should be real"
        )

        fm = smod.fit_mask()
        for fmi in fm:
            self.assertTrue(torch.sum(fmi).item() == 0, "this fit_mask should not mask any pixels")

    def test_groupmodel_saveload(self):
        np.random.seed(12345)
        tar = make_basic_sersic(N=51, M=51)

        psf = ap.models.Moffat_PSF(
            name="psf model 1",
            target=make_basic_gaussian_psf(N=11),
            parameters={
                "center": {"value": [5, 5], "locked": True},
                "n": 2.0,
                "Rd": 3.0,
                "I0": {"value": 0.0, "locked": True},
            },
        )

        mod1 = ap.models.Sersic_Galaxy(
            name="base model 1",
            target=tar,
            parameters={"center": {"value": [5, 5], "locked": False}},
            psf=psf,
            psf_mode="full",
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
            name="group model",
            filename="test_save_group_model.yaml",
        )
        self.assertEqual(len(smod.models), len(newmod.models), "Group model should load sub models")

        self.assertEqual(newmod.parameters.size, 16, "Group model size should sum all parameters")

        self.assertTrue(
            torch.all(newmod.parameters.vector_values() == smod.parameters.vector_values()),
            "Save/load should extract all parameters",
        )


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

        self.assertTrue(
            torch.all(smod().data >= 0),
            "PSF group sample should be greater than or equal to zero",
        )

    def test_psfgroupmodel_saveload(self):
        np.random.seed(12345)
        tar = make_basic_gaussian_psf()

        psf1 = ap.models.Moffat_PSF(
            name="psf model 1",
            target=tar,
            parameters={
                "n": 2.0,
                "Rd": 3.0,
            },
        )

        psf2 = ap.models.Sersic_PSF(
            name="psf model 2",
            target=tar,
            parameters={
                "n": 2.0,
                "Re": 3.0,
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
            name="group model",
            filename="test_save_psfgroup_model.yaml",
        )
        self.assertEqual(len(smod.models), len(newmod.models), "Group model should load sub models")

        self.assertEqual(newmod.parameters.size, 4, "Group model size should sum all parameters")

        self.assertTrue(
            torch.all(newmod.parameters.vector_values() == smod.parameters.vector_values()),
            "Save/load should extract all parameters",
        )

    def test_psfgroupmodel_fitting(self):

        np.random.seed(124)
        pixelscale = 1.0
        psf1 = ap.utils.initialize.moffat_psf(1.0, 4.0, 101, pixelscale, normalize=False)
        psf2 = ap.utils.initialize.moffat_psf(3.0, 2.0, 101, pixelscale, normalize=False)
        psf = psf1 + 0.5 * psf2
        psf /= psf.sum()
        star = psf * 10  # flux of 10
        variance = star / 1e5
        star += np.random.normal(scale=np.sqrt(variance))

        psf_target2 = ap.image.PSF_Image(
            data=star.copy() / star.sum(),  # empirical PSF from cutout
            pixelscale=pixelscale,
        )
        psf_target2.normalize()

        point_target = ap.image.Target_Image(
            data=star,  # cutout of star
            pixelscale=pixelscale,
            variance=variance,
        )

        moffat_component1 = ap.models.AstroPhot_Model(
            name="psf part1",
            model_type="moffat psf model",
            target=psf_target2,
            parameters={
                "n": 1.5,
                "Rd": 4.5,
                "I0": {"value": -3.0, "locked": False},
            },
            normalize_psf=False,
        )

        moffat_component2 = ap.models.AstroPhot_Model(
            name="psf part2",
            model_type="moffat psf model",
            target=psf_target2,
            parameters={
                "n": 2.6,
                "Rd": 1.7,
                "I0": {"value": -2.3, "locked": False},
            },
            normalize_psf=False,
        )

        full_psf_model = ap.models.AstroPhot_Model(
            name="full psf",
            model_type="psf group model",
            target=psf_target2,
            models=[moffat_component1, moffat_component2],
            normalize_psf=True,
        )
        full_psf_model.initialize()

        model = ap.models.AstroPhot_Model(
            name="star",
            model_type="point model",
            target=point_target,
            psf=full_psf_model,
        )
        model.initialize()

        ap.fit.LM(model, verbose=1).fit()

        self.assertTrue(
            abs(model["flux"].value.item() - 1.0) < 1e-2, "Star flux should be accurate"
        )
        self.assertTrue(
            model["flux"].uncertainty.item() < 1e-2, "Star flux uncertainty should be small"
        )

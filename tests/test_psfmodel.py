import unittest
import astrophot as ap
import torch
import numpy as np
from utils import make_basic_sersic, make_basic_gaussian_psf
#torch.autograd.set_detect_anomaly(True)
######################################################################
# PSF Model Objects
######################################################################

class TestAllPSFModelBasics(unittest.TestCase):
    def test_all_psfmodel_sample(self):

        target = make_basic_gaussian_psf()
        for model_type in ap.models.PSF_Model.List_Model_Names(useable=True):
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
            print(MODEL.parameters)
            img = MODEL()
            self.assertTrue(
                torch.all(torch.isfinite(img.data)),
                "Model should evaluate a real number for the full image",
            )
            self.assertIsInstance(str(MODEL), str, "String representation should return string")
            self.assertIsInstance(repr(MODEL), str, "Repr should return string")

class TestEigenPSF(unittest.TestCase):
    def test_init(self):
        target = make_basic_gaussian_psf(N = 51, rand = 666)
        dat = target.data.detach()
        dat[dat < 0] = 0
        target = ap.image.PSF_Image(data = dat, pixelscale = target.pixelscale)
        basis = np.stack(list(make_basic_gaussian_psf(N = 51, sigma = s, rand = int(4923*s)).data for s in np.linspace(8, 1, 5)))
        # basis = np.random.rand(10,51,51)
        EM = ap.models.AstroPhot_Model(
            model_type = "eigen psf model",
            eigen_basis = basis,
            eigen_pixelscale = 1,
            target = target,
        )

        EM.initialize()

        res = ap.fit.LM(EM, verbose = 1).fit()

        self.assertEqual(res.message, "success")

class TestPixelPSF(unittest.TestCase):
    def test_init(self):
        target = make_basic_gaussian_psf(N = 11)
        target.data[target.data < 0] = 0
        target = ap.image.PSF_Image(data = target.data / torch.sum(target.data), pixelscale = target.pixelscale)
        
        PM = ap.models.AstroPhot_Model(
            model_type = "pixelated psf model",
            target = target,
        )

        PM.initialize()
        
        self.assertTrue(torch.allclose(PM().data, target.data))

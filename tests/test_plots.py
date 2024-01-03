import unittest

import numpy as np
import matplotlib.pyplot as plt

import astrophot as ap
from utils import make_basic_sersic, make_basic_gaussian_psf

class TestPlots(unittest.TestCase):
    """
    Can't test visuals, so this only tests that the code runs
    """
    def test_target_image(self):
        target = make_basic_sersic()

        try:
            fig, ax = plt.subplots()
        except Exception as e:
            print("skipping test_target_image because matplotlib is not installed properly")
            return

        ap.plots.target_image(fig, ax, target)
        plt.close()

    def test_psf_image(self):
        target = make_basic_gaussian_psf()

        fig, ax = plt.subplots()

        ap.plots.psf_image(fig, ax, target)
        plt.close()
        
    def test_target_image_list(self):
        target1 = make_basic_sersic()
        target2 = make_basic_sersic()
        target = ap.image.Target_Image_List([target1,target2])
        
        try:
            fig, ax = plt.subplots(2)
        except Exception as e:
            print("skipping test_target_image_list because matplotlib is not installed properly")
            return

        ap.plots.target_image(fig, ax, target)
        plt.close()
        
    def test_model_image(self):
        target = make_basic_sersic()
        
        new_model = ap.models.AstroPhot_Model(
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
        new_model.initialize()

        try:
            fig, ax = plt.subplots()
        except Exception as e:
            print("skipping test because matplotlib is not installed properly")
            return

        ap.plots.model_image(fig, ax, new_model)

        plt.close()

    def test_residual_image(self):
        target = make_basic_sersic()
        
        new_model = ap.models.AstroPhot_Model(
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
        new_model.initialize()

        try:
            fig, ax = plt.subplots()
        except Exception as e:
            print("skipping test because matplotlib is not installed properly")
            return

        ap.plots.residual_image(fig, ax, new_model)

        plt.close()

    def test_model_windows(self):

        target = make_basic_sersic()
        
        new_model = ap.models.AstroPhot_Model(
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
        new_model.initialize()

        try:
            fig, ax = plt.subplots()
        except Exception as e:
            print("skipping test because matplotlib is not installed properly")
            return

        ap.plots.model_window(fig, ax, new_model)

        plt.close()
        

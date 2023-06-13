import unittest

import numpy as np
import torch
import matplotlib.pyplot as plt

from autophot import image
import autophot as ap
from utils import make_basic_sersic

class TestPlots(unittest.TestCase):
    """
    Can't test visuals, so this only tests that the code runs
    """
    def test_target_image(self):
        target = make_basic_sersic()

        fig, ax = plt.subplots()

        ap.plots.target_image(fig, ax, target)
        plt.close()
        
    def test_target_image_list(self):
        target1 = make_basic_sersic()
        target2 = make_basic_sersic()
        target = ap.image.Target_Image_List([target1,target2])
        
        fig, ax = plt.subplots(2)

        ap.plots.target_image(fig, ax, target)
        plt.close()
        
    def test_model_image(self):
        target = make_basic_sersic()
        
        new_model = ap.models.AutoPhot_Model(
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

        fig, ax = plt.subplots()

        ap.plots.model_image(fig, ax, new_model)

        plt.close()

    def test_residual_image(self):
        target = make_basic_sersic()
        
        new_model = ap.models.AutoPhot_Model(
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

        fig, ax = plt.subplots()

        ap.plots.residual_image(fig, ax, new_model)

        plt.close()

    def test_model_windows(self):

        target = make_basic_sersic()
        
        new_model = ap.models.AutoPhot_Model(
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

        fig, ax = plt.subplots()

        ap.plots.model_window(fig, ax, new_model)

        plt.close()
        

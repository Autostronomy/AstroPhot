import unittest
from autoprof import image
from autoprof import models
from autoprof.utils.parametric_profiles import sersic_np
from autoprof.utils.conversions.coordinates import Axis_Ratio_Cartesian_np
import torch
import numpy as np
import matplotlib.pyplot as plt

class TestModel(unittest.TestCase):

    def test_model_creation(self):
        np.random.seed(12345)
        shape = (10,15)
        tar = image.Target_Image(
            data = np.random.normal(loc = 0, scale = 1.4, size = shape),
            pixelscale = 0.8,
            variance = np.ones(shape)*(1.4**2),
            psf = np.array([[0.05, 0.1, 0.05],[0.1, 0.4, 0.1],[0.05, 0.1, 0.05]]),
        )

        mod = models.BaseModel(name = "base model", target = tar, parameters = {"center": {"value": [5,5], "locked": True}})

        mod.initialize()
        
        self.assertFalse(mod.locked, "default model should not be locked")
        
        mod.sample()

        self.assertTrue(torch.all(mod.model_image.data == 0), "BaseModel model_image should be zeros")

        mod.compute_loss()
        
        self.assertAlmostEqual(mod.loss.detach().item(), 147.4986368304884, 5, "Loss calculation returns incorrect value")


class TestSersic(unittest.TestCase):

    def test_sersic_creation(self):
        np.random.seed(12345)
        N = 50
        Width = 20
        shape = (N+10,N)
        true_params = [2,5,10,-3, 5, 0.7, np.pi/4]
        IXX, IYY = np.meshgrid(np.linspace(-Width, Width, shape[1]), np.linspace(-Width, Width, shape[0]))
        QPAXX, QPAYY = Axis_Ratio_Cartesian_np(true_params[5], IXX - true_params[3], IYY - true_params[4], true_params[6])
        Z0 = sersic_np(np.sqrt(QPAXX**2 + QPAYY**2), true_params[0], true_params[1], true_params[2]) + np.random.normal(loc = 0, scale = 0.1, size = shape)
        tar = image.Target_Image(
            data = Z0,
            pixelscale = 0.8,
            variance = np.ones(Z0.shape)*(0.1**2),
        )

        mod = models.Sersic_Galaxy(name = "sersic model", target = tar, parameters = {"center": {"value": [-3.2 + N/2, 5.1 + (N+10)/2]}})
        
        self.assertFalse(mod.locked, "default model should not be locked")
        
        mod.initialize()

        mod.sample()

        mod.compute_loss()
        
        self.assertLess(mod.loss.detach().item(), 15000, "Loss calculation returns value too high")

        mod.loss.backward()

        for p in mod.parameters:
            self.assertFalse(mod.parameters[p].grad is None, "Gradient should be evaluated for all model parameters")

        
            
    def test_sersic_fit(self):
        np.random.seed(12345)
        N = 50
        Width = 20
        shape = (N+10,N)
        true_params = [2,5,10,-3, 5, 0.7, np.pi/4]
        IXX, IYY = np.meshgrid(np.linspace(-Width, Width, shape[1]), np.linspace(-Width, Width, shape[0]))
        QPAXX, QPAYY = Axis_Ratio_Cartesian_np(true_params[5], IXX - true_params[3], IYY - true_params[4], true_params[6])
        Z0 = sersic_np(np.sqrt(QPAXX**2 + QPAYY**2), true_params[0], true_params[1], true_params[2]) + np.random.normal(loc = 0, scale = 0.1, size = shape)
        tar = image.Target_Image(
            data = Z0,
            pixelscale = 0.8,
            variance = np.ones(Z0.shape)*(0.1**2),
        )

        mod = models.Sersic_Galaxy(name = "sersic model", target = tar, parameters = {"center": {"value": [-3.2 + N/2, 5.1 + (N+10)/2]}})
        
        self.assertFalse(mod.locked, "default model should not be locked")
        
        mod.initialize()
        mod_initparams = {}
        for p in mod.parameters:
            mod_initparams[p] = np.copy(mod[p].representation.detach().numpy())

        optimizer = torch.optim.Adam(mod.get_parameters_representation(), lr = 0.1)

        for epoch in range(100):
            optimizer.zero_grad()
            
            mod.sample()
            
            mod.compute_loss()
            
            mod.loss.backward()
            
            optimizer.step()
            
            optimizer.zero_grad()

        for p in mod.parameters:
            self.assertFalse(np.any(mod[p].representation.detach().numpy() == mod_initparams[p]), f"parameter {p} should update with optimization")
            
class TestSuper(unittest.TestCase):

    def test_supermodel_creation(self):
        np.random.seed(12345)
        shape = (10,15)
        tar = image.Target_Image(
            data = np.random.normal(loc = 0, scale = 1.4, size = shape),
            pixelscale = 0.8,
            variance = np.ones(shape)*(1.4**2),
        )

        mod1 = models.BaseModel(name = "base model 1", target = tar, parameters = {"center": {"value": [5,5], "locked": True}})
        mod2 = models.BaseModel(name = "base model 2", target = tar, parameters = {"center": {"value": [5,5], "locked": True}})

        smod = models.Super_Model(name = "super model", model_list = [mod1, mod2], target = tar)
            
        self.assertFalse(smod.locked, "default model state should not be locked")
        
        smod.initialize()

        smod.sample()

        self.assertTrue(torch.all(smod.model_image.data == 0), "model_image should be zeros")

        smod.compute_loss()
        
        self.assertAlmostEqual(smod.loss.detach().item(), 147.4986368304884, 5, "Loss calculation returns incorrect value")

    def test_supermodel_fit(self):
        np.random.seed(12345)
        N = 50
        Width = 20
        shape = (N+10,N)
        true_params1 = [2,4,10,-3, 5, 0.7, np.pi/4]
        true_params2 = [1.2,6,8,2, -3, 0.5, -np.pi/4]
        IXX, IYY = np.meshgrid(np.linspace(-Width, Width, shape[1]), np.linspace(-Width, Width, shape[0]))
        QPAXX, QPAYY = Axis_Ratio_Cartesian_np(true_params1[5], IXX - true_params1[3], IYY - true_params1[4], true_params1[6])
        Z0 = sersic_np(np.sqrt(QPAXX**2 + QPAYY**2), true_params1[0], true_params1[1], true_params1[2])
        QPAXX, QPAYY = Axis_Ratio_Cartesian_np(true_params2[5], IXX - true_params2[3], IYY - true_params2[4], true_params2[6])
        Z0 += sersic_np(np.sqrt(QPAXX**2 + QPAYY**2), true_params2[0], true_params2[1], true_params2[2])
        Z0 += np.random.normal(loc = 0, scale = 0.1, size = shape)
        tar = image.Target_Image(
            data = Z0,
            pixelscale = 0.8,
            variance = np.ones(Z0.shape)*(0.1**2),
        )

        mod1 = models.Sersic_Galaxy(name = "sersic model 1", target = tar, parameters = {"center": {"value": [-3.2 + N/2, 5.1 + (N+10)/2]}})
        mod2 = models.Sersic_Galaxy(name = "sersic model 2", target = tar, parameters = {"center": {"value": [2.1 + N/2, -3.1 + (N+10)/2]}})
        
        smod = models.Super_Model(name = "super model", model_list = [mod1, mod2], target = tar)
            
        self.assertFalse(smod.locked, "default model should not be locked")
        
        smod.initialize()

        mod1_initparams = {}
        for p in mod1.parameters:
            mod1_initparams[p] = np.copy(mod1[p].representation.detach().numpy())
        mod2_initparams = {}
        for p in mod2.parameters:
            mod2_initparams[p] = np.copy(mod2[p].representation.detach().numpy())
            
        optimizer = torch.optim.Adam(smod.get_parameters_representation(), lr = 0.1)

        for epoch in range(100):
            optimizer.zero_grad()
            
            smod.sample()
            
            smod.compute_loss()
            
            smod.loss.backward()
            
            optimizer.step()
            
            optimizer.zero_grad()

        for p in mod1.parameters:
            self.assertFalse(np.any(mod1[p].representation.detach().numpy() == mod1_initparams[p]), f"mod1 parameter {p} should update with optimization")
        for p in mod2.parameters:
            self.assertFalse(np.any(mod2[p].representation.detach().numpy() == mod2_initparams[p]), f"mod2 parameter {p} should update with optimization")

if __name__ == "__main__":
    unittest.main()
        

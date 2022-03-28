import unittest
from autoprof import models
from autoprof import image
import numpy as np

class TestState(unittest.TestCase):
    def test_parameter_setting(self):
        new_model = models.Model(
            parameters={
                "dummy_variable": 0,
                "dummy_fixed": 0,
                "dummy_limit_lower": 0.5,
                "dummy_limit_upper": 0.5,
                "dummy_limit_range": 0.5,
            },
            fixed=["dummy_fixed"],
            limits={
                "dummy_limit_lower": [0, None],
                "dummy_limit_upper": [None, 1],
                "dummy_limit_range": [0, 1],
            },
        )

        # Free variable
        new_model.set_representation("dummy_variable", 5)
        self.assertEqual(
            new_model.get_representation("dummy_variable"),
            5,
            msg="should be able to set variable in model",
        )

        # Fixed variable
        new_model.set_representation("dummy_fixed", 5)
        self.assertEqual(
            new_model.get_representation("dummy_fixed"),
            0,
            msg="fixed variable should not be changed",
        )

        # Lower limit
        new_model.set_representation("dummy_limit_lower", 5)
        self.assertAlmostEqual(
            new_model.get_value("dummy_limit_lower"),
            5,
            delta=1,
            msg="lower limit variable should not have upper limit",
        )
        new_model.set_representation("dummy_limit_lower", -5)
        self.assertGreater(
            new_model.get_value("dummy_limit_lower"),
            0,
            msg="lower limit variable should not be lower than lower limit",
        )
        new_model.add_representation("dummy_limit_lower", -5)
        self.assertGreater(
            new_model.get_value("dummy_limit_lower"),
            0,
            msg="lower limit variable should not be lower than lower limit",
        )

        # Upper limit
        new_model.set_representation("dummy_limit_upper", -5)
        self.assertAlmostEqual(
            new_model.get_value("dummy_limit_upper"),
            -5,
            delta=1,
            msg="upper limit variable should not have lower limit",
        )
        new_model.set_representation("dummy_limit_upper", 5)
        self.assertLess(
            new_model.get_value("dummy_limit_upper"),
            1,
            msg="upper limit variable should not be greater than upper limit",
        )
        new_model.add_representation("dummy_limit_upper", 5)
        self.assertLess(
            new_model.get_value("dummy_limit_upper"),
            1,
            msg="upper limit variable should not be greater than upper limit",
        )

        # Range limit
        new_model.set_representation("dummy_limit_range", -5)
        self.assertGreater(
            new_model.get_value("dummy_limit_range"),
            0,
            msg="range limit variable should not be less than lower limit",
        )
        new_model.add_representation("dummy_limit_range", -5)
        self.assertGreater(
            new_model.get_value("dummy_limit_range"),
            0,
            msg="range limit variable should not be less than lower limit",
        )
        new_model.set_representation("dummy_limit_range", 5)
        self.assertLess(
            new_model.get_value("dummy_limit_range"),
            1,
            msg="range limit variable should not be greater than upper limit",
        )
        new_model.add_representation("dummy_limit_range", 5)
        self.assertLess(
            new_model.get_value("dummy_limit_range"),
            1,
            msg="range limit variable should not be greater than upper limit",
        )

    def test_iteration(self):
        new_model = models.Model(
            parameters={
                "dummy_variable": 0,
                "dummy_fixed": 0,
                "dummy_limit_lower": 0.5,
                "dummy_limit_upper": 0.5,
                "dummy_limit_range": 0.5,
            },
            fixed=["dummy_fixed"],
            limits={
                "dummy_limit_lower": [0, None],
                "dummy_limit_upper": [None, 1],
                "dummy_limit_range": [0, 1],
            },
        )
        testimage_shape = (10,20)
        new_model.set_image(image.AP_Image(np.zeros(testimage_shape), pixelscale = 1.))
        new_model.initialize()
    
        new_model.step_iteration()
        new_model.sample_model()
        new_model.update_loss(np.ones(testimage_shape))
        
        new_model.step_iteration()
        new_model.sample_model()
        new_model.update_loss(np.zeros(testimage_shape))
        
        self.assertAlmostEqual(new_model.loss[0], 0, 'loss didnt update to expected value')
        self.assertAlmostEqual(new_model.loss[1], 1, 'past loss doesnt return expected value')
        
if __name__ == "__main__":
    unittest.main()

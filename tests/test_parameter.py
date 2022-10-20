import unittest
from autoprof.models import Parameter, Pointing_Parameter
import torch

class TestParameter(unittest.TestCase):
    def test_parameter_setting(self):

        with torch.no_grad():
            base_param = Parameter('base param')
            base_param.set_value(1.)
            self.assertEqual(base_param.value, 1, msg = "Value should be set to 1")
            self.assertEqual(base_param.representation, 1, msg = "Representation should be set to 1")
        
            base_param.value = 2.
            self.assertEqual(base_param.value, 2, msg = "Value should be set to 2")
            self.assertEqual(base_param.representation, 2, msg = "Representation should be set to 2")
            
            base_param.value += 2.
            self.assertEqual(base_param.value, 4, msg = "Value should be set to 4")
            self.assertEqual(base_param.representation, 4, msg = "Representation should be set to 4")
            
            # Test a locked parameter that it does not change
            locked_param = Parameter('locked param', value = 1., locked = True)
            locked_param.set_value(2.)
            self.assertEqual(locked_param.value, 1, msg = "Locked value should remain at 1")
            
            locked_param.value = 2.
            self.assertEqual(locked_param.value, 1, msg = "Locked value should remain at 1")
            
            locked_param.set_value(2., override_locked = True)
            self.assertEqual(locked_param.value, 2, msg = "Locked value should be forced to update to 2")
        
    def test_parameter_limits(self):
        
        # Lower limit parameter
        lowlim_param = Parameter('lowlim param', limits = (1,None))
        lowlim_param.representation = 100.
        self.assertAlmostEqual(
            lowlim_param.value,
            100,
            delta=1,
            msg="lower limit variable should not have upper limit",
        )
        lowlim_param.representation = -100.
        self.assertGreater(
            lowlim_param.value,
            1,
            msg="lower limit variable should not be lower than lower limit",
        )

        # Upper limit parameter
        uplim_param = Parameter('uplim param', limits = (None, 1))
        uplim_param.representation = -100.
        self.assertAlmostEqual(
            uplim_param.value,
            -100,
            delta=1,
            msg="upper limit variable should not have lower limit",
        )
        uplim_param.representation = 100.
        self.assertLess(
            uplim_param.value,
            1,
            msg="upper limit variable should not be greater than upper limit",
        )

        # Range limit parameter
        range_param = Parameter('range param', limits = (-1, 1))
        range_param.representation = -100.
        self.assertGreater(
            range_param.value,
            -1,
            msg="range limit variable should not be less than lower limit",
        )
        range_param.representation = 100.
        self.assertLess(
            range_param.value,
            1,
            msg="range limit variable should not be greater than upper limit",
        )

        # Cyclic Range limit parameter
        cyrange_param = Parameter('cyrange param', limits = (-1, 1), cyclic = True)
        cyrange_param.representation = 2.
        self.assertAlmostEqual(
            cyrange_param.value,
            0,
            delta=0.1,
            msg="cyclic variable should loop in range (upper)",
        )
        cyrange_param.representation = -2.
        self.assertAlmostEqual(
            cyrange_param.value,
            0,
            delta=0.1,
            msg="cyclic variable should loop in range (lower)",
        )

    def test_parameter_operations(self):

        base_param1 = Parameter('base param1', value = 2.)
        base_param2 = Parameter('base param2', value = 1.)

        self.assertEqual(base_param1 - base_param2, 1, msg= "parameter difference not evaluated properly")

        cyclic_param1 = Parameter('cyclic param1', value = -0.9, limits = (-1, 1), cyclic = True)
        cyclic_param2 = Parameter('cyclic param2', value = 0.9, limits = (-1, 1), cyclic = True)

        self.assertAlmostEqual((cyclic_param1 - cyclic_param2).detach().numpy(), -0.2, msg= "cyclic parameter difference should wrap") # fixme check positive/negative


    def test_parameter_array(self):
        
        param_array1 = Parameter("array1", value = list(float(3 + i) for i in range(5)))
        param_array2 = Parameter("array2", value = list(float(i) for i in range(5)))

        self.assertTrue(torch.all((param_array1 - param_array2) == 3), msg = "parameter array difference not as expected")

        param_array2.value = list(float(3) for i in range(5))
        self.assertTrue(torch.all(param_array2.value == 3), msg = "parameter array value should be updated")

        for P in param_array2:
            self.assertEqual(P, 3, "individual elements of parameter array should be updated")

        self.assertEqual(len(param_array2), 5, "parameter array should have length attribute")

    def test_pointing_parameter(self):
        original_param = Parameter("original", value = 5, limits = (0, 10))

        point_param = Pointing_Parameter("pointer", original_param)

        self.assertEqual(point_param.value, 5, "Pointer should take on original parameter value")
        original_param.value = 6
        self.assertEqual(point_param.value, 6, "Pointer should take on new value for original parameter")
        point_param.value = 7        
        self.assertEqual(original_param.value, 7, "Pointer should update original parameter when setting value")
        self.assertEqual(original_param.representation, point_param.representation, "Pointer should track original parameter representation")

        self.assertEqual(original_param.limits, point_param.limits, "Pointer should take on original parameter limit properties")
        # fixme try setting _value
        
    def test_pointing_parameter_array(self):
        original_array = Parameter("original", value = torch.arange(1,6), limits = (0,10))
        point_array = Pointing_Parameter("pointer", original_array)

        self.assertEqual(point_array.value[2], 3, "Pointer should take on original parameter value")
        original_array.value = torch.arange(3,8)
        self.assertEqual(point_array.value[2], 5, "Pointer should take on new value for original parameter")
        point_array.value = torch.arange(4, 9)        
        self.assertEqual(original_array.value[2], 6, "Pointer should update original parameter when setting value")
        self.assertEqual(original_array.representation[2], point_array.representation[2], "Pointer should track original parameter representation")

        self.assertEqual(original_array.limits, point_array.limits, "Pointer should take on original parameter limit properties")
        # fixme try setting _value

    def test_parameter_gradients(self):
        params = Parameter("input params", value = torch.ones(3))
        X = torch.sum(params.value * 3)
        X.backward()
        self.assertTrue(torch.all(params.grad == 3), "Parameters should track gradient")
        
if __name__ == "__main__":
    unittest.main()

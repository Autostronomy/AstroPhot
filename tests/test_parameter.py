import unittest
from autoprof.models import Parameter

class TestParameter(unittest.TestCase):
    def test_parameter_setting(self):

        # Test simply setting a parameter value and retrieving it
        base_param = Parameter('base param')
        base_param.set_value(1)
        self.assertEqual(base_param.value, 1, msg = "Value should be set to 1")
        self.assertEqual(base_param.representation, 1, msg = "Representation should be set to 1")

        # Test a fixed parameter that it does not change
        fixed_param = Parameter('fixed param', value = 1, fixed = True)
        fixed_param.set_value(2)
        self.assertEqual(fixed_param.value, 1, msg = "Fixed value should be set to 1")

    def test_parameter_limits(self):
        
        # Lower limit parameter
        lowlim_param = Parameter('lowlim param', limits = (1,None))
        lowlim_param.set_representation(100)
        self.assertAlmostEqual(
            lowlim_param.value,
            100,
            delta=1,
            msg="lower limit variable should not have upper limit",
        )
        lowlim_param.set_representation(-100)
        self.assertGreater(
            lowlim_param.value,
            1,
            msg="lower limit variable should not be lower than lower limit",
        )

        # Upper limit parameter
        uplim_param = Parameter('uplim param', limits = (None, 1))
        uplim_param.set_representation(-100)
        self.assertAlmostEqual(
            uplim_param.value,
            -100,
            delta=1,
            msg="upper limit variable should not have lower limit",
        )
        uplim_param.set_representation(100)
        self.assertLess(
            uplim_param.value,
            1,
            msg="upper limit variable should not be greater than upper limit",
        )

        # Range limit parameter
        range_param = Parameter('range param', limits = (-1, 1))
        range_param.set_representation(-100)
        self.assertGreater(
            range_param.value,
            -1,
            msg="range limit variable should not be less than lower limit",
        )
        range_param.set_representation(100)
        self.assertLess(
            range_param.value,
            1,
            msg="range limit variable should not be greater than upper limit",
        )

        # Cyclic Range limit parameter
        cyrange_param = Parameter('cyrange param', limits = (-1, 1), cyclic = True)
        cyrange_param.set_representation(2)
        self.assertAlmostEqual(
            cyrange_param.value,
            0,
            delta=0.1,
            msg="cyclic variable should loop in range (upper)",
        )
        cyrange_param.set_representation(-2)
        self.assertAlmostEqual(
            cyrange_param.value,
            0,
            delta=0.1,
            msg="cyclic variable should loop in range (lower)",
        )
        
if __name__ == "__main__":
    unittest.main()

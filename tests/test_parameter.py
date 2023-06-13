import unittest
from autophot.models import Parameter
import torch


class TestParameter(unittest.TestCase):
    def test_parameter_setting(self):

        with torch.no_grad():
            base_param = Parameter("base param")
            base_param.set_value(1.0)
            self.assertEqual(base_param.value, 1, msg="Value should be set to 1")
            self.assertEqual(
                base_param.representation, 1, msg="Representation should be set to 1"
            )

            base_param.value = 2.0
            self.assertEqual(base_param.value, 2, msg="Value should be set to 2")
            self.assertEqual(
                base_param.representation, 2, msg="Representation should be set to 2"
            )

            base_param.value += 2.0
            self.assertEqual(base_param.value, 4, msg="Value should be set to 4")
            self.assertEqual(
                base_param.representation, 4, msg="Representation should be set to 4"
            )

            # Test a locked parameter that it does not change
            locked_param = Parameter("locked param", value=1.0, locked=True)
            locked_param.set_value(2.0)
            self.assertEqual(
                locked_param.value, 1, msg="Locked value should remain at 1"
            )

            locked_param.value = 2.0
            self.assertEqual(
                locked_param.value, 1, msg="Locked value should remain at 1"
            )

            locked_param.set_value(2.0, override_locked=True)
            self.assertEqual(
                locked_param.value,
                2,
                msg="Locked value should be forced to update to 2",
            )

    def test_parameter_limits(self):

        # Lower limit parameter
        lowlim_param = Parameter("lowlim param", limits=(1, None))
        lowlim_param.representation = 100.0
        self.assertAlmostEqual(
            lowlim_param.value,
            100,
            delta=1,
            msg="lower limit variable should not have upper limit",
        )
        lowlim_param.representation = -100.0
        self.assertGreater(
            lowlim_param.value,
            1,
            msg="lower limit variable should not be lower than lower limit",
        )

        # Upper limit parameter
        uplim_param = Parameter("uplim param", limits=(None, 1))
        uplim_param.representation = -100.0
        self.assertAlmostEqual(
            uplim_param.value,
            -100,
            delta=1,
            msg="upper limit variable should not have lower limit",
        )
        uplim_param.representation = 100.0
        self.assertLess(
            uplim_param.value,
            1,
            msg="upper limit variable should not be greater than upper limit",
        )

        # Range limit parameter
        range_param = Parameter("range param", limits=(-1, 1))
        range_param.representation = -100.0
        self.assertGreater(
            range_param.value,
            -1,
            msg="range limit variable should not be less than lower limit",
        )
        range_param.representation = 100.0
        self.assertLess(
            range_param.value,
            1,
            msg="range limit variable should not be greater than upper limit",
        )

        # Cyclic Range limit parameter
        cyrange_param = Parameter("cyrange param", limits=(-1, 1), cyclic=True)
        cyrange_param.representation = 2.0
        self.assertAlmostEqual(
            cyrange_param.value,
            0,
            delta=0.1,
            msg="cyclic variable should loop in range (upper)",
        )
        cyrange_param.representation = -2.0
        self.assertAlmostEqual(
            cyrange_param.value,
            0,
            delta=0.1,
            msg="cyclic variable should loop in range (lower)",
        )

    def test_parameter_array(self):

        param_array1 = Parameter("array1", value=list(float(3 + i) for i in range(5)))
        param_array2 = Parameter("array2", value=list(float(i) for i in range(5)))

        param_array2.value = list(float(3) for i in range(5))
        self.assertTrue(
            torch.all(param_array2.value == 3),
            msg="parameter array value should be updated",
        )

        for P in param_array2:
            self.assertEqual(
                P, 3, "individual elements of parameter array should be updated"
            )

        self.assertEqual(
            len(param_array2), 5, "parameter array should have length attribute"
        )

    def test_parameter_gradients(self):
        V = torch.ones(3)
        V.requires_grad = True
        params = Parameter("input params", value=V)
        X = torch.sum(params.value * 3)
        X.backward()
        self.assertTrue(torch.all(V.grad == 3), "Parameters should track gradient")


if __name__ == "__main__":
    unittest.main()

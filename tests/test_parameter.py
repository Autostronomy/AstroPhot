import unittest
from astrophot.param import Node as BaseNode, Parameter_Node
import torch
import numpy as np

class Node(BaseNode):
    """
    Dummy class for testing purposes
    """

    def value(self):
        return None

class TestNode(unittest.TestCase):

    def test_node_init(self):
        node1 = Node("node1")
        node2 = Node("node2", locked = True)

    def test_node_link(self):
        node1 = Node("node1")
        node2 = Node("node2")
        node3 = Node("node3", locked = True)

        node1.link(node2, node3)

        self.assertIs(node1["node2"], node2, "node getitem should fetch correct node")

        for Na, Nb in zip(node1.flat().keys(), (node2, node3)):
            self.assertIs(Na, Nb, "node flat should produce correct order")

        node4 = Node("node4")

        node2.link(node4)

        self.assertIs(node1["node2:node4"], node4, "node getitem should fetch correct node")
        
        for Na, Nb in zip(node1.flat(include_locked=False).keys(), (node4,)):
            self.assertIs(Na, Nb, "node flat should produce correct order")


        node1.dump()

        self.assertEqual(len(node1.nodes), 0, "dump should clear all nodes")

    def test_state(self):
        
        node1 = Node("node1")
        node2 = Node("node2")
        node3 = Node("node3", locked = True)

        node1.link(node2, node3)

        state = node1.get_state()

class TestParameter(unittest.TestCase):
    @torch.no_grad()
    def test_parameter_setting(self):
        base_param = Parameter_Node("base param")
        base_param.value = 1.0
        self.assertEqual(base_param.value, 1, msg="Value should be set to 1")
        
        base_param.value = 2.0
        self.assertEqual(base_param.value, 2, msg="Value should update to 2")
        
        base_param.value += 2.0
        self.assertEqual(base_param.value, 4, msg="Value should update to 4")

        # Test a locked parameter that it does not change
        locked_param = Parameter_Node("locked param", value=1.0, locked=True)
        locked_param.value = 2.0
        self.assertEqual(
            locked_param.value, 1, msg="Locked value should remain at 1"
        )

        locked_param.value = 2.0
        self.assertEqual(
            locked_param.value, 1, msg="Locked value should remain at 1"
        )
        
    def test_parameter_limits(self):

        # Lower limit parameter
        lowlim_param = Parameter_Node("lowlim param", limits=(1, None))
        lowlim_param.value = 100.0
        self.assertEqual(
            lowlim_param.value,
            100,
            msg="lower limit variable should not have upper limit",
        )
        with self.assertRaises(AssertionError):
            lowlim_param.value = -100.0

        # Upper limit parameter
        uplim_param = Parameter_Node("uplim param", limits=(None, 1))
        uplim_param.value = -100.0
        self.assertEqual(
            uplim_param.value,
            -100,
            msg="upper limit variable should not have lower limit",
        )
        with self.assertRaises(AssertionError):
            uplim_param.value = 100.0

        # Range limit parameter
        range_param = Parameter_Node("range param", limits=(-1, 1))
        with self.assertRaises(AssertionError):
            range_param.value = 100.0
        with self.assertRaises(AssertionError):
            range_param.value = -100.0

        # Cyclic Range limit parameter
        cyrange_param = Parameter_Node("cyrange param", limits=(-1, 1), cyclic=True)
        cyrange_param.value = 2.0
        self.assertEqual(
            cyrange_param.value,
            0,
            msg="cyclic variable should loop in range (upper)",
        )
        cyrange_param.value = -2.0
        self.assertEqual(
            cyrange_param.value,
            0,
            msg="cyclic variable should loop in range (lower)",
        )

    def test_parameter_array(self):

        param_array1 = Parameter_Node("array1", value=list(float(3 + i) for i in range(5)))
        param_array2 = Parameter_Node("array2", value=list(float(i) for i in range(5)))

        param_array2.value = list(float(3) for i in range(5))
        self.assertTrue(
            torch.all(param_array2.value == 3),
            msg="parameter array value should be updated",
        )

        self.assertEqual(
            len(param_array2), 5, "parameter array should have length attribute"
        )

    def test_parameter_gradients(self):
        V = torch.ones(3)
        V.requires_grad = True
        params = Parameter_Node("input params", value=V)
        X = torch.sum(params.value * 3)
        X.backward()
        self.assertTrue(torch.all(V.grad == 3), "Parameters should track gradient")

    def test_parameter_state(self):

        P = Parameter_Node("state", value = 1., uncertainty = 0.5, limits = (-2, 2), locked = True, prof = 1.)

        P2 = Parameter_Node("v2")
        P2.set_state(P.get_state())

        self.assertEqual(P.value, P2.value, "state should preserve value")
        self.assertEqual(P.uncertainty, P2.uncertainty, "state should preserve uncertainty")
        self.assertEqual(P.prof, P2.prof, "state should preserve prof")
        self.assertEqual(P.locked, P2.locked, "state should preserve locked")
        self.assertEqual(P.limits[0].tolist(), P2.limits[0].tolist(), "state should preserve limits")
        self.assertEqual(P.limits[1].tolist(), P2.limits[1].tolist(), "state should preserve limits")

        S = str(P)

# class TestParameterGroup(unittest.TestCase):

#     def test_generation(self):
#         P = Parameter("state", value = 1., uncertainty = 0.5, limits = (-1, 1), locked = True, prof = 1.)

#         P2 = Parameter("v2")
#         P2.set_state(P.get_state())

#         PG = Parameter_Group("group", parameters = [P,P2])

#         PG_copy = PG.copy()

#     def test_vectors(self):
#         P1 = Parameter("test1", value = 1., uncertainty = 0.5, limits = (-1, 1), locked = False, prof = 1.)

#         P2 = Parameter("test2", value = 2., uncertainty = 5., limits = (None, 1), locked = False)

#         PG = Parameter_Group("group", parameters = [P1,P2])

#         names = PG.get_name_vector()
#         self.assertEqual(names, ["test1", "test2"], "get name vector should produce ordered list of names")

#         uncertainty = PG.get_uncertainty_vector()
#         self.assertTrue(np.all(uncertainty.detach().cpu().numpy() == np.array([0.5,5.])), "get uncertainty vector should track uncertainty")

#     def test_inspection(self):
#         P1 = Parameter("test1", value = 1., uncertainty = 0.5, limits = (-1, 1), locked = False, prof = 1.)

#         P2 = Parameter("test2", value = 2., uncertainty = 5., limits = (None, 1), locked = False)

#         PG = Parameter_Group("group", parameters = [P1,P2])

#         self.assertEqual(len(PG), 2, "parameter group should only have two parameters here")

#         string = str(PG)
        
if __name__ == "__main__":
    unittest.main()

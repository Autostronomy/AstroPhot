import unittest
from astrophot.param import Node as BaseNode, Parameter_Node, Param_Mask
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

        for Na, Nb in zip(node1.flat().values(), (node2, node3)):
            self.assertIs(Na, Nb, "node flat should produce correct order")

        node4 = Node("node4")

        node2.link(node4)

        self.assertIs(node1["node2:node4"], node4, "node getitem should fetch correct node")
        
        for Na, Nb in zip(node1.flat(include_locked=False).values(), (node4,)):
            self.assertIs(Na, Nb, "node flat should produce correct order")


        node1.dump()

        self.assertEqual(len(node1.nodes), 0, "dump should clear all nodes")

    def test_state(self):
        
        node1 = Node("node1")
        node2 = Node("node2")
        node3 = Node("node3", locked = True)

        node1.link(node2, node3)

        state = node1.get_state()

        S = str(node1)
        R = repr(node1)

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

    def test_parameter_value(self):

        P1 = Parameter_Node("test1", value = 0.5, uncertainty = 0.5, limits = (-1, 1), locked = False, prof = 1.)

        P2 = Parameter_Node("test2", value = P1)

        P3 = Parameter_Node("test3", value = lambda P: P["test1"].value**2, link = (P1,))

        self.assertEqual(P1.value.item(), 0.5, "Parameter should store value")
        self.assertEqual(P2.value.item(), 0.5, "Pointing parameter should fetch value")
        self.assertEqual(P3.value.item(), 0.25, "Function parameter should compute value")
        

class TestParameterVector(unittest.TestCase):
    def test_param_vector_creation(self):

        P1 = Parameter_Node("test1", value = 0.5, uncertainty = 0.5, limits = (-1, 1), locked = False, prof = 1.)
        P2 = Parameter_Node("test2", value = 2., uncertainty = 5., locked = False)
        P3 = Parameter_Node("test3", value = [4.,5.], uncertainty = [5.,5.], locked = False)
        P4 = Parameter_Node("test4", value = P2)
        P5 = Parameter_Node("test5", value = lambda P: P["test1"].value**2, link = (P1,))
        PG = Parameter_Node("testgroup", link = (P1, P2, P3, P4, P5))
        
        self.assertTrue(torch.all(PG.vector_values() == torch.tensor([0.5,2.,4.,5.], dtype=P1.value.dtype, device = P1.value.device)), "Vector store all leaf node values")
        self.assertEqual(PG.vector_mask().numel(), 4, "Vector should take all/only leaf node masks")
        self.assertEqual(PG.vector_identities().size, 4, "Vector should take all/only leaf node identities")

    def test_vector_masking(self):
        
        P1 = Parameter_Node("test1", value = 0.5, uncertainty = 0.3, limits = (-1, 1), locked = False, prof = 1.)
        P2 = Parameter_Node("test2", value = 2., uncertainty = 1., locked = False)
        P3 = Parameter_Node("test3", value = [4.,5.], uncertainty = [5.,3.], locked = False)
        P4 = Parameter_Node("test4", value = P2)
        P5 = Parameter_Node("test5", value = lambda P: P["test1"].value**2, link = (P1,))
        PG = Parameter_Node("testgroup", link = (P1, P2, P3, P4, P5))

        mask = torch.tensor([1,0,0,1], dtype = torch.bool, device=P1.value.device)

        with Param_Mask(PG, mask):
            self.assertTrue(torch.all(PG.vector_values() == torch.tensor([0.5,5.], dtype=P1.value.dtype, device = P1.value.device)), "Vector store all leaf node values")
            self.assertTrue(torch.all(PG.vector_uncertainty() == torch.tensor([0.3,3.], dtype=P1.value.dtype, device = P1.value.device)), "Vector store all leaf node uncertainty")
            self.assertEqual(PG.vector_mask().numel(), 4, "Vector should take all/only leaf node masks")
            self.assertEqual(PG.vector_identities().size, 2, "Vector should take all/only leaf node identities")

            # Nested masking
            new_mask = torch.tensor([1,0], dtype = torch.bool, device=P1.value.device)
            with Param_Mask(PG, new_mask):
                self.assertTrue(torch.all(PG.vector_values() == torch.tensor([0.5], dtype=P1.value.dtype, device = P1.value.device)), "Vector store all leaf node values")
                self.assertTrue(torch.all(PG.vector_uncertainty() == torch.tensor([0.3], dtype=P1.value.dtype, device = P1.value.device)), "Vector store all leaf node uncertainty")
                self.assertEqual(PG.vector_mask().numel(), 4, "Vector should take all/only leaf node masks")
                self.assertEqual(PG.vector_identities().size, 1, "Vector should take all/only leaf node identities")

            self.assertTrue(torch.all(PG.vector_values() == torch.tensor([0.5,5.], dtype=P1.value.dtype, device = P1.value.device)), "Vector store all leaf node values")
            self.assertTrue(torch.all(PG.vector_uncertainty() == torch.tensor([0.3,3.], dtype=P1.value.dtype, device = P1.value.device)), "Vector store all leaf node uncertainty")
            self.assertEqual(PG.vector_mask().numel(), 4, "Vector should take all/only leaf node masks")
            self.assertEqual(PG.vector_identities().size, 2, "Vector should take all/only leaf node identities")


    def test_vector_representation(self):
        
        P1 = Parameter_Node("test1", value = 0.5, uncertainty = 0.3, limits = (-1, 1), locked = False, prof = 1.)
        P2 = Parameter_Node("test2", value = 2., uncertainty = 1., locked = False)
        P3 = Parameter_Node("test3", value = [4.,5.], uncertainty = [5.,3.], limits = ((0., 1.), None), locked = False)
        P4 = Parameter_Node("test4", value = P2)
        P5 = Parameter_Node("test5", value = lambda P: P["test1"].value**2, link = (P1,))
        P6 = Parameter_Node("test6", value = ((5,6),(7,8)), uncertainty = 0.1 * np.zeros((2,2)), limits = (None, 10*np.ones((2,2))))
        PG = Parameter_Node("testgroup", link = (P1, P2, P3, P4, P5, P6))

        mask = torch.tensor([1,1,0,1,0,1,0,1], dtype = torch.bool, device=P1.value.device)

        self.assertEqual(len(PG.vector_representation()), 8, "representation should collect all values")
        with Param_Mask(PG, mask):
            # round trip
            vec = PG.vector_values().clone()
            rep = PG.vector_representation()
            PG.vector_set_representation(rep)
            self.assertTrue(torch.all(vec == PG.vector_values()), "representation should be reversible")
            self.assertEqual(PG.vector_values().numel(), 5, "masked values shouldn't be shown")

        S = str(PG)
        R = repr(PG)

if __name__ == "__main__":
    unittest.main()

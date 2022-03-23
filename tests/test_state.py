import unittest
from autoprof.utils.state import State

class TestState(unittest.TestCase):

    def test_option(self):
        new_state = State(ap_dummy = 'test')
        self.assertEqual(new_state.options['ap_dummy'], 'test', 'State.options should accept parameters')
        self.assertIn('ap_dummy', new_state.options, 'State.options should accept in statements')

    def test_options(self):
        multi_tests = ['test1', 'test2', 'test3']
        new_state = State(ap_dummy = multi_tests)
        for mt, S in zip(multi_tests, new_state):
            self.assertEqual(S.options['ap_dummy'], mt, 'State.options should accept list parameters and produce iterable state')
        

if __name__ == '__main__':
    unittest.main()

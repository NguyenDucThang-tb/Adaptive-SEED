import unittest
from argparse import Namespace
from types import MethodType
from seed_video.runtime import Runtime
from seed_video.components.adaptive import adaptive_weight
from seed_video.components.gates import correct_margin


class ComponentTests(unittest.TestCase):
    def test_adaptive_bounds(self):
        self.assertEqual(adaptive_weight(.9, -1), 0)
        self.assertEqual(adaptive_weight(.9, 0), 0)
        self.assertAlmostEqual(adaptive_weight(.9, .5), .45)
        self.assertEqual(adaptive_weight(.9, 20), .9)
        self.assertAlmostEqual(adaptive_weight(.9, .5, 2), .225)

    def test_invalid_parameters(self):
        for values in [(1, 1, 0), (-1, 1, 1), (1, float('nan'), 1)]:
            with self.assertRaises(ValueError):
                adaptive_weight(*values)

    def test_callback_binding_and_current_state(self):
        state = Runtime(Namespace())
        state.YES, state.NO = 0, 1
        state.correct_margin = MethodType(correct_margin, state)
        self.assertEqual(state.correct_margin([3., 1.], 0), 2.)
        self.assertEqual(state.correct_margin([3., 1.], 1), -2.)
        # Preserve code hashing used by keep/repair.
        self.assertIs(state.correct_margin.__code__, correct_margin.__code__)

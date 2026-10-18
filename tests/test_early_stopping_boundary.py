import unittest

from recbole.utils.utils import early_stopping


class EarlyStoppingBoundaryTest(unittest.TestCase):
    def test_stops_after_exactly_configured_non_improving_rounds(self):
        best = 1.0
        step = 0
        for expected_step in range(1, 30):
            best, step, stop, update = early_stopping(
                0.9, best, step, max_step=30, bigger=True, min_delta=1e-4
            )
            self.assertEqual(step, expected_step)
            self.assertFalse(stop)
            self.assertFalse(update)
        best, step, stop, update = early_stopping(
            0.9, best, step, max_step=30, bigger=True, min_delta=1e-4
        )
        self.assertEqual(step, 30)
        self.assertTrue(stop)
        self.assertFalse(update)


if __name__ == '__main__':
    unittest.main()

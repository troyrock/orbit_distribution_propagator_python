import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distribution_propagator.statistics import (
    dkw_epsilon, estimate_mixing, first_sustained, phase_metrics, sample_sigma,
)


class StatisticsTests(unittest.TestCase):
    def test_two_lobes_are_not_mixed(self):
        metrics = phase_metrics([0.0] * 100 + [math.pi] * 100, 72)
        self.assertLess(metrics.resultants[0], 1e-15)
        self.assertAlmostEqual(metrics.resultants[1], 1)
        self.assertAlmostEqual(metrics.max_gap, math.pi)

    def test_uniform_and_branch_cut(self):
        angles = (np.arange(72) + 0.5) * math.tau / 72
        metrics = phase_metrics(angles, 72)
        self.assertEqual(metrics.histogram, [1] * 72)
        self.assertEqual(metrics.occupied_fraction, 1)
        self.assertLess(max(metrics.resultants), 1e-15)
        self.assertAlmostEqual(phase_metrics([-.01, .01], 72).max_gap, math.tau - .02)

    def test_unwrapped_width_and_nonfinite(self):
        metrics = phase_metrics(np.linspace(-2 * math.tau, 2 * math.tau, 1001), 72)
        self.assertAlmostEqual(metrics.q95_width, 3.8 * math.tau)
        for values in ([math.nan], [math.inf], [0, math.nan]):
            with self.assertRaises(ValueError):
                phase_metrics(values, 72)
            with self.assertRaises(ValueError):
                sample_sigma(values)

    def test_event_brackets(self):
        event = first_sustained([0, 1, 2, 3, 4], [False, True, False, True, True], 2)
        self.assertEqual((event.lower_s, event.upper_s, event.confirmed_at), (2, 3, 4))
        self.assertFalse(first_sustained([0, 1], [True, True], 3).found)
        self.assertEqual(first_sustained([0, 1], [True, True], 2).upper_s, 0)

    def test_dkw_and_gaussian_shear(self):
        self.assertLess(dkw_epsilon(20000), .01)
        mu, a = 3.986004418e14, 26560000.0
        self.assertIsNone(estimate_mixing(np.zeros(3), np.full(3, a), mu, .05))
        # Resolve the analytic variance well above subtraction roundoff in rates.
        axes = np.array([a - 1000000, a, a + 1000000])
        rates = np.sqrt(mu / axes) / axes
        expected = math.sqrt(-2 * math.log(.05)) / np.std(rates, ddof=1)
        self.assertAlmostEqual(estimate_mixing(np.zeros(3), axes, mu, .05) / expected, 1, places=11)


if __name__ == "__main__":
    unittest.main()

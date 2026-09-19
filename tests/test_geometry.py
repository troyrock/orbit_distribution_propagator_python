"""Vectorized analysis must preserve the independently tested scalar geometry."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distribution_propagator.geometry import cartesian_batch, rtn_batch
from distribution_propagator.orbit import from_keplerian, to_cartesian, rtn_basis

MU = 3.986004418e14


class GeometryTests(unittest.TestCase):
    def test_batch_scalar_and_orbital_invariants(self):
        elements = np.array([from_keplerian(4e7, e, i, .73, 1.19, anomaly)
                             for e in (0, .02, .7, .95)
                             for i in (0, .9, 2.8)
                             for anomaly in (-2e5, -.01, 0, 2.9, 2e5)])
        actual = cartesian_batch(elements.reshape(4, -1, 6), MU).reshape(-1, 6)
        expected = np.array([to_cartesian(e, MU) for e in elements])
        self.assertLess(np.max(np.linalg.norm(actual[:, :3] - expected[:, :3], axis=1)), 1e-6)
        self.assertLess(np.max(np.linalg.norm(actual[:, 3:] - expected[:, 3:], axis=1)), 1e-8)
        radius = np.linalg.norm(actual[:, :3], axis=1)
        energy = np.sum(actual[:, 3:]**2, axis=1) / 2 - MU / radius
        np.testing.assert_allclose(energy, -MU / (2 * elements[:, 0]), rtol=2e-12)
        h = np.linalg.norm(np.cross(actual[:, :3], actual[:, 3:]), axis=1)
        expected_h = np.sqrt(MU * elements[:, 0] * (1 - elements[:, 1]**2 - elements[:, 2]**2))
        np.testing.assert_allclose(h, expected_h, rtol=2e-12)

    def test_rtn_orientation_and_single_state(self):
        orbit = from_keplerian(26560000, .02, 1.0, .3, .4, .5)
        cart = cartesian_batch(orbit, MU)
        basis = rtn_batch(cart)
        np.testing.assert_allclose(basis, rtn_basis(cart), atol=3e-16)
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=3e-16)
        self.assertAlmostEqual(np.linalg.det(basis), 1)

    def test_invalid_geometry(self):
        for value in ([1, 2], [1, 1.1, 0, 0, 0, 0], [math.nan] * 6):
            with self.assertRaises(ValueError):
                cartesian_batch(value, MU)
        with self.assertRaises(ValueError):
            rtn_batch(np.zeros((3, 6)))


if __name__ == "__main__":
    unittest.main()

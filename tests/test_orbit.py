import math
import unittest

import numpy as np

from distribution_propagator.orbit import (
    covariance_factor, from_cartesian, from_keplerian, rtn_basis,
    to_cartesian, validate_elements,
)

MU = 3.986004418e14


class OrbitTests(unittest.TestCase):
    def test_circular_analytic_cartesian(self):
        a = 26560000.0
        speed = math.sqrt(MU/a)
        for phase in [0, math.pi/2, math.pi, 3*math.pi/2, 63*math.tau+.4]:
            actual = to_cartesian([a, 0, 0, 0, 0, phase], MU)
            expected = [a*math.cos(phase), a*math.sin(phase), 0,
                        -speed*math.sin(phase), speed*math.cos(phase), 0]
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-12)

    def test_independent_perifocal_oracle(self):
        # Construct expected Cartesian states directly from true anomaly,
        # independently of the equinoctial forward/inverse pair under test.
        cases = [(26560000, .2, .7, 1.1, 2.3, -.9),
                 (42164000, .01, .05, -.8, .3, 2.5),
                 (42000000, .65, 1.2, 2.2, -1.7, math.pi),
                 (70000000, .7, 2.6, -.3, 1.9, -2.2),
                 (100000000, .9, 1.57, 3.1, .1, 1e-4)]
        for a, e, inc, node, arg, true_anomaly in cases:
            cn, sn, cw, sw = math.cos(node), math.sin(node), math.cos(arg), math.sin(arg)
            ci, si = math.cos(inc), math.sin(inc)
            p = np.array([cn*cw-sn*sw*ci, sn*cw+cn*sw*ci, sw*si])
            q = np.array([-cn*sw-sn*cw*ci, -sn*sw+cn*cw*ci, cw*si])
            parameter, cf, sf = a*(1-e*e), math.cos(true_anomaly), math.sin(true_anomaly)
            expected = np.concatenate((parameter/(1+e*cf)*(cf*p+sf*q),
                                       math.sqrt(MU/parameter)*(-sf*p+(e+cf)*q)))
            eccentric = 2*math.atan2(math.sqrt(1-e)*math.sin(true_anomaly/2),
                                     math.sqrt(1+e)*math.cos(true_anomaly/2))
            elements = from_keplerian(a, e, inc, node, arg, eccentric-e*math.sin(eccentric))
            state = to_cartesian(elements, MU)
            np.testing.assert_allclose(state[:3], expected[:3], atol=3e-6, rtol=2e-12)
            np.testing.assert_allclose(state[3:], expected[3:], atol=3e-9, rtol=2e-12)
            recovered = from_cartesian(expected, MU)
            self.assertAlmostEqual(recovered[0]/a, 1, delta=3e-13)
            np.testing.assert_allclose(recovered[1:5], elements[1:5], atol=2e-13)
            self.assertAlmostEqual(math.remainder(recovered[5]-elements[5], math.tau), 0, delta=3e-13)
            basis = rtn_basis(state)
            np.testing.assert_allclose(basis@basis.T, np.eye(3), atol=2e-14)
            self.assertAlmostEqual(np.linalg.det(basis), 1, delta=2e-14)

    def test_reject_invalid_orbits(self):
        bad = [[7000000, 1, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0],
               [7000000, 0, 0, 1e7, 0, 0], [7000000, 0, 0, 0, 0, float('nan')]]
        for elements in bad:
            with self.subTest(elements=elements), self.assertRaises(ValueError):
                to_cartesian(elements, MU)
        for state in [[0]*6, [7000000, 0, 0, 1, 0, 0], [7000000, 0, 0, 0, 100000, 0]]:
            with self.subTest(state=state), self.assertRaises(ValueError):
                from_cartesian(state, MU)
        with self.assertRaises(ValueError):
            from_keplerian(7000000, .01, math.pi, 0, 0, 0)
        with self.assertRaises(ValueError):
            validate_elements([7000000, .2, 0, 0, 0, 0], MU, 6500000)
        with self.assertRaises(ValueError):
            to_cartesian([7000000, 0, 0, 0, 0, 0], -MU)

    def test_psd_covariance_rank_one_and_scaled(self):
        direction = np.array([1e6, -.001, .5, 0, 3, 1e-6])
        rank_one = np.outer(direction, direction)
        lower = covariance_factor(rank_one)
        np.testing.assert_allclose(lower@lower.T, rank_one, rtol=2e-12, atol=1e-300)
        lower0 = np.array([[1,0,0,0,0,0], [.3,math.sqrt(.91),0,0,0,0],
                          [.2,-.15,math.sqrt(.9375),0,0,0], [.4,-.3,0,math.sqrt(.75),0,0],
                          [-.2,.25,0,.1,math.sqrt(.8875),0], [0,0,0,0,0,0]])
        scales = np.array([1e6, 3e3, 7e2, 2e-3, 4e-6, 0])
        lower0 *= scales[:,None]
        covariance = lower0@lower0.T
        recovered = covariance_factor(covariance)
        np.testing.assert_allclose(recovered, lower0, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(covariance_factor(np.zeros((6,6))), np.zeros((6,6)))

    def test_reject_invalid_covariances(self):
        bad = []
        indefinite = np.eye(6); indefinite[0,1] = indefinite[1,0] = 1.1
        asymmetric = np.eye(6); asymmetric[0,1] = .1
        zero_axis = np.eye(6); zero_axis[0,0] = 0; zero_axis[0,1] = zero_axis[1,0] = 1e-30
        negative = np.eye(6); negative[0,0] = -1e-30
        nonfinite = np.eye(6); nonfinite[1,1] = float('nan')
        bad.extend([indefinite, asymmetric, zero_axis, negative, nonfinite, np.eye(5)])
        for covariance in bad:
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                covariance_factor(covariance)


if __name__ == '__main__':
    unittest.main()

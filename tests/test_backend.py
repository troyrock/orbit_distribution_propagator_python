"""Native DSST, adaptive integration, conversion, and independent Java gates."""

from dataclasses import replace
import csv
import math
from pathlib import Path
import unittest

from distribution_propagator.backend import Backend, BackendConfig, _make_orbit


INITIAL = (26560000.0, 0.01, -0.004, 0.4, 0.3, 0.8)
FIXTURE = Path(__file__).parent / "data" / "orekit_long_horizon.csv"


class BackendTests(unittest.TestCase):
    def assert_elements_close(self, actual, expected, semimajor=5e-6, other=5e-13):
        for index, (a, e) in enumerate(zip(actual, expected)):
            self.assertLessEqual(abs(a - e), semimajor if index == 0 else other,
                                 f"component {index}: {a} versus {e}")

    def test_kepler_twenty_years_and_fsal(self):
        config = BackendConfig(force_model="kepler")
        backend = Backend(config, INITIAL)
        n = math.sqrt(config.mu / INITIAL[0]) / INITIAL[0]
        for target in (0.0, 86400.0, 365.25 * 86400.0, 20.0 * 365.25 * 86400.0):
            state = backend.advance(target)
            self.assertEqual(state.mean[:5], INITIAL[:5])
            self.assertLessEqual(abs(state.mean[5] - INITIAL[5] - n * target), 3e-11)
            self.assertEqual(state.mean, state.osculating)
            self.assertEqual(backend.time, target)
        stats = backend.stats
        self.assertGreater(stats.accepted_steps, 0)
        self.assertEqual(stats.derivative_evaluations,
                         1 + 6 * (stats.accepted_steps + stats.rejected_steps))

    def test_mean_osculating_roundtrip_varied_geometry(self):
        orbits = (
            INITIAL, (26560000.0, 0.0, 0.0, 0.4, 0.3, 0.8),
            (30000000.0, 0.6, 0.1, 0.99, 0.05, 2.8),
            (30000000.0, 0.02, -0.04, 1.7, 0.1, -2.0),
        )
        for model in ("j2", "j2_j2sq"):
            for initial in orbits:
                with self.subTest(model=model, orbit=initial):
                    config = BackendConfig(force_model=model, initial_type="mean")
                    converted = Backend(config, initial).advance(0.0)
                    self.assertGreater(abs(converted.osculating[0] - converted.mean[0]), 1.0)
                    restored = Backend(replace(config, initial_type="osculating"),
                                       converted.osculating).advance(0.0)
                    self.assert_elements_close(restored.mean, initial)
                    self.assert_elements_close(restored.osculating, converted.osculating)

    def test_tightened_accuracy_and_output_cadence(self):
        normal = BackendConfig(initial_type="mean")
        tight = replace(normal, relative_tolerance=1e-13, absolute_tolerance_m=1e-5,
                        absolute_tolerance_elements=1e-14, max_step_s=3600.0)
        target = 365.25 * 86400.0
        one_output = Backend(normal, INITIAL).advance(target)
        reference = Backend(tight, INITIAL).advance(target)
        scheduled = Backend(normal, INITIAL)
        for index in range(1, 101):
            many_outputs = scheduled.advance(target * index / 100.0)
        self.assert_elements_close(one_output.mean, reference.mean, 1e-3, 3e-9)
        self.assert_elements_close(many_outputs.mean, reference.mean, 1e-3, 3e-9)

    def test_mean_output_does_not_change_trajectory(self):
        config = BackendConfig(initial_type="mean")
        mean = Backend(replace(config, output_type="mean"), INITIAL)
        osculating = Backend(config, INITIAL)
        for target in (0.0, 86400.0, 20.0 * 86400.0):
            a, b = mean.advance(target), osculating.advance(target)
            self.assertEqual(a.mean, a.osculating)
            self.assertEqual(a.mean, b.mean)

    def test_repeated_output_keeps_steps_and_state(self):
        backend = Backend(BackendConfig(), INITIAL)
        first = backend.advance(100000.0)
        statistics = backend.stats
        self.assertEqual(backend.advance(100000.0), first)
        self.assertEqual(backend.stats, statistics)

    def test_cached_native_force_rates_equal_public_uncached_path(self):
        # Hansen caching must not freeze any state-dependent root or derivative.
        for model in ("j2", "j2_j2sq"):
            backend = Backend(BackendConfig(force_model=model, initial_type="mean"), INITIAL)
            for orbit in (INITIAL, (30000000.0, 0.6, 0.1, 0.99, 0.05, 12.8),
                          (30000000.0, 0.0, 0.0, 1.7, 0.1, -2.0)):
                with self.subTest(model=model, orbit=orbit):
                    native_orbit = _make_orbit(orbit, 3600.0, backend.config.mu)
                    cached = backend._rates(orbit, 3600.0)
                    uncached = backend._propagator.computeDerivatives(native_orbit)[:6]
                    self.assertEqual(cached, uncached)

    def test_high_eccentricity_against_independent_fixed_step_integrator(self):
        # Different integrator and uncached public native force dispatch, with
        # the same strict limits as the sibling C++ fixed-RK4 regression.
        initial = (30000000.0, 0.6, 0.1, 0.99, 0.05, 2.8)
        config = BackendConfig(initial_type="mean")
        target = 14.0 * 86400.0
        adaptive = Backend(config, initial).advance(target)
        reference = Backend(config, initial)._propagator
        reference.propagationType = "MEAN"
        reference.setInitialState(_make_orbit(initial, 0.0, config.mu), "MEAN")
        orbit = reference.propagate(target, step=300.0)
        expected = (orbit.a, orbit.equinoctial_ex, orbit.equinoctial_ey,
                    orbit.hx, orbit.hy, orbit.lm)
        self.assert_elements_close(adaptive.mean, expected, 1e-5, 5e-11)

    def test_minimum_step_failure_never_silently_relaxes_accuracy(self):
        config = BackendConfig(initial_type="mean", relative_tolerance=1e-30,
                               absolute_tolerance_m=1e-30, absolute_tolerance_elements=1e-30,
                               min_step_s=3600.0, max_step_s=3600.0)
        backend = Backend(config, INITIAL)
        with self.assertRaisesRegex(RuntimeError, "minimum step"):
            backend.advance(3600.0)
        self.assertEqual(backend.time, 0.0)
        self.assertEqual(backend.stats.rejected_steps, 1)

    def test_invalid_inputs_and_backward_time(self):
        for change in (
            {"mu": 0.0}, {"earth_radius_m": math.nan}, {"j2": -1.0},
            {"max_step_s": 0.0}, {"min_step_s": 1e6}, {"relative_tolerance": 0.0},
            {"absolute_tolerance_elements": math.inf}, {"force_model": "drag"},
            {"initial_type": "bad"}, {"output_type": "bad"},
        ):
            with self.subTest(config=change), self.assertRaises(ValueError):
                Backend(replace(BackendConfig(), **change), INITIAL)
        for orbit in ((-1.0, 0, 0, 0, 0, 0), (2e7, 1, 0, 0, 0, 0),
                      (2e7, 0, 0, 0, 0, math.inf), (2e7,)):
            with self.subTest(orbit=orbit), self.assertRaises(ValueError):
                Backend(BackendConfig(), orbit)
        backend = Backend(BackendConfig(), INITIAL)
        backend.advance(1.0)
        for time in (0.0, math.nan, math.inf):
            with self.subTest(time=time), self.assertRaises(ValueError):
                backend.advance(time)

    def test_independent_java_orekit_32_case_fixture(self):
        # These fixed limits are copied from the validated C++ sibling. They
        # bound numerical parity for this force model, not physical accuracy.
        from distribution_propagator.orbit import to_cartesian

        with FIXTURE.open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 32)
        backends = {}
        output_cache = {}
        worst_position = worst_velocity = 0.0
        for row in rows:
            key = (row["model"], row["initial_type"])
            if key not in backends:
                backends[key] = Backend(BackendConfig(force_model=key[0], initial_type=key[1]), INITIAL)
            target = float(row["days"]) * 86400.0
            cache_key = (*key, target)
            if cache_key not in output_cache:
                output_cache[cache_key] = backends[key].advance(target)
            actual = getattr(output_cache[cache_key], row["output_type"])
            expected = tuple(float(row[name]) for name in ("a", "ex", "ey", "hx", "hy", "lm"))
            with self.subTest(model=key, days=row["days"], output=row["output_type"]):
                self.assert_elements_close(actual, expected, 1e-4, 8e-10)
                cartesian = to_cartesian(actual, BackendConfig().mu)
                reference = tuple(float(row[name]) for name in ("x", "y", "z", "vx", "vy", "vz"))
                position = math.dist(cartesian[:3], reference[:3])
                velocity = math.dist(cartesian[3:], reference[3:])
                worst_position = max(worst_position, position)
                worst_velocity = max(worst_velocity, velocity)
                self.assertLessEqual(position, 0.02)
                self.assertLessEqual(velocity, 3e-6)
        print(f"Java Orekit parity: 32 cases, max position {worst_position:.10g} m, "
              f"max velocity {worst_velocity:.10g} m/s")


if __name__ == "__main__":
    unittest.main()

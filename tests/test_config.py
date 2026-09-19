from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from distribution_propagator.config import Config, config_template, estimated_memory_bytes, output_times, read_config, validate_config
from distribution_propagator.orbit import day


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def write(self, text):
        file = self.directory/'config.cfg'
        file.write_text(text, encoding='utf-8')
        return file

    def test_template_and_full_uint64_seed(self):
        c = read_config(self.write(config_template()))
        self.assertEqual(c.samples, 5000)
        self.assertEqual(c.accuracy_check, 8)
        self.assertAlmostEqual(c.nominal[5], math.pi/3, delta=1e-15)
        c = read_config(self.write('samples=20\nseed=18446744073709551615\n'))
        self.assertEqual(c.seed, (1<<64)-1)

    def test_relative_covariance_and_empirical_paths(self):
        covariance = np.diag([100, 1e-8, 4e-8, 0, 0, 1e-4])
        covariance[0,5] = covariance[5,0] = .02
        np.savetxt(self.directory/'cov.csv', covariance, delimiter=',')
        c = read_config(self.write('covariance_csv=cov.csv\n'))
        np.testing.assert_array_equal(c.covariance, covariance)
        c = read_config(self.write('empirical_samples_csv=posterior.csv\n'))
        self.assertEqual(c.empirical_samples_csv, self.directory/'posterior.csv')

    def test_parser_rejects_invalid_inputs(self):
        cases = ['unrecognized=1', 'samples=10\nsamples=20', 'samples=-1', 'samples=2.5',
                 'sigma=1,2,3,4,5', 'sigma=1,2,-3,4,5,6', 'uncertainty_coordinates=cartesian',
                 'duration_days=1oops', 'sigma=0,0,0,0,0,0\ncovariance_csv=cov.csv',
                 'orbit_equinoctial=26560000,0,0,0,0,0\norbit_keplerian_deg=26560000,0,0,0,0,0',
                 'seed=18446744073709551616', 'samples', '=1', 'samples=',
                 'duration_days=nan', 'duration_days=1_0', 'sigma=1e309,0,0,0,0,0']
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_config(self.write(text))

    def test_schedule_includes_exact_end_without_duplicate(self):
        c = Config(duration_days=2.5, output_step_days=1)
        self.assertEqual(output_times(c), [0, day, 2*day, 2.5*day])
        self.assertEqual(output_times(replace(c, duration_days=2)), [0, day, 2*day])
        self.assertEqual(output_times(replace(c, duration_days=.25)), [0, .25*day])

    def test_invalid_configuration_before_allocation(self):
        c = Config()
        for changes in [dict(output_step_days=0), dict(output_step_days=1e-12), dict(max_memory_mb=0),
                        dict(samples=1), dict(threads=62), dict(threads=-1), dict(samples=2.5),
                        dict(seed=-1), dict(coverage_occupied_fraction=1.01), dict(persistence=0),
                        dict(mixing_max_tv=float('nan')), dict(duration_days=float('inf'))]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(replace(c, **changes))

    def test_memory_estimates_account_for_processes_bins_and_html(self):
        c = Config(threads=1)
        baseline = estimated_memory_bytes(c, 5000)
        self.assertGreater(estimated_memory_bytes(replace(c, threads=8), 5000), baseline)
        self.assertGreater(estimated_memory_bytes(replace(c, phase_bins=100000), 5000), baseline)
        self.assertGreater(baseline, estimated_memory_bytes(replace(c, visual_samples=0), 5000))
        self.assertLess(estimated_memory_bytes(Config(), 50000), c.max_memory_mb*1024**2)

    def test_memory_estimate_covers_vectorized_analysis_scratch(self):
        # Fixed workers, no display, and >16 samples keep process, IPC, and
        # serialization terms constant. Measured analysis alone peaks near
        # 393 bytes/particle, besides initial storage and retained frames.
        c = Config(threads=1, visual_samples=0, duration_days=1, output_step_days=1)
        extra = estimated_memory_bytes(c, 201000)-estimated_memory_bytes(c, 200000)
        self.assertGreaterEqual(extra, 1000*(2*64+640))

    def test_memory_estimate_reserves_worker_serialization_copies(self):
        c = Config(threads=1, visual_samples=0, duration_days=2, output_step_days=1)
        extra_worker = estimated_memory_bytes(replace(c, threads=2), 1000)-estimated_memory_bytes(c, 1000)
        self.assertEqual(extra_worker, 64*1024**2+4*16*3*64)


if __name__ == '__main__':
    unittest.main()

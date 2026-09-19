"""Library arraylike inputs must also support the output contract."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distribution_propagator.backend import BackendConfig
from distribution_propagator.config import Config
from distribution_propagator.output import write_results
from distribution_propagator.simulation import run_simulation


class OutputTests(unittest.TestCase):
    def test_arraylike_config_and_numpy_integer_counts_export(self):
        config = Config(backend=BackendConfig(force_model="kepler", initial_type="mean", output_type="mean"),
                        nominal=(26560000., 0, 0, 0, 0, 0), covariance=np.zeros((6, 6)).tolist(),
                        samples=np.int64(2), threads=np.int64(1), phase_bins=np.int64(8),
                        seed=np.uint64(2**64 - 1), persistence=np.int64(1),
                        duration_days=.001, output_step_days=.001, write_html=False)
        simulation = run_simulation(config)
        with tempfile.TemporaryDirectory() as directory:
            write_results(simulation, directory)
            result = json.loads((Path(directory) / "run.json").read_text())
            self.assertEqual(result["metadata"]["seed_string"], str(2**64 - 1))
            self.assertEqual(result["metadata"]["nominal_elements"], list(config.nominal))
            self.assertEqual(result["summary"]["workers_used"], 1)
            before = (Path(directory) / "summary.json").read_bytes()
            with self.assertRaises(ValueError):
                write_results(simulation, directory)
            self.assertEqual((Path(directory) / "summary.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

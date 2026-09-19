# Distribution propagator — Python

Propagate initial Earth-orbit uncertainty using the **native Python Orekit DSST
port**. This independent implementation preserves the C++ tool's configuration,
random sample sequence, metrics, output schema, and offline HTML visualization.
Normal execution requires no C++ executable, compiler, or Java runtime.

See [DESCRIPTION.md](DESCRIPTION.md) for algorithms and data structures,
[SCIENCE.md](docs/SCIENCE.md) for the ball/banana/ribbon critique and sample-size
reasoning, and [VALIDATION.md](docs/VALIDATION.md) for measured evidence.

## Setup

Requires Python 3.10+, NumPy 1.24+, and the local native `DSST-python` port.
There is no compilation step. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy
$env:DSST_PYTHON_SOURCE = 'D:\orekit\DSST-python\src'
.\.venv\Scripts\python.exe run.py --help
```

Set `DSST_PYTHON_SOURCE` to the **src directory containing dsst**. Alternatively
install the local port into the same environment using
`.\.venv\Scripts\python.exe -m pip install D:/orekit/DSST-python`.
An explicit source setting takes priority; otherwise an installed package is
used, then the local `D:/orekit/DSST-python/src` fallback. Verified native
revision: `48001696aea61b4a5629f42ef509c0c8f0d83c87` (Orekit 13.1.6 formulas).
The upstream checkout is not modified. On Linux/macOS use `python3`, your
checkout's source path, and `.venv/bin/python`.

This workstation also has a ready Python/NumPy runtime:

```powershell
$python = 'C:\Users\trockwood\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $python run.py --help
```

Optional installation with your selected interpreter's `-m pip install .`
provides the `distribution-propagator` command and
`python -m distribution_propagator`. NumPy is a declared dependency;
the native DSST port must still be installed or located as above.

## Run and visualize

Use your selected interpreter (e.g. `.\.venv\Scripts\python.exe` or `& $python`)
in place of `python` below:

```powershell
python run.py --config examples/meo_ball.cfg --output outputs/meo-ball --workers 8
python run.py --config examples/meo_ball.cfg --samples 256 --accuracy-check 2 --output outputs/pilot
```

Open `outputs/meo-ball/visualization.html` in a browser. The offline viewer
supports rotation, zoom, pan, orbital-plane/edge-on views, exact-snapshot
playback, phase histograms and metric histories. Particle IDs stay stable.
Metrics use all particles; only displayed points are subsampled.

The full demonstration uses 5,000 particles, 60 days, and half-day snapshots.
It starts with isotropic 100 km position and 1 m/s velocity standard deviations
in RTN around a 26,560 km semimajor-axis orbit (e=0.02, i=55°). This is a broad
illustrative posterior, not measured orbit-determination uncertainty.
`meo_energy_shear.cfg` shows semimajor-axis shear; `meo_correlated.cfg` reads a
full correlated RTN covariance from `meo_covariance.csv`.

## Configuration and input distributions

Generate an example with `python run.py --write-example my_case.cfg`. Config
files use strict `key = value` entries with `#` comments. Unknown/duplicate
keys fail. Distances are metres, velocities metres/second, and angles radians
unless `_deg` is specified. Duration and output cadence are days; integrator
step limits ending in `_s` are seconds.

Choose one nominal orbit:

- `orbit_keplerian_deg = a,e,i,Omega,omega,M` (last four angles in degrees).
- `orbit_equinoctial = a,ex,ey,hx,hy,lambda` (continuous mean longitude last).
- `orbit_cartesian = x,y,z,vx,vy,vz` in a common inertial equatorial frame.

`initial_type` and `output_type` independently select `mean` or `osculating`.
Orbit-determination states and instantaneous physical visualization normally
use osculating elements. Set `uncertainty_coordinates` to `equinoctial`,
`cartesian`, or `rtn`, and choose one distribution input:

- `sigma`: six Gaussian standard deviations (diagonal covariance).
- `covariance_csv`: six headerless rows of six values, including all desired
  position/velocity correlations; positive semidefinite matrices are accepted.
- `empirical_samples_csv`: equally weighted, headerless six-value rows. Absolute
  states for equinoctial/Cartesian inputs; offsets from nominal for RTN inputs.
  The file's row count overrides `samples`.

CSV paths are relative to the config file. RTN order is
`[dR,dT,dN,dvR,dvT,dvN]`. Velocity errors are inertial vector components resolved
in the nominal epoch RTN basis, not rotating-coordinate derivatives. Covariance
units are products of the corresponding state units. Invalid orbits abort the
run without clipping, redrawing, or dropping particles. The default 1,000 km
perigee-altitude floor is a configured model-domain check, not assurance that
omitted perturbations are negligible.

## Controls and output

CLI overrides: `--samples`, `--seed`, `--workers`, `--visual-samples`,
`--duration-days`, `--output-step-days`, and `--accuracy-check`. `--threads`
aliases `--workers`; the compatible config key remains `threads`. In Python
these select **processes**. Zero selects at most eight automatically, one runs
directly, and explicit values up to 61 are accepted for Windows portability.

`--no-html` disables HTML; `--visual-samples 0` removes JSON display points.
`--export-states` adds full Cartesian histories and can create large files.
A new or empty output directory is required.

| File | Contents |
| --- | --- |
| `visualization.html` | Standalone viewer with embedded data. |
| `run.json` | Settings, provenance, summary, display cloud and all-ensemble metrics. |
| `summary.json` | Events, numerical checks, counters; written last as completion marker. |
| `initial_samples.csv` | Every initial equinoctial state and stable particle ID. |
| `metrics.csv` | Phase, geometric and coverage/mixing diagnostics at each epoch. |
| `states.csv` | Optional Cartesian history, mean longitude and mean semimajor axis. |

Schema version 1 and CSV columns match the C++ tool, with additional Python
provenance and worker count. `seed_string` preserves all 64 seed bits in browsers.
Missing events and unchecked accuracy results are `null`. Partial I/O failures
leave an incomplete directory without `summary.json`; rerun to a new directory.

## Interpretation, accuracy, and performance

Phase is **continuous DSST mean longitude minus nominal mean longitude**;
the wrapped histogram is not true anomaly. Defaults require three consecutive
saved epochs for each event:

- Coverage: largest empty phase gap ≤5° and all 72 phase bins occupied.
- Mixing: coverage plus all four circular harmonics ≤0.05 and histogram
  total variation from uniform ≤0.15.
- Central 95% wrap: unwrapped 2.5–97.5% interval at least one revolution.

Outputs report a preceding/first-passing epoch bracket and later confirmation
time. Refine cadence to resolve the sampled crossing. A mixing flag alone
does not establish a thin ribbon; inspect phase-matched RTN thickness and the
3D cloud. The Gaussian Kepler shear estimate is an interpretation aid.

Adaptive Dormand–Prince 5(4) integrates native DSST mean forces; short-period
terms are reconstructed at saved epochs. Defaults match C++: relative tolerance
`1e-11`, absolute semimajor-axis tolerance `0.001 m`, other element tolerance
`1e-12`, maximum step one day. `--accuracy-check N` repropagates the first N
particles with tenfold tighter tolerances and fourfold smaller maximum step.

Cached Hansen tables, separate processes, bounded queues, contiguous float64
histories and vectorized diagnostics improve speed without relaxing tolerances.
`max_memory_mb` is a conservative allocation estimate including processes and
serialization, not an OS hard limit. Reduce samples, saved epochs, display
points, or workers if it exceeds budget. Small jobs may be faster with one worker.

Start with 2,000–5,000 particles; use 20,000–50,000 for onset studies and check
doubled counts and at least three seeds. At one fixed time, 18,445 IID particles
give a 95% DKW scalar-CDF error bound of ±0.01, which does not directly constrain
mixing time or 3D density. Arbitrary empirical samples need not be IID.

```powershell
python tools/study.py --config examples/meo_ball.cfg --output outputs/convergence `
  --counts 2000,5000,20000 --seeds 11,22,33 --workers 8 --cadences 0.5,0.25
python tools/study.py --config examples/meo_ball.cfg --output outputs/worker-study `
  --counts 500 --seeds 11 --workers 1 --benchmark-workers 1,2,4,8 --accuracy-check 2
```

Studies save individual results, logs, hashes, and aggregate JSON/CSV. Worker
sweeps require byte-identical sample and metric files. Measured timings are in
[VALIDATION.md](docs/VALIDATION.md); Python is slower than compiled C++.

## Tests and Python API

From an uninstalled source checkout, first add `src` to Python's import path.
In PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -v
python tools/backend_probe.py outputs/python-reference.csv tests/data/orekit_long_horizon.csv
python tools/compare_cpp.py --cpp ../distribution_propagator/build/distribution_propagator.exe `
  --output outputs/cpp-comparison --long
python tools/verify_package.py
```

On Linux/macOS, the equivalent test command is
`PYTHONPATH=src python -m unittest discover -s tests -v`.

Standard tests require neither pytest nor a compiler. The independent Java
fixture is checked without a JVM. Optional C++ comparison uses an existing
executable as an oracle. The package check builds/installs offline in a local
isolated directory; pip, setuptools 68+ and NumPy must already be available.
Browser testing needs Node/Playwright and Chromium:
`node tools/test_viewer.cjs outputs/meo-ball/visualization.html`.

Import `Config`, `BackendConfig`, `run_simulation`, and `write_results` from
their modules for library use. Put process-parallel calls under
`if __name__ == '__main__':`; use `threads=1` in notebooks.

Current forces: Earth monopole, J2, and J2-squared. Drag, Sun/Moon, SRP,
higher harmonics, tesseral resonances, maneuvers, measurement updates, and
process noise are outside this version. Numerical agreement through one year
for this force set does not validate omitted physical effects over that horizon.

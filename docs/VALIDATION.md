# Validation and measured performance

Verified on Windows, 2026-09-19, using CPython 3.12.14 and NumPy 2.3.5.
The native Python DSST dependency was clean at
`48001696aea61b4a5629f42ef509c0c8f0d83c87`; the unchanged sibling C++ application
was at `d88bb04`. Results below apply to the implemented Earth monopole/J2/J2²
model, not omitted physical perturbations.

## Automated gates

**62 standard-library unittest tests passed.** Coverage includes:

- Independent orbital geometry, energy/angular-momentum invariants, scalar/
  vectorized conversions and nonsingular circular/equatorial cases.
- PSD correlated covariance, degenerate axes, MT19937-64 oracle vectors,
  Gaussian sample moments, RTN velocity conventions, empirical inputs.
- Twenty-year analytic Kepler phase, FSAL accounting, minimum-step failure,
  mean/osculating inversion, varied geometries, cached/uncached native rates,
  and a 14-day eccentric-orbit comparison against native fixed-step RK4.
- Independent Java Orekit fixture, circular harmonics including opposing lobes,
  unwrapped quantiles, persistence/confirmation times, Gaussian shear and DKW.
- CLI errors/overrides, nonintegral final cadence, full uint64 seeds, module
  execution, output protection, worker determinism, and convergence-study tools.

Full log: `outputs/final-tests.log` (generated and ignored by Git). Run with
the source import path documented in README, or install the package first.
The observed 77.97-second suite duration overlapped the demonstration run and
is not a benchmark.

## Independent Java reference

`tests/data/orekit_long_horizon.csv` is copied unchanged from the C++ project's
independently generated Java Orekit fixture. It contains 32 combinations of
J2/J2-squared force selection and mean/osculating state types through 365 days.
The fixture generator/provenance remains in the sibling C++ project; normal
tests need no Java installation.

| Quantity | Maximum observed discrepancy | Unchanged acceptance limit |
| --- | ---: | ---: |
| Cartesian position norm | 0.008781014 m | 0.02 m |
| Cartesian velocity norm | 1.274485e-6 m/s | 3e-6 m/s |
| Semimajor axis | 8.83e-7 m | 1e-4 m |
| Longitude | 3.320e-10 rad | 8e-10 rad |

Other equinoctial components also satisfy their original 8e-10 limit. No
reference value or acceptance tolerance was changed to make Python pass.

## Cross-language equivalence

`tools/compare_cpp.py` passed ten paired scenarios at 30 days and ten with
perturbed horizons extended to 365.25 days. Each set uses the three supported
force selections, mean/mean, mean/osculating, and osculating/osculating modes,
plus a 512-particle Kepler full-wrap transition. Every run samples and
propagates independently in the respective language.

On this workstation, tested initial elements and exported states were
numerically identical. Maximum phase-sigma difference was 7.11e-15 rad and
RTN-sigma difference 2.61e-8 m. Histograms, flags and event brackets matched
exactly. Such exact state equality is an observation, not a cross-platform
guarantee; the comparison script retains explicit tight numerical tolerances.

The full-wrap case detected coverage in [9.5,10] days, mixing in [12.5,13]
days, and central-95% wrapping in [11,11.5] days using its stated test thresholds.
Saved comparison reports include source hashes, C++ binary hash, tolerances,
per-case maxima, and timing:

- `outputs/cpp-parity-final-30d/comparison.json`
- `outputs/cpp-parity-final-year/comparison.json`

## Worker performance and reproducibility

Measured with the ball configuration, 500 particles, seed 20260919, J2/J2²,
60 days, 121 epochs, no HTML or display points, and two tighter accuracy checks.
Worker benchmarks ran sequentially. Times include sampling, propagation,
analysis and checks; wall time also includes interpreter startup and file I/O.

| Worker processes | Simulation seconds | Wall seconds | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 78.601 | 79.432 | 1.00 |
| 2 | 44.349 | 45.118 | 1.77 |
| 4 | 24.953 | 25.758 | 3.15 |
| 8 | 14.342 | 15.145 | 5.48 |

Every worker count produced identical SHA-256 hashes for initial samples and
metrics. Tightened-reference maximum errors were identical: 1.490e-6 m position
and 5.684e-14 rad phase. This small ensemble did not meet the strict default
sustained mixing criterion during the horizon; it is a performance experiment,
not a converged onset study. Evidence: `outputs/worker-performance/study.json`
and `study.csv`. Hardware/load and problem size affect speedups.

## Finished 5,000-particle example

`outputs/meo-ball-final/visualization.html` contains the complete demonstration
with 5,000 particles, 1,500 display particles, 121 saved epochs, eight workers,
and eight tighter accuracy checks. Its deliberately broad initial RTN cloud
has 100 km position and 1 m/s velocity standard deviations per axis.

| Event | Sampled onset bracket | Confirmation epoch |
| --- | --- | --- |
| Five-degree coverage | 7–7.5 days | 8.5 days |
| Central 95% width reaches one turn | 11–11.5 days | 12.5 days |
| Configured phase mixing | 15.5–16 days | 17 days |

These event brackets exactly match the existing C++ demonstration with the
same configuration and seed. The first eight tighter integrations differed
by at most 2.993e-6 m and 1.137e-13 rad. Propagation work was 610,000 accepted
steps, zero rejected steps, and 3,665,000 derivative evaluations. Python
simulation elapsed time was 159.69 seconds while other validation jobs also
ran; do not use that time as an isolated cross-language speed comparison.

This single-seed demonstration is not a convergence claim for actual OD
uncertainty. Use independent seeds, increased sample count, and finer output
cadence before reporting a scientific onset estimate.

## Viewer and installation

The copied viewer template is unchanged. Browser checks passed on synthetic
data, a 61-frame native DSST run, and the final 121-frame demonstration:
desktop/mobile layout, cloud/orbit framing, rotation, pan, zoom, saved-epoch
playback, canvas rendering, and no network requests or browser errors.
Screenshots are in `outputs/meo-ball-final/screenshots`.

`tools/verify_package.py` built a wheel offline with no build isolation,
installed it into a local target directory, confirmed imports came from that
installed target, and ran eight particles/three epochs/two spawned workers.
The packaged viewer resource produced a complete HTML file. The wheel contains
no native binaries and no global packages were changed. Packaging requires
the runtime/build dependencies already installed; the tool saves its evidence
under `build/package-verification`.

The JSON summary in `validation_results.json` preserves compact measured
evidence in Git; large histories, screenshots and timing logs stay outside Git.

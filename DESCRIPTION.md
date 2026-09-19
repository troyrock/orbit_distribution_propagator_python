# Data structures and numerical algorithms

This document describes implemented behavior. [README.md](README.md) explains
setup and usage; [docs/SCIENCE.md](docs/SCIENCE.md) discusses physical interpretation.

## Modules and data flow

| Module | Responsibility |
| --- | --- |
| `config.py` | Dataclass defaults, strict file parsing, epoch schedule, memory estimates. |
| `orbit.py` | Scalar coordinate conversion, orbit validity, RTN bases, PSD covariance factor. |
| `sampling.py` | C++-compatible MT19937-64/Box–Muller, Gaussian and empirical ensembles. |
| `backend.py` | Native DSST adapter, adaptive mean propagation, short-period reconstruction. |
| `geometry.py` | Vectorized Cartesian and RTN geometry for cloud analysis. |
| `statistics.py` | Quantiles, circular statistics, persistent events, analytic mixing estimate. |
| `simulation.py` | Independent process workers, stable histories, diagnostics and accuracy checks. |
| `output.py` | Scientific CSV/JSON and packaged offline HTML resource. |
| `cli.py` | CLI validation and orchestration. |

Execution validates input, samples the initial posterior, generates output
epochs, propagates a nominal reference, propagates every particle, analyzes
each frame, optionally repeats selected particles with tighter tolerances,
and writes results. Invalid particles or numerical failures abort the run.
Output creation starts only after the numerical calculation succeeds.

`Config` and `BackendConfig` are dataclasses. `BackendState` holds two six-element
tuples: `mean`, plus the selected output orbit in the compatibility field named
`osculating` (which contains mean elements when `output_type=mean`).
`BackendStats` counts accepted/rejected steps and derivative evaluations.
`Simulation` retains configuration, initial particles, a dense state history,
frame dictionaries, and a summary dictionary.

## Coordinates and storage

All particles share an initial epoch and an inertial geocentric equatorial
frame. Time is elapsed seconds; fixed-axis monopole/J2/J2-squared forces need
no calendar epoch or external ephemeris. Distances and velocities use SI units.
The six equinoctial elements are:

```text
[a, ex, ey, hx, hy, lambda]
ex = e*cos(omega+Omega)       ey = e*sin(omega+Omega)
hx = tan(i/2)*cos(Omega)      hy = tan(i/2)*sin(Omega)
lambda = M+omega+Omega       # continuous mean longitude, radians
```

Mean longitude is not true anomaly and is allowed to accumulate many turns.
The convention is regular for circular/equatorial prograde orbits, but rejects
near-retrograde singularities. Cartesian states are `[x,y,z,vx,vy,vz]`.
RTN basis matrices have radial, transverse, normal unit vectors as rows.

Initial particles form a NumPy float64 array of shape `(N,6)`. Retained history
has shape `(F,N,8)`: six selected output elements, continuous mean longitude,
and mean semimajor axis. This costs exactly `64*F*N` bytes. Frame dictionaries
retain only the first `min(visual_samples,N)` display positions, a 181-point
reference ellipse, and full-ensemble metrics. All computations use float64;
JSON round-trip precision and 17-significant-digit CSV preserve that precision.

Scalar and vectorized conversion solve the elliptic equinoctial Kepler
equation `F-ex*sin(F)+ey*cos(F)=lambda` after reducing longitude for trig calls.
A bracketed Newton iteration avoids unconstrained Newton divergence. Analytic
equinoctial plane bases map orbital-plane position/velocity to inertial space.
Tests check energy, angular momentum, independent geometry, and batch/scalar
agreement over large accumulated angles and eccentricity up to 0.95.

## Initial posterior sampling

Covariance order and units follow the declared uncertainty coordinates.
Factorization first scales by diagonal standard deviations, tests symmetry
and definiteness in correlation units, and uses a lower-triangular PSD factor.
Zero-variance rows must have zero cross-covariances. This avoids comparing
metre-squared and dimensionless entries with an inappropriate common threshold.

The Gaussian generator implements the standard 312-word MT19937-64 recurrence
and the C++ application's explicit 53-bit open-interval Box–Muller mapping.
It deliberately does not use Python `random` or NumPy MT19937, whose streams
are different. For each particle, six normals are multiplied by the lower
factor in fixed order. Sampling happens before process scheduling; the first
N samples are a stable prefix across ensemble sizes and worker counts.

Equinoctial errors add directly. Cartesian errors add to the nominal Cartesian
state. RTN position and inertial velocity-error vectors rotate through the
same nominal epoch basis before addition. These are not derivatives in a
rotating coordinate frame; no additional angular-velocity cross product is used.

Empirical inputs use every equally weighted row. Equinoctial/Cartesian rows
are absolute states, while RTN rows are offsets. Cartesian conversions lift
longitude onto the turn nearest nominal; explicitly supplied equinoctial
longitude preserves its winding. There is no implicit resampling or weighting.
Every initial and saved output orbit must be finite, bound, nonsingular, and
above the configured perigee floor. Invalid samples are neither clipped nor
redrawn, avoiding a silent change to the posterior.

## Native DSST propagation

The adapter imports the native `dsst` Python package. It constructs its
`DSSTZonal` J2 and `DSSTJ2SquaredClosedForm`/`ZeisModel` forces as configured,
using native auxiliary elements and short-period machinery. Kepler mean motion
is included exactly once. This application does not call C++, a JVM, or a
compiled DSST wrapper.

For an osculating initial state, native force-consistent inversion produces
the initial mean orbit. Native initialization and integration lifecycle hooks
are respected. Each saved osculating output reconstructs short-period terms
from the current mean state at that exact epoch; selected mean output skips
that reconstruction. Each particle owns independent mutable force state.

The upstream convenience propagator uses fixed-step RK4. This adapter instead
uses an embedded Dormand–Prince 5(4) integrator around the same native mean
force formulas. The weighted error uses absolute tolerances for semimajor axis
and dimensionless/angular elements plus relative scaling. Longitude scaling
is bounded so accumulated revolutions do not silently loosen phase accuracy.
Continuous longitude uses compensated accumulation to reduce long-run loss
of small increments. Step growth/shrinkage and min/max limits are bounded;
failure to satisfy accuracy at the minimum step is an error, not acceptance.

The final derivative is reused at the next accepted step (FSAL). Invariant
Hansen objects are cached within each force model rather than repeatedly
allocated. Tests compare cached and uncached native rates exactly. Output
times are reached by shortening a step, including a final nonintegral cadence
interval, so display playback never invents intermediate propagated states.

The optional accuracy check repropagates the first requested particles with
all three tolerances divided by ten and maximum step divided by four, bounded
below by the minimum step. Maximum Cartesian position and continuous phase
differences cover every saved frame. These are convergence diagnostics for
the selected model, not uncertainty in the real orbit.

## Parallel execution and memory

Workers are Python processes because native force calculations are Python
code. `spawn` is used on all platforms for isolation and Windows consistency.
One worker executes directly; automatic selection is `min(8,cpu_count,N)`.
Explicit worker counts are capped at 61 for Windows portability.

A task contains at most 16 particles and returns a contiguous `(F,batch,8)`
array plus integer work counters. At most twice the worker count of tasks is
in flight. Results go into predetermined particle slices, independent of
completion order. Finished futures are released before queue refill. Cloud
reductions run in the parent in fixed particle order. Worker-count comparisons
require identical initial samples, metrics, and complete exported histories.

Preflight estimates retained arrays, 640 bytes per particle for initial/analysis
scratch, Python display/JSON objects, interpreter reserves, and four bounded
transport buffers per worker (including serialization copies). Loading an
empirical file checks the estimate incrementally. JSON and HTML are streamed
without building duplicate full-document strings. The configured memory budget
is an estimate, not an operating-system enforcement mechanism.

## Statistics and event definitions

At each time, `phase_i = mean_lambda_i - nominal_mean_lambda`. Sample standard
deviation uses `N-1`. Quantiles use linear interpolation at index `p*(N-1)`.
The central 95% width is the unwrapped 97.5% minus 2.5% quantile.

Wrapped phases in `[0,2*pi)` give fixed bins, largest adjacent gap including
the zero-crossing gap, occupied-bin fraction, and:

```text
R_k = hypot(mean(cos(k*phase)), mean(sin(k*phase))), k=1..4
TV  = 0.5 * sum(abs(bin_count/N - 1/B))
```

Coverage tests maximum gap and occupancy. Mixing additionally tests every
harmonic and TV. Requiring several harmonics catches simple opposing-lobe
distributions that have small R1 but are not uniform. Thresholds depend on
sample count and angular resolution and remain configurable.

`first_sustained` finds the first run of `persistence` passing snapshots. Its
interval brackets the preceding and first-passing epoch; its confirmation
index is converted to elapsed seconds for `*_confirmed_at_s` output. Events
already present at time zero receive `[0,0]`. Unobserved events are null.
Saved-epoch persistence is not a continuous-time guarantee.

Global RTN standard deviations describe the cloud relative to the nominal
position. Phase-matched thickness compares each particle to the nominal
ellipse at the same selected-output mean longitude and resolves that residual
in the matched RTN basis. This removes the main geometric phase separation
for a thickness diagnostic; it does not replace the full nonlinear cloud or
prove thinness. Mean semimajor-axis spread is reported separately.

The analytic estimate uses sampled initial mean elements and rates
`n=sqrt(mu/a)/a`, with Welford accumulation of phase variance, rate variance,
and their covariance. It solves the stable positive quadratic root for
`Var(phi0+n*t)=-2*log(R_target)`. It assumes Gaussian Kepler shear and is not
the numerical coverage/mixing criterion. No rate variance yields null unless
the initial phase variance already exceeds the analytic target.

The summary also reports `sqrt(log(2/0.05)/(2*N))`, the 95% DKW error for one
scalar CDF at a fixed time with IID samples. Its applicability is qualified
for arbitrary empirical ensembles. It is not a mixing-time confidence interval.

## Output integrity and extension

All files are created exclusively in an empty directory. `summary.json` is
written last as completion marker. Nonfinite JSON is rejected and embedded
`<` characters escaped to prevent premature script termination. The package
includes the unchanged C++ viewer resource, with exactly one data insertion
token. It makes no network requests and animates saved epochs only.

Extension boundaries are explicit: add force configuration/native lifecycle
support in `backend.py`, coordinate transforms in `orbit.py`, distribution
designs in `sampling.py`, statistics in `statistics.py`, and schema/viewer
changes together. New forces require their own independent long-horizon
fixtures, mean/osculating checks, convergence checks, and physical scope notes.
Weighted/correlated sample designs require revised statistics and uncertainty
bounds. Avoid changing reference fixtures or acceptance tolerances merely to
make a regression pass.

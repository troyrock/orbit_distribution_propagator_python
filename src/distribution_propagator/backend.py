"""Adaptive propagation with the independent, native Orekit DSST Python port.

No C++ executable, shared library, JVM, or subprocess is used.  Force evaluation
and short-period reconstruction remain in the upstream DSST implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Sequence

Elements = tuple[float, float, float, float, float, float]


@dataclass
class BackendConfig:
    mu: float = 3.986004418e14
    earth_radius_m: float = 6378137.0
    j2: float = 1.08262668e-3
    force_model: str = "j2_j2sq"
    initial_type: str = "osculating"
    output_type: str = "osculating"
    relative_tolerance: float = 1e-11
    absolute_tolerance_m: float = 1e-3
    absolute_tolerance_elements: float = 1e-12
    min_step_s: float = 1e-3
    max_step_s: float = 86400.0

    def validate(self) -> None:
        for name in (
            "mu", "earth_radius_m", "relative_tolerance", "absolute_tolerance_m",
            "absolute_tolerance_elements", "min_step_s", "max_step_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_step_s > self.max_step_s:
            raise ValueError("minimum step exceeds maximum step")
        if not math.isfinite(self.j2) or self.j2 < 0.0:
            raise ValueError("J2 must be finite and nonnegative")
        if self.force_model not in ("kepler", "j2", "j2_j2sq"):
            raise ValueError("force_model must be kepler, j2, or j2_j2sq")
        if self.initial_type not in ("mean", "osculating"):
            raise ValueError("initial_type must be mean or osculating")
        if self.output_type not in ("mean", "osculating"):
            raise ValueError("output_type must be mean or osculating")


@dataclass(frozen=True)
class BackendState:
    mean: Elements
    # Contains mean elements when output_type="mean"; retain that output label.
    osculating: Elements


@dataclass
class BackendStats:
    accepted_steps: int = 0
    rejected_steps: int = 0
    derivative_evaluations: int = 0


@lru_cache(maxsize=1)
def _runtime() -> SimpleNamespace:
    # Protect the read-only upstream checkout, including lazily imported files.
    sys.dont_write_bytecode = True
    requested_source = os.environ.get("DSST_PYTHON_SOURCE")
    if requested_source:
        source = Path(requested_source).expanduser().resolve()
        if not (source / "dsst" / "dsst_propagator.py").is_file():
            raise ImportError("DSST_PYTHON_SOURCE must point to DSST-python/src")
        sys.path.insert(0, str(source))
    elif importlib.util.find_spec("dsst") is None:
        source = Path("D:/orekit/DSST-python/src")
        if not (source / "dsst" / "dsst_propagator.py").is_file():
            raise ImportError(
                "Native DSST Python port is missing. Install DSST-python or set "
                "DSST_PYTHON_SOURCE to its src directory."
            )
        sys.path.insert(0, str(source))

    from dsst.dsst_propagator import DSSTPropagator
    from dsst.forces.dsst_gravity_context import SphericalHarmonicsProviderData
    from dsst.forces.dsst_zonal import DSSTZonal
    from dsst.forces.dsstj_2_squared_closed_form import DSSTJ2SquaredClosedForm
    from dsst.forces.zeis_model import ZeisModel
    from dsst.utilities.auxiliary_elements import AuxiliaryElements, EquinoctialOrbitData

    return SimpleNamespace(
        Propagator=DSSTPropagator, Gravity=SphericalHarmonicsProviderData,
        Zonal=DSSTZonal, J2Squared=DSSTJ2SquaredClosedForm, Zeis=ZeisModel,
        Auxiliary=AuxiliaryElements, Orbit=EquinoctialOrbitData,
        source=str(Path(sys.modules["dsst"].__file__).resolve().parent),
    )


def backend_source() -> str:
    """Return the actual native DSST package path used in this process."""
    return _runtime().source


def _validate_elements(values: Sequence[float]) -> None:
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError("orbit must contain six finite equinoctial elements")
    if values[0] <= 0.0 or math.hypot(values[1], values[2]) >= 1.0:
        raise ValueError("DSST backend requires a bound elliptic orbit")


def _elements(orbit: object) -> Elements:
    return (orbit.a, orbit.equinoctial_ex, orbit.equinoctial_ey,
            orbit.hx, orbit.hy, orbit.lm)


def _make_orbit(values: Sequence[float], time: float, mu: float):
    _validate_elements(values)
    a, ex, ey, hx, hy, longitude = values
    eccentricity = math.hypot(ex, ey)
    motion = math.sqrt(mu / a) / a
    reduced = math.remainder(longitude, math.tau)
    correction = 0.0
    for _ in range(100):
        angle = reduced + correction
        f2 = ex * math.sin(angle) - ey * math.cos(angle)
        f1 = 1.0 - ex * math.cos(angle) - ey * math.sin(angle)
        f0 = correction - f2
        shift = 2.0 * f0 * f1 / (2.0 * f1 * f1 - f0 * f2)
        correction -= shift
        if abs(shift) <= 2e-15:
            break
    else:
        raise RuntimeError("equinoctial Kepler equation did not converge")
    le = longitude + correction
    reduced_le = reduced + correction
    numerator = ex * math.sin(reduced_le) - ey * math.cos(reduced_le)
    denominator = (1.0 + math.sqrt(1.0 - eccentricity * eccentricity)
                   - ex * math.cos(reduced_le) - ey * math.sin(reduced_le))
    lv = le + 2.0 * math.atan2(numerator, denominator)
    return _runtime().Orbit(
        date=time, frame="inertial_equatorial", e=eccentricity,
        keplerian_mean_motion=motion, keplerian_period=math.tau / motion,
        a=a, equinoctial_ex=ex, equinoctial_ey=ey, hx=hx, hy=hy,
        lm=longitude, le=le, lv=lv, mu=mu,
    )


_A = (
    (), (1.0 / 5.0,), (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
    (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
)
_C = (0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0)
_B4 = (5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
       -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0)
_ERROR_WEIGHTS = tuple((_A[6][j] if j < 6 else 0.0) - _B4[j] for j in range(7))


class Backend:
    """One particle's independent DSST force state and adaptive DP5(4) solver.

    Calls must have nondecreasing elapsed times in seconds. Different particles
    can run in separate processes; do not concurrently call one instance.
    """

    def __init__(self, config: BackendConfig, initial: Sequence[float]):
        config = replace(config)
        config.validate()
        _validate_elements(initial)
        # Keep NumPy scalar types supplied by ensemble sampling outside this
        # scalar hot loop and make the backend's tuple-of-floats API consistent.
        initial = tuple(float(value) for value in initial)
        native = _runtime()
        self.config = config
        self._native = native
        self._propagator = native.Propagator(propagationType="OSCULATING")
        self._propagator.setMu(config.mu)
        self._zonal = None
        self._j2_squared = None
        self._zonal_hansen = None
        if config.force_model != "kepler":
            gravity = native.Gravity(
                ae=config.earth_radius_m, mu=config.mu, c20=-config.j2,
                max_degree=2, max_order=0,
            )
            self._zonal = native.Zonal(gravity)
            self._propagator.addForceModel(self._zonal)
            if config.force_model == "j2_j2sq":
                self._j2_squared = native.J2Squared(native.Zeis(), gravity)
                self._propagator.addForceModel(self._j2_squared)
        self._central = self._propagator.getAllForceModels()[-1]
        orbit = _make_orbit(initial, 0.0, config.mu)
        if config.initial_type == "osculating":
            self._propagator.beforeIntegration(orbit)
            orbit = self._propagator.computeMeanState(orbit, epsilon=1e-14, maxIterations=200)
        self._current = _elements(orbit)
        _validate_elements(self._current)
        self._propagator.beforeIntegration(_make_orbit(self._current, 0.0, config.mu))
        if self._zonal is not None:
            # Reuse only invariant polynomial tables. Native createUAnddU still
            # refreshes every orbit-dependent Hansen root at each stage.
            self._zonal_hansen = self._zonal.createHansenObjects()
        self._statistics = BackendStats()
        self._elapsed = 0.0
        self._next_step = min(config.max_step_s, max(config.min_step_s, 3600.0))
        self._longitude_compensation = 0.0
        self._first_rate = None

    @property
    def stats(self) -> BackendStats:
        return replace(self._statistics)

    @property
    def time(self) -> float:
        return self._elapsed

    def _rates(self, state: Sequence[float], date: float) -> list[float]:
        orbit = _make_orbit(state, date, self.config.mu)
        auxiliary = self._native.Auxiliary(orbit, self._propagator.I)
        rates = [0.0] * 6
        if self._zonal is not None:
            context = self._zonal.initializeStep(auxiliary)
            potential = self._zonal.createUAnddU(date, context, auxiliary, self._zonal_hansen)
            rates = self._zonal.computeMeanElementRates(context, potential)
        if self._j2_squared is not None:
            second_order = self._j2_squared.getMeanElementRate(orbit, auxiliary)
            for j in range(6):
                rates[j] += second_order[j]
        central = self._propagator.elementRates(self._central, orbit, auxiliary, [self.config.mu])
        for j in range(6):
            rates[j] += central[j]
        self._statistics.derivative_evaluations += 1
        if not all(math.isfinite(value) for value in rates):
            raise RuntimeError("nonfinite native DSST derivative")
        return rates

    def advance(self, target: float) -> BackendState:
        if not math.isfinite(target) or target < self._elapsed:
            raise ValueError("output times must be finite and nondecreasing")
        config = self.config
        attempts = 0
        while self._elapsed < target:
            attempts += 1
            if attempts > 10_000_000:
                raise RuntimeError("integration step limit exceeded")
            remaining = target - self._elapsed
            h = min(self._next_step, config.max_step_s, remaining)
            if h <= 0.0 or self._elapsed + h == self._elapsed:
                raise RuntimeError("integration time resolution exhausted")
            if self._first_rate is None:
                self._first_rate = self._rates(self._current, self._elapsed)
            k = [self._first_rate]
            for stage in range(1, 7):
                increment = []
                for component in range(6):
                    # Explicit accumulation preserves the C++ evaluation order;
                    # Python 3.12+ sum() can use compensated summation instead.
                    weighted = 0.0
                    for j in range(stage):
                        weighted += _A[stage][j] * k[j][component]
                    increment.append(h * weighted)
                candidate = [self._current[j] + increment[j] for j in range(6)]
                k.append(self._rates(candidate, self._elapsed + _C[stage] * h))
            error = 0.0
            for component in range(6):
                weighted = 0.0
                for stage in range(7):
                    weighted += _ERROR_WEIGHTS[stage] * k[stage][component]
                magnitude = (1.0 if component == 5 else
                             max(abs(self._current[component]), abs(candidate[component])))
                absolute = (config.absolute_tolerance_m if component == 0
                            else config.absolute_tolerance_elements)
                scale = absolute + config.relative_tolerance * magnitude
                error = max(error, abs(h * weighted) / scale)
            if not math.isfinite(error):
                raise RuntimeError("nonfinite integration error estimate")
            factor = 5.0 if error == 0.0 else max(0.1, min(5.0, 0.9 * error ** -0.2))
            if error <= 1.0:
                phase_delta = increment[5] - self._longitude_compensation
                next_longitude = self._current[5] + phase_delta
                self._longitude_compensation = (next_longitude - self._current[5]) - phase_delta
                candidate[5] = next_longitude
                self._current = tuple(candidate)
                self._elapsed = target if h == remaining else self._elapsed + h
                self._first_rate = k[6]
                self._statistics.accepted_steps += 1
                self._next_step = max(config.min_step_s, min(config.max_step_s, h * factor))
            else:
                self._statistics.rejected_steps += 1
                if h <= config.min_step_s or remaining < config.min_step_s:
                    raise RuntimeError("requested accuracy cannot be met at minimum step")
                self._next_step = max(config.min_step_s, h * min(0.9, factor))
        osculating = self._current
        if config.output_type == "osculating":
            orbit = _make_orbit(self._current, self._elapsed, config.mu)
            # Clear interpolation slots and evaluate at this exact output date.
            self._propagator.beforeIntegration(orbit)
            osculating = _elements(self._propagator.computeOsculatingOrbit(orbit))
            _validate_elements(osculating)
        return BackendState(self._current, osculating)

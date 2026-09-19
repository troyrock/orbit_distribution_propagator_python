"""Strict, C++-compatible configuration parser and allocation preflight."""

from dataclasses import dataclass, field
import math
import numbers
import os
from pathlib import Path
import re

import numpy as np

from .backend import BackendConfig
from .orbit import covariance_factor, day, from_cartesian, from_keplerian, pi, validate_elements


@dataclass
class Config:
    backend: BackendConfig = field(default_factory=BackendConfig)
    nominal: np.ndarray = field(default_factory=lambda: from_keplerian(26560000, .02, 55*pi/180, .3, .4, .2))
    uncertainty_coordinates: str = "equinoctial"
    covariance: np.ndarray = field(default_factory=lambda: np.diag(np.square([100000, .0001, .0001, .00005, .00005, .0001])))
    empirical_samples_csv: Path | None = None
    samples: int = 5000
    threads: int = 0
    visual_samples: int = 1500
    phase_bins: int = 72
    persistence: int = 3
    seed: int = 20260919
    duration_days: float = 120.0
    output_step_days: float = 1.0
    minimum_altitude_m: float = 1000000.0
    coverage_max_gap_deg: float = 5.0
    coverage_occupied_fraction: float = 1.0
    mixing_max_resultant: float = .05
    mixing_max_tv: float = .15
    max_memory_mb: float = 2048.0
    accuracy_check: int = 0
    write_html: bool = True
    export_states: bool = False


def numeric_values(text, count=None):
    tokens = text.replace(',', ' ').split()
    # float() also accepts Python-specific underscores and Unicode digits;
    # the external file format deliberately uses ordinary decimal literals.
    number_pattern = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
    if any(re.fullmatch(number_pattern, token) is None for token in tokens):
        raise ValueError(f"Invalid numeric list: {text}")
    try:
        values = np.array([float(token) for token in tokens])
    except ValueError as error:
        raise ValueError(f"Invalid numeric list: {text}") from error
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite configuration value")
    if count is not None and len(values) != count:
        raise ValueError(f"Expected exactly {count} numeric values")
    return values


def _integer(text):
    if not re.fullmatch(r"\+?[0-9]+", text):
        raise ValueError(f"Expected nonnegative integer: {text}")
    value = int(text)
    if value > (1 << 64)-1:
        raise ValueError("Integer overflow")
    return value


def _read_covariance(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            line = raw.partition('#')[0].strip()
            if line:
                if len(rows) == 6:
                    raise ValueError("Covariance must have six rows")
                rows.append(numeric_values(line, 6))
    if len(rows) != 6:
        raise ValueError("Covariance must have six rows")
    return np.array(rows)


def read_config(path):
    path = Path(path)
    kv = {}
    with path.open(encoding="utf-8-sig") as stream:
        for lineno, raw in enumerate(stream, 1):
            line = raw.partition('#')[0].strip()
            if not line:
                continue
            if '=' not in line:
                raise ValueError(f"Missing '=' at config line {lineno}")
            key, value = (part.strip() for part in line.split('=', 1))
            if not key or not value or key in kv:
                raise ValueError(f"Empty or duplicate config key: {key}")
            kv[key] = value
    c = Config()
    for key in ("force_model", "initial_type", "output_type"):
        if key in kv:
            setattr(c.backend, key, kv.pop(key))
    if "uncertainty_coordinates" in kv:
        c.uncertainty_coordinates = kv.pop("uncertainty_coordinates")
    for key in ("mu", "earth_radius_m", "j2", "relative_tolerance", "absolute_tolerance_m",
                "absolute_tolerance_elements", "min_step_s", "max_step_s"):
        if key in kv:
            setattr(c.backend, key, float(numeric_values(kv.pop(key), 1)[0]))
    for key in ("duration_days", "output_step_days", "minimum_altitude_m", "coverage_max_gap_deg",
                "coverage_occupied_fraction", "mixing_max_resultant", "mixing_max_tv", "max_memory_mb"):
        if key in kv:
            setattr(c, key, float(numeric_values(kv.pop(key), 1)[0]))
    for key in ("samples", "threads", "visual_samples", "phase_bins", "persistence", "accuracy_check", "seed"):
        if key in kv:
            setattr(c, key, _integer(kv.pop(key)))
    kep, eq, cart = (kv.pop(key, None) for key in ("orbit_keplerian_deg", "orbit_equinoctial", "orbit_cartesian"))
    if sum(item is not None for item in (kep, eq, cart)) > 1:
        raise ValueError("Specify only one nominal orbit format")
    if kep is not None:
        values = numeric_values(kep, 6)
        c.nominal = from_keplerian(*values[:2], *(values[2:]*pi/180))
    elif eq is not None:
        c.nominal = numeric_values(eq, 6)
    elif cart is not None:
        c.nominal = from_cartesian(numeric_values(cart, 6), c.backend.mu)
    sig, cov, emp = (kv.pop(key, None) for key in ("sigma", "covariance_csv", "empirical_samples_csv"))
    if sum(item is not None for item in (sig, cov, emp)) > 1:
        raise ValueError("Specify one of sigma, covariance_csv, empirical_samples_csv")
    if sig is not None:
        values = numeric_values(sig, 6)
        if np.any(values < 0):
            raise ValueError("sigma cannot be negative")
        with np.errstate(over="ignore"):
            c.covariance = np.diag(values*values)
    elif cov is not None:
        c.covariance = _read_covariance(path.parent/cov)
    elif emp is not None:
        c.empirical_samples_csv = path.parent/emp
    if c.uncertainty_coordinates != "equinoctial" and all(v is None for v in (sig, cov, emp)):
        raise ValueError("Non-equinoctial uncertainty requires explicit sigma or covariance_csv")
    if kv:
        raise ValueError(f"Unknown config key: {sorted(kv)[0]}")
    validate_config(c)
    return c


def validate_config(c):
    positive = lambda x: math.isfinite(x) and x > 0
    b = c.backend
    if (not all(positive(getattr(b, key)) for key in ("mu", "earth_radius_m", "relative_tolerance",
            "absolute_tolerance_m", "absolute_tolerance_elements", "min_step_s", "max_step_s"))
            or b.min_step_s > b.max_step_s or not math.isfinite(b.j2) or b.j2 < 0):
        raise ValueError("Invalid DSST constants, tolerance or step limits")
    if b.force_model not in {"kepler", "j2", "j2_j2sq"} or b.initial_type not in {"mean", "osculating"} or b.output_type not in {"mean", "osculating"}:
        raise ValueError("Invalid force_model, initial_type or output_type")
    for key in ("samples", "threads", "visual_samples", "phase_bins", "persistence", "accuracy_check", "seed"):
        value = getattr(c, key)
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= value < 1 << 64:
            raise ValueError(f"Expected nonnegative 64-bit integer for {key}")
    if not 2 <= c.samples <= 100000000 or c.threads > 61 or not 4 <= c.phase_bins <= 100000 or c.persistence == 0:
        raise ValueError("Invalid sample/thread/bin/persistence count")
    if not all(positive(v) for v in (c.duration_days, c.output_step_days, c.max_memory_mb)) or c.duration_days*day > 1e13 or c.duration_days/c.output_step_days > 1000000:
        raise ValueError("Invalid/excessive duration, cadence or memory budget")
    if not math.isfinite(c.minimum_altitude_m) or c.minimum_altitude_m < 0:
        raise ValueError("minimum_altitude_m must be nonnegative")
    if not (0 < c.coverage_max_gap_deg <= 360 and 0 < c.coverage_occupied_fraction <= 1 and 0 < c.mixing_max_resultant < 1 and 0 < c.mixing_max_tv < 1):
        raise ValueError("Invalid coverage/mixing thresholds")
    if c.uncertainty_coordinates not in {"equinoctial", "cartesian", "rtn"}:
        raise ValueError("uncertainty_coordinates must be equinoctial, cartesian, or rtn")
    validate_elements(c.nominal, b.mu, b.earth_radius_m+c.minimum_altitude_m)
    covariance_factor(c.covariance)


def output_times(c):
    end, step = c.duration_days*day, c.output_step_days*day
    if not (math.isfinite(end) and math.isfinite(step) and end > 0 and step > 0) or end/step > 1000000:
        raise ValueError("Invalid/excessive output schedule")
    return [0.0] + [i*step for i in range(1, math.floor(end/step)+1) if i*step < end] + [end]


def estimated_memory_bytes(c, n):
    """Conservative storage budget including Python JSON/HTML copies and workers.

    The Python tool caps automatic process parallelism at eight workers.
    NumPy retains dense numerical arrays; reserve Python interpreter overhead,
    bounded interprocess result buffers, and Python objects during serialization.
    """
    frames = math.ceil(c.duration_days/c.output_step_days)+1
    displayed = min(c.visual_samples, n)
    workers = min(n, c.threads or min(8, os.cpu_count() or 1))
    return (frames*(n*64+c.phase_bins*8+1024)+n*304
            +frames*((displayed+181)*384+c.phase_bins*128+8192)
            +2*workers*min(16, n)*frames*64
            +(workers*64+64)*1024*1024)


def config_template():
    return """# SI units except degrees explicitly named and time in days.
force_model = j2_j2sq
initial_type = osculating
output_type = osculating
orbit_keplerian_deg = 26560000, 0.02, 55, 20, 30, 10
uncertainty_coordinates = equinoctial
# Deliberately broad demonstration; replace with your actual OD covariance.
sigma = 100000, 0.0001, 0.0001, 0.00005, 0.00005, 0.0001
samples = 5000
seed = 20260919
threads = 0
duration_days = 120
output_step_days = 1
visual_samples = 1500
phase_bins = 72
coverage_max_gap_deg = 5
coverage_occupied_fraction = 1
mixing_max_resultant = 0.05
mixing_max_tv = 0.15
persistence = 3
minimum_altitude_m = 1000000
relative_tolerance = 1e-11
absolute_tolerance_m = 0.001
absolute_tolerance_elements = 1e-12
min_step_s = 0.001
max_step_s = 86400
max_memory_mb = 2048
accuracy_check = 8
"""

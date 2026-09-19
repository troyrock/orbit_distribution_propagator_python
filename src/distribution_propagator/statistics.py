"""Circular diagnostics; definitions match the C++ simulator."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

TAU = 2.0 * math.pi


@dataclass(frozen=True)
class PhaseMetrics:
    sigma: float
    q95_width: float
    max_gap: float
    occupied_fraction: float
    total_variation: float
    resultants: tuple[float, ...]
    histogram: list[int]


@dataclass(frozen=True)
class EventInterval:
    found: bool = False
    lower_s: float = 0.0
    upper_s: float = 0.0
    confirmed_at: int = 0


def sample_sigma(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Statistics require a finite one-dimensional sequence")
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def quantile_sorted(values: Sequence[float], fraction: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not 0.0 <= fraction <= 1.0:
        raise ValueError("Invalid quantile input")
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) < 0):
        raise ValueError("Quantile input must be finite and sorted")
    location = fraction * (values.size - 1)
    lower = int(location)
    upper = min(lower + 1, values.size - 1)
    return float(values[lower] + (location - lower) * (values[upper] - values[lower]))


def phase_metrics(unwrapped_phase: Sequence[float], bins: int) -> PhaseMetrics:
    values = np.asarray(unwrapped_phase, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Phase requires a nonempty finite one-dimensional sequence")
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 4:
        raise ValueError("At least four integer phase bins are required")
    ordered = np.sort(values)
    width = quantile_sorted(ordered, 0.975) - quantile_sorted(ordered, 0.025)
    phase = np.remainder(values, TAU)
    indices = np.minimum(bins - 1, (phase / TAU * bins).astype(np.int64))
    histogram = np.bincount(indices, minlength=bins)
    resultants = tuple(float(math.hypot(float(np.mean(np.cos(k * phase))),
                                      float(np.mean(np.sin(k * phase)))))
                       for k in range(1, 5))
    wrapped_sorted = np.sort(phase)
    wrap_gap = wrapped_sorted[0] + TAU - wrapped_sorted[-1]
    max_gap = max(float(wrap_gap), float(np.max(np.diff(wrapped_sorted), initial=0)))
    return PhaseMetrics(
        sample_sigma(values), width, max_gap,
        float(np.count_nonzero(histogram) / bins),
        float(0.5 * np.sum(np.abs(histogram / values.size - 1.0 / bins))),
        resultants, histogram.tolist(),
    )


def dkw_epsilon(n: int, alpha: float = 0.05) -> float:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("Sample count must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def first_sustained(times: Sequence[float], flags: Sequence[bool],
                    persistence: int) -> EventInterval:
    if len(times) != len(flags) or persistence < 1 or isinstance(persistence, bool):
        raise ValueError("Invalid event series")
    if not isinstance(persistence, int):
        raise ValueError("Persistence must be an integer")
    if any(not math.isfinite(t) for t in times) or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("Event times must be finite and nondecreasing")
    run = 0
    for index, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run >= persistence:
            first = index + 1 - persistence
            return EventInterval(True, float(times[max(0, first - 1)]), float(times[first]), index)
    return EventInterval()


def estimate_mixing(mean_longitude: np.ndarray, mean_a: np.ndarray, mu: float,
                    max_resultant: float) -> float | None:
    """Gaussian Kepler-shear scale, not the full numerical mixing criterion."""
    # Welford update deliberately mirrors the C++ reference's accumulation.
    origin = float(mean_longitude[0])
    mp = mn = vp = vn = covariance = 0.0
    for count, (longitude, axis) in enumerate(zip(mean_longitude, mean_a), 1):
        phase = float(longitude) - origin
        rate = math.sqrt(mu / float(axis)) / float(axis)
        dp, dn = phase - mp, rate - mn
        mp += dp / count
        mn += dn / count
        vp += dp * (phase - mp)
        vn += dn * (rate - mn)
        covariance += dp * (rate - mn)
    denominator = len(mean_a) - 1
    if denominator < 1:
        raise ValueError("Mixing estimate requires at least two samples")
    vp, vn, covariance = vp / denominator, vn / denominator, covariance / denominator
    target = -2 * math.log(max_resultant)
    if vp >= target:
        return 0.0
    if vn <= np.finfo(float).tiny:
        return None
    root = math.sqrt(covariance * covariance + vn * (target - vp))
    return ((target - vp) / (root + covariance) if covariance >= 0
            else (root - covariance) / vn)

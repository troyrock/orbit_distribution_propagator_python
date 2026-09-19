"""Deterministic ensemble propagation with bounded process parallelism."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
import multiprocessing
import os
import time

import numpy as np

from .backend import Backend, BackendStats
from .config import estimated_memory_bytes, output_times, validate_config
from .geometry import cartesian_batch, rtn_batch
from .orbit import validate_elements
from .sampling import sample_initial
from .statistics import dkw_epsilon, estimate_mixing, first_sustained, phase_metrics, sample_sigma


@dataclass
class Simulation:
    config: object
    initial: np.ndarray
    # Frame, particle, [six output elements, continuous mean longitude, mean a].
    states: np.ndarray
    frames: list
    summary: dict


def worker_count(config, n):
    return min(n, config.threads or min(8, os.cpu_count() or 1))


def _propagate_batch(start, initial, times, config, minimum_radius):
    """Top-level, picklable worker; each particle owns all mutable DSST state."""
    values = np.empty((len(times), len(initial), 8), dtype=np.float64)
    totals = BackendStats()
    for local, elements in enumerate(initial):
        epoch = 0.0
        try:
            backend = Backend(config, elements)
            for frame, epoch in enumerate(times):
                state = backend.advance(epoch)
                validate_elements(state.osculating, config.mu, minimum_radius)
                values[frame, local, :6] = state.osculating
                values[frame, local, 6:] = state.mean[5], state.mean[0]
            for key in vars(totals):
                setattr(totals, key, getattr(totals, key) + getattr(backend.stats, key))
        except Exception as error:
            raise RuntimeError(f"Particle {start + local} at {epoch:g} s: {error}") from error
    return start, values, totals


def _propagate(initial, times, config, progress):
    n = len(initial)
    states = np.empty((len(times), n, 8), dtype=np.float64)
    workers = worker_count(config, n)
    totals = BackendStats()
    completed, last_report = 0, time.perf_counter()
    # Keep enough tasks for all workers, including tiny validation ensembles.
    batch_size = min(16, max(1, (n + 4 * workers - 1) // (4 * workers)))
    next_start = 0

    def accept(result):
        nonlocal completed
        start, values, stats = result
        count = values.shape[1]
        states[:, start:start + count, :] = values
        completed += count
        for key in vars(totals):
            setattr(totals, key, getattr(totals, key) + getattr(stats, key))

    def report(force=False):
        nonlocal last_report
        now = time.perf_counter()
        if progress and (force or now - last_report >= 5):
            progress(f"Propagated {completed} / {n} particles")
            last_report = now

    args = (times, config.backend, config.backend.earth_radius_m + config.minimum_altitude_m)
    if workers == 1:
        for start in range(0, n, batch_size):
            accept(_propagate_batch(start, initial[start:start + batch_size], *args))
            report()
    else:
        # Spawn is portable and avoids sharing the native port's mutable caches.
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = set()
            try:
                while pending or next_start < n:
                    while next_start < n and len(pending) < 2 * workers:
                        end = min(n, next_start + batch_size)
                        pending.add(pool.submit(_propagate_batch, next_start,
                                                initial[next_start:end], *args))
                        next_start = end
                    done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        accept(future.result())
                    if done:
                        del future
                        done.clear()
                    report()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
    report(force=True)
    return states, totals, workers


def _analyze_frame(epoch, values, reference, config):
    mu = config.backend.mu
    ref = np.asarray(reference.osculating)
    refcart = cartesian_batch(ref, mu)
    cart = cartesian_batch(values[:, :6], mu)
    residual = (cart[:, :3] - refcart[:3]) @ rtn_batch(refcart).T
    matched = np.broadcast_to(ref, (len(values), 6)).copy()
    matched[:, 5] = values[:, 5]
    matchcart = cartesian_batch(matched, mu)
    tube = np.einsum("nij,nj->ni", rtn_batch(matchcart), cart[:, :3] - matchcart[:, :3])
    phase = phase_metrics(values[:, 6] - reference.mean[5], config.phase_bins)
    coverage = (phase.max_gap <= config.coverage_max_gap_deg * np.pi / 180
                and phase.occupied_fraction >= config.coverage_occupied_fraction)
    mixed = (coverage and max(phase.resultants) <= config.mixing_max_resultant
             and phase.total_variation <= config.mixing_max_tv)
    curve = np.broadcast_to(ref, (181, 6)).copy()
    curve[:, 5] = np.arange(181) * (2 * np.pi) / 180
    metrics = {
        "phase_sigma_rad": phase.sigma, "phase_q95_width_rad": phase.q95_width,
        "resultants": list(phase.resultants), "max_gap_deg": phase.max_gap * 180 / np.pi,
        "occupied_fraction": phase.occupied_fraction, "total_variation": phase.total_variation,
        "histogram": phase.histogram, "rtn_sigma_m": np.std(residual, axis=0, ddof=1).tolist(),
        "tube_rtn_sigma_m": np.std(tube, axis=0, ddof=1).tolist(),
        "semimajor_sigma_m": sample_sigma(values[:, 7]), "coverage": bool(coverage), "mixed": bool(mixed),
    }
    return {"time_s": epoch, "positions_m": cart[:config.visual_samples, :3].tolist(),
            "reference_elements": ref.tolist(),
            "reference_orbit_m": cartesian_batch(curve, mu)[:, :3].tolist(), "metrics": metrics}


def run_simulation(config, progress=None):
    """Run an ensemble; no output files are written until all checks succeed.

    Scripts using multiple workers must call this beneath ``if __name__ ==
    '__main__':``. Use threads=1 in an interactive interpreter or notebook.
    """
    validate_config(config)
    # Normalize accepted NumPy integer inputs to portable multiprocessing/JSON values.
    config = replace(config, **{key: int(getattr(config, key)) for key in (
        "samples", "threads", "visual_samples", "phase_bins", "persistence", "accuracy_check", "seed")})
    start = time.perf_counter()
    initial = sample_initial(config)
    config = replace(config, samples=len(initial))
    if estimated_memory_bytes(config, len(initial)) > config.max_memory_mb * 1024**2:
        raise ValueError("Estimated state/display/worker storage exceeds max_memory_mb; "
                         "reduce samples, snapshots, visual samples or workers")
    times = output_times(config)
    nominal = Backend(config.backend, config.nominal)
    references = [nominal.advance(epoch) for epoch in times]
    states, stats, workers = _propagate(initial, times, config, progress)
    frames = [_analyze_frame(epoch, values, ref, config)
              for epoch, values, ref in zip(times, states, references)]
    summary = {"sample_count": len(initial), "frame_count": len(times),
               "threads_used": workers, "workers_used": workers}
    for name, flags in (
        ("coverage", [f["metrics"]["coverage"] for f in frames]),
        ("mixing", [f["metrics"]["mixed"] for f in frames]),
        ("central95_wrap", [f["metrics"]["phase_q95_width_rad"] >= 2 * np.pi for f in frames]),
    ):
        event = first_sustained(times, flags, config.persistence)
        summary[name + "_time_s"] = event.upper_s if event.found else None
        summary[name + "_bracket_s"] = [event.lower_s, event.upper_s] if event.found else None
        summary[name + "_confirmed_at_s"] = times[event.confirmed_at] if event.found else None
    summary.update({
        "analytic_mixing_time_s": estimate_mixing(states[0, :, 6], states[0, :, 7],
                                                   config.backend.mu, config.mixing_max_resultant),
        "analytic_estimate_model": "Gaussian Kepler phase shear using initial sampled mean elements; not the full numerical mixing criterion",
        "dkw_cdf_error_95": dkw_epsilon(len(initial)),
        "dkw_applicability": ("Only if empirical particles are IID draws; not guaranteed for arbitrary supplied ensembles"
                              if config.empirical_samples_csv else "IID initial Gaussian sampling; pointwise-in-time CDF bound"),
        "accuracy_checked_samples": min(config.accuracy_check, len(initial)),
        "accuracy_max_position_m": None, "accuracy_max_phase_rad": None,
        **vars(stats),
    })
    if config.accuracy_check:
        tighter = replace(config.backend,
                          relative_tolerance=config.backend.relative_tolerance * .1,
                          absolute_tolerance_m=config.backend.absolute_tolerance_m * .1,
                          absolute_tolerance_elements=config.backend.absolute_tolerance_elements * .1,
                          max_step_s=max(config.backend.min_step_s, config.backend.max_step_s / 4))
        position_error = phase_error = 0.0
        for particle in range(summary["accuracy_checked_samples"]):
            check = Backend(tighter, initial[particle])
            for frame, epoch in enumerate(times):
                fine = check.advance(epoch)
                coarse = states[frame, particle]
                delta = cartesian_batch(fine.osculating, tighter.mu)[:3] - cartesian_batch(coarse[:6], tighter.mu)[:3]
                position_error = max(position_error, float(np.linalg.norm(delta)))
                phase_error = max(phase_error, abs(fine.mean[5] - coarse[6]))
        summary["accuracy_max_position_m"] = position_error
        summary["accuracy_max_phase_rad"] = float(phase_error)
    summary["elapsed_seconds"] = time.perf_counter() - start
    return Simulation(config, initial, states, frames, summary)

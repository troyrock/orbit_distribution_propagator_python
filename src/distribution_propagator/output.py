"""C++-compatible scientific exports and a self-contained offline HTML viewer."""

import csv
from importlib import resources
import json
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np

from .backend import backend_source
from .geometry import cartesian_batch


def ensure_empty_output(output):
    path = Path(output)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output directory must be empty: {path}")


def _revision(source):
    root = Path(source).parent.parent
    if not (root / ".git").exists():
        return "unknown (installed package without Git metadata)"
    try:
        prefix = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
        revision = subprocess.run(prefix + ["rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=True, timeout=5).stdout.strip()
        dirty = subprocess.run(prefix + ["status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        return revision + (" (modified)" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown (Git unavailable)"


def run_document(simulation):
    c, b = simulation.config, simulation.config.backend
    source = backend_source()
    settings = {key: getattr(c, key) for key in (
        "duration_days", "output_step_days", "minimum_altitude_m", "persistence",
        "coverage_max_gap_deg", "coverage_occupied_fraction", "mixing_max_resultant",
        "mixing_max_tv", "max_memory_mb")}
    settings.update({key: getattr(b, key) for key in ("relative_tolerance", "absolute_tolerance_m",
                      "absolute_tolerance_elements", "min_step_s", "max_step_s")})
    metadata = {
        "samples": len(simulation.initial), "visual_samples": min(c.visual_samples, len(simulation.initial)),
        "force_model": b.force_model, "input_type": b.initial_type, "output_type": b.output_type,
        "uncertainty_coordinates": c.uncertainty_coordinates,
        "sampling": "equal-weight empirical ensemble" if c.empirical_samples_csv else "IID Gaussian",
        "seed": int(c.seed), "seed_string": str(c.seed), "mu": b.mu, "earth_radius_m": b.earth_radius_m,
        "j2": b.j2, "elapsed_seconds": simulation.summary["elapsed_seconds"],
        "dsst_revision": _revision(source), "dsst_source": source,
        "implementation": "native Python DSST", "python_version": platform.python_version(),
        "numpy_version": np.__version__, "phase_bins": int(c.phase_bins),
        "phase_definition": "Continuous mean longitude minus nominal mean longitude; histogram modulo 2pi. Not true anomaly.",
        "coverage_definition": f"Largest empty phase gap <= {c.coverage_max_gap_deg:g} degrees and occupied-bin fraction >= {c.coverage_occupied_fraction:g}; sustained for {c.persistence} snapshots",
        "mixing_definition": f"Coverage plus all R1..R4 <= {c.mixing_max_resultant:g} and histogram total variation <= {c.mixing_max_tv:g}; sustained for {c.persistence} snapshots",
        "model_scope": "Earth monopole/J2/J2-squared as selected; no drag, Sun/Moon, SRP, higher zonals, tesseral gravity, maneuvers, measurement updates or process noise. Inertial equatorial axes, relative epoch.",
        "nominal_elements": np.asarray(c.nominal, dtype=float).tolist(),
        "covariance": None if c.empirical_samples_csv else np.asarray(c.covariance, dtype=float).tolist(), "settings": settings,
    }
    return {"schema_version": 1, "metadata": metadata, "summary": simulation.summary,
            "frames": simulation.frames}


def _row(writer, values):
    writer.writerow(format(value, ".17g") if isinstance(value, (float, np.floating)) else value
                    for value in values)


def write_results(simulation, output):
    """Write to an unused directory, publishing summary.json last as completion marker."""
    ensure_empty_output(output)
    output = Path(output)
    document = run_document(simulation)
    template = resources.files("distribution_propagator").joinpath("resources/viewer.html").read_text(encoding="utf-8")
    if template.count("__DISTRIBUTION_DATA__") != 1:
        raise RuntimeError("Invalid embedded viewer template")
    output.mkdir(parents=True, exist_ok=True)
    # Stream encoded pieces; never make another full in-memory JSON/HTML copy.
    encoder = json.JSONEncoder(allow_nan=False, separators=(",", ":"))
    with (output / "run.json").open("x", encoding="utf-8", newline="\n") as stream:
        for piece in encoder.iterencode(document):
            stream.write(piece.replace("<", "\\u003c"))
        stream.write("\n")
    if simulation.config.write_html:
        before, after = template.split("__DISTRIBUTION_DATA__")
        with (output / "visualization.html").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(before)
            with (output / "run.json").open(encoding="utf-8") as data:
                shutil.copyfileobj(data, stream)
            stream.write(after)
    with (output / "initial_samples.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow("particle,a_m,ex,ey,hx,hy,mean_longitude_rad".split(","))
        for particle, state in enumerate(simulation.initial):
            _row(writer, [particle, *state])
    with (output / "metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow("time_s,time_days,phase_sigma_rad,phase_q95_width_rad,R1,R2,R3,R4,max_gap_deg,occupied_fraction,total_variation,r_sigma_m,t_sigma_m,n_sigma_m,tube_r_sigma_m,tube_t_sigma_m,tube_n_sigma_m,semimajor_sigma_m,coverage,mixed".split(","))
        for frame in simulation.frames:
            m = frame["metrics"]
            _row(writer, [frame["time_s"], frame["time_s"] / 86400, m["phase_sigma_rad"], m["phase_q95_width_rad"],
                          *m["resultants"], m["max_gap_deg"], m["occupied_fraction"], m["total_variation"],
                          *m["rtn_sigma_m"], *m["tube_rtn_sigma_m"], m["semimajor_sigma_m"],
                          int(m["coverage"]), int(m["mixed"])])
    if simulation.config.export_states:
        with (output / "states.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow("time_s,particle,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,mean_longitude_rad,mean_a_m".split(","))
            for frame, states in zip(simulation.frames, simulation.states):
                cart = cartesian_batch(states[:, :6], simulation.config.backend.mu)
                for particle, state in enumerate(states):
                    _row(writer, [frame["time_s"], particle, *cart[particle], *state[6:]])
    with (output / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(simulation.summary, stream, allow_nan=False, indent=2)
        stream.write("\n")

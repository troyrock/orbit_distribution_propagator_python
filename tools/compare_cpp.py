#!/usr/bin/env python3
"""Optional cross-language oracle comparison of independently executed simulators.

The Python simulator never uses the C++ executable. This validation script runs
both separately, on the same reproducible Gaussian draws, and compares initial
samples, every exported state, every displayed position, scientific metrics,
histograms, classifications, and event brackets. It retains all run artifacts.

Example:
    python tools/compare_cpp.py --cpp ../distribution_propagator/build-msvc-ninja/distribution_propagator.exe --output outputs/parity

Defaults deliberately use small ensembles for fast validation, not inference
about population mixing. The additional Kepler case uses 512 samples over 80
days to exercise actual coverage and mixing transitions. --long extends all
perturbed cases to one year. See the tolerance rationales in TOLERANCES below.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TOLERANCES = {
    "initial_a_m": 1e-6,  # Cartesian conversion roundoff at MEO radius.
    "initial_elements": 2e-13,
    "position_m": 0.01,  # 10 x configured 1 mm local a tolerance; tested globally.
    "velocity_m_s": 1e-6,
    "mean_longitude_rad": 5e-10,  # 13 mm arc length at the nominal MEO radius.
    "mean_a_m": 1e-4,
    "phase_rad": 5e-10,
    "resultant": 5e-10,
    "gap_deg": 1e-7,
    "sigma_m": 0.02,  # Pairwise geometric differences contribute to spread.
    "semimajor_sigma_m": 1e-4,
    "probability": 1e-12,
}


class ComparisonError(AssertionError):
    """A measured discrepancy exceeded the documented comparison contract."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    def reject(value):
        raise ComparisonError(f"Nonfinite JSON value {value} in {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def exact(actual, expected, label: str) -> None:
    if actual != expected:
        raise ComparisonError(f"{label}: Python {actual!r} != C++ {expected!r}")


def close(actual, expected, tolerance: float, label: str, maxima: dict) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ComparisonError(f"{label}: comparison contains nonfinite values")
    error = abs(actual - expected)
    maxima[label] = max(maxima.get(label, 0.0), error)
    if error > tolerance:
        raise ComparisonError(f"{label}: |{actual:.17g} - {expected:.17g}| = {error:.8g} > {tolerance:g}")


def compare_case(python_output: Path, cpp_output: Path) -> dict:
    py = read_json(python_output / "run.json")
    cpp = read_json(cpp_output / "run.json")
    maxima = {}
    exact(py["schema_version"], cpp["schema_version"], "schema_version")
    exact(len(py["frames"]), len(cpp["frames"]), "frame_count")
    pinit = read_csv(python_output / "initial_samples.csv")
    cinit = read_csv(cpp_output / "initial_samples.csv")
    exact(len(pinit), len(cinit), "initial sample count")
    exact(set(pinit[0]), set(cinit[0]), "initial sample CSV schema")
    for p, c in zip(pinit, cinit):
        exact(p["particle"], c["particle"], "initial particle ordering")
        for field in ("a_m", "ex", "ey", "hx", "hy", "mean_longitude_rad"):
            tolerance = TOLERANCES["initial_a_m" if field == "a_m" else "initial_elements"]
            close(float(p[field]), float(c[field]), tolerance, "initial_" + field, maxima)

    pstates = read_csv(python_output / "states.csv")
    cstates = read_csv(cpp_output / "states.csv")
    exact(len(pstates), len(cstates), "exported state count")
    exact(set(pstates[0]), set(cstates[0]), "state CSV schema")
    for p, c in zip(pstates, cstates):
        exact(p["particle"], c["particle"], "state particle ordering")
        exact(float(p["time_s"]), float(c["time_s"]), "state epoch")
        for columns, name in ((("x_m", "y_m", "z_m"), "position_m"),
                              (("vx_m_s", "vy_m_s", "vz_m_s"), "velocity_m_s")):
            distance = math.dist([float(p[k]) for k in columns], [float(c[k]) for k in columns])
            close(distance, 0.0, TOLERANCES[name], "state_" + name, maxima)
        for key in ("mean_longitude_rad", "mean_a_m"):
            close(float(p[key]), float(c[key]), TOLERANCES[key], key, maxima)

    metric_scalars = {
        "phase_sigma_rad": "phase_rad", "phase_q95_width_rad": "phase_rad",
        "max_gap_deg": "gap_deg", "occupied_fraction": "probability",
        "total_variation": "probability", "semimajor_sigma_m": "semimajor_sigma_m",
    }
    for p, c in zip(py["frames"], cpp["frames"]):
        exact(p["time_s"], c["time_s"], "frame epoch")
        for key in ("positions_m", "reference_orbit_m"):
            exact(len(p[key]), len(c[key]), key + " length")
            for point, expected in zip(p[key], c[key]):
                close(math.dist(point, expected), 0.0, TOLERANCES["position_m"], key, maxima)
        pm, cm = p["metrics"], c["metrics"]
        exact(set(pm), set(cm), "metric schema")
        for field in ("histogram", "coverage", "mixed"):
            exact(pm[field], cm[field], field)
        for field, tolerance_name in metric_scalars.items():
            close(pm[field], cm[field], TOLERANCES[tolerance_name], field, maxima)
        for field, tolerance_name in (("resultants", "resultant"), ("rtn_sigma_m", "sigma_m"),
                                      ("tube_rtn_sigma_m", "sigma_m")):
            exact(len(pm[field]), len(cm[field]), field + " length")
            for value, expected in zip(pm[field], cm[field]):
                close(value, expected, TOLERANCES[tolerance_name], field, maxima)

    ps = read_json(python_output / "summary.json")
    cs = read_json(cpp_output / "summary.json")
    for field in ("sample_count", "frame_count", "coverage_time_s", "coverage_bracket_s",
                  "mixing_time_s", "mixing_bracket_s", "central95_wrap_time_s",
                  "central95_wrap_bracket_s"):
        exact(ps[field], cs[field], field)
    exact(ps["accuracy_checked_samples"], 0, "accuracy checks disabled during timing")
    exact(ps["accuracy_max_position_m"], None, "unchecked position error")
    exact(ps["accuracy_max_phase_rad"], None, "unchecked phase error")
    if cs["analytic_mixing_time_s"] is None:
        exact(ps["analytic_mixing_time_s"], None, "analytic mixing estimate")
    else:
        close(ps["analytic_mixing_time_s"], cs["analytic_mixing_time_s"],
              max(1e-6, abs(cs["analytic_mixing_time_s"]) * 1e-8), "analytic_mixing_time_s", maxima)
    return {"status": "passed", "sample_count": len(pinit), "frame_count": len(py["frames"]),
            "max_abs_errors": maxima,
            "coverage_time_s": ps["coverage_time_s"], "mixing_time_s": ps["mixing_time_s"],
            "central95_wrap_time_s": ps["central95_wrap_time_s"],
            "python_simulation_seconds": ps["elapsed_seconds"],
            "cpp_simulation_seconds": cs["elapsed_seconds"]}


def run(command: list[str], log_path: Path, timeout: float) -> float:
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + json.dumps(command) + "\n")
        log.flush()
        child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if child.returncode:
        raise ComparisonError(f"Child returned {child.returncode}; inspect {log_path}")
    return time.perf_counter() - start


def config_text(force: str, initial: str, output: str, count: int, days: float, wrap: bool = False) -> str:
    values = {
        "force_model": force, "initial_type": initial, "output_type": output,
        "orbit_keplerian_deg": "26560000,0.02,55,20,30,10",
        "uncertainty_coordinates": "rtn", "sigma": "100000,100000,100000,1,1,1",
        "samples": count, "seed": 20260919, "threads": 1, "visual_samples": min(count, 48),
        "duration_days": days, "output_step_days": 0.5 if wrap else days / 4,
        "phase_bins": 24, "persistence": 2, "coverage_max_gap_deg": 10,
        "coverage_occupied_fraction": 1, "mixing_max_resultant": 0.15, "mixing_max_tv": 0.2,
        "minimum_altitude_m": 1000000, "relative_tolerance": "1e-11",
        "absolute_tolerance_m": 0.001, "absolute_tolerance_elements": "1e-12",
        "min_step_s": 0.001, "max_step_s": 86400, "accuracy_check": 0,
    }
    return "\n".join(f"{key} = {value}" for key, value in values.items()) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpp", required=True, type=Path, help="Existing C++ simulator; used only as an oracle")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--entrypoint", type=Path, default=ROOT / "run.py")
    parser.add_argument("--output", required=True, type=Path, help="New or empty comparison directory")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--duration-days", type=float, default=30)
    parser.add_argument("--long", action="store_true", help="Use one-year perturbed cases")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args(argv)
    for key in ("cpp", "python", "entrypoint", "output"):
        setattr(args, key, getattr(args, key).resolve())
    for path in (args.cpp, args.python, args.entrypoint):
        if not path.is_file():
            parser.error(f"Required file does not exist: {path}")
    if args.samples < 2 or not math.isfinite(args.duration_days) or args.duration_days <= 0:
        parser.error("Samples must be >= 2 and duration must be positive and finite")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("Timeout must be positive and finite")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    days = 365.25 if args.long else args.duration_days
    cases = [(f"{force}_{initial}_{output}", force, initial, output, args.samples, days, False)
             for force in ("kepler", "j2", "j2_j2sq")
             for initial, output in (("mean", "mean"), ("mean", "osculating"), ("osculating", "osculating"))]
    cases.append(("kepler_full_wrap", "kepler", "mean", "mean", 512, 80, True))
    report = {"status": "running", "cpp_executable": str(args.cpp), "cpp_sha256": sha256(args.cpp),
              "python_executable": str(args.python), "entrypoint": str(args.entrypoint),
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))},
              "tolerances": TOLERANCES, "cases": [],
              "interpretation": "Cross-language implementation agreement, not independent validation of omitted physical effects."}
    report_path = args.output / "comparison.json"
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save()
    try:
        for name, force, initial, output, count, horizon, wrap in cases:
            directory = args.output / name
            directory.mkdir()
            config = directory / "input.cfg"
            config.write_text(config_text(force, initial, output, count, horizon, wrap), encoding="utf-8")
            print(f"Running {name}: {count} particles, {horizon:g} days", flush=True)
            walls = {}
            for language, prefix in (("python", [str(args.python), str(args.entrypoint)]),
                                     ("cpp", [str(args.cpp)])):
                command = prefix + ["--config", str(config), "--output", str(directory / language),
                                    "--no-html", "--export-states"]
                walls[language + "_wall_seconds"] = run(command, directory / f"{language}.log", args.timeout_seconds)
            measured = compare_case(directory / "python", directory / "cpp")
            if wrap and (measured["coverage_time_s"] is None or measured["mixing_time_s"] is None):
                raise ComparisonError("Full-wrap case failed to exercise coverage and mixing events")
            report["cases"].append({"name": name, **measured, **walls})
            save()
            print(f"  PASS; max position difference {measured['max_abs_errors']['state_position_m']:.6g} m", flush=True)
        report["status"] = "passed"
    except (ComparisonError, OSError, KeyError, ValueError, subprocess.TimeoutExpired) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save()
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 1
    save()
    print(f"Passed {len(cases)} independent paired cases. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

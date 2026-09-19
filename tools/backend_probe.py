"""Write 32 native Python DSST cases and compare a Java or C++ reference CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distribution_propagator.backend import Backend, BackendConfig
from distribution_propagator.orbit import to_cartesian


FIELDS = ("model", "initial_type", "days", "output_type", "a", "ex", "ey",
          "hx", "hy", "lm", "x", "y", "z", "vx", "vy", "vz")


def generate(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(FIELDS)
        for model in ("j2", "j2_j2sq"):
            for initial_type in ("mean", "osculating"):
                config = BackendConfig(force_model=model, initial_type=initial_type)
                backend = Backend(config, (26560000.0, 0.01, -0.004, 0.4, 0.3, 0.8))
                for days in (0.0, 1.0, 30.0, 365.0):
                    result = backend.advance(days * 86400.0)
                    for output_type in ("mean", "osculating"):
                        values = getattr(result, output_type)
                        writer.writerow((model, initial_type, days, output_type,
                                         *values, *to_cartesian(values, config.mu)))


def compare(actual: Path, reference: Path) -> tuple[float, float]:
    with actual.open(newline="", encoding="utf-8") as source:
        actual_rows = list(csv.DictReader(source))
    with reference.open(newline="", encoding="utf-8") as source:
        reference_rows = list(csv.DictReader(source))
    if len(actual_rows) != 32 or len(reference_rows) != 32:
        raise ValueError("parity fixture must contain 32 cases")
    worst_position = worst_velocity = 0.0
    for a, r in zip(actual_rows, reference_rows):
        if any(a[name] != r[name] for name in ("model", "initial_type", "output_type")):
            raise ValueError("mismatched parity row metadata")
        if float(a["days"]) != float(r["days"]):
            raise ValueError("mismatched parity row time")
        av, rv = ([float(row[name]) for name in FIELDS[4:]] for row in (a, r))
        if not all(math.isfinite(value) for value in (*av, *rv)):
            raise ValueError("nonfinite parity result")
        for index in range(6):
            limit = 1e-4 if index == 0 else 8e-10
            if abs(av[index] - rv[index]) > limit:
                raise ValueError(f"element parity limit exceeded: {r}")
        worst_position = max(worst_position, math.dist(av[6:9], rv[6:9]))
        worst_velocity = max(worst_velocity, math.dist(av[9:12], rv[9:12]))
    if worst_position > 0.02 or worst_velocity > 3e-6:
        raise ValueError(f"Cartesian parity limit exceeded: {worst_position} m, {worst_velocity} m/s")
    return worst_position, worst_velocity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("reference", type=Path, nargs="?")
    args = parser.parse_args()
    try:
        generate(args.output)
        if args.reference:
            position, velocity = compare(args.output, args.reference)
            print(f"DSST parity: 32 cases, max position {position:.10g} m, "
                  f"max velocity {velocity:.10g} m/s")
    except (OSError, ValueError, RuntimeError, ImportError) as exception:
        print(str(exception), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

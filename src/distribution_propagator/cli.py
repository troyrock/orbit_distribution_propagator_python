"""Command-line entry point; multiprocessing-safe on Windows and POSIX."""

import argparse
from pathlib import Path
import sys

from .config import Config, config_template, read_config, validate_config
from .output import ensure_empty_output, write_results
from .simulation import run_simulation


def main(argv=None):
    parser = argparse.ArgumentParser(description="Propagate orbit uncertainty with native Python Orekit DSST.")
    parser.add_argument("--config", type=Path, help="C++-compatible key=value configuration")
    parser.add_argument("--output", type=Path, default=Path("outputs/run"), help="Unused output directory")
    parser.add_argument("--write-example", type=Path, help="Write example configuration and exit")
    for flag in ("samples", "seed", "visual-samples", "accuracy-check"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--threads", "--workers", dest="threads", type=int,
                        help="Worker processes (0=automatic, up to 8; 1=serial)")
    parser.add_argument("--duration-days", type=float)
    parser.add_argument("--output-step-days", type=float)
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--export-states", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.write_example:
            with args.write_example.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(config_template())
            print(f"Wrote {args.write_example.resolve()}")
            return 0
        config = read_config(args.config) if args.config else Config()
        for key in ("samples", "seed", "threads", "visual_samples", "accuracy_check", "duration_days", "output_step_days"):
            if getattr(args, key) is not None:
                setattr(config, key, getattr(args, key))
        config.write_html = not args.no_html
        config.export_states = args.export_states
        validate_config(config)
        ensure_empty_output(args.output)
        simulation = run_simulation(config, progress=lambda text: print(text, file=sys.stderr, flush=True))
        write_results(simulation, args.output)
        print(f"Completed {len(simulation.initial)} particles, {len(simulation.frames)} snapshots "
              f"in {simulation.summary['elapsed_seconds']:.3f} s using {simulation.summary['workers_used']} worker(s).")
        for event in ("coverage", "mixing"):
            value = simulation.summary[event + "_time_s"]
            print(f"{event.capitalize()}: " + (f"{value / 86400:.6g} days" if value is not None else "not observed"))
        print(f"Results: {args.output.resolve()}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; no completed simulation was published.", file=sys.stderr)
        return 130

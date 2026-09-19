#!/usr/bin/env python3
"""Source-checkout launcher; installation is optional."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from distribution_propagator.cli import main

    raise SystemExit(main())

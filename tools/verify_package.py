"""Build and smoke-test an installed wheel offline without changing site-packages.

Run with a Python environment that already provides NumPy, pip and setuptools.
The native DSST port must be installed or discoverable through DSST_PYTHON_SOURCE.
All generated artifacts remain in a unique directory below build/ for review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]

_INSPECT = r'''
from importlib import resources
import json
from pathlib import Path
import sys
import distribution_propagator
from distribution_propagator.backend import backend_source

target, output = map(Path, sys.argv[1:])
module = Path(distribution_propagator.__file__).resolve()
if not module.is_relative_to(target.resolve()):
    raise RuntimeError(f"Smoke test imported source checkout instead of installation: {module}")
viewer = resources.files("distribution_propagator").joinpath("resources/viewer.html")
template = viewer.read_text(encoding="utf-8")
html = (output / "visualization.html").read_text(encoding="utf-8")
summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
if template.count("__DISTRIBUTION_DATA__") != 1 or "__DISTRIBUTION_DATA__" in html:
    raise RuntimeError("Packaged viewer substitution failed")
if (summary["sample_count"], summary["frame_count"], summary["workers_used"]) != (8, 3, 2):
    raise RuntimeError("Installed multiprocessing smoke test has unexpected dimensions")
print(json.dumps({"installed_module": str(module), "backend_source": backend_source(),
    "viewer_resource_bytes": len(viewer.read_bytes()),
    "generated_html_bytes": (output / "visualization.html").stat().st_size,
    "sample_count": summary["sample_count"], "frame_count": summary["frame_count"],
    "workers_used": summary["workers_used"], "elapsed_seconds": summary["elapsed_seconds"]}))
'''


def verify(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="installed-wheel-", dir=directory.resolve()))
    wheels, target, output = work / "wheels", work / "installed", work / "output"
    python = [sys.executable, "-B"]
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(python + ["-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                            "--no-index", "--wheel-dir", str(wheels), str(ROOT)],
                   cwd=work, env=env, check=True)
    wheel_files = list(wheels.glob("*.whl"))
    if len(wheel_files) != 1:
        raise RuntimeError("Expected exactly one distribution wheel")
    wheel = wheel_files[0]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if "distribution_propagator/resources/viewer.html" not in names:
            raise RuntimeError("Wheel omitted the HTML viewer resource")
        if any(name.lower().endswith((".so", ".dll", ".pyd", ".exe")) for name in names):
            raise RuntimeError("Native binary unexpectedly included in pure Python wheel")
    subprocess.run(python + ["-m", "pip", "install", "--no-deps", "--no-index",
                            "--target", str(target), str(wheel)], cwd=work, env=env, check=True)
    # Replace inherited source paths: the application must come from the wheel.
    # The interpreter's existing third-party libraries (NumPy) stay available.
    env["PYTHONPATH"] = str(target)
    subprocess.run(python + ["-m", "distribution_propagator", "--config",
                            str(ROOT / "examples" / "meo_ball.cfg"), "--output", str(output),
                            "--samples", "8", "--workers", "2", "--duration-days", "0.2",
                            "--output-step-days", "0.1", "--visual-samples", "8",
                            "--accuracy-check", "1", "--export-states"],
                   cwd=work, env=env, check=True)
    inspect = subprocess.run(python + ["-c", _INSPECT, str(target), str(output)],
                             cwd=work, env=env, check=True, text=True, capture_output=True)
    result = json.loads(inspect.stdout)
    result.update({"wheel": str(wheel), "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                   "python": sys.executable, "network_used": False,
                   "installation_scope": "isolated --target directory; no global changes"})
    report = work / "verification.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Packaging verification report: {report}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "build" / "package-verification")
    args = parser.parse_args()
    try:
        verify(args.directory)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Packaging verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

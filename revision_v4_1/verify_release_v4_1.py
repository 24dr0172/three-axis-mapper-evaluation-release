#!/usr/bin/env python3
"""v4.1 reproducibility-release verifier.

The publication manuscript is distributed separately. This entrypoint verifies
the release manifest, protocol crosswalk, retained figures, and scientific
evidence without requiring the manuscript source or bibliography in this tree.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/three_axis_v41_mplconfig")

ROOT = Path(__file__).resolve().parents[1]
REV = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"PASS {msg}")


def load_legacy():
    path = ROOT / "code/verification/verify_release.py"
    spec = importlib.util.spec_from_file_location("verify_release_legacy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    # importlib writes a .pyc for the legacy module in-process; the env var
    # cannot stop that after interpreter start, and the fresh pyc would fail
    # the manifest path-set check below.
    sys.dont_write_bytecode = True
    for cache in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    mpl = REV / ".mplconfig"
    if mpl.exists():
        shutil.rmtree(mpl, ignore_errors=True)

    legacy = load_legacy()
    for cache in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    legacy.verify_manifest()
    ok("v4.1 SHA256SUMS path-set equals the release tree")
    legacy.verify_figures()
    legacy.verify_exact_mapper_distance()
    legacy.verify_quality_survival()
    legacy.verify_track_c()
    legacy.verify_c7()
    legacy.verify_c1_c3()
    legacy.verify_axis1_repairs()
    legacy.verify_c5_c6()
    legacy.verify_c8()
    legacy.verify_c9()
    legacy.verify_axis3_conventional()
    legacy.verify_n1_r10_ball()
    legacy.verify_e2m()
    legacy.verify_synthesis()
    legacy.verify_legacy_baseline()
    ok("legacy scientific claim checks passed")

    figure_readme = (ROOT / "manuscript/figures/README.md").read_text().lower()
    if "dbscan noise/retention" not in figure_readme or "finite filtration" not in figure_readme:
        fail("v4.1 figure interpretation language is missing")
    ok("v4.1 figure interpretation language is present")

    print("V4_1_RELEASE_CHECKS_PASSED")


if __name__ == "__main__":
    main()

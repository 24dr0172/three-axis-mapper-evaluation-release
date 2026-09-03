#!/usr/bin/env python3
"""Regenerate manuscript figures and all release verification outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(command: list[str], log: Path | None = None) -> None:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log:
        log.write_text(result.stdout)
    if result.returncode:
        raise SystemExit(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT.parent / "three_axis_mapper_final_release_reproduced")
    parser.add_argument("--skip-construction-replay", action="store_true")
    parser.add_argument("--skip-axis1-recomputation", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "-B", "code/verification/verify_release.py"],
        out / "CLAIM_VERIFICATION.txt")
    run([sys.executable, "-B", "code/figures/generate_manuscript_figures.py",
         "--output-dir", str(out / "figures")])
    
    can_recompute_axis1 = not args.skip_axis1_recomputation
    try:
        import sklearn  # noqa: F401
    except ImportError:
        can_recompute_axis1 = False
        (out / "AXIS1_CORRECTION_REPLAY.log").write_text(
            "SKIPPED: scikit-learn not installed in current environment. Recomputation of raw Axis-I constructions skipped.\n"
        )

    if can_recompute_axis1:
        run([sys.executable, "-B", "code/reproduction/recompute_axis1_corrected.py",
             "--phase", "all", "--output-root", str(out / "axis1_corrected")],
            out / "AXIS1_CORRECTION_REPLAY.log")

    if not args.skip_construction_replay:
        try:
            import sklearn  # noqa: F401
            run([sys.executable, "-B", "code/reproduction/replay_core_constructions.py",
                 "--output", str(out / "CORE_CONSTRUCTION_REPLAY.json")],
                out / "CORE_CONSTRUCTION_REPLAY.log")
        except ImportError:
            (out / "CORE_CONSTRUCTION_REPLAY.log").write_text(
                "SKIPPED: scikit-learn not installed in current environment. Core construction replay skipped.\n"
            )

    files = sorted(path for path in out.rglob("*") if path.is_file())
    report = {
        "scope": "all manuscript-derived figures and claim checks, corrected Axis-I scientific replay, plus representative core construction replay",
        "historical_ledgers_rewritten": False,
        "files": {path.relative_to(out).as_posix(): sha256(path) for path in files},
    }
    (out / "REGENERATION_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"regenerated {len(files)} outputs in {out}")


if __name__ == "__main__":
    main()

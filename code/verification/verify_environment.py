#!/usr/bin/env python3
"""Fail-closed verification of the portable release environment."""

from __future__ import annotations

import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "environment/requirements.txt"


def fail(message: str) -> None:
    print(f"FAIL environment: {message}", file=sys.stderr)
    raise SystemExit(2)


def required_versions() -> dict[str, str]:
    required: dict[str, str] = {}
    for raw in REQUIREMENTS.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line or line.count("==") != 1:
            fail(f"portable requirement is not an exact pin: {line}")
        name, version = (part.strip() for part in line.split("==", 1))
        if not name or not version:
            fail(f"malformed portable requirement: {line}")
        required[name] = version
    if not required:
        fail("no portable package pins found")
    return required


def main() -> None:
    if platform.python_implementation() != "CPython":
        fail(f"CPython required; found {platform.python_implementation()}")
    if sys.version_info[:2] != (3, 12):
        fail(f"CPython 3.12.x required; found {platform.python_version()}")

    required = required_versions()
    mismatches = []
    for dist, expected in required.items():
        try:
            found = metadata.version(dist)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{dist}: missing (expected {expected})")
            continue
        if found != expected:
            mismatches.append(f"{dist}: {found} (expected {expected})")

    if mismatches:
        fail("; ".join(mismatches))

    print(f"PASS environment CPython {platform.python_version()}")
    for dist, expected in required.items():
        print(f"PASS environment {dist}=={expected}")
    print("ENVIRONMENT_CHECK_PASSED")


if __name__ == "__main__":
    main()

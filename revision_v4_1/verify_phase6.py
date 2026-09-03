#!/usr/bin/env python3
"""Phase 6 reproducibility and release-engineering checks."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = Path(__file__).resolve().parent


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"PASS {msg}")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    proc = subprocess.run(
        [sys.executable, str(REV / "verify_phase5.py")],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or "PHASE5_CHECKS_PASSED" not in proc.stdout:
        fail(f"Phase 5 regression\n{proc.stdout}\n{proc.stderr}")
    ok("Phase 5 still passes")

    proc = subprocess.run(
        [sys.executable, str(REV / "verify_manuscript_parameters.py")],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or "MANUSCRIPT_PARAMETER_CHECKS_PASSED" not in proc.stdout:
        fail(f"parameter verifier\n{proc.stdout}\n{proc.stderr}")
    ok("manuscript parameter verifier passed")

    pre = REV / "PRE_V4_1_SHA256SUMS.txt"
    if not pre.is_file():
        fail("missing pre-v4.1 SHA256SUMS snapshot")
    allowed = ("manuscript/", "revision_v4_1/", "SHA256SUMS.txt",
               "CITATION.cff", "README.md", "pyproject.toml")
    drifted = []
    for line in pre.read_text().splitlines():
        want, rel = line.split(None, 1)
        rel = rel.strip()
        if any(rel == a or rel.startswith(a) for a in allowed):
            continue
        path = ROOT / rel
        if not path.is_file() or digest(path) != want:
            drifted.append(rel)
    if drifted:
        fail(f"frozen pre-v4.1 hashes drifted: {drifted[:8]}")
    ok("evidence/contracts/src/code match the pre-v4.1 SHA256SUMS subset")

    fig = json.loads((ROOT / "manuscript/figures/FIGURE_DATA_V4_1.json").read_text())
    if fig["figure_07"]["doses"]["0.05"]["n_eligible"] != 30:
        fail("fig07 figure-data eligibility")
    c1 = REV / "derived/C1_FULL_DOSE_RESPONSE_V4_1.json"
    if fig["figure_07"]["source_sha256"][str(c1.relative_to(ROOT))] != digest(c1):
        fail("fig07 figure-data not bound to C1 derived JSON")
    ok("v4.1 figure-data object is bound to derived C1 evidence")

    nmap = list(csv.DictReader((ROOT / "manuscript/V4_1_NUMBER_EVIDENCE_MAP.csv").open()))
    if len(nmap) < 12:
        fail(f"number map too short: {len(nmap)}")
    for row in nmap:
        ev = ROOT / row["controlling_evidence"]
        if not ev.exists():
            fail(f"number-map evidence missing: {row['controlling_evidence']}")
    ok(f"number-evidence map has {len(nmap)} rows with existing evidence paths")

    if not (REV / "CHANGELOG_V4_1.txt").is_file():
        fail("missing changelog")
    log = (REV / "CHANGELOG_V4_1.txt").read_text()
    for token in ("Disclosure changes", "New derived analyses", "New robustness runs"):
        if token not in log:
            fail(f"changelog missing {token}")
    ok("v4.1 changelog classifies disclosure / derived / robustness work")

    env = json.loads((REV / "derived/ENVIRONMENT_PREFLIGHT_V4_1.json").read_text())
    if env["python_required"] != "CPython 3.12.x":
        fail("env preflight lost 3.12 requirement")
    if "verify_environment.py" not in env["fail_closed_entrypoint"]:
        fail("env preflight lost fail-closed entrypoint")
    ok("environment preflight distinguishes 3.12 portable vs 3.13.5 legacy")

    prov = list(csv.DictReader((REV / "derived/CAMPAIGN_REPLAY_PROVENANCE_V4_1.csv").open()))
    classes = {r["replay_class"] for r in prov}
    if "deterministic reconstruction" not in classes:
        fail("provenance missing reconstruction class")
    if "evidence-ledger recomputation plus metadata re-audit" not in classes:
        fail("provenance missing III-2 metadata re-audit class")
    ok("campaign replay-provenance table is present")

    cff = (ROOT / "CITATION.cff").read_text()
    if 'version: "4.1.0"' not in cff:
        fail("CITATION.cff not bumped to 4.1.0")
    if "10.5281" in cff or "zenodo" in cff.lower():
        fail("invented archive DOI in CITATION.cff")
    ok("release version is 4.1.0 with no invented archive DOI")
    print("PHASE6_CHECKS_PASSED")


if __name__ == "__main__":
    main()

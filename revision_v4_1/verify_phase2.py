#!/usr/bin/env python3
"""Phase 2 acceptance checks against frozen ledgers and the revised manuscript."""

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


def f6(x) -> str:
    return f"{float(x):.6f}"


def check_frozen() -> None:
    allowed = ("manuscript/",)
    manifest = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines():
        expected, rel = line.split(None, 1)
        manifest[rel.strip()] = expected
    bad = [
        rel
        for rel, want in manifest.items()
        if not rel.startswith(allowed) and digest(ROOT / rel) != want
    ]
    if bad:
        fail(f"frozen hashes changed: {bad[:8]}")
    ok("frozen evidence/contracts/code unchanged")


def check_phase1_still_passes() -> None:
    proc = subprocess.run(
        [sys.executable, str(REV / "verify_phase1.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        fail(f"Phase 1 verifier regressed:\n{proc.stdout}\n{proc.stderr}")
    if "PHASE1_CHECKS_PASSED" not in proc.stdout:
        fail("Phase 1 verifier did not print PHASE1_CHECKS_PASSED")
    ok("Phase 1 checks still pass")


def check_generated_tables() -> None:
    proc = subprocess.run(
        [sys.executable, str(REV / "generate_phase2_tables.py")],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        fail(f"phase2 table generator failed: {proc.stderr}")
    tex = (ROOT / "manuscript/tables/m6a2_stability.tex").read_text()
    cells = list(csv.DictReader((ROOT / "evidence/baseline/m6a2/cell_summaries.csv").open()))
    paired_obj = json.loads((ROOT / "revision_v4_1/derived/M6A2_WASSERSTEIN_COMPANION_V4_1.json").read_text())
    paired = [r for r in paired_obj["paired_method_differences"] if r["metric_id"] == "bottleneck_distance"]
    for row in paired:
        for token in (
            f6(row["conventional_mean_on_pairs"]), f6(row["fmapper_mean_on_pairs"]),
            f6(row["mean_conventional_minus_fmapper"]), f6(row["ci95_low"]), f6(row["ci95_high"]),
        ):
            if token not in tex:
                fail(f"M6-A2 TeX missing paired token {token} at sigma={row['noise_sigma']}")
    # ordering
    for sigma in ("0.01", "0.03", "0.05", "0.08", "0.1"):
        c = next(r for r in cells if r["method_id"] == "conventional" and r["noise_sigma"] == sigma)
        f = next(r for r in cells if r["method_id"] == "f_mapper" and r["noise_sigma"] == sigma)
        if not (float(c["bottleneck_mean"]) < float(f["bottleneck_mean"])):
            fail(f"bottleneck ordering failed at sigma={sigma}")
    if "matched replication IDs and perturbation seeds" not in tex:
        fail("M6-A2 pairing statement missing")
    r10 = (ROOT / "manuscript/tables/r10_accounting.tex").read_text()
    for token in ("282", "205", "30", "47"):
        if token not in r10:
            fail(f"R10 table missing {token}")
    ok("M6-A2 and R10 tables match frozen ledgers")


def check_manuscript() -> None:
    tex = (ROOT / "manuscript/mapper_manuscript.tex").read_text()
    required = {
        "M6-A2 subsection": r"\subsection{Filtered structural stability on Swiss}",
        "M6-A2 table input": r"\input{tables/m6a2_stability.tex}",
        "R10 appendix": r"\section{Structural-robustness accounting}",
        "R10 table input": r"\input{tables/r10_accounting.tex}",
        "N1 evidence map": "N1 is a descriptive matched-status compilation",
        "N1 not confirmatory": "is not a confirmatory claim",
        "IB1 named": "IB1/III-4",
        "E2M III-3 named": "E2M III-3 is a three-case Circle finite diagnostic, not a Reeb theorem",
        "QS scoped": "non-adjudicating provenance",
        "Ball ineligible id": r"BALL\_IB4\_III5\_CONS\_CIRCLE\_M10\_K04",
        "Ball reason": r"selected\_center\_count<5",
        "Ensemble qualified": "project-specified metric-medoid assignment for Silhouette ranking",
        "pi tuple": r"\pi=(\pi_1,\dots,\pi_m)",
        "componentwise": "componentwise",
        "DNA": r"$D_M^{\mathrm{NA}}$",
        "not contradictory": "not contradictory",
        "no new threshold": "We do not convert the bottleneck diagnostic into a new binary threshold",
        "forensic id": r"SAME\_FILTER\_FORENSIC\_PROOF\_III2\_V5",
        "positional swap": "positional argument swap",
        "env table": r"\input{tables/environment_provenance.tex}",
        "metadata repair": "metadata repair, not generation of new scientific numbers",
        "Circle/Tripod scope": "not empirical coverage of arbitrary Reeb-graph combinatorics",
    }
    missing = [name for name, token in required.items() if token not in tex]
    if missing:
        fail(f"manuscript missing: {missing}")
    if "unmodified source" not in tex:
        fail("Ensemble results not distinguished from unmodified source algorithm")
    if "bookkeeping defect" in tex:
        fail("vague bookkeeping-defect wording remains")
    ok("Phase 2 manuscript statements present")


def check_inclusion_closed() -> None:
    tex = (ROOT / "manuscript/mapper_manuscript.tex").read_text()
    for token in ("M6-A2", "R10", "N1 is a descriptive", "IB1/III-4", r"selected\_center\_count<5"):
        if token not in tex:
            fail(f"inclusion gap still open: {token}")
    ok("Phase 0 inclusion gaps for M6-A2, R10, N1, IB1/III-4, Ball row are closed")


def main() -> None:
    check_frozen()
    check_phase1_still_passes()
    check_generated_tables()
    check_manuscript()
    check_inclusion_closed()
    print("PHASE2_CHECKS_PASSED")


if __name__ == "__main__":
    main()

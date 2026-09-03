#!/usr/bin/env python3
"""Phase 3 derived-analysis checks. No new Mapper constructions."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = Path(__file__).resolve().parent
DERIVED = REV / "derived"


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


def f6(x: float) -> str:
    return f"{float(x):.6f}"


def main() -> None:
    allowed = ("manuscript/",)
    manifest = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines():
        expected, rel = line.split(None, 1)
        manifest[rel.strip()] = expected
    bad = [rel for rel, want in manifest.items()
           if not rel.startswith(allowed) and digest(ROOT / rel) != want]
    if bad:
        fail(f"frozen hashes changed: {bad[:8]}")
    ok("frozen evidence/contracts/code unchanged")

    proc = subprocess.run([sys.executable, str(REV / "verify_phase2.py")],
                          capture_output=True, text=True)
    if proc.returncode != 0 or "PHASE2_CHECKS_PASSED" not in proc.stdout:
        fail(f"Phase 2 regression\n{proc.stdout}\n{proc.stderr}")
    ok("Phase 2 still passes")

    env = dict(**os.environ, PHASE3_WRITE_FIGURES="never")
    proc = subprocess.run([sys.executable, str(REV / "generate_phase3_derived.py")],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        fail(f"phase3 generator failed: {proc.stderr or proc.stdout}")
    ok("phase3 generator reran")

    c1 = json.loads((DERIVED / "C1_FULL_DOSE_RESPONSE_V4_1.json").read_text())
    if c1["mapper_constructions_generated"] != 0:
        fail("C1 derived analysis generated Mapper constructions")
    for dose, rec in c1["doses"].items():
        if rec["n_eligible"] != 30 or rec["eligibility_reason_counts"].get("success") != 30:
            fail(f"C1 {dose} eligibility not 30/30 success")
    ok("C1 four-dose 30/30 success on all pairs")

    w = json.loads((DERIVED / "M6A2_WASSERSTEIN_COMPANION_V4_1.json").read_text())
    if w["uniform_ordering"]:
        fail("Wasserstein companion must not claim uniform Conventional<F-Mapper ordering")
    if w["primary_ordering_metric"] != "bottleneck_extended_pd":
        fail("Wasserstein file lost primary-metric label")
    paired_b = [r for r in w["paired_method_differences"] if r["metric_id"] == "bottleneck_distance"]
    if [r["n_pairs"] for r in paired_b] != [30, 30, 30, 30, 29]:
        fail("M6-A2 paired bottleneck denominators drifted")
    excludes_zero = [r["ci95_high"] < 0 or r["ci95_low"] > 0 for r in paired_b]
    if excludes_zero != [True, True, False, False, False]:
        fail("M6-A2 paired bottleneck interval interpretation drifted")
    ok("M6-A2 Wasserstein companion does not steal the bottleneck ordering")

    c8 = json.loads((DERIVED / "C8_QUALITY_FIDELITY_DISCORDANCE_V4_1.json").read_text())
    rules = c8["predeclared_rules"]
    if "75th percentile" not in rules["high_quality"]:
        fail("C8 high-quality rule is not within-grid quartile")
    if "not a fixed Silhouette cutoff" not in rules["not_a_universal_threshold"]:
        fail("C8 missing non-universal-threshold rule")
    if (c8["n_eligible"], c8["n_exact"], c8["n_mismatch"]) != (387, 315, 72):
        fail("C8 387/315/72 split drifted")
    sil = c8["bootstrap_mean_difference_mismatch_minus_match"]["silhouette_macro"]
    db = c8["bootstrap_mean_difference_mismatch_minus_match"]["davies_bouldin_macro"]
    if not (sil["ci95_low"] > 0 and db["ci95_high"] < 0):
        fail("C8 bootstrap intervals do not separate in the expected direction")
    ok("C8 discordance and 72-vs-315 bootstrap match frozen eligible grid")

    tax = json.loads((DERIVED / "FAILURE_TAXONOMY_V4_1.json").read_text())
    if "not reconstructed" not in tax["scope"]["iii2b"]:
        fail("III-2B reconstruction prohibition missing")
    if tax["iii2b_cell_level"]["n_cells"] != 72:
        fail("III-2B must stay cell-level (72 cells)")
    if tax["iii2b_cell_level"]["totals"]["n_planned"] != 2160:
        fail("III-2B planned total drifted")
    if tax["c8_run_level"]["attempted"] != 530:
        fail("C8 attempted count drifted")
    ok("failure taxonomy is C8 run-level and III-2B cell-level only")

    for suffix in ("pdf", "svg", "png"):
        p = ROOT / "manuscript/figures" / f"fig07_c1_perturbation_response.{suffix}"
        if not p.is_file() or p.stat().st_size < 1000:
            fail(f"missing C1 dose figure {p.name}")
    ok("fig07 C1 perturbation-response figure exists")

    tex = (ROOT / "manuscript/mapper_manuscript.tex").read_text()
    for p in (ROOT / "manuscript/tables").glob("*.tex"):
        tex += "\n" + p.read_text()
    needed = {
        "fig:c1dose": r"\label{fig:c1dose}",
        "tab:c8discord": r"\label{tab:c8discord}",
        "tab:failtax": r"\label{tab:failtax}",
        "tab:m6a2w": r"\label{tab:m6a2w}",
        "discordance rate": f6(c8["discordance_rate"]),
        "sil CI": f6(sil["point_estimate"]),
        "no universal threshold": "do not define a general Silhouette threshold",
    }
    missing = [k for k, v in needed.items() if v not in tex]
    if missing:
        fail(f"manuscript missing {missing}")
    ok("manuscript cites Phase 3 derived objects")

    # JSON/tex derived objects must remain byte-stable; the C1 PDF is
    # matplotlib-metadata unstable, so the verifier never rewrites it.
    registry = list(csv.DictReader((REV / "DERIVED_EVIDENCE_REGISTRY.csv").open()))
    stable = {
        "P3_C1_DOSE", "P3_M6A2_W", "P3_C8_DISCORD", "P3_FAILTAX",
    }
    drifted = []
    for row in registry:
        if row["artifact_id"] not in stable:
            continue
        path = ROOT / row["relative_path"]
        if digest(path) != row["sha256"]:
            drifted.append(row["artifact_id"])
    if drifted:
        fail(f"phase3 derived JSON hash drift: {drifted}")
    ok("phase3 derived JSON hashes match the registry")
    print("PHASE3_CHECKS_PASSED")


if __name__ == "__main__":
    main()

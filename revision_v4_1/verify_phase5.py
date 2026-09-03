#!/usr/bin/env python3
"""Phase 5 statistics and reporting-policy checks."""

from __future__ import annotations

import csv
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


def require_all(text: str, tokens: list[str], label: str) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        fail(f"{label} missing {missing}")


def main() -> None:
    proc = subprocess.run(
        [sys.executable, str(REV / "verify_phase4.py")],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or "PHASE4_CHECKS_PASSED" not in proc.stdout:
        fail(f"Phase 4 regression\n{proc.stdout}\n{proc.stderr}")
    ok("Phase 4 still passes")

    proc = subprocess.run(
        [sys.executable, str(REV / "verify_plan_surfaces.py")],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or "PLAN_SURFACE_CHECKS_PASSED" not in proc.stdout:
        fail(f"plan-surface regression\n{proc.stdout}\n{proc.stderr}")
    ok("plan-surface checks still pass")

    tex = (ROOT / "manuscript/mapper_manuscript.tex").read_text()
    claims = (ROOT / "manuscript/CLAIMS.txt").read_text()
    results = (ROOT / "manuscript/RESULTS.txt").read_text()
    index = list(
        csv.DictReader((ROOT / "manuscript/CLAIM_EVIDENCE_INDEX.csv").open())
    )

    required_fields = {
        "claim_id",
        "tier",
        "estimand",
        "evidence",
        "resolution_rule",
        "uncertainty",
        "manuscript_location",
        "status",
    }
    if len(index) != 35:
        fail(f"claim index row count changed: {len(index)}")
    missing_fields = required_fields - set(index[0])
    if missing_fields:
        fail(f"claim index missing fields {sorted(missing_fields)}")
    confirmatory = [row for row in index if row["tier"] == "confirmatory"]
    if not confirmatory:
        fail("claim index has no confirmatory claims")
    for row in confirmatory:
        blank = [field for field in required_fields if not row.get(field, "").strip()]
        if blank:
            fail(f"confirmatory claim {row['claim_id']} has blank fields {blank}")
    ok(f"{len(confirmatory)} confirmatory claims carry estimand/rule/uncertainty fields")

    policy_tokens = [
        "Confirmatory claims are limited and separately identified",
        "resampling unit",
        "number of independent clusters or streams",
        "interval method",
        "predeclared resolution rules",
        "does not claim familywise statistical significance",
        "A p-value is therefore not required for valid inference here",
        "exploratory IB4F p-values do not adjudicate",
    ]
    require_all(tex, policy_tokens, "manuscript statistics policy")
    ok("multiplicity, tier mapping, and no-blanket-p-value policy are explicit")

    require_all(
        claims + "\n" + results,
        [
            "Statistical reporting policy",
            "familywise-significance claim",
            "exploratory tier",
            "predeclared resolution rule",
        ],
        "release summaries",
    )
    ok("CLAIMS and RESULTS use the Phase 5 reporting policy")

    c7 = next((row for row in index if row["claim_id"] == "C7_TRACK_C_INTERACTION"), None)
    if c7 is None:
        fail("C7 claim-index row missing")
    require_all(
        c7["estimand"] + " " + c7["resolution_rule"] + " " + c7["uncertainty"],
        [
            "beta_12",
            "10 replicate streams",
            "stream-level nonparametric percentile bootstrap",
            "10 clusters",
            "Terminal T2",
            "Small-n limitation",
        ],
        "C7 claim-index disclosure",
    )
    c7_text_start = tex.find(r"\widehat\beta_{12}=-0.023328")
    c7_text_end = tex.find(r"\subsection{When the stability estimand", c7_text_start)
    if c7_text_start < 0 or c7_text_end < 0:
        fail("C7 manuscript block not found")
    c7_text = tex[c7_text_start:c7_text_end]
    require_all(
        c7_text,
        [
            "stream-level nonparametric percentile bootstrap",
            "10 independent stream clusters",
            "CR1 sandwich estimator",
            "terminal state T2",
            "only 10 stream clusters",
            "small cluster count is a limitation of the interval",
        ],
        "C7 manuscript disclosure",
    )
    ok("C7 reports unit, 10 clusters, both intervals, rule, and adjacent small-n limit")

    forbidden = [
        "valid inference requires p-values",
        "all descriptive analyses are statistically significant",
        "familywise significance is established",
    ]
    present = [phrase for phrase in forbidden if phrase in tex]
    if present:
        fail(f"forbidden statistical language remains {present}")
    ok("no blanket significance or p-value requirement remains")

    print("PHASE5_CHECKS_PASSED")


if __name__ == "__main__":
    main()

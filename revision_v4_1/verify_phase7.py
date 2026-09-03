#!/usr/bin/env python3
"""Phase 7 writing and publication-hygiene checks."""

from __future__ import annotations

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


def main() -> None:
    proc = subprocess.run(
        [sys.executable, str(REV / "verify_phase6.py")],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or "PHASE6_CHECKS_PASSED" not in proc.stdout:
        fail(f"Phase 6 regression\n{proc.stdout}\n{proc.stderr}")
    ok("Phase 6 still passes")

    tex = (ROOT / "manuscript/mapper_manuscript.tex").read_text()
    bib = (ROOT / "manuscript/BIBLIOGRAPHY.txt").read_text()
    readme = (ROOT / "README.md").read_text()
    reviewer_surfaces = "\n".join(
        [
            tex,
            (ROOT / "manuscript/CLAIM_EVIDENCE_INDEX.csv").read_text(),
            (ROOT / "manuscript/CLAIMS.txt").read_text(),
            (ROOT / "manuscript/RESULTS.txt").read_text(),
            (REV / "CLAIM_TIER_TABLE.csv").read_text(),
            (REV / "MASTER_EXPERIMENTAL_DESIGN_TABLE.csv").read_text(),
            (REV / "CHANGELOG_V4_1.txt").read_text(),
            *(path.read_text() for path in sorted((ROOT / "manuscript/tables").glob("*.tex"))),
        ]
    )

    abstract_end = tex.find(r"\end{abstract}")
    abstract = tex[:abstract_end]
    for token in (
        "Kang--Lim RCESCC Ensemble Mapper with project-specified metric-medoid Silhouette assignment",
        "not universal defects",
        "design-specific",
    ):
        if token not in abstract:
            fail(f"abstract missing {token}")
    ok("abstract keeps Ensemble qualification and design-specific failure language")

    if "11 spokes" in tex:
        fail("spokes jargon remains")
    if "525 Swiss/Digits construction units" in tex:
        fail("C3 construction-units jargon remains")
    if "sealed FCM implementation" in tex:
        fail("sealed FCM implementation jargon remains")
    if "sealed computational reference" in tex:
        fail("sealed computational reference jargon remains")
    if "were audited against the canonical filter-definition registry and requalified" in tex:
        fail("requalified jargon remains untranslated")
    if r"not\_exposed\_by\_sealed\_run\_fcm\_1d" not in tex:
        fail("frozen ledger token not_exposed_by_sealed_run_fcm_1d was renamed")
    if "11 controlled cells" not in tex:
        fail("controlled-cells replacement missing")
    if "metadata re-audited" not in tex:
        fail("metadata re-audit wording missing")
    stale_jargon = [
        "11 spokes",
        "33 spokes",
        "Swiss (500 units), Digits (25 units)",
        "height y; requalified same-filter identities",
        "2208 run records requalified",
        "Same-filter identity uses the requalified 2208-run table",
        "Requalification repaired stored filter-identity metadata",
        "requalification SAME_FILTER_FORENSIC_PROOF_III2_V5",
        "Axis-III requalification is metadata re-audit",
        "requalification wording",
    ]
    present = [phrase for phrase in stale_jargon if phrase in reviewer_surfaces]
    if present:
        fail(f"reviewer-facing terminology leaks remain: {present}")
    for replacement in (
        "11 controlled cells",
        "33 controlled cells",
        "Swiss (500 records), Digits (25 records)",
        "metadata-re-audited 2208-run table",
        "2208 run records metadata re-audited",
    ):
        if replacement not in reviewer_surfaces:
            fail(f"reviewer-facing replacement missing: {replacement}")
    if "RUN_LEVEL_REQUALIFIED.csv" not in reviewer_surfaces:
        fail("stable RUN_LEVEL_REQUALIFIED.csv filename was renamed")
    ok("terminology replacements are in place; ledger field names preserved")

    if r"\label{eq:fcmobj}" not in tex:
        fail("FCM objective unlabeled")
    if r"\pi=(\pi_1,\dots,\pi_m)" not in tex:
        fail("pi tuple definition missing")
    ok("FCM objective is labelled and pi is defined before use")

    for src, label in ((tex, "manuscript"), (bib, "bibliography")):
        if "doi:10.1186/s12859-025-06085-5" not in src:
            fail(f"{label} missing D-Mapper publisher DOI")
        if "doi:10.1089/cmb.2024.0919" not in src:
            fail(f"{label} missing implicit-interval Mapper publisher DOI")
        if "JACT" in src:
            fail(f"{label} changed Zhou 2022 to JACT")
        if "Research in Computational Topology 2" not in src:
            fail(f"{label} lost Zhou 2022 Springer chapter venue")
    ok("Tao--Ge papers have distinct publisher DOIs; Zhou 2022 venue unchanged")

    if "No public archive DOI is claimed" not in tex:
        fail("manuscript archive-DOI disclaimer missing")
    if "drafting assistance" not in tex or "code review" not in tex:
        fail("manuscript AI disclosure incomplete")
    if "approved every scientific decision" not in tex:
        fail("authors-approved-decisions statement missing")
    if "drafting assistance" not in readme or "principal investigators" not in readme.lower():
        fail("README AI/authorship disclosure incomplete")
    ok("AI disclosure and no-invented-DOI policy are present")
    print("PHASE7_CHECKS_PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 6 release-engineering artifacts. Additive; does not edit sealed science."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = Path(__file__).resolve().parent
DERIVED = REV / "derived"
TEX = ROOT / "manuscript" / "tables"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(name: str, obj) -> Path:
    DERIVED.mkdir(parents=True, exist_ok=True)
    path = DERIVED / name
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return path


def write_number_map() -> Path:
    rows = [
        {
            "manuscript_number": "0.626863",
            "quantity": "C1 alpha=0.05 median D_M^NA",
            "controlling_evidence": "revision_v4_1/derived/C1_FULL_DOSE_RESPONSE_V4_1.json",
            "source_authority": "evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl",
            "claim_id": "C1_SWISS",
            "tier": "exploratory (failed confirmatory pattern; four-level descriptive)",
        },
        {
            "manuscript_number": "0.843945 / 0.919686 / 0.937897",
            "quantity": "C1 alpha=0.10/0.20/0.30 median D_M^NA",
            "controlling_evidence": "revision_v4_1/derived/C1_FULL_DOSE_RESPONSE_V4_1.json",
            "source_authority": "evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl",
            "claim_id": "C1_SWISS",
            "tier": "exploratory",
        },
        {
            "manuscript_number": "30/30 x 4",
            "quantity": "C1 eligibility at each positive perturbation level",
            "controlling_evidence": "revision_v4_1/derived/C1_FULL_DOSE_RESPONSE_V4_1.json",
            "source_authority": "evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl",
            "claim_id": "C1_SWISS",
            "tier": "failure-accounting",
        },
        {
            "manuscript_number": "tab:m6a2 bottleneck means/CIs",
            "quantity": "M6-A2 method x noise bottleneck displacement",
            "controlling_evidence": "manuscript/tables/m6a2_stability.tex",
            "source_authority": "evidence/baseline/m6a2/",
            "claim_id": "M6A2_STABILITY",
            "tier": "confirmatory (bottleneck primary)",
        },
        {
            "manuscript_number": "tab:m6a2w Wasserstein means/CIs",
            "quantity": "M6-A2 Wasserstein companion (not primary ordering)",
            "controlling_evidence": "revision_v4_1/derived/M6A2_WASSERSTEIN_COMPANION_V4_1.json",
            "source_authority": "evidence/baseline/m6a2/",
            "claim_id": "M6A2_STABILITY",
            "tier": "exploratory companion",
        },
        {
            "manuscript_number": "282 / 205 / 30 / 47",
            "quantity": "R10 attempted / eligible / successful-ineligible / typed degenerate",
            "controlling_evidence": "manuscript/tables/r10_accounting.tex",
            "source_authority": "evidence/baseline/r10/R10_V4_RUN_ACCOUNTING.json",
            "claim_id": "R10_STRUCTURAL",
            "tier": "failure-accounting",
        },
        {
            "manuscript_number": "0.371134 (36/97)",
            "quantity": "C8 within-grid high-quality mismatch discordance",
            "controlling_evidence": "revision_v4_1/derived/C8_QUALITY_FIDELITY_DISCORDANCE_V4_1.json",
            "source_authority": "evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl",
            "claim_id": "C8_FMAPPER_AXIS3",
            "tier": "exploratory",
        },
        {
            "manuscript_number": "0.034752 [0.030035,0.039450]",
            "quantity": "C8 Silhouette mean difference mismatch-minus-match",
            "controlling_evidence": "revision_v4_1/derived/C8_QUALITY_FIDELITY_DISCORDANCE_V4_1.json",
            "source_authority": "evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl",
            "claim_id": "C8_FMAPPER_AXIS3",
            "tier": "exploratory",
        },
        {
            "manuscript_number": "-0.052745 [-0.059958,-0.045493]",
            "quantity": "C8 Davies-Bouldin mean difference mismatch-minus-match",
            "controlling_evidence": "revision_v4_1/derived/C8_QUALITY_FIDELITY_DISCORDANCE_V4_1.json",
            "source_authority": "evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl",
            "claim_id": "C8_FMAPPER_AXIS3",
            "tier": "exploratory",
        },
        {
            "manuscript_number": "530 / 408 / 111 / 11",
            "quantity": "C8 run-level attempted / success / FCM non-convergence / coverage-gap",
            "controlling_evidence": "revision_v4_1/derived/FAILURE_TAXONOMY_V4_1.json",
            "source_authority": "evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl",
            "claim_id": "C8_FMAPPER_AXIS3",
            "tier": "failure-accounting",
        },
        {
            "manuscript_number": "2160 / 1201",
            "quantity": "III-2B cell-level planned eligible constructions / joint recoveries",
            "controlling_evidence": "revision_v4_1/derived/FAILURE_TAXONOMY_V4_1.json",
            "source_authority": "evidence/baseline/axis3_iii2/III2B_RECOVERY_SUMMARY.csv",
            "claim_id": "AXIS3_III2",
            "tier": "confirmatory (cell-level; run-level journals not reconstructed)",
        },
        {
            "manuscript_number": "292 / 50 / 118 / 205",
            "quantity": "FCM non-convergence at 1e-7/1e-4/1e-5/1e-6 and max_iter=300",
            "controlling_evidence": "revision_v4_1/derived/FCM_SOLVER_SENSITIVITY_AUDIT_V4_1_SUMMARY.json",
            "source_authority": "revision_v4_1/derived/FCM_SOLVER_SENSITIVITY_AUDIT_V4_1.jsonl",
            "claim_id": "C3_FMAPPER; C8_FMAPPER_AXIS3",
            "tier": "exploratory robustness; not confirmatory",
        },
        {
            "manuscript_number": "266 / 26",
            "quantity": "Frozen FCM failures recovered / remaining at 1e-7 max_iter=1000",
            "controlling_evidence": "revision_v4_1/derived/FCM_SOLVER_SENSITIVITY_AUDIT_V4_1_SUMMARY.json",
            "source_authority": "revision_v4_1/derived/FCM_SOLVER_SENSITIVITY_AUDIT_V4_1.jsonl",
            "claim_id": "C3_FMAPPER; C8_FMAPPER_AXIS3",
            "tier": "exploratory robustness; not confirmatory",
        },
        {
            "manuscript_number": "1055 / 1035",
            "quantity": "FCM audit construction records / unique dataset-seed-c jobs",
            "controlling_evidence": "revision_v4_1/derived/FCM_SOLVER_SENSITIVITY_AUDIT_V4_1_SUMMARY.json",
            "source_authority": "evidence/campaigns/c3/PLAN.jsonl; evidence/campaigns/c8/PLAN.jsonl",
            "claim_id": "C3_FMAPPER; C8_FMAPPER_AXIS3",
            "tier": "exploratory robustness; not confirmatory",
        },
        {
            "manuscript_number": "skipped_interpreter_unavailable",
            "quantity": "ENV_EQUIVALENCE_CHECK_V4_1 status",
            "controlling_evidence": "revision_v4_1/derived/ENV_EQUIVALENCE_CHECK_V4_1_SKIP.json",
            "source_authority": "plan section 6.2",
            "claim_id": "(not a scientific claim)",
            "tier": "provenance",
        },
    ]
    path = ROOT / "manuscript/V4_1_NUMBER_EVIDENCE_MAP.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_figure_data_v41() -> Path:
    c1_path = DERIVED / "C1_FULL_DOSE_RESPONSE_V4_1.json"
    c1 = json.loads(c1_path.read_text())
    fig_pdf = ROOT / "manuscript/figures/fig07_c1_perturbation_response.pdf"
    obj = {
        "analysis_id": "FIGURE_DATA_V4_1",
        "note": (
            "Additive figure-data object for v4.1 derived figures. "
            "Does not replace manuscript/figures/FIGURE_DATA.json, which remains "
            "the authority for fig01--fig06 and the quick-release cmp."
        ),
        "figure_07": {
            "stem": "fig07_c1_perturbation_response",
            "analysis_id": c1["analysis_id"],
            "mapper_constructions_generated": c1["mapper_constructions_generated"],
            "doses": c1["doses"],
            "source_sha256": {
                "revision_v4_1/derived/C1_FULL_DOSE_RESPONSE_V4_1.json": digest(c1_path),
                "evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl": digest(
                    ROOT / "evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl"
                ),
                "evidence/campaigns/c1/C1_PAIRED_QUALITY_LEDGER.jsonl": digest(
                    ROOT / "evidence/campaigns/c1/C1_PAIRED_QUALITY_LEDGER.jsonl"
                ),
                "evidence/campaigns/c1/C1_QUALITY_LEDGER.jsonl": digest(
                    ROOT / "evidence/campaigns/c1/C1_QUALITY_LEDGER.jsonl"
                ),
            },
            "figure_files": {
                suffix: digest(ROOT / "manuscript/figures" / f"fig07_c1_perturbation_response.{suffix}")
                for suffix in ("pdf", "svg", "png")
            },
            "pdf_present": fig_pdf.is_file() and fig_pdf.stat().st_size > 1000,
        },
    }
    path = ROOT / "manuscript/figures/FIGURE_DATA_V4_1.json"
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return path


def write_provenance() -> Path:
    rows = [
        ("C4_FMAPPER", "representative construction replay",
         "Corrected Axis-I F-Mapper fixed-cover campaign; representative replay in run_quick/run_all."),
        ("CIA2_CONVENTIONAL", "evidence-ledger recomputation",
         "Released CIA2 construction ledger; Digits 0/20 is reference-ineligible all_points_unassigned."),
        ("C1_SWISS", "evidence-ledger recomputation",
         "Paired comparison and quality ledgers; v4.1 four-level perturbation series is derived, 0 new constructions."),
        ("C2_SELECTION", "deterministic reconstruction",
         "260 unit / 780 arm rows reconstructed after a logging-stage crash from journaled unit outcomes; used only for the uniform selection-failure conclusion."),
        ("C3_FMAPPER", "evidence-ledger recomputation",
         "C3 construction ledger is the confirmatory status authority (339/181/5)."),
        ("C7_TRACK_C", "evidence-ledger recomputation",
         "Exact-joint endpoint correction on already-generated Track C evidence; 0 new Mapper constructions."),
        ("C6A_CONVENTIONAL", "evidence-ledger recomputation",
         "Frozen-cover Swiss ledger; primary estimand not measurable."),
        ("C6B_BALL", "evidence-ledger recomputation",
         "Frozen-landmark Swiss ledger."),
        ("C8_FMAPPER_AXIS3", "evidence-ledger recomputation",
         "C8 construction ledger; discordance/taxonomy are derived summaries."),
        ("C5_FMAPPER", "evidence-ledger recomputation",
         "Finite-case cells; frequencies taken from C8."),
        ("C9_BALL_CIRCLE", "evidence-ledger recomputation",
         "Saturated k=2.0 arm; vacuous DiD."),
        ("M6A2_STABILITY", "evidence-ledger recomputation",
         "302-run Swiss extended-persistence comparison; Wasserstein companion is derived."),
        ("R10_STRUCTURAL", "evidence-ledger recomputation",
         "282-construction accounting from R10_V4_RUN_ACCOUNTING.json."),
        ("N1_MATCHED", "evidence-ledger recomputation",
         "Descriptive matched-status compilation; not confirmatory."),
        ("AXIS3_III1", "evidence-ledger recomputation",
         "Conventional III-1 Circle/Tripod finite-case baselines; produced under legacy CPython 3.13.5."),
        ("AXIS3_III2", "evidence-ledger recomputation plus metadata re-audit",
         "III-2B cell-level recovery from III2B_RECOVERY_SUMMARY.csv. Same-filter metadata re-audit: SAME_FILTER_FORENSIC_PROOF_III2_V5; scientific numeric fields unchanged. Run-level construction journals are not reconstructed."),
        ("IB1_III4", "evidence-ledger recomputation",
         "34/34 eligible Circle/Tripod perturbation rows."),
        ("FCM_SOLVER_SENSITIVITY_AUDIT_V4_1", "new robustness arm on frozen lens values",
         "FCM-only reclassification; 0 Mapper constructions; not a replacement for C3/C8."),
        ("ENV_EQUIVALENCE_CHECK_V4_1", "skipped_interpreter_unavailable",
         "CPython 3.13.5 not provisioned; provenance table retained."),
    ]
    path = DERIVED / "CAMPAIGN_REPLAY_PROVENANCE_V4_1.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["campaign_id", "replay_class", "note"])
        writer.writerows(rows)
    TEX.mkdir(parents=True, exist_ok=True)
    lines = [
        r"% Auto-generated by generate_phase6_release.py. Do not hand-edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Reproducibility provenance for headline campaigns. Replay class is one of: full executable replay; representative construction replay; deterministic reconstruction; evidence-ledger recomputation. The Axis-III metadata repair is a metadata re-audit, not scientific-number generation.}",
        r"\label{tab:replayprov}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{@{}lX@{}}",
        r"\toprule",
        r"Campaign & Replay class \\",
        r"\midrule",
    ]
    for campaign, cls, _note in rows:
        camp = campaign.replace("_", r"\_")
        lines.append(f"{camp} & {cls} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    (TEX / "replay_provenance.tex").write_text("\n".join(lines))
    return path


def write_changelog() -> Path:
    text = """Three-axis Mapper evaluation — v4.1 changelog
=============================================

Release version
    Reproducibility package version 4.1.0. Prior releases remain immutable
    archives. This changelog classifies v4.1 work; it does not replace
    sealed scientific ledgers.

Disclosure changes (text-only; no new science)
    C1 original confirmatory pattern reported as not observed; four-level perturbation
    series retained as descriptive axis disagreement with 30/30 eligibility.
    C7 factorial, model, both intervals, and no-new-constructions statement.
    CIA2 Digits 0/20 explained as all_points_unassigned reference-ineligible.
    Every FCM non-convergence names tol=1e-7, max_iter=300, linspace init.
    Ensemble identity qualified as Kang-Lim RCESCC with project-specified
    metric-medoid Silhouette assignment.
    Ball ineligible row named with selected_center_count<5.
    D_M versus D_M^NA preserved; pi tuple defined before use.
    Axis-III invariant recovery not conflated with zero bottleneck distance.
    Metadata repair named by forensic proof SAME_FILTER_FORENSIC_PROOF_III2_V5.
    Statistics policy: confirmatory vs exploratory vs failure-accounting vs
    algebraic; no blanket p-value requirement; C7 small-n beside the interval.
    Writing hygiene: abstract qualification, terminology, FCM objective label,
    publisher-verified Tao--Ge DOIs, AI disclosure, no invented archive DOI.

New derived analyses (0 Mapper constructions)
    C1_FULL_DOSE_RESPONSE_V4_1 and fig07.
    M6A2_WASSERSTEIN_COMPANION_V4_1 (not the primary ordering).
    C8_QUALITY_FIDELITY_DISCORDANCE_V4_1 and 72-vs-315 bootstrap.
    FAILURE_TAXONOMY_V4_1 (C8 run-level; III-2B cell-level only).
    Master experimental-design table generated from ledgers.

New robustness runs (authorized campaign IDs only)
    FCM_SOLVER_SENSITIVITY_AUDIT_V4_1: FCM-only reclassification on frozen
    C3/C8 lens values. Labelled robustness/sensitivity, not confirmatory.
    ENV_EQUIVALENCE_CHECK_V4_1: skipped_interpreter_unavailable.

Not in this revision
    Random FCM initialization; Kang-Lim original crisp-assignment rerun;
    C6-a refitted-cover control; arbitrary bottleneck threshold; reconstruction
    of missing III-2B run-level journals; Zhou 2022 venue change.
"""
    path = REV / "CHANGELOG_V4_1.txt"
    path.write_text(text)
    return path


def write_environment_preflight() -> Path:
    required = {}
    for raw in (ROOT / "environment/requirements.txt").read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = (part.strip() for part in line.split("==", 1))
        required[name] = version
    found = {}
    missing = []
    mismatch = []
    for name, want in required.items():
        try:
            got = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        found[name] = got
        if got != want:
            mismatch.append({"name": name, "found": got, "required": want})
    py = platform.python_version()
    obj = {
        "record_id": "ENVIRONMENT_PREFLIGHT_V4_1",
        "python_implementation": platform.python_implementation(),
        "python_version": py,
        "python_required": "CPython 3.12.x",
        "python_ok": platform.python_implementation() == "CPython"
        and sys.version_info[:2] == (3, 12),
        "portable_requirements": required,
        "found_versions": found,
        "missing_packages": missing,
        "version_mismatches": mismatch,
        "legacy_axis3_python": "3.13.5",
        "legacy_axis3_available": False,
        "env_equivalence_check": "skipped_interpreter_unavailable",
        "fail_closed_entrypoint": "code/verification/verify_environment.py",
        "authoritative_replay_environment": (
            "CPython 3.12.x with environment/requirements.txt; "
            "ENVIRONMENT_LOCK_MANIFEST.json is the production lock digest."
        ),
        "legacy_evidence_environment": (
            "Conventional III-1 and III-2: CPython 3.13.5 with "
            "environment/requirements_axis3_legacy.txt."
        ),
        "pdflatex": False,
        "latexmk": False,
        "pdf_compile": "skipped_toolchain_unavailable",
        "note": (
            "This host's default interpreter is CPython 3.12.3. The portable "
            "preflight remains fail-closed: verify_environment.py must see every "
            "pinned distribution. Missing pins here do not rewrite sealed "
            "environment/ files."
        ),
    }
    return dump("ENVIRONMENT_PREFLIGHT_V4_1.json", obj)


def write_sha256sums() -> Path:
    skip_parts = {"__pycache__", ".mplconfig"}
    lines = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == "SHA256SUMS.txt":
            continue
        if any(part in skip_parts for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix == ".pyc" or path.name.endswith("~"):
            continue
        lines.append(f"{digest(path)}  {rel}")
    out = ROOT / "SHA256SUMS.txt"
    out.write_text("\n".join(lines) + "\n")
    return out


def frozen_subset_ok() -> None:
    pre = REV / "PRE_V4_1_SHA256SUMS.txt"
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
        raise SystemExit(f"frozen pre-v4.1 hashes drifted: {drifted[:8]}")


def main() -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    DERIVED.mkdir(parents=True, exist_ok=True)
    write_number_map()
    write_figure_data_v41()
    write_provenance()
    write_changelog()
    write_environment_preflight()
    frozen_subset_ok()
    print("PHASE6_ARTIFACTS_WRITTEN")


if __name__ == "__main__":
    main()

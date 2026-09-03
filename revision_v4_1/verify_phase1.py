#!/usr/bin/env python3
"""Phase 1 acceptance checks: design table vs ledgers, manuscript vs plan 3.1-3.5."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
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


def jl(rel: str) -> list[dict]:
    return [json.loads(x) for x in (ROOT / rel).read_text().splitlines() if x.strip()]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_sealed_untouched() -> None:
    """Manuscript revision is authorized; frozen evidence/code/contracts are not."""
    pre = REV / "PRE_V4_1_SHA256SUMS.txt"
    manifest_path = pre if pre.is_file() else ROOT / "SHA256SUMS.txt"
    allowed_prefixes = ("manuscript/", "revision_v4_1/", "CITATION.cff", "README.md")
    manifest = {}
    for line in manifest_path.read_text().splitlines():
        expected, rel = line.split(None, 1)
        manifest[rel.strip()] = expected
    mismatches = [
        rel for rel, want in manifest.items()
        if not any(rel == a or rel.startswith(a) for a in allowed_prefixes)
        and (not (ROOT / rel).is_file() or digest(ROOT / rel) != want)
    ]
    if mismatches:
        fail(f"frozen evidence/code/contract hashes changed: {mismatches[:8]}")
    if pre.is_file():
        ok("frozen ledgers/contracts/code match the pre-v4.1 SHA256SUMS subset")
        return
    manuscript_changed = [
        rel for rel in manifest
        if rel.startswith("manuscript/") and digest(ROOT / rel) != manifest[rel]
    ]
    if not manuscript_changed:
        fail("expected manuscript files to change in Phase 1")
    ok(f"frozen ledgers/contracts/code unchanged; manuscript paths revised ({len(manuscript_changed)})")


def check_design_table() -> dict:
    proc = subprocess.run(
        [sys.executable, str(REV / "generate_master_design_table.py")],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        fail(f"design table generator failed: {proc.stderr}")
    rows = list(csv.DictReader((REV / "MASTER_EXPERIMENTAL_DESIGN_TABLE.csv").open()))
    by = {r["campaign_id"]: r for r in rows}

    cia = by["CIA2_CONVENTIONAL"]["dbscan"]
    if "eps=0.15" not in cia or "Digits" not in cia:
        fail(f"CIA2 Digits eps=0.15 missing from design table: {cia}")
    if "1.455055840852852" in cia:
        fail("CIA2 row must not carry the C3 Digits scale-rule eps")
    ok("CIA2 Digits DBSCAN eps=0.15 in generated table")

    c3 = by["C3_FMAPPER"]["dbscan"]
    if "1.455055840852852" not in c3:
        fail(f"C3 Digits eps missing: {c3}")
    ok("C3 Digits eps=1.455055840852852 in generated table")

    c1 = by["C1_SWISS"]
    if "30 paired realizations at each of four positive doses" not in c1["replicate_unit"]:
        fail(c1["replicate_unit"])
    if "N=2500" not in c1["n_generator_rule"] or "hole excision" not in c1["n_generator_rule"]:
        fail(c1["n_generator_rule"])
    if c1["method"] != "Conventional Mapper":
        fail(f"C1 method drifted: {c1['method']}")
    ok("C1 30+30x4 accounting, Swiss N=2500 hole excision, Conventional method")

    tc = by["TRACK_C_C7"]
    if "4 x 4 x 3 = 48" not in tc["replicate_unit"]:
        fail(tc["replicate_unit"])
    if "alpha=0 is the clean reference" not in tc["noise_model"]:
        fail(tc["noise_model"])
    if "no new Mapper constructions" not in tc["primary_endpoint"]:
        fail(tc["primary_endpoint"])
    ok("Track C/C7 48 positive-dose cells and no-new-constructions statement")

    tex = (ROOT / "manuscript/tables/master_experimental_design.tex").read_text()
    for token in ("0.15", "1.455055840852852", "2500", "48 perturbed cells", "PRIMARY"):
        if token not in tex:
            fail(f"generated TeX missing {token}")
    ok("generated TeX contains the known-correction tokens")
    return by


def round6(x: float) -> float:
    return float(f"{x:.6f}")


def check_c1_numbers(tex: str) -> None:
    cmp_rows = jl("evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl")
    paired = jl("evidence/campaigns/c1/C1_PAIRED_QUALITY_LEDGER.jsonl")
    want_d = {}
    want_s = {}
    for alpha in (0.05, 0.1, 0.2, 0.3):
        ds = [r["d_common_id"] for r in cmp_rows if r["alpha"] == alpha]
        elig = sum(1 for r in cmp_rows if r["alpha"] == alpha and r["eligible"])
        if elig != 30 or len(ds) != 30:
            fail(f"C1 alpha={alpha} eligibility {elig}/30")
        want_d[alpha] = round6(statistics.median(ds))
        sp = [r["silhouette_perturbed"] for r in paired if r["alpha"] == alpha]
        want_s[alpha] = round6(statistics.median(sp))
    for alpha, val in want_d.items():
        token = f"{val:.6f}"
        if token not in tex:
            fail(f"manuscript missing C1 median D at alpha={alpha}: {token}")
    for alpha, val in want_s.items():
        token = f"{val:.6f}"
        if token not in tex:
            fail(f"manuscript missing C1 median Silhouette at alpha={alpha}: {token}")
    if "30/30" not in tex:
        fail("manuscript missing 30/30 eligibility denominators")
    if "not observed" not in tex.lower():
        fail("manuscript does not state C1 primary pattern not observed")
    if "most direct counterexample" in tex.lower():
        fail("unlabelled 'most direct counterexample' language remains")
    ok("C1 four-dose medians, 30/30 denominators, not-observed language, no cherry-pick slogan")


def check_c7(tex: str) -> None:
    required = [
        r"\alpha\in\{0.05,0.10,0.20,0.30\}",
        r"n\in\{6,10,15,20\}",
        r"\eps\in\{0.08,0.15,0.25\}",
        r"4\times 4\times 3=48",
        "No new Mapper constructions",
        r"\widehat\beta_{12}=-0.023328",
        "[-0.026719,-0.019658]",
        "[-0.027730,-0.018927]",
        "10 independent stream clusters",
        r"\beta_n(n-10)",
        r"\beta_e(\eps-0.15)",
        "categorical $\\alpha$ effects",
    ]
    missing = [item for item in required if item not in tex]
    if missing:
        fail(f"C7 manuscript missing {missing}")
    fit = json.loads((ROOT / "evidence/campaigns/c7/C7_INTERACTION_FIT.json").read_text())
    if fit["mapper_constructions_generated"] != 0:
        fail("C7 ledger says new constructions were generated")
    if abs(fit["beta_12"]["point_estimate"] + 0.023328450168527137) > 1e-15:
        fail("C7 beta12 drifted")
    ok("C7 factorial, model, both intervals, 10 clusters, no new constructions")


def check_cia2_and_fcm(tex: str) -> None:
    if "all points are unassigned" not in tex:
        fail("CIA2 Digits all-points-unassigned explanation missing")
    if "not a universal property of Conventional Mapper on Digits" not in tex:
        fail("CIA2 Digits scope sentence missing")
    if r"\mathrm{tol}=10^{-7}" not in tex or r"\mathrm{max\_iter}=300" not in tex:
        fail("frozen FCM solver criterion missing from manuscript")
    if "linspace" not in tex:
        fail("deterministic linspace initialization missing")
    if "not_exposed_by_sealed_run_fcm_1d" not in tex.replace("\\_", "_"):
        fail("C3/C8 not-exposed iteration traces not reported")
    ok("CIA2 Digits explanation and FCM solver-criterion language")


def check_input_table(tex: str) -> None:
    if r"\input{tables/master_experimental_design.tex}" not in tex:
        fail("manuscript does not input the generated design table")
    ok("manuscript inputs generated master design table")


def main() -> None:
    check_sealed_untouched()
    check_design_table()
    tex = (ROOT / "manuscript/mapper_manuscript.tex").read_text()
    check_c1_numbers(tex)
    check_c7(tex)
    check_cia2_and_fcm(tex)
    check_input_table(tex)
    print("PHASE1_CHECKS_PASSED")


if __name__ == "__main__":
    main()

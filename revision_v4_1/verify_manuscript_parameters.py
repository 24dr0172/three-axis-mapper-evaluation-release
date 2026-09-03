#!/usr/bin/env python3
"""Check key parameters in the standalone manuscript and release crosswalk.

Usage:
    python verify_manuscript_parameters.py [path/to/manuscript.tex]

When no path is supplied, the checker looks for the current manuscript beside
the reviewer-release directory.  The publication source is intentionally not
duplicated inside the release.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"PASS {msg}")


def jl(rel: str) -> list[dict]:
    return [json.loads(x) for x in (ROOT / rel).read_text().splitlines() if x.strip()]


def main() -> None:
    tex_path = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else ROOT.parent / "written_version_with_source_fixes.tex"
    )
    if not tex_path.is_file():
        fail(
            "standalone manuscript not found; pass its path as the first argument"
        )
    tex = tex_path.read_text()
    protocol = (ROOT / "manuscript/README.md").read_text()
    manuscript_surface = tex + "\n" + protocol
    design = list(csv.DictReader((ROOT / "revision_v4_1/MASTER_EXPERIMENTAL_DESIGN_TABLE.csv").open()))
    by = {r["campaign_id"]: r for r in design}

    tokens = {
        "FCM tol": r"\mathrm{tol}=10^{-7}",
        "FCM max_iter": r"\mathrm{max\_iter}=300",
        "CIA2 eps": "eps = 0.15",
        "C3 Digits eps": "1.455055840852852",
        "Swiss sample-then-excise N": "N_base = 2500",
        "C7 48 cells": "48 perturbed cells",
        "overlap 0.30": "0.30",
        "linspace": "linspace",
        "retained-observation fraction": r"r_N=\frac{|I_s|}{|I_0|}",
        "retained-edge fraction": r"r_E=\frac{|E_0\cap E_s|}{|E_0|}",
        "edge-Jaccard dissimilarity": r"d_J(E_0,E_s)=1-\frac{|E_0\cap E_s|}{|E_0\cup E_s|}",
        "Jaccard antecedent": "jaccard1901",
        "FCM simplex constraint": r"\sum_{j=1}^{c}u_{ij}=1",
        "type-restricted Wasserstein": r"\label{eq:typewasserstein}",
        "Axis I/II protocol table": r"\label{tab:protocol-axis12}",
        "Axis III protocol table": r"\label{tab:protocol-axis3}",
        "C6A ordinary endpoint distinction": "C6A uses ordinary `D_M`",
    }
    missing = [k for k, v in tokens.items() if v not in manuscript_surface]
    if missing:
        fail(f"manuscript/crosswalk missing design tokens {missing}")
    ok("manuscript and protocol crosswalk name solver, cover, N, and factorial tokens")

    if "eps=0.15" not in by["CIA2_CONVENTIONAL"]["dbscan"]:
        fail("CIA2 design-table Digits eps")
    if "1.455055840852852" not in by["C3_FMAPPER"]["dbscan"]:
        fail("C3 design-table Digits eps")
    if "tol=1e-7" not in by["C3_FMAPPER"]["fcm_solver"] or "max_iter=300" not in by["C3_FMAPPER"]["fcm_solver"]:
        fail("C3 FCM solver string")
    if "30 paired realizations at each of four positive doses" not in by["C1_SWISS"]["replicate_unit"]:
        fail("C1 replicate accounting")
    if "10 replicate streams" not in by["TRACK_C_C7"]["replicate_unit"]:
        fail("C7 stream count")
    ok("generated design table still carries known-correction parameters")

    c1 = jl("evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl")
    elig = sum(1 for r in c1 if r.get("eligible"))
    if elig != 120:
        fail(f"C1 eligible pairs {elig}")
    c3 = jl("evidence/campaigns/c3/C3_CONSTRUCTION_LEDGER.jsonl")
    if len(c3) != 525:
        fail(f"C3 n={len(c3)}")
    c8 = jl("evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl")
    if len(c8) != 530:
        fail(f"C8 n={len(c8)}")
    ok("ledger denominators C1=120 eligible, C3=525, C8=530")

    for token in (
        r"\(339\) constructions succeed",
        r"\(181\) terminate",
        r"14/20",
        r"21/30",
    ):
        if token not in tex:
            fail(f"frozen confirmatory denominator missing: {token}")
    ok("reported construction and agreement denominators remain in the manuscript")

    scientific_disclosures = {
        "narrative review method": "narrative critical review",
        "review corpus size": r"\(65\) sources",
        "C1 failed confirmatory disclosure": "confirmatory pattern was not observed",
        "C1 noise-unit qualification": "differs from the fraction of unique observations excluded",
        "M6A2 paired analysis": "paired Conventional-minus-F-Mapper",
        "M6A2 paired bootstrap seed": "seed 20260830",
        "literature/case separation": "Statements beginning from the present case studies are restricted to the released finite designs",
        "small-grid denominator": r"\(n=11\) evaluated cells",
        "R10 evidence map": r"\(282\) attempted",
        "N1 evidence map": r"\(906\) matched units",
        "IB1/III-4 evidence map": r"\(34/34\)",
    }
    missing = [
        key for key, value in scientific_disclosures.items() if value not in tex
    ]
    if missing:
        fail(f"manuscript missing scientific disclosures {missing}")
    forbidden = ("[11]", "[15, 24, 5]", r"(5\times5)", r"(3\times3)")
    present = [token for token in forbidden if token in tex]
    if present:
        fail(f"manuscript retains hardcoded citations or out-of-math tokens {present}")

    # Keep scope qualifications available where they matter without allowing any
    # one defensive disclaimer family to recur throughout the manuscript.
    defensive_families = {
        "ranking or benchmark": r"\b(?:not (?:a )?(?:universal )?(?:ranking|benchmark)|universal ranking|one ranking number)\b",
        "theorem or convergence claim": r"\b(?:not (?:a )?theorem|not (?:a )?convergence result)\b",
        "intrinsic probability": r"\b(?:not (?:an )?intrinsic (?:failure|success) probability|intrinsic failure probability)\b",
        "causal or population claim": r"\b(?:not (?:a )?causal effect|no population inference|without population inference|not population inference)\b",
        "Reeb-distance claim": r"\b(?:not (?:a )?Reeb distance|does not define a Reeb distance)\b",
        "universal metric": r"\bnot (?:a )?universal metric\b",
    }
    repeated = {
        family: len(re.findall(pattern, tex, flags=re.IGNORECASE))
        for family, pattern in defensive_families.items()
        if len(re.findall(pattern, tex, flags=re.IGNORECASE)) > 2
    }
    if repeated:
        fail(f"defensive disclaimer families repeated more than twice {repeated}")
    ok("review method, statistical units, denominators, and evidence map are explicit")
    ok("no defensive disclaimer family occurs more than twice")
    print("MANUSCRIPT_PARAMETER_CHECKS_PASSED")


if __name__ == "__main__":
    main()

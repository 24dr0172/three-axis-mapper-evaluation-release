#!/usr/bin/env python3
"""Check plan v4.1 reviewer-visible surfaces against the outstanding-critique list."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"PASS {msg}")


def main() -> None:
    tex = (ROOT / "manuscript/mapper_manuscript.tex").read_text()
    for p in sorted((ROOT / "manuscript/tables").glob("*.tex")):
        tex += "\n" + p.read_text()
    index = list(csv.DictReader((ROOT / "manuscript/CLAIM_EVIDENCE_INDEX.csv").open()))
    fields = list(index[0].keys())
    required_fields = [
        "claim_id", "axis", "type", "evidence",
        "tier", "estimand", "resolution_rule", "uncertainty",
        "manuscript_location", "status",
    ]
    missing_f = [f for f in required_fields if f not in fields]
    if missing_f:
        fail(f"claim index missing fields {missing_f}")
    if len(index) != 35:
        fail(f"claim index rows {len(index)}")
    if any(not r["estimand"] or not r["resolution_rule"] for r in index):
        fail("blank estimand or resolution_rule")
    ok("CLAIM_EVIDENCE_INDEX.csv has 35 rows and tier/estimand/rule/location/status")

    freeze_tokens = [
        "69de1663642f69bffd8db2e1aba09b09fe21b6cb3e5976b79567d47fe5d9beee",
        "Text-only",
        "Derived analysis",
        "Targeted robustness",
        r"FCM\_SOLVER\_SENSITIVITY\_AUDIT\_V4\_1",
        r"\input{tables/freeze_and_boundary.tex}",
    ]
    miss = [t for t in freeze_tokens if t not in tex]
    if miss:
        fail(f"freeze/boundary missing {miss}")
    ok("manuscript ships pre-v4.1 snapshot hash and analysis boundary")

    critique = {
        "C1 four-dose table": r"\label{tab:c1doses}",
        "C1 not observed": "originally adjudicated C1 primary pattern is not observed",
        "C1 no slogan": "most direct counterexample",
        "C7 alpha grid": r"\alpha\in\{0.05,0.10,0.20,0.30\}",
        "C7 48 cells": r"4\times 4\times 3=48",
        "C7 no new constructions": "No new Mapper constructions were generated",
        "C7 CR1 interval": "[-0.027730,-0.018927]",
        "master design": r"\input{tables/master_experimental_design.tex}",
        "CIA2 degenerate": "all points are unassigned",
        "FCM tol": r"\mathrm{tol}=10^{-7}",
        "FCM max_iter": r"\mathrm{max\_iter}=300",
        "FCM traces unavailable": r"not\_exposed\_by\_sealed\_run\_fcm\_1d",
        "M6-A2": r"\input{tables/m6a2_stability.tex}",
        "R10": r"\input{tables/r10_accounting.tex}",
        "Ball reason": r"selected\_center\_count<5",
        "pi vector": r"\pi=(\pi_1,\dots,\pi_m)",
        "DNA defined": r"$D_M^{\mathrm{NA}}$",
        "forensic proof": r"SAME\_FILTER\_FORENSIC\_PROOF\_III2\_V5",
        "env table": r"\input{tables/environment_provenance.tex}",
        "no bookkeeping defect": "bookkeeping defect",
    }
    if critique["C1 no slogan"] in tex:
        fail("unlabelled most-direct-counterexample language remains")
    if critique["no bookkeeping defect"] in tex:
        fail("bookkeeping defect wording remains")
    needed = {k: v for k, v in critique.items() if k not in {"C1 no slogan", "no bookkeeping defect"}}
    missing = [k for k, v in needed.items() if v not in tex]
    if missing:
        fail(f"manuscript still missing: {missing}")
    ok("critique manuscript items are present in the working tree")
    print("PLAN_SURFACE_CHECKS_PASSED")


if __name__ == "__main__":
    main()

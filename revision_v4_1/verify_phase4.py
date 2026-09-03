#!/usr/bin/env python3
"""Phase 4 robustness-audit checks. Does not rerun FCM or edit C3/C8."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = Path(__file__).resolve().parent
DERIVED = REV / "derived"
CAMPAIGN = "FCM_SOLVER_SENSITIVITY_AUDIT_V4_1"
ENV_CAMPAIGN = "ENV_EQUIVALENCE_CHECK_V4_1"


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

    for rel in (
        "evidence/campaigns/c3/C3_CONSTRUCTION_LEDGER.jsonl",
        "evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl",
        "evidence/campaigns/c3/PLAN.jsonl",
        "evidence/campaigns/c8/PLAN.jsonl",
        "src/mapper_framework/f_mapper.py",
    ):
        if digest(ROOT / rel) != manifest[rel]:
            fail(f"{rel} hash changed")
    ok("C3/C8 ledgers, plans, and sealed FCM unchanged")

    proc = subprocess.run([sys.executable, str(REV / "verify_phase3.py")],
                          capture_output=True, text=True)
    if proc.returncode != 0 or "PHASE3_CHECKS_PASSED" not in proc.stdout:
        fail(f"Phase 3 regression\n{proc.stdout}\n{proc.stderr}")
    ok("Phase 3 still passes")

    jsonl = DERIVED / f"{CAMPAIGN}.jsonl"
    summary_path = DERIVED / f"{CAMPAIGN}_SUMMARY.json"
    if not jsonl.is_file() or not summary_path.is_file():
        fail("missing FCM audit jsonl or summary")
    rows = [json.loads(x) for x in jsonl.read_text().splitlines() if x.strip()]
    summary = json.loads(summary_path.read_text())
    if summary["campaign_id"] != CAMPAIGN:
        fail("wrong campaign_id")
    if summary["mapper_constructions_generated"] != 0:
        fail("Phase 4 generated Mapper constructions")
    if "not a replacement" not in summary["label"]:
        fail("summary lost robustness/not-confirmatory label")
    if "Not recomputed" not in summary["downstream_quality_fidelity"]:
        fail("missing downstream not-recomputed statement")
    if len(rows) != 1055:
        fail(f"expected 1055 unit rows, got {len(rows)}")
    if summary["n_unique_fcm_jobs"] != 1035:
        fail(f"unique FCM jobs {summary['n_unique_fcm_jobs']}")
    if summary["frozen_match_count"] != 1055:
        fail("audit 1e-7/300 does not match every frozen flag")
    if any(not r.get("audit_matches_frozen_1e-7_300") for r in rows):
        fail("jsonl contains a frozen-flag mismatch")
    if any(r.get("campaign_id") != CAMPAIGN for r in rows):
        fail("jsonl campaign_id drift")
    if summary["frozen_nonconverged"] != 292:
        fail(f"frozen nonconverged {summary['frozen_nonconverged']} != 181+111")
    if set(summary["tols"]) != {"1e-4", "1e-5", "1e-6", "1e-7"}:
        fail(f"tol grid {summary['tols']}")
    # Looser tolerance cannot increase the non-convergence count.
    seq = [summary["by_tol_nonconverged"][t] for t in ("1e-4", "1e-5", "1e-6", "1e-7")]
    if seq != sorted(seq):
        fail(f"nonconvergence not monotone in tol: {seq}")
    if summary["by_tol_nonconverged"]["1e-7"] != 292:
        fail("1e-7/300 nonconvergence must equal frozen 292")
    if not summary.get("smoke_vs_sealed", {}).get("all_match"):
        fail("missing sealed-vs-vectorized smoke match")
    ok("FCM audit matches frozen 1e-7/300 on 1055 units; 1035 unique jobs; 0 constructions")

    env = json.loads((DERIVED / f"{ENV_CAMPAIGN}_SKIP.json").read_text())
    if env["campaign_id"] != ENV_CAMPAIGN:
        fail("env skip campaign id")
    if env["status"] != "skipped_interpreter_unavailable":
        fail("env check was not recorded as skipped")
    if env["mapper_constructions_generated"] != 0:
        fail("env skip generated constructions")
    ok("ENV_EQUIVALENCE_CHECK_V4_1 skipped (no CPython 3.13)")

    tex = (ROOT / "manuscript/mapper_manuscript.tex").read_text()
    for p in (ROOT / "manuscript/tables").glob("*.tex"):
        tex += "\n" + p.read_text()
    needed = {
        "campaign id": r"FCM\_SOLVER\_SENSITIVITY\_AUDIT\_V4\_1",
        "tab:fcmsolv": r"\label{tab:fcmsolv}",
        "robustness": "robustness/sensitivity",
        "not replacement": "not a replacement for the frozen confirmatory",
        "not recomputed": "not recomputed",
        "input table": r"\input{tables/fcm_solver_sensitivity.tex}",
        "env skip": r"skipped\_interpreter\_unavailable",
    }
    missing = [k for k, v in needed.items() if v not in tex]
    if missing:
        fail(f"manuscript missing {missing}")
    table = (ROOT / "manuscript/tables/fcm_solver_sensitivity.tex").read_text()
    if str(summary["frozen_nonconverged"]) not in table:
        fail("table missing frozen nonconvergence count")
    if "not a replacement" not in table:
        fail("table caption lost robustness label")
    ok("manuscript cites the FCM audit as robustness, not confirmatory")

    registry = {r["artifact_id"]: r for r in csv.DictReader((REV / "DERIVED_EVIDENCE_REGISTRY.csv").open())}
    for aid, rel in (
        ("P4_FCM_JSONL", f"revision_v4_1/derived/{CAMPAIGN}.jsonl"),
        ("P4_FCM_SUMMARY", f"revision_v4_1/derived/{CAMPAIGN}_SUMMARY.json"),
        ("P4_FCM_TEX", "manuscript/tables/fcm_solver_sensitivity.tex"),
        ("P4_ENV_SKIP", f"revision_v4_1/derived/{ENV_CAMPAIGN}_SKIP.json"),
    ):
        if aid not in registry:
            fail(f"registry missing {aid}")
        if registry[aid]["relative_path"] != rel:
            fail(f"registry path for {aid}")
        if digest(ROOT / rel) != registry[aid]["sha256"]:
            fail(f"registry hash drift {aid}")
    ok("Phase 4 derived objects are hashed in the registry")
    print("PHASE4_CHECKS_PASSED")


if __name__ == "__main__":
    main()

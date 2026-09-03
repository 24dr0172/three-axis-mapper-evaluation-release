#!/usr/bin/env python3
"""Write the reviewer-visible claim index with v4.1 tier fields.

Keeps the original four columns first so verify_release.py still works.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIER = Path(__file__).resolve().parent / "CLAIM_TIER_TABLE.csv"
INDEX = ROOT / "manuscript" / "CLAIM_EVIDENCE_INDEX.csv"

# Post Phase-1/2 manuscript locations (overrides stale Phase-0 notes).
LOCATION = {
    "QS_CIRCLE": "appendix evidence map; scoped out of adjudication",
    "C1_SWISS": "main text sec:quality tab:c1doses and fig:c1quality; failed confirmatory; four-dose series",
    "C2_SELECTION": "main text sec:quality failure accounting",
    "C3_FMAPPER": "main text sec:quality construction reliability; frozen FCM solver named",
    "C4_FMAPPER": "main text sec:stability tab:axis1fixed",
    "C4_BALL": "main text sec:stability tab:axis1fixed",
    "C4_ENSEMBLE": "main text qualified Ensemble constructibility",
    "CIA2_CONVENTIONAL": "main text sec:stability; Digits 0/20 as reference-ineligible all_points_unassigned",
    "C5_CIRCLE": "main text sec:fidelity finite-case; frequencies from C8",
    "C5_TRIPOD": "main text sec:fidelity; non-convergent under frozen FCM solver",
    "C6A_CONVENTIONAL": "main text sec:stability undefined estimand; fig:interactions B",
    "C6B_BALL": "main text sec:stability; fig:interactions C",
    "IB4F_FMAPPER": "appendix/exploratory; not a C3 confirmatory interaction",
    "IV1_COVER": "main text sec:integration",
    "IV2": "main text sec:integration",
    "IV3_STABILITY_FIDELITY": "main text sec:integration",
    "IV5": "appendix descriptive companion to IV-1",
    "IB1_III4_CIRCLE": "appendix evidence map; 34/34 eligible campaign",
    "IB1_III4_TRIPOD": "appendix evidence map; lower-leaf 3-leaf/1-branch preserved",
    "M6A2_STABILITY": "main text Filtered structural stability on Swiss; tab:m6a2",
    "IB3_II3_PARAMETER": "appendix descriptive",
    "IB2_II2_COVER": "appendix descriptive",
    "AXIS3_III1_CIRCLE": "main text sec:fidelity direct baseline",
    "AXIS3_III1_TRIPOD": "main text sec:fidelity direct baseline",
    "AXIS3_III2": "main text sec:fidelity surfaces; metadata re-audit SAME_FILTER_FORENSIC_PROOF_III2_V5",
    "N1_MATCHED": "appendix evidence map; descriptive only",
    "R10_STRUCTURAL": "appendix Structural-robustness accounting; tab:r10account",
    "BALL_PARAMETER": "main text Ball Mapper and metric shape; ineligible row selected_center_count<5",
    "E2M_ENSEMBLE": "main text qualified Ensemble constructibility",
    "E2M_II1": "appendix descriptive",
    "E2M_III3": "appendix evidence map; finite-case diagnostic not a Reeb theorem",
    "TRACK_C_FACTORIAL": "main text sec:stability Track C design",
    "C7_TRACK_C_INTERACTION": "main text sec:stability full factorial/model/both intervals",
    "C8_FMAPPER_AXIS3": "main text sec:quality and sec:fidelity",
    "C9_BALL_CIRCLE": "main text sec:stability; fig:interactions D",
}

OUT_FIELDS = [
    "claim_id",
    "axis",
    "type",
    "evidence",
    "tier",
    "estimand",
    "resolution_rule",
    "uncertainty",
    "manuscript_location",
    "status",
]


def main() -> None:
    rows = list(csv.DictReader(TIER.open()))
    if len(rows) != 35:
        raise SystemExit(f"expected 35 tier rows, got {len(rows)}")
    missing = [r["claim_id"] for r in rows if r["claim_id"] not in LOCATION]
    if missing:
        raise SystemExit(f"missing location override: {missing}")
    out = []
    for r in rows:
        out.append(
            {
                "claim_id": r["claim_id"],
                "axis": r["axis"],
                "type": r["index_type"],
                "evidence": r["evidence_authority"],
                "tier": r["v4_1_tier"],
                "estimand": r["estimand"],
                "resolution_rule": r["resolution_rule"],
                "uncertainty": r["uncertainty"],
                "manuscript_location": LOCATION[r["claim_id"]],
                "status": r["current_terminal_state"],
            }
        )
    with INDEX.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {INDEX} rows={len(out)}")


if __name__ == "__main__":
    main()

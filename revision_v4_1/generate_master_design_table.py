#!/usr/bin/env python3
"""Generate the v4.1 master experimental-design table from frozen authorities.

Do not hand-edit the CSV or the TeX snippet. Re-run this script.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = Path(__file__).resolve().parent / "MASTER_EXPERIMENTAL_DESIGN_TABLE.csv"
TEX_PATH = ROOT / "manuscript" / "tables" / "master_experimental_design.tex"
FIELDS = [
    "campaign_id",
    "method",
    "dataset",
    "lens",
    "n_generator_rule",
    "cover_or_radius",
    "dbscan",
    "fcm_solver",
    "noise_model",
    "replicate_unit",
    "primary_endpoint",
    "authority",
]


def jl(rel: str) -> list[dict]:
    return [json.loads(x) for x in (ROOT / rel).read_text().splitlines() if x.strip()]


def uniq(rows: list[dict], key: str):
    return sorted({r.get(key) for r in rows}, key=lambda x: (x is None, str(x)))


def fmt_set(values, digits=None) -> str:
    out = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and digits is not None:
            out.append(f"{v:.{digits}g}" if digits < 6 else f"{v:.{digits}f}".rstrip("0").rstrip("."))
        else:
            out.append(str(v))
    return ", ".join(out)


def clusterer_by_dataset(rows: list[dict], spec_key="clusterer_spec") -> str:
    parts = []
    for ds in uniq(rows, "dataset_id"):
        sub = [r for r in rows if r.get("dataset_id") == ds]
        specs = {json.dumps(r.get(spec_key), sort_keys=True) for r in sub}
        if len(specs) != 1:
            parts.append(f"{ds}: mixed")
            continue
        spec = json.loads(next(iter(specs)))
        if not isinstance(spec, dict):
            parts.append(f"{ds}: {spec}")
            continue
        eps = spec.get("eps")
        ms = spec.get("min_samples")
        parts.append(f"{ds}: eps={eps}, min_samples={ms}")
    return "; ".join(parts)


def build_rows() -> list[dict]:
    c1_plan = jl("replay_inputs/axis1_corrections/C1_PLAN.jsonl")
    c1_con = jl("evidence/campaigns/c1/C1_CONSTRUCTION_LEDGER.jsonl")
    c1_cmp = jl("evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl")
    c3 = jl("evidence/campaigns/c3/C3_CONSTRUCTION_LEDGER.jsonl")
    c4 = jl("evidence/campaigns/c4_fmapper/C4R_CONSTRUCTION_LEDGER.jsonl")
    cia = jl("evidence/campaigns/conventional_fixed_cover/CIA2_CONSTRUCTION_LEDGER.jsonl")
    c6a = jl("evidence/campaigns/c6a/C6A_LEDGER.jsonl")
    c6b_contract = json.loads((ROOT / "contracts/C6B_BALL_SWISS_NOISE_X_RADIUS_V1.json").read_text())
    c7 = json.loads((ROOT / "evidence/campaigns/c7/C7_INTERACTION_FIT.json").read_text())
    tc_con = jl("evidence/baseline/track_c/IB5_INTERACTIONS_CONSTRUCTION_LEDGER.jsonl")
    c8 = jl("evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl")
    c8_contract = json.loads((ROOT / "contracts/C8_FMAPPER_CIRCLE_GRID_AND_AXIS3_REPLICATION_V1.json").read_text())
    c9_contract = json.loads((ROOT / "contracts/C9_BALL_CIRCLE_NOISE_X_RADIUS_V1.json").read_text())
    iii1c = json.loads((ROOT / "evidence/baseline/axis3_iii1/CIRCLE_RESULT.json").read_text())
    iii1t = json.loads((ROOT / "evidence/baseline/axis3_iii1/TRIPOD_RESULT.json").read_text())
    iii2b = list(csv.DictReader((ROOT / "evidence/baseline/axis3_iii2/III2B_RECOVERY_SUMMARY.csv").open()))
    c5 = jl("evidence/campaigns/c5/C5TA_LEDGER.jsonl")

    n_c1_ref = sum(1 for r in c1_con if r["kind"] == "reference")
    n_c1_pair = Counter(r["alpha"] for r in c1_cmp)
    c1_eps = uniq(c1_plan, "dbscan_eps")
    c1_min = uniq(c1_plan, "dbscan_min_samples")
    c1_n = uniq(c1_plan, "n_intervals")
    c1_p = uniq(c1_plan, "overlap_frac")
    c1_method = uniq(c1_plan, "method")
    c1_lens = uniq(c1_plan, "lens_id")
    if n_c1_ref != 30 or set(n_c1_pair) != {0.05, 0.1, 0.2, 0.3} or any(v != 30 for v in n_c1_pair.values()):
        raise SystemExit("C1 dose accounting does not match 30 refs + 30 pairs at four positive doses")

    cia_digits = [r for r in cia if r["dataset_id"] == "digits_1797x64_scaled16"]
    cia_digit_eps = uniq(cia_digits, "dbscan_eps")
    if cia_digit_eps != [0.15]:
        raise SystemExit(f"CIA2 Digits eps expected 0.15, got {cia_digit_eps}")

    c3_digits = [r for r in c3 if r["dataset_id"] == "digits_1797x64_scaled16"]
    c3_digit_eps = sorted({r["clusterer_spec"]["eps"] for r in c3_digits})
    if c3_digit_eps != [1.455055840852852]:
        raise SystemExit(f"C3 Digits eps mismatch: {c3_digit_eps}")

    tc_alpha = [a for a in uniq(tc_con, "alpha") if a not in (None,)]
    # reference rows have alpha null; perturbed rows carry the four positive doses
    tc_pert = [r for r in tc_con if r.get("kind") != "reference" and r.get("alpha") not in (None, 0, 0.0)]
    # construction ledger mixes ref/pert; comparison ledger is the 48-cell authority
    tc_cmp = jl("evidence/baseline/track_c/IB5_INTERACTIONS_COMPARISON_LEDGER.jsonl")
    cells = {(r["n_intervals"], r["eps"], r["alpha"]) for r in tc_cmp}
    if len(cells) != 48:
        raise SystemExit(f"Track C analysed cells expected 48, got {len(cells)}")
    if {a for _, _, a in cells} != {0.05, 0.1, 0.2, 0.3}:
        raise SystemExit("Track C analysed cells are not the four positive doses")

    c3_tol = uniq(c3, "fcm_tol")
    c3_max = uniq(c3, "fcm_max_iter")
    c8_tol = uniq(c8, "fcm_tol")
    c8_max = uniq(c8, "fcm_max_iter")
    fcm_frozen = "tol=1e-7, max_iter=300, deterministic linspace init; fcm_seed retained but inert"
    if c3_tol != [1e-7] or c3_max != [300] or c8_tol != [1e-7] or c8_max != [300]:
        raise SystemExit("FCM solver hyperparameters are not uniform at the frozen criterion")

    iii2_n = sorted({int(r["sample_size"]) for r in iii2b})
    iii2_cells = len(iii2b)

    rows = [
        dict(
            campaign_id="C4_FMAPPER",
            method="F-Mapper",
            dataset="Circle, Swiss, Digits",
            lens="dataset canonical (Circle height y; Swiss radial_xz; Digits frozen PC1)",
            n_generator_rule="Circle N=1000 uniform; Swiss N_base=2500 with hole excision (variable realized N); Digits 1797x64 frozen PCA artifact",
            cover_or_radius="fuzzy cover c=8, tau=0.10, frozen on the reference then applied to 50%/80% subsamples",
            dbscan=clusterer_by_dataset(c4),
            fcm_solver=fcm_frozen,
            noise_model="none (subsampling of clean realizations)",
            replicate_unit="20 defined comparisons per dataset x retention cell",
            primary_endpoint="median exact joint D_M^NA at 50% vs 80% retention",
            authority="evidence/campaigns/c4_fmapper/C4R_CONSTRUCTION_LEDGER.jsonl",
        ),
        dict(
            campaign_id="CIA2_CONVENTIONAL",
            method="Conventional Mapper",
            dataset="Circle, Swiss, Digits",
            lens="dataset canonical",
            n_generator_rule="Circle N=1000; Swiss N_base=2500 hole excision; Digits 1797x64 frozen PCA",
            cover_or_radius="regular cover n_intervals=10, overlap=0.30, frozen reference intervals",
            dbscan="all three datasets including Digits: eps=0.15, min_samples=3 (Digits is NOT the C3 scale-rule eps)",
            fcm_solver="n/a",
            noise_model="none (subsampling of clean realizations)",
            replicate_unit="20 planned comparisons per dataset x retention cell",
            primary_endpoint="exact joint D_M^NA; defined-denominator counts. Digits: clean reference degenerate/all_points_unassigned, so 0/20 defined",
            authority="evidence/campaigns/conventional_fixed_cover/CIA2_CONSTRUCTION_LEDGER.jsonl",
        ),
        dict(
            campaign_id="C1_SWISS",
            method="Conventional Mapper",
            dataset="Swiss roll with hole",
            lens=fmt_set(c1_lens),
            n_generator_rule="generate_swiss_roll_with_hole(N=2500, resample_to_exact_N=False); hole excision; realized N varies by data_seed",
            cover_or_radius=f"regular cover n_intervals={fmt_set(c1_n)}, overlap={fmt_set(c1_p)}; clean realized pullbacks frozen within replicate",
            dbscan=f"eps={fmt_set(c1_eps)}, min_samples={fmt_set(c1_min)}",
            fcm_solver="n/a",
            noise_model="coordinate Gaussian, sigma=alpha*epsilon0 with epsilon0=11.634735480276268; alpha in {0.05,0.10,0.20,0.30}",
            replicate_unit=f"{n_c1_ref} clean references; 30 paired realizations at each of four positive doses (150 constructions)",
            primary_endpoint="originally adjudicated C1 pattern at alpha=0.05; terminal PRIMARY_PATTERN_NOT_OBSERVED. Full four-dose D_M^NA and Silhouette series retained as descriptive",
            authority="evidence/campaigns/c1/FINAL_ACCOUNTING.json; replay_inputs/axis1_corrections/C1_PLAN.jsonl",
        ),
        dict(
            campaign_id="C3_FMAPPER",
            method="F-Mapper",
            dataset="Swiss (500 records), Digits (25 records)",
            lens="Swiss radial_xz; Digits frozen PC1",
            n_generator_rule="Swiss N_base=2500 hole excision; Digits 1797x64 frozen PCA; one Digits realization per grid cell",
            cover_or_radius="fuzzy cover c in {5,8,10,15,20}, tau in {0.10,0.15,0.20,0.30,0.40}",
            dbscan=clusterer_by_dataset(c3),
            fcm_solver=fcm_frozen,
            noise_model="none (construction-reliability grid)",
            replicate_unit="Swiss: 20 data realizations per cell; Digits: 1 realization per cell",
            primary_endpoint="construction-status counts (success / FCM non-convergence under frozen solver / coverage gap)",
            authority="evidence/campaigns/c3/C3_CONSTRUCTION_LEDGER.jsonl",
        ),
        dict(
            campaign_id="TRACK_C_C7",
            method="Conventional Mapper",
            dataset="unit_circle_S1",
            lens="height y",
            n_generator_rule="generate_clean_circle(N=1000, radius=1.0, sampling=uniform)",
            cover_or_radius="regular cover n in {6,10,15,20}, overlap=0.30; clean realized pullbacks frozen within configuration and replicate",
            dbscan="eps in {0.08,0.15,0.25}, min_samples=3",
            fcm_solver="n/a",
            noise_model="four positive coordinate-noise doses alpha in {0.05,0.10,0.20,0.30}; alpha=0 is the clean reference, not one of the 48 perturbed cells",
            replicate_unit="10 replicate streams; 4 x 4 x 3 = 48 perturbed cells; 600 constructions / 480 eligible comparisons",
            primary_endpoint="C7: beta_12 (n_intervals x eps) after exact-joint endpoint correction; no new Mapper constructions",
            authority="evidence/campaigns/c7/C7_INTERACTION_FIT.json; evidence/baseline/track_c/IB5_INTERACTIONS_COMPARISON_LEDGER.jsonl",
        ),
        dict(
            campaign_id="C6A_CONVENTIONAL",
            method="Conventional Mapper",
            dataset="Swiss roll with hole",
            lens="radial_xz",
            n_generator_rule="Swiss N_base=2500 hole excision; realized N recorded per row (e.g. 2217 on replicate 0)",
            cover_or_radius="frozen regular cover n in {10,15}, overlap=0.30",
            dbscan="eps=1.015739105123552, min_samples=3",
            fcm_solver="n/a",
            noise_model="alpha in {0,0.05,0.10,0.20,0.30}; primary contrast at alpha=0.05",
            replicate_unit="20 replicates; 200 frozen-cover constructions",
            primary_endpoint="paired contrast D(n=15)-D(n=10) at alpha=0.05; terminal PRIMARY_ESTIMAND_NOT_MEASURABLE",
            authority="evidence/campaigns/c6a/C6A_LEDGER.jsonl; contracts/C6A_CONVENTIONAL_SWISS_NOISE_X_COVER_V1.json",
        ),
        dict(
            campaign_id="C6B_BALL",
            method="Ball Mapper",
            dataset="Swiss roll with hole",
            lens="n/a (metric cover)",
            n_generator_rule="Swiss N_base=2500 hole excision",
            cover_or_radius=f"frozen landmarks; epsilon0={c6b_contract['epsilon0']}; k in {c6b_contract['radius_multipliers']}",
            dbscan="n/a",
            fcm_solver="n/a",
            noise_model=f"alpha in {c6b_contract['alphas']}",
            replicate_unit="20 replicates; 300 frozen-landmark constructions",
            primary_endpoint="median edge-Jaccard difference-in-differences; interval spans zero",
            authority="contracts/C6B_BALL_SWISS_NOISE_X_RADIUS_V1.json; evidence/campaigns/c6b/C6B_LEDGER.jsonl",
        ),
        dict(
            campaign_id="C9_BALL_CIRCLE",
            method="Ball Mapper",
            dataset="unit_circle_S1",
            lens="n/a (metric cover); lens_id recorded as circle_height_y in the contract only",
            n_generator_rule="generate_clean_circle(N=1000, radius=1.0, sampling=uniform)",
            cover_or_radius=f"frozen landmarks; epsilon0={c9_contract['frozen_scales']['ball_epsilon0']}; k in {c9_contract['factors']['radius_multipliers']}",
            dbscan="n/a",
            fcm_solver="n/a",
            noise_model="alpha in {0,0.05,0.10,0.20,0.30}; sigma=alpha*epsilon0",
            replicate_unit="20 replicates; 300 constructions",
            primary_endpoint="edge-Jaccard DiD; vacuous because k=2.0 is a saturated two-node one-edge graph",
            authority="contracts/C9_BALL_CIRCLE_NOISE_X_RADIUS_V1.json; evidence/campaigns/c9/C9_LEDGER.jsonl",
        ),
        dict(
            campaign_id="C8_FMAPPER_AXIS3",
            method="F-Mapper",
            dataset="Circle grid N=1000 (500); Tripod N=1000 (30)",
            lens="Circle height y; Tripod height y",
            n_generator_rule="Circle uniform N=1000; Tripod N=1000",
            cover_or_radius="fuzzy cover c in {5,8,10,15,20}, tau in {0.10,0.15,0.20,0.30,0.40}; primary cell (c=10, tau=0.10)",
            dbscan="eps=0.15, min_samples=3 (III-1 Axis-III clusterer, not the Swiss scale rule)",
            fcm_solver=fcm_frozen,
            noise_model="none (independent data realizations)",
            replicate_unit="Circle 20 per cell; Tripod 30 at the primary cell; 530 constructions",
            primary_endpoint="Tier-A invariant agreement at (c=10, tau=0.10); full grid descriptive",
            authority="evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl; contracts/C8_FMAPPER_CIRCLE_GRID_AND_AXIS3_REPLICATION_V1.json",
        ),
        dict(
            campaign_id="C5_FMAPPER",
            method="F-Mapper",
            dataset="one Circle case, one Tripod case",
            lens="height y",
            n_generator_rule="same frozen C8/C5 cell configuration",
            cover_or_radius="c=10, tau=0.10",
            dbscan="eps=0.15, min_samples=3",
            fcm_solver=fcm_frozen,
            noise_model="none",
            replicate_unit="n=1 per domain",
            primary_endpoint="finite-case invariant agreement; Tripod non-convergent under frozen solver; frequencies from C8",
            authority="evidence/campaigns/c5/C5TA_LEDGER.jsonl",
        ),
        dict(
            campaign_id="AXIS3_III1",
            method="Conventional Mapper",
            dataset="Circle; Tripod",
            lens="height y (same-filter as the analytic/PL reference)",
            n_generator_rule=f"Circle N={iii1c['sample_size']}; Tripod N={iii1t['sample_size']}",
            cover_or_radius="n_intervals=10, overlap=0.30",
            dbscan="eps=0.15, min_samples=3",
            fcm_solver="n/a",
            noise_model="none (direct baseline)",
            replicate_unit="one finite case per domain",
            primary_endpoint="graph/nerve invariants vs same-filter reference; type-restricted bottleneck diagnostic",
            authority="evidence/baseline/axis3_iii1/CIRCLE_RESULT.json; TRIPOD_RESULT.json",
        ),
        dict(
            campaign_id="AXIS3_III2",
            method="Conventional Mapper",
            dataset="Circle; Tripod",
            lens="height y; metadata-re-audited same-filter identities",
            n_generator_rule=f"III-2A N=1000 (50 cover cells); III-2B N in {iii2_n} over {iii2_cells} cells, 30 matched realizations per cell",
            cover_or_radius="III-2A 5x5 cover landscape; III-2B reduced 3x3 cover grid",
            dbscan="Axis-III frozen clusterer (eps=0.15, min_samples=3)",
            fcm_solver="n/a",
            noise_model="none (sample-size x cover surface)",
            replicate_unit="III-2B: 2160 eligible runs; 2208 run records metadata re-audited",
            primary_endpoint="joint graph-and-nerve invariant recovery (46/50 cells; 1201/2160 runs)",
            authority="evidence/baseline/axis3_iii2/RUN_LEVEL_REQUALIFIED.csv",
        ),
    ]
    # keep C7 design fields referenced so a drift in the fit file fails loudly
    if c7["design"]["replicate_streams"] != 10:
        raise SystemExit("C7 replicate stream count drifted")
    if c5[0]["status"] not in {"success", "fcm_non_convergence"}:
        raise SystemExit("C5 ledger unreadable")
    if c8_contract["frozen_configuration"]["fcm_tol"] != 1e-07:
        raise SystemExit("C8 contract FCM tol drifted")
    return rows


def tex_escape(s: str) -> str:
    s = s.replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace(r"D\_M^NA", r"$D_M^{\mathrm{NA}}$")
    s = s.replace(r"PRIMARY\_PATTERN\_NOT\_OBSERVED", r"\texttt{PRIMARY\_PATTERN\_NOT\_OBSERVED}")
    s = s.replace(r"PRIMARY\_ESTIMAND\_NOT\_MEASURABLE", r"\texttt{PRIMARY\_ESTIMAND\_NOT\_MEASURABLE}")
    s = s.replace(r"all\_points\_unassigned", r"\texttt{all\_points\_unassigned}")
    return s


def write_tex(rows: list[dict]) -> None:
    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    compact = {
        "C4_FMAPPER": "C4 F-Mapper",
        "CIA2_CONVENTIONAL": "CIA2 Conventional",
        "C1_SWISS": "C1 Swiss",
        "C3_FMAPPER": "C3 F-Mapper",
        "TRACK_C_C7": "Track C / C7",
        "C6A_CONVENTIONAL": "C6-a Conventional",
        "C6B_BALL": "C6-b Ball",
        "C9_BALL_CIRCLE": "C9 Ball Circle",
        "C8_FMAPPER_AXIS3": "C8 F-Mapper",
        "C5_FMAPPER": "C5 F-Mapper",
        "AXIS3_III1": "III-1 Conventional",
        "AXIS3_III2": "III-2 Conventional",
    }
    lines = [
        "% Auto-generated by revision_v4_1/generate_master_design_table.py. Do not hand-edit.",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Master experimental-design table generated from controlling ledgers and contracts. "
        "Swiss $N$ is a 2500-draw base with hole excision (realized $N$ varies). "
        "Track~C analysed cells use the four positive noise doses; $\\alpha=0$ is the clean reference. "
        "CIA2 Digits uses DBSCAN $\\varepsilon=0.15$; C3 Digits uses $\\varepsilon=1.455055840852852$.}",
        "\\label{tab:masterdesign}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}p{0.12\\textwidth}"
        ">{\\raggedright\\arraybackslash}p{0.13\\textwidth}"
        ">{\\raggedright\\arraybackslash}X"
        ">{\\raggedright\\arraybackslash}p{0.22\\textwidth}@{}}",
        "\\toprule",
        "Campaign & Method / data & Design (generator, cover, clusterer, solver, noise, replicates) & Primary endpoint \\\\",
        "\\midrule",
    ]
    for row in rows:
        design = (
            f"{row['n_generator_rule']}. Cover: {row['cover_or_radius']}. "
            f"DBSCAN: {row['dbscan']}. FCM: {row['fcm_solver']}. "
            f"Noise: {row['noise_model']}. {row['replicate_unit']}."
        )
        cell = " & ".join(
            tex_escape(x)
            for x in (
                compact[row["campaign_id"]],
                f"{row['method']}; {row['dataset']}",
                design,
                row["primary_endpoint"],
            )
        )
        lines.append(cell + " \\\\")
        lines.append("\\addlinespace")
    if lines[-1] == "\\addlinespace":
        lines.pop()
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""])
    TEX_PATH.write_text("\n".join(lines))


def main() -> None:
    rows = build_rows()
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_tex(rows)
    print(f"wrote {CSV_PATH} rows={len(rows)}")
    print(f"wrote {TEX_PATH}")


if __name__ == "__main__":
    main()

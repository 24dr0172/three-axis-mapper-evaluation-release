#!/usr/bin/env python3
"""Phase 3 derived analyses from frozen evidence. No new Mapper constructions."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DERIVED = Path(__file__).resolve().parent / "derived"
FIG = ROOT / "manuscript" / "figures"
TEX = ROOT / "manuscript" / "tables"
SEED = 20260830
B = 10_000

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/three_axis_v41_mplconfig")))
import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
GRAY = "#6B7280"
BLACK = "#111827"
LIGHT = "#E5E7EB"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jl(rel: str) -> list[dict]:
    return [json.loads(x) for x in (ROOT / rel).read_text().splitlines() if x.strip()]


def dump(name: str, obj: dict) -> Path:
    DERIVED.mkdir(parents=True, exist_ok=True)
    path = DERIVED / name
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return path


def f6(x: float) -> str:
    return f"{float(x):.6f}"


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    diffs = np.empty(B, dtype=float)
    na, nb = len(a), len(b)
    for i in range(B):
        diffs[i] = rng.choice(a, size=na, replace=True).mean() - rng.choice(b, size=nb, replace=True).mean()
    point = float(a.mean() - b.mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "point_estimate": point,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "B": B,
        "seed": SEED,
        "n_a": int(na),
        "n_b": int(nb),
        "method": "independent two-sample percentile bootstrap of mean(a)-mean(b)",
    }


def c1_dose_response() -> dict:
    cmp_rows = jl("evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl")
    paired = jl("evidence/campaigns/c1/C1_PAIRED_QUALITY_LEDGER.jsonl")
    quality_rows = jl("evidence/campaigns/c1/C1_QUALITY_LEDGER.jsonl")
    quality = {r["construction_id"]: r for r in quality_rows}
    by_d = {}
    for alpha in (0.05, 0.1, 0.2, 0.3):
        dvals = [r["d_common_id"] for r in cmp_rows if r["alpha"] == alpha]
        reasons = Counter(r.get("reason") for r in cmp_rows if r["alpha"] == alpha)
        elig = sum(1 for r in cmp_rows if r["alpha"] == alpha and r.get("eligible"))
        ps = [r for r in paired if r["alpha"] == alpha]
        sil_c = [r["silhouette_clean"] for r in ps]
        sil_p = [r["silhouette_perturbed"] for r in ps]
        delta = [p - c for p, c in zip(sil_p, sil_c)]
        clean_ids = [r["construction_id"].replace("__NOISE__", "__REF__").rsplit("__A", 1)[0] for r in ps]
        pert_ids = [r["construction_id"] for r in ps]
        q_clean = [quality[x] for x in clean_ids]
        q_pert = [quality[x] for x in pert_ids]
        noise_c = [r["noise_fraction_macro"] for r in q_clean]
        noise_p = [r["noise_fraction_macro"] for r in q_pert]
        noise_inc_c = [r["noise_fraction_incidence"] for r in q_clean]
        noise_inc_p = [r["noise_fraction_incidence"] for r in q_pert]
        eligible_frac_c = [r["n_pullbacks_eligible_silhouette"] / r["n_pullbacks_nonempty"] for r in q_clean]
        eligible_frac_p = [r["n_pullbacks_eligible_silhouette"] / r["n_pullbacks_nonempty"] for r in q_pert]
        by_d[f"{alpha:.2f}"] = {
            "n_pairs": len(dvals),
            "n_eligible": elig,
            "eligibility_reason_counts": dict(reasons),
            "median_d_m_na": float(np.median(dvals)),
            "median_silhouette_clean": float(np.median(sil_c)),
            "median_silhouette_perturbed": float(np.median(sil_p)),
            "median_silhouette_delta": float(np.median(delta)),
            "median_noise_fraction_macro_clean": float(np.median(noise_c)),
            "median_noise_fraction_macro_perturbed": float(np.median(noise_p)),
            "median_noise_fraction_incidence_clean": float(np.median(noise_inc_c)),
            "median_noise_fraction_incidence_perturbed": float(np.median(noise_inc_p)),
            "median_eligible_silhouette_pullback_fraction_clean": float(np.median(eligible_frac_c)),
            "median_eligible_silhouette_pullback_fraction_perturbed": float(np.median(eligible_frac_p)),
            "d_values": dvals,
            "silhouette_delta": delta,
            "noise_fraction_macro_clean_values": noise_c,
            "noise_fraction_macro_perturbed_values": noise_p,
        }
        if elig != 30 or reasons.get("success") != 30:
            raise SystemExit(f"C1 alpha={alpha} eligibility drifted")
    obj = {
        "analysis_id": "C1_FULL_DOSE_RESPONSE_V4_1",
        "mapper_constructions_generated": 0,
        "source": {
            "comparison": "evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl",
            "comparison_sha256": sha256(ROOT / "evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl"),
            "paired": "evidence/campaigns/c1/C1_PAIRED_QUALITY_LEDGER.jsonl",
            "paired_sha256": sha256(ROOT / "evidence/campaigns/c1/C1_PAIRED_QUALITY_LEDGER.jsonl"),
            "quality": "evidence/campaigns/c1/C1_QUALITY_LEDGER.jsonl",
            "quality_sha256": sha256(ROOT / "evidence/campaigns/c1/C1_QUALITY_LEDGER.jsonl"),
        },
        "doses": {k: {kk: vv for kk, vv in v.items() if kk not in {
                    "d_values", "silhouette_delta",
                    "noise_fraction_macro_clean_values", "noise_fraction_macro_perturbed_values"
                  }} for k, v in by_d.items()},
        "_plot": by_d,
    }
    # Matplotlib PDF metadata is not byte-stable across reruns. Verify mode
    # must not rewrite an already-hashed figure.
    write_fig = os.environ.get("PHASE3_WRITE_FIGURES", "if_missing")
    FIG.mkdir(parents=True, exist_ok=True)
    existing = all((FIG / f"fig07_c1_perturbation_response.{s}").is_file()
                   for s in ("pdf", "svg", "png"))
    if write_fig == "never" or (write_fig == "if_missing" and existing):
        return obj
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.65), constrained_layout=True)
    labels = ["0.05", "0.10", "0.20", "0.30"]
    keys = ["0.05", "0.10", "0.20", "0.30"]
    dbox = [by_d[k]["d_values"] for k in keys]
    axes[0].boxplot(dbox, labels=labels, medianprops={"color": BLACK, "lw": 1.6})
    axes[0].set_xlabel("α")
    axes[0].set_ylabel(r"$D_M^{\mathrm{NA}}$")
    axes[0].set_title("Common-cover distance")
    axes[0].text(-0.18, 1.08, "A", transform=axes[0].transAxes, fontsize=11, fontweight="bold")
    sbox = [by_d[k]["silhouette_delta"] for k in keys]
    axes[1].boxplot(sbox, labels=labels, medianprops={"color": ORANGE, "lw": 1.6})
    axes[1].axhline(0, color=GRAY, lw=0.8, ls="--")
    axes[1].set_xlabel("α")
    axes[1].set_ylabel("Δ Silhouette")
    axes[1].set_title("Conditional quality")
    axes[1].text(-0.18, 1.08, "B", transform=axes[1].transAxes, fontsize=11, fontweight="bold")
    clean_noise = [by_d[k]["median_noise_fraction_macro_clean"] for k in keys]
    pert_noise = [by_d[k]["median_noise_fraction_macro_perturbed"] for k in keys]
    x = np.arange(len(keys))
    axes[2].plot(x, clean_noise, marker="o", lw=1.4, color=GRAY, label="clean")
    axes[2].plot(x, pert_noise, marker="o", lw=1.6, color=ORANGE, label="perturbed")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_xlabel("α")
    axes[2].set_ylabel("Median DBSCAN noise fraction")
    axes[2].set_title("Pullback-level noise")
    axes[2].legend(frameon=False, loc="upper left")
    axes[2].text(-0.18, 1.08, "C", transform=axes[2].transAxes, fontsize=11, fontweight="bold")
    elig = [by_d[k]["n_eligible"] for k in keys]
    axes[3].bar(labels, elig, color=GREEN, width=0.6)
    axes[3].set_ylim(0, 32)
    axes[3].set_xlabel("α")
    axes[3].set_ylabel("Defined pairs / 30")
    axes[3].set_title("Defined comparisons")
    for xlab, yval in zip(labels, elig):
        axes[3].text(xlab, yval + 0.4, f"{yval}/30", ha="center", fontsize=8)
    axes[3].text(-0.18, 1.08, "D", transform=axes[3].transAxes, fontsize=11, fontweight="bold")
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"fig07_c1_perturbation_response.{suffix}", bbox_inches="tight",
                    pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return obj


def m6a2_wasserstein() -> dict:
    cells = list(csv.DictReader((ROOT / "evidence/baseline/m6a2/cell_summaries.csv").open()))
    ci = list(csv.DictReader((ROOT / "evidence/baseline/m6a2/bootstrap_intervals.csv").open()))
    per_run = list(csv.DictReader((ROOT / "evidence/baseline/m6a2/per_run_metrics.csv").open()))
    rows = []
    for row in ci:
        if row["metric_id"] != "wasserstein_extended_pd" or row["noise_condition_id"] == "sigma_0.00":
            continue
        rows.append({
            "method_id": row["method_id"],
            "noise_sigma": row["noise_sigma"],
            "n_valid": int(row["n_valid_observations"]),
            "mean": float(row["mean"]),
            "ci95_low": float(row["ci_95_low"]),
            "ci95_high": float(row["ci_95_high"]),
        })
    ordering = []
    for sigma in ("0.01", "0.03", "0.05", "0.08", "0.1"):
        c = next(r for r in rows if r["method_id"] == "conventional" and r["noise_sigma"] == sigma)
        f = next(r for r in rows if r["method_id"] == "f_mapper" and r["noise_sigma"] == sigma)
        ordering.append({
            "noise_sigma": sigma,
            "conventional_mean_below_fmapper": c["mean"] < f["mean"],
        })
    paired = []
    rng = np.random.default_rng(SEED)
    for metric in ("bottleneck_distance", "wasserstein_distance"):
        for sigma in ("0.01", "0.03", "0.05", "0.08", "0.1"):
            by_rep = {}
            for row in per_run:
                if row["noise_sigma"] != sigma or row["status"] != "success" or not row[metric]:
                    continue
                by_rep.setdefault(int(row["replication_id"]), {})[row["method_id"]] = (
                    float(row[metric]), row["perturbation_seed"]
                )
            for values in by_rep.values():
                if {"conventional", "f_mapper"}.issubset(values) and values["conventional"][1] != values["f_mapper"][1]:
                    raise RuntimeError("M6A2 paired methods do not share the perturbation seed")
            differences = np.asarray([
                values["conventional"][0] - values["f_mapper"][0]
                for _, values in sorted(by_rep.items())
                if {"conventional", "f_mapper"}.issubset(values)
            ])
            conventional = np.asarray([
                values["conventional"][0] for _, values in sorted(by_rep.items())
                if {"conventional", "f_mapper"}.issubset(values)
            ])
            f_mapper = np.asarray([
                values["f_mapper"][0] for _, values in sorted(by_rep.items())
                if {"conventional", "f_mapper"}.issubset(values)
            ])
            boot = differences[rng.integers(0, len(differences), size=(5000, len(differences)))].mean(axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            paired.append({
                "metric_id": metric,
                "noise_sigma": sigma,
                "n_pairs": int(len(differences)),
                "conventional_mean_on_pairs": float(conventional.mean()),
                "fmapper_mean_on_pairs": float(f_mapper.mean()),
                "mean_conventional_minus_fmapper": float(differences.mean()),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "bootstrap": {
                    "method": "paired percentile bootstrap of replicate-level method differences",
                    "B": 5000,
                    "seed": SEED,
                    "unit": "matched replication_id and perturbation_seed",
                },
            })
    return {
        "analysis_id": "M6A2_WASSERSTEIN_COMPANION_V4_1",
        "mapper_constructions_generated": 0,
        "role": "companion descriptive summary; not the primary M6-A2 ordering claim",
        "primary_ordering_metric": "bottleneck_extended_pd",
        "source": {
            "bootstrap": "evidence/baseline/m6a2/bootstrap_intervals.csv",
            "bootstrap_sha256": sha256(ROOT / "evidence/baseline/m6a2/bootstrap_intervals.csv"),
            "cells": "evidence/baseline/m6a2/cell_summaries.csv",
            "cells_sha256": sha256(ROOT / "evidence/baseline/m6a2/cell_summaries.csv"),
            "per_run": "evidence/baseline/m6a2/per_run_metrics.csv",
            "per_run_sha256": sha256(ROOT / "evidence/baseline/m6a2/per_run_metrics.csv"),
        },
        "rows": rows,
        "paired_method_differences": paired,
        "conventional_mean_below_fmapper_by_sigma": ordering,
        "uniform_ordering": all(x["conventional_mean_below_fmapper"] for x in ordering),
        "cell_count_check": len(cells),
    }


def c8_discordance() -> dict:
    ledger = ROOT / "evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl"
    rows = [r for r in jl("evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl")
            if r.get("arm_id") == "A_CIRCLE_GRID"]
    eligible = [r for r in rows if r.get("tier_a_eligible")]
    if len(rows) != 500 or len(eligible) != 387:
        raise SystemExit("C8 Circle grid counts drifted")
    sil = np.asarray([r["silhouette_macro"] for r in eligible], dtype=float)
    db = np.asarray([r["davies_bouldin_macro"] for r in eligible], dtype=float)
    exact = np.asarray([bool(r["tier_a_exact_agreement"]) for r in eligible])
    q75 = float(np.quantile(sil, 0.75))
    high = sil >= q75
    mismatch = ~exact
    n_high = int(high.sum())
    n_high_mismatch = int((high & mismatch).sum())
    # Spearman via rank correlation
    sil_rank = sil.argsort().argsort().astype(float)
    exact_rank = exact.astype(float)
    rho = float(np.corrcoef(sil_rank, exact_rank)[0, 1])
    match_sil = sil[exact]
    miss_sil = sil[mismatch]
    match_db = db[exact]
    miss_db = db[mismatch]
    if len(miss_sil) != 72 or len(match_sil) != 315:
        raise SystemExit("C8 72/315 split drifted")
    rng = np.random.default_rng(SEED)
    sil_diff = bootstrap_mean_diff(miss_sil, match_sil, rng)
    rng = np.random.default_rng(SEED)
    db_diff = bootstrap_mean_diff(miss_db, match_db, rng)
    return {
        "analysis_id": "C8_QUALITY_FIDELITY_DISCORDANCE_V4_1",
        "mapper_constructions_generated": 0,
        "predeclared_rules": {
            "population": "C8 Circle grid constructions with tier_a_eligible true (n=387)",
            "fidelity": "tier_a_exact_agreement",
            "quality": "silhouette_macro",
            "high_quality": "silhouette_macro >= 75th percentile of the eligible set (inclusive, within-grid)",
            "discordance_rate": "P(topology-mismatched | high-quality)",
            "rank_association": "Pearson correlation of ranks (Spearman rho) between silhouette_macro and exact_agreement",
            "not_a_universal_threshold": "The quartile is computed inside this eligible grid; it is not a fixed Silhouette cutoff for Mapper output in general",
        },
        "source": {
            "ledger": "evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl",
            "sha256": sha256(ledger),
        },
        "n_circle": 500,
        "n_eligible": 387,
        "n_exact": 315,
        "n_mismatch": 72,
        "eligible_silhouette_q75": q75,
        "n_high_quality": n_high,
        "n_high_quality_and_mismatch": n_high_mismatch,
        "discordance_rate": n_high_mismatch / n_high,
        "spearman_silhouette_vs_exact": rho,
        "group_means": {
            "mismatch_silhouette": float(miss_sil.mean()),
            "match_silhouette": float(match_sil.mean()),
            "mismatch_davies_bouldin": float(miss_db.mean()),
            "match_davies_bouldin": float(match_db.mean()),
        },
        "bootstrap_mean_difference_mismatch_minus_match": {
            "silhouette_macro": sil_diff,
            "davies_bouldin_macro": db_diff,
        },
    }


def failure_taxonomy() -> dict:
    c8 = jl("evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl")
    acc = json.loads((ROOT / "evidence/campaigns/c8/C8_RUN_ACCOUNTING.json").read_text())
    by_arm = {}
    for arm in ("A_CIRCLE_GRID", "B_TRIPOD_AXIS3_REPLICATION"):
        sub = [r for r in c8 if r.get("arm_id") == arm]
        by_arm[arm] = {
            "n": len(sub),
            "status_counts": dict(Counter(r["status"] for r in sub)),
            "tier_a_eligible": sum(bool(r.get("tier_a_eligible")) for r in sub),
            "tier_a_exact_agreement": sum(bool(r.get("tier_a_exact_agreement")) for r in sub),
        }
    iii2b = list(csv.DictReader((ROOT / "evidence/baseline/axis3_iii2/III2B_RECOVERY_SUMMARY.csv").open()))
    cells = []
    tot = Counter()
    for r in iii2b:
        rec = {
            "cell_id": r["cell_id"],
            "benchmark": r["benchmark"],
            "sample_size": int(r["sample_size"]),
            "n_planned": int(r["n_planned"]),
            "n_attempts_executed": int(r["n_attempts_executed"]),
            "n_success": int(r["n_success"]),
            "n_eligible": int(r["n_eligible"]),
            "n_failed_or_missing": int(r["n_failed_or_missing"]),
            "k_joint_recovered": int(r["k_joint_recovered"]),
        }
        cells.append(rec)
        for k in ("n_planned", "n_attempts_executed", "n_success", "n_eligible",
                  "n_failed_or_missing", "k_joint_recovered"):
            tot[k] += rec[k]
    if tot["n_planned"] != 2160 or tot["k_joint_recovered"] != 1201:
        raise SystemExit("III-2B cell totals drifted")
    return {
        "analysis_id": "FAILURE_TAXONOMY_V4_1",
        "mapper_constructions_generated": 0,
        "scope": {
            "c8": "run-level outcomes from the shipped C8 construction ledger",
            "iii2b": "cell-level only from III2B_RECOVERY_SUMMARY.csv; run-level III-2B construction journals are not reconstructed",
        },
        "source": {
            "c8_ledger_sha256": sha256(ROOT / "evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl"),
            "c8_accounting_sha256": sha256(ROOT / "evidence/campaigns/c8/C8_RUN_ACCOUNTING.json"),
            "iii2b_sha256": sha256(ROOT / "evidence/baseline/axis3_iii2/III2B_RECOVERY_SUMMARY.csv"),
        },
        "c8_run_level": {
            "planned": acc["planned_constructions"],
            "attempted": acc["attempted_constructions"],
            "status_counts": acc["status_counts"],
            "by_arm": by_arm,
        },
        "iii2b_cell_level": {
            "n_cells": len(cells),
            "totals": dict(tot),
            "cells": cells,
        },
    }


def write_tex(c8: dict, tax: dict, wass: dict) -> None:
    TEX.mkdir(parents=True, exist_ok=True)
    sil = c8["bootstrap_mean_difference_mismatch_minus_match"]["silhouette_macro"]
    db = c8["bootstrap_mean_difference_mismatch_minus_match"]["davies_bouldin_macro"]
    (TEX / "c8_discordance.tex").write_text("\n".join([
        r"% Auto-generated by generate_phase3_derived.py. Do not hand-edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{C8 Circle-grid quality--fidelity discordance among 387 eligible constructions. High quality is the within-grid top quartile of Silhouette, not a universal threshold. Bootstrap intervals compare the 72 topology-mismatched eligible constructions with the 315 exact matches.}",
        r"\label{tab:c8discord}",
        r"\small",
        r"\begin{tabular}{@{}lc@{}}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        f"Eligible / exact / mismatched & $387$ / $315$ / $72$ \\\\",
        f"Within-grid Silhouette 75th percentile & ${f6(c8['eligible_silhouette_q75'])}$ \\\\",
        f"High-quality (top quartile) & ${c8['n_high_quality']}$ \\\\",
        f"High-quality and topology-mismatched & ${c8['n_high_quality_and_mismatch']}$ \\\\",
        f"Discordance rate $P(\\text{{mismatch}}\\mid\\text{{high-quality}})$ & ${f6(c8['discordance_rate'])}$ \\\\",
        f"Spearman $\\rho$ (Silhouette vs exact agreement) & ${f6(c8['spearman_silhouette_vs_exact'])}$ \\\\",
        f"Mean Silhouette mismatch $-$ match (95\\% CI) & ${f6(sil['point_estimate'])}$ $[{f6(sil['ci95_low'])},{f6(sil['ci95_high'])}]$ \\\\",
        f"Mean Davies--Bouldin mismatch $-$ match (95\\% CI) & ${f6(db['point_estimate'])}$ $[{f6(db['ci95_low'])},{f6(db['ci95_high'])}]$ \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]))
    c8s = tax["c8_run_level"]["status_counts"]
    tot = tax["iii2b_cell_level"]["totals"]
    (TEX / "failure_taxonomy.tex").write_text("\n".join([
        r"% Auto-generated by generate_phase3_derived.py. Do not hand-edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Failure taxonomy from shipped records only. C8 is run-level. III-2B is cell-level; run-level III-2B construction journals are not reconstructed.}",
        r"\label{tab:failtax}",
        r"\small",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Campaign & Attempted & Success & Non-convergence / failed & Other typed \\",
        r"\midrule",
        f"C8 (run-level) & {tax['c8_run_level']['attempted']} & {c8s['success']} & {c8s['fcm_non_convergence']} & {c8s['coverage_gap']} coverage-gap \\\\",
        f"III-2B (cell totals) & {tot['n_attempts_executed']} & {tot['n_success']} & {tot['n_failed_or_missing']} failed/missing & {tot['k_joint_recovered']} joint recoveries \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]))
    lines = [
        r"% Auto-generated by generate_phase3_derived.py. Do not hand-edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{M6-A2 Wasserstein companion (not the primary ordering). Conventional mean bottleneck displacement is below F-Mapper at all five nonzero levels; Wasserstein means do not share that uniform order.}",
        r"\label{tab:m6a2w}",
        r"\small",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Method & $\sigma$ & $n$ & Mean $W_1$ & 95\% CI \\",
        r"\midrule",
    ]
    for method, label in (("conventional", "Conventional"), ("f_mapper", "F-Mapper")):
        sub = [r for r in wass["rows"] if r["method_id"] == method]
        for i, r in enumerate(sub):
            sig = "0.10" if r["noise_sigma"] == "0.1" else r["noise_sigma"]
            mcell = label if i == 0 else ""
            lines.append(
                f"{mcell} & ${sig}$ & {r['n_valid']} & ${f6(r['mean'])}$ & "
                f"$[{f6(r['ci95_low'])},{f6(r['ci95_high'])}]$ \\\\"
            )
        if method == "conventional":
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TEX / "m6a2_wasserstein.tex").write_text("\n".join(lines))


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    c1 = c1_dose_response()
    plot = c1.pop("_plot")
    dump("C1_FULL_DOSE_RESPONSE_V4_1.json", c1)
    wass = m6a2_wasserstein()
    dump("M6A2_WASSERSTEIN_COMPANION_V4_1.json", wass)
    c8 = c8_discordance()
    dump("C8_QUALITY_FIDELITY_DISCORDANCE_V4_1.json", c8)
    tax = failure_taxonomy()
    dump("FAILURE_TAXONOMY_V4_1.json", tax)
    write_tex(c8, tax, wass)
    print("C1 doses", list(c1["doses"]))
    print("C8 discordance_rate", c8["discordance_rate"], "rho", c8["spearman_silhouette_vs_exact"])
    print("C8 sil diff", c8["bootstrap_mean_difference_mismatch_minus_match"]["silhouette_macro"]["point_estimate"])
    print("wasserstein uniform", wass["uniform_ordering"])
    print("III2B cells", tax["iii2b_cell_level"]["n_cells"])
    _ = plot
    print("PHASE3_DERIVED_WRITTEN")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the six manuscript figures directly from released evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/three_axis_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/figures"

BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#D55E00"
GOLD = "#E69F00"
MAGENTA = "#CC79A7"
GRAY = "#6B7280"
LIGHT = "#E5E7EB"
BLACK = "#111827"

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
    "svg.fonttype": "none",
    "svg.hashsalt": "three-axis-mapper-2026",
})


def jsonl(relative: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / relative).read_text().splitlines() if line.strip()]


def csv_rows(relative: str) -> list[dict]:
    with (ROOT / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(relative: str) -> str:
    h = hashlib.sha256()
    with (ROOT / relative).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quantiles(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    return {
        "n": int(len(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
    }


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", color=BLACK)


def draw_heatmap(ax, matrix: np.ndarray):
    """Draw heatmap cells as explicit quadrilaterals for reliable PDF output."""
    nrows, ncols = matrix.shape
    image = ax.pcolormesh(
        np.arange(ncols + 1) - 0.5,
        np.arange(nrows + 1) - 0.5,
        matrix,
        cmap="viridis",
        vmin=0,
        vmax=1,
        shading="flat",
        antialiased=False,
    )
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(-0.5, nrows - 0.5)
    return image


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        if suffix == "pdf":
            metadata = {"Creator": "Three-axis Mapper release",
                        "CreationDate": None, "ModDate": None}
        elif suffix == "svg":
            metadata = {"Creator": "Three-axis Mapper release", "Date": None}
        else:
            metadata = {"Software": "Three-axis Mapper release"}
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.04,
                    facecolor="white", metadata=metadata)
    plt.close(fig)


def figure_framework() -> dict:
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    columns = [
        (0.025, BLUE, "AXIS I", "Stability under\nperturbation",
         "State object + perturbation\nState correspondence", "Observation / edge /\nfiltered-graph change"),
        (0.35, GREEN, "AXIS II", "Local cluster\nvalidation",
         "Pullback partitions\nSilhouette / Davies-Bouldin", "Coverage / noise\nConstruction status"),
        (0.675, ORANGE, "AXIS III", "Topological\nagreement",
         "Fixed domain + lens\nGraph / nerve invariants", "Reference invariants /\nextended persistence"),
    ]
    for x, color, axis, title, rule, endpoint in columns:
        box = FancyBboxPatch((x, 0.23), 0.30, 0.63,
                             boxstyle="round,pad=0.012,rounding_size=0.018",
                             linewidth=1.2, edgecolor=color, facecolor="white")
        ax.add_patch(box)
        ax.add_patch(FancyBboxPatch((x, 0.74), 0.30, 0.12,
                                    boxstyle="round,pad=0.012,rounding_size=0.018",
                                    linewidth=0, facecolor=color))
        ax.text(x + 0.15, 0.80, axis, color="white", fontweight="bold",
                ha="center", va="center", fontsize=11)
        ax.text(x + 0.15, 0.65, title, color=BLACK, fontweight="bold",
                ha="center", va="center", fontsize=9.5)
        ax.text(x + 0.15, 0.49, rule, color=BLACK, ha="center", va="center",
                fontsize=8.5, linespacing=1.35)
        ax.plot([x + 0.04, x + 0.26], [0.39, 0.39], color=LIGHT, lw=1)
        ax.text(x + 0.15, 0.31, endpoint, color=GRAY, ha="center", va="center",
                fontsize=8.2, linespacing=1.3)
    save(fig, "fig01_three_axis_framework")
    return {"type": "conceptual", "axes": ["I", "II", "III"]}


def figure_axis1_subsampling() -> dict:
    fmapper = jsonl("evidence/campaigns/c4_fmapper/C4R_CONSTRUCTION_LEDGER.jsonl")
    conventional = jsonl("evidence/campaigns/conventional_fixed_cover/CIA2_CONSTRUCTION_LEDGER.jsonl")
    ball = jsonl("evidence/campaigns/c4_ball_ensemble/C4V2_BALL_LEDGER.jsonl")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)
    fractions = [0.5, 0.8]
    datasets = [("swiss_roll_with_hole", "Swiss", BLUE),
                ("digits_1797x64_scaled16", "Digits", ORANGE)]
    summary = {"fmapper": {}, "conventional": {}, "ball": {}}

    ax = axes[0]
    for dataset, label, color in datasets:
        meds, low, high = [], [], []
        for fraction in fractions:
            vals = [r["d_common_id_distance"] for r in fmapper
                    if r["arm"] == "fixed_cover" and r["dataset_id"] == dataset
                    and r["fraction"] == fraction and r["d_common_id_distance"] is not None]
            q = quantiles(vals)
            summary["fmapper"][f"{label}_{fraction}"] = q
            meds.append(q["median"]); low.append(q["q25"]); high.append(q["q75"])
        ax.plot(fractions, meds, marker="o", lw=1.8, color=color, label=label)
        ax.fill_between(fractions, low, high, color=color, alpha=0.14, linewidth=0)
    ax.set_title("F-Mapper fixed cover")
    ax.set_xlabel("Retained fraction")
    ax.set_ylabel("Joint common-cover distance")
    ax.set_xticks(fractions, ["50%", "80%"])
    ax.legend(frameon=False, loc="upper right")
    panel_label(ax, "A")

    ax = axes[1]
    conv_ds = [("unit_circle_S1", "Circle", GREEN),
               ("swiss_roll_with_hole", "Swiss", BLUE),
               ("digits_1797x64_scaled16", "Digits", ORANGE)]
    offsets = [-0.03, 0.0, 0.03]
    label_heights = {"Circle": 0.004, "Swiss": 0.015, "Digits": 0.026}
    for (dataset, label, color), offset in zip(conv_ds, offsets):
        meds = []
        for fraction in fractions:
            vals = [r["d_common_id_distance"] for r in conventional
                    if r["arm"] == "fixed_cover" and r["dataset_id"] == dataset
                    and r["fraction"] == fraction and r["d_common_id_distance"] is not None]
            q = quantiles(vals) if vals else {"n": 0, "q25": None, "median": None, "q75": None}
            summary["conventional"][f"{label}_{fraction}"] = q
            meds.append(np.nan if not vals else q["median"])
            ax.text(fraction + offset, label_heights[label],
                    f"{len(vals)}/20", color=color, ha="center", va="bottom", fontsize=7)
        ax.plot(np.asarray(fractions) + offset, meds, marker="o", lw=1.5, color=color, label=label)
    ax.set_title("Conventional fixed cover")
    ax.set_xlabel("Retained fraction")
    ax.set_ylabel("Joint common-cover distance")
    ax.set_xticks(fractions, ["50%", "80%"])
    ax.set_ylim(-0.005, 0.10)
    ax.legend(frameon=False, loc="upper right")
    panel_label(ax, "B")

    ax = axes[2]
    vals_by_fraction = []
    for fraction in fractions:
        vals = [r["retained_edge_fraction"] for r in ball
                if r["dataset_id"] == "swiss_roll_with_hole" and r["fraction"] == fraction]
        vals_by_fraction.append(vals)
        summary["ball"][f"Swiss_{fraction}"] = quantiles(vals)
    for fraction, vals, color in zip(fractions, vals_by_fraction, [SKY, BLUE]):
        q = quantiles(vals)
        lo, hi = float(np.min(vals)), float(np.max(vals))
        # Explicit line geometry avoids a PDF-backend boxplot rendering issue.
        ax.plot([fraction, fraction], [lo, q["q25"]], color=BLACK, lw=0.9)
        ax.plot([fraction, fraction], [q["q75"], hi], color=BLACK, lw=0.9)
        ax.plot([fraction - 0.018, fraction + 0.018], [lo, lo], color=BLACK, lw=0.9)
        ax.plot([fraction - 0.018, fraction + 0.018], [hi, hi], color=BLACK, lw=0.9)
        ax.plot([fraction, fraction], [q["q25"], q["q75"]], color=color, lw=7,
                solid_capstyle="butt")
        ax.plot([fraction - 0.035, fraction + 0.035], [q["median"], q["median"]],
                color=BLACK, lw=1.5)
    ax.set_title("Ball Mapper, Swiss")
    ax.set_xlabel("Retained fraction")
    ax.set_ylabel("Retained-edge fraction")
    ax.set_xticks(fractions, ["50%", "80%"])
    ax.set_ylim(0, 1.03)
    panel_label(ax, "C")
    save(fig, "fig02_axis1_fixed_cover_subsampling")
    return summary


def figure_fmapper_reliability() -> dict:
    c3 = jsonl("evidence/campaigns/c3/C3_CONSTRUCTION_LEDGER.jsonl")
    c8 = jsonl("evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl")
    intervals = [5, 8, 10, 15, 20]
    thresholds = [0.10, 0.15, 0.20, 0.30, 0.40]
    datasets = [([r for r in c3 if r["dataset_id"] == "swiss_roll_with_hole"], "Swiss roll", "C3"),
                ([r for r in c3 if r["dataset_id"] == "digits_1797x64_scaled16"], "Digits", "C3"),
                ([r for r in c8 if r["dataset_id"] == "unit_circle"], "Circle", "C8")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    summary = {}
    for panel, (ax, (source, title, campaign)) in enumerate(zip(axes, datasets)):
        matrix = np.zeros((len(intervals), len(thresholds)))
        for i, c in enumerate(intervals):
            for j, tau in enumerate(thresholds):
                cell = [r for r in source if int(r["n_intervals"]) == c
                        and float(r["threshold"]) == tau]
                matrix[i, j] = sum(r["status"] == "success" for r in cell) / len(cell)
                summary[f"{campaign}|{title}|c={c}|tau={tau}"] = {
                    "n": len(cell), "success": sum(r["status"] == "success" for r in cell),
                    "rate": float(matrix[i, j])}
        image = draw_heatmap(ax, matrix)
        for i in range(len(intervals)):
            for j in range(len(thresholds)):
                color = "white" if matrix[i, j] < 0.20 or matrix[i, j] > 0.75 else BLACK
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color=color, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("FCM threshold τ")
        ax.set_ylabel("Number of fuzzy cover sets c")
        ax.set_xticks(range(len(thresholds)), [f"{x:.2f}" for x in thresholds])
        ax.set_yticks(range(len(intervals)), intervals)
        panel_label(ax, chr(ord("A") + panel))
    cbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    cbar.set_label("Construction success rate")
    save(fig, "fig03_fmapper_construction_reliability")
    return summary


def figure_c1_quality() -> dict:
    paired = [r for r in jsonl("evidence/campaigns/c1/C1_PAIRED_QUALITY_LEDGER.jsonl")
              if r["alpha"] == 0.05]
    comparisons = [r for r in jsonl("evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl")
                   if r["alpha"] == 0.05]
    d_by_construction = {r["construction_id"]: r["d_common_id"] for r in comparisons}
    clean = np.asarray([r["silhouette_clean"] for r in paired])
    perturbed = np.asarray([r["silhouette_perturbed"] for r in paired])
    delta = perturbed - clean
    dvals = np.asarray([d_by_construction[r["construction_id"]] for r in paired])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    ax = axes[0]
    for a, b in zip(clean, perturbed):
        ax.plot([0, 1], [a, b], color=LIGHT, lw=0.75, zorder=1)
    ax.scatter(np.zeros_like(clean), clean, color=BLUE, s=18, zorder=2, label="Clean")
    ax.scatter(np.ones_like(perturbed), perturbed, color=ORANGE, s=18, zorder=2, label="α=0.05")
    ax.plot([0, 1], [np.median(clean), np.median(perturbed)], color=BLACK, lw=2.2,
            marker="D", ms=5, zorder=3, label="Median")
    ax.set_xticks([0, 1], ["Clean", "α=0.05"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("Silhouette (macro)")
    ax.set_title("Paired Swiss quality")
    panel_label(ax, "A")
    ax = axes[1]
    ax.axhline(0, color=GRAY, lw=0.8, ls="--")
    ax.scatter(dvals, delta, color=GREEN, s=26, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.scatter([np.median(dvals)], [np.median(delta)], color=BLACK, marker="D", s=42,
               label="Median pair", zorder=3)
    ax.set_xlabel("Joint common-cover distance")
    ax.set_ylabel("Δ Silhouette (perturbed − clean)")
    ax.set_title("Common-cover change vs quality change")
    ax.legend(frameon=False, loc="lower left")
    panel_label(ax, "B")
    save(fig, "fig04_c1_swiss_quality_response")
    return {"n_pairs": len(paired), "clean": quantiles(clean), "perturbed": quantiles(perturbed),
            "delta": quantiles(delta), "d_common_id": quantiles(dvals)}


def figure_axis3_recovery() -> dict:
    rows = csv_rows("evidence/baseline/axis3_iii2/III2B_RECOVERY_SUMMARY.csv")
    c8 = jsonl("evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl")
    samples = sorted({int(r["sample_size"]) for r in rows})
    covers = sorted({(int(r["n_intervals"]), float(r["overlap_frac"])) for r in rows})
    datasets = [("unit_circle_S1", "Circle"), ("branching_tripod_Y", "Tripod")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    summary = {"conventional": {}, "fmapper_circle_grid": {}, "fmapper_primary": {}}
    for panel, (ax, (dataset, title)) in enumerate(zip(axes.flat[:2], datasets)):
        matrix = np.zeros((len(covers), len(samples)))
        for i, (n, p) in enumerate(covers):
            for j, sample in enumerate(samples):
                row = next(r for r in rows if r["benchmark"] == dataset
                           and int(r["sample_size"]) == sample
                           and int(r["n_intervals"]) == n and float(r["overlap_frac"]) == p)
                value = float(row["p_joint_attempt_level"])
                matrix[i, j] = value
                summary["conventional"][f"{dataset}|N={sample}|L={n}|p={p}"] = {
                    "n": int(row["n_attempts_executed"]),
                    "joint_recovered": int(row["k_joint_recovered"]), "rate": value}
        image = draw_heatmap(ax, matrix)
        for i in range(len(covers)):
            for j in range(len(samples)):
                color = "white" if matrix[i, j] < 0.42 or matrix[i, j] > 0.82 else BLACK
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color=color, fontsize=7.3)
        ax.set_title(title)
        ax.set_xlabel("Sample size N")
        ax.set_ylabel("Cover (intervals, overlap)")
        ax.set_xticks(range(len(samples)), samples)
        ax.set_yticks(range(len(covers)), [f"{n}, {p:.1f}" for n, p in covers])
        panel_label(ax, chr(ord("A") + panel))
    cbar = fig.colorbar(image, ax=axes[0, :], shrink=0.72, pad=0.02)
    cbar.set_label("Joint recovery proportion")

    ax = axes[1, 0]
    circle = [r for r in c8 if r["dataset_id"] == "unit_circle"]
    fm_intervals = [5, 8, 10, 15, 20]
    fm_thresholds = [.10, .15, .20, .30, .40]
    matrix = np.zeros((len(fm_intervals), len(fm_thresholds)))
    for i, c in enumerate(fm_intervals):
        for j, tau in enumerate(fm_thresholds):
            cell = [r for r in circle if r["n_intervals"] == c and r["threshold"] == tau]
            exact = sum(r["tier_a_exact_agreement"] for r in cell)
            matrix[i, j] = exact / len(cell)
            summary["fmapper_circle_grid"][f"c={c}|tau={tau}"] = {
                "attempted": len(cell), "eligible": sum(r["tier_a_eligible"] for r in cell),
                "exact": exact, "attempt_level_rate": float(matrix[i, j])}
    fm_image = draw_heatmap(ax, matrix)
    for i in range(len(fm_intervals)):
        for j in range(len(fm_thresholds)):
            color = "white" if matrix[i, j] < .42 or matrix[i, j] > .82 else BLACK
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color=color, fontsize=7.3)
    ax.set_title("F-Mapper Circle grid")
    ax.set_xlabel("FCM threshold τ")
    ax.set_ylabel("Fuzzy cover sets c")
    ax.set_xticks(range(len(fm_thresholds)), [f"{x:.2f}" for x in fm_thresholds])
    ax.set_yticks(range(len(fm_intervals)), fm_intervals)
    panel_label(ax, "C")
    cbar = fig.colorbar(fm_image, ax=ax, shrink=.72, pad=.02)
    cbar.set_label("Exact recovery / attempts")

    ax = axes[1, 1]
    circle_primary = [r for r in circle if r["n_intervals"] == 10 and r["threshold"] == .10]
    tripod = [r for r in c8 if r["dataset_id"] == "branching_tripod_y"]
    groups = [("Circle", circle_primary), ("Tripod", tripod)]
    attempted = [sum(r["tier_a_exact_agreement"] for r in group) / len(group) for _, group in groups]
    conditional = [sum(r["tier_a_exact_agreement"] for r in group)
                   / sum(r["tier_a_eligible"] for r in group) for _, group in groups]
    x = np.arange(2); width = .34
    ax.bar(x - width/2, attempted, width, color=BLUE, label="All attempts")
    ax.bar(x + width/2, conditional, width, color=GREEN, label="Conditional eligible")
    for i, (_, group) in enumerate(groups):
        exact = sum(r["tier_a_exact_agreement"] for r in group)
        eligible = sum(r["tier_a_eligible"] for r in group)
        ax.text(i - width/2, attempted[i] + .025, f"{exact}/{len(group)}", ha="center", fontsize=8)
        ax.text(i + width/2, conditional[i] + .025, f"{exact}/{eligible}", ha="center", fontsize=8)
        summary["fmapper_primary"][groups[i][0].lower()] = {
            "attempted": len(group), "eligible": eligible, "exact": exact,
            "attempt_level_rate": attempted[i], "conditional_rate": conditional[i]}
    ax.set_xticks(x, [name for name, _ in groups])
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Proportion matching reference invariants")
    ax.set_title("F-Mapper selected parameter cell")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "D")
    save(fig, "fig05_conventional_axis3_recovery_surface")
    return summary


def figure_interactions() -> dict:
    track = jsonl("evidence/baseline/track_c/IB5_INTERACTIONS_COMPARISON_LEDGER.jsonl")
    c7 = json.loads((ROOT / "evidence/campaigns/c7/C7_INTERACTION_FIT.json").read_text())
    c6a = jsonl("evidence/campaigns/c6a/C6A_LEDGER.jsonl")
    c6b = jsonl("evidence/campaigns/c6b/C6B_LEDGER.jsonl")
    c9 = jsonl("evidence/campaigns/c9/C9_LEDGER.jsonl")
    alphas = [0.0, 0.05, 0.10, 0.20, 0.30]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
    summary = {"c7": {}, "c6a": {}, "c6b": {}, "c9": {}}
    ax = axes[0, 0]
    eps_levels = [.08, .15, .25]
    for n, color in zip((6, 10, 15, 20), (SKY, BLUE, GREEN, ORANGE)):
        means = [float(np.mean([r["d_common_id"] for r in track
                                if r["n_intervals"] == n and r["eps"] == eps]))
                 for eps in eps_levels]
        summary["c7"][f"n={n}"] = {f"eps={eps}": value for eps, value in zip(eps_levels, means)}
        ax.plot(eps_levels, means, marker="o", lw=1.6, color=color, label=f"n={n}")
    interval = c7["beta_12"]["primary_interval"]
    summary["c7"]["beta_12"] = c7["beta_12"]
    ax.set_xlabel("DBSCAN radius ε")
    ax.set_ylabel("Mean joint common-cover distance")
    ax.set_title("Circle: cover × clustering")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.text(.03, .96, f"β₁₂={c7['beta_12']['point_estimate']:.4f}\n95% [{interval['lo_2.5']:.4f}, {interval['hi_97.5']:.4f}]",
            transform=ax.transAxes, va="top", fontsize=7.5, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78, "pad": 1.2})
    panel_label(ax, "A")

    ax = axes[0, 1]
    med, low, high = [], [], []
    for alpha in alphas:
        vals = [r["n_outside_frozen_cover"] for r in c6a if r["alpha"] == alpha]
        q = quantiles(vals); summary["c6a"][str(alpha)] = q
        med.append(q["median"]); low.append(q["q25"]); high.append(q["q75"])
    ax.plot(alphas, med, color=ORANGE, marker="o", lw=1.8)
    ax.fill_between(alphas, low, high, color=ORANGE, alpha=0.16, linewidth=0)
    ax.set_xlabel("Noise dose α")
    ax.set_ylabel("Observations outside fixed cover")
    ax.set_title("Conventional: coverage loss")
    ax.text(0.03, 0.93, "0/20 primary pairs measurable", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=ORANGE, fontweight="bold")
    panel_label(ax, "B")
    ax = axes[1, 0]
    for k, color in [(1.0, BLUE), (1.5, GREEN), (2.0, MAGENTA)]:
        med, low, high = [], [], []
        for alpha in alphas:
            vals = [r["edge_jaccard_distance"] for r in c6b
                    if r["alpha"] == alpha and r["radius_multiplier"] == k]
            q = quantiles(vals); summary["c6b"][f"alpha={alpha}|k={k}"] = q
            med.append(q["median"]); low.append(q["q25"]); high.append(q["q75"])
        ax.plot(alphas, med, color=color, marker="o", lw=1.7, label=f"radius × {k:g}")
        ax.fill_between(alphas, low, high, color=color, alpha=0.10, linewidth=0)
    ax.set_xlabel("Noise dose α")
    ax.set_ylabel("Edge-Jaccard distance")
    ax.set_title("Ball Mapper: fixed landmarks")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.98, 0.05, "DiD 95% CI spans 0", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=BLACK, fontweight="bold")
    panel_label(ax, "C")

    ax = axes[1, 1]
    for k, color in [(1.0, BLUE), (1.5, GREEN), (2.0, MAGENTA)]:
        fractions = []
        for alpha in alphas[1:]:
            vals = [r["edge_jaccard_distance"] for r in c9
                    if r["alpha"] == alpha and r["radius_multiplier"] == k]
            value = sum(v != 0.0 for v in vals) / len(vals)
            fractions.append(value)
            summary["c9"][f"alpha={alpha}|k={k}"] = {"n": len(vals), "fraction_nonzero": value}
        ax.plot(alphas[1:], fractions, marker="o", lw=1.7, color=color, label=f"radius × {k:g}")
    ax.set_xlabel("Noise dose α")
    ax.set_ylabel("Fraction with nonzero edge change")
    ax.set_ylim(-.02, 1.0)
    ax.set_title("Ball Circle: saturation diagnostic")
    ax.legend(frameon=False, loc="upper left")
    ax.text(.98, .95, "k=2: 0/80 nonzero\nprimary contrast uninformative", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=MAGENTA, fontweight="bold")
    panel_label(ax, "D")
    save(fig, "fig06_noise_by_structure_interactions")
    return summary


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    data = {
        "figure_01": figure_framework(),
        "figure_02": figure_axis1_subsampling(),
        "figure_03": figure_fmapper_reliability(),
        "figure_04": figure_c1_quality(),
        "figure_05": figure_axis3_recovery(),
        "figure_06": figure_interactions(),
        "source_sha256": {
            rel: sha256(rel) for rel in [
                "evidence/campaigns/c4_fmapper/C4R_CONSTRUCTION_LEDGER.jsonl",
                "evidence/campaigns/conventional_fixed_cover/CIA2_CONSTRUCTION_LEDGER.jsonl",
                "evidence/campaigns/c4_ball_ensemble/C4V2_BALL_LEDGER.jsonl",
                "evidence/campaigns/c3/C3_CONSTRUCTION_LEDGER.jsonl",
                "evidence/campaigns/c7/C7_INTERACTION_FIT.json",
                "evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl",
                "evidence/campaigns/c9/C9_LEDGER.jsonl",
                "evidence/campaigns/c1/C1_PAIRED_QUALITY_LEDGER.jsonl",
                "evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl",
                "evidence/baseline/axis3_iii2/III2B_RECOVERY_SUMMARY.csv",
                "evidence/campaigns/c6a/C6A_LEDGER.jsonl",
                "evidence/campaigns/c6b/C6B_LEDGER.jsonl",
            ]
        },
    }
    (OUT / "FIGURE_DATA.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print("generated 6 figures in PDF, SVG, and PNG plus FIGURE_DATA.json")


if __name__ == "__main__":
    main()

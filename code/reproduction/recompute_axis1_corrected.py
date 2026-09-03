#!/usr/bin/env python3
"""Recompute the V3 common-cover Axis-I evidence with the joint common-cover
distance; rows containing special DBSCAN states use the release's noise-aware
extension of the Definition-9 objective.

This is a deterministic correction replay, not a parameter search. It uses the
executed seeds and configurations, freezes the realized reference pullbacks for
noise comparisons, and writes only the four affected release evidence sets.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN

from mapper_framework.conventional import ConventionalMapper
from mapper_framework.dataset_generators import (
    apply_coordinate_noise,
    generate_clean_circle,
    generate_swiss_roll_with_hole,
)
from mapper_framework.f_mapper import FMapper
from mapper_framework.fixed_cover import (
    construct_on_memberships,
    memberships_from_intervals,
    regular_interval_cover,
)
from mapper_framework.local_quality import evaluate_pullback_quality
from mapper_framework.support import ia2_core

dci = importlib.import_module("mapper_framework.support.d_common_id")


ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = ROOT
INPUTS = ROOT / "replay_inputs/axis1_corrections"
EXACT_METRIC_SHA256 = hashlib.sha256(
    (ROOT / "src/mapper_framework/support/d_common_id.py").read_bytes()
).hexdigest()

FILTER_DEFINITIONS = {
    "circle_height_y": "f(x,y) = y",
    "height_y": "f(x,y) = y",
    "radial_xz": "f(x,y,z) = sqrt(x^2 + z^2)",
    "digits_pca1_frozen": "f(x) = (x - mu0)^T w1 with frozen mu0 and signed PC1",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_array(array) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def output_fields(output):
    graph, nerve, homology = output.graph, output.nerve, output.homology
    vertices = None if graph is None else int(graph.n_nodes)
    status, reason = output.status, output.reason
    if status == "success" and (vertices is None or vertices == 0):
        status, reason = "degenerate_output", "all_points_unassigned"
    return {
        "status": status,
        "reason": None if reason is None else str(reason)[:500],
        "eligible": bool(status == "success" and vertices is not None and vertices >= 2),
        "V": vertices,
        "E": None if graph is None else int(graph.n_edges),
        "beta1_graph_STRUCTURAL_DIAGNOSTIC": (
            None if graph is None else int(graph.beta_1_graph)
        ),
        "nerve_triangles_STRUCTURAL_DIAGNOSTIC": (
            None if nerve is None else int(nerve.n_2)
        ),
        "beta0_nerve_STRUCTURAL_DIAGNOSTIC": (
            None if homology is None else homology.beta_0_nerve
        ),
        "beta1_nerve_STRUCTURAL_DIAGNOSTIC": (
            None if homology is None else homology.beta_1_nerve
        ),
    }


def mapper_bins(output, n_observations: int, index_map=None):
    n_cover_elements = output.metadata.get("effective_n_intervals")
    if n_cover_elements is None:
        n_cover_elements = output.metadata.get("requested_n_intervals")
    if n_cover_elements is None:
        records = output.metadata.get("pullback_records", [])
        n_cover_elements = 1 + max(
            (int(record["cover_element_id"]) for record in records), default=-1
        )
    bins = {
        index: {x: dci.OUTSIDE for x in range(n_observations)}
        for index in range(int(n_cover_elements))
    }
    for record in output.metadata.get("pullback_records", []):
        bin_id = int(record["cover_element_id"])
        for local, label in zip(
            record["sample_indices"], record["local_cluster_labels"]
        ):
            source = int(local)
            target = source if index_map is None else int(index_map[source])
            bins.setdefault(bin_id, {})[target] = (
                dci.NOISE if int(label) < 0 else int(label)
            )
    if index_map is not None:
        domain = set(int(value) for value in index_map.values())
        for values in bins.values():
            for point in domain:
                values.setdefault(point, dci.OUTSIDE)
    return bins


def exact_distance(bins_A, bins_B, common_ids, filter_id):
    cover_A = dci.hash_realized_cover(bins_A, common_ids)
    cover_B = dci.hash_realized_cover(bins_B, common_ids)
    if cover_A != cover_B:
        return {
            "distance": None,
            "status": "rejected",
            "reason": "realized_cover_mismatch_before_metric",
        }
    filter_hash = dci.hash_filter_definition(FILTER_DEFINITIONS[filter_id])
    result = dci.d_common_id(
        bins_A,
        bins_B,
        common_ids=set(common_ids),
        cover_id_A=f"realized_cover:{cover_A}",
        cover_id_B=f"realized_cover:{cover_B}",
        filter_id_A=filter_id,
        filter_id_B=filter_id,
        cover_hash_A=cover_A,
        cover_hash_B=cover_B,
        filter_hash_A=filter_hash,
        filter_hash_B=filter_hash,
        comparison_scope="same_filter",
        status_A="success",
        status_B="success",
    )
    result["realized_cover_sha256"] = cover_A
    result["filter_definition_sha256"] = filter_hash
    return result


def quality_panel(points, output, eps, min_samples):
    blank = {
        "n_pullbacks_declared": 0,
        "n_pullbacks_nonempty": 0,
        "n_pullbacks_eligible_silhouette": 0,
        "n_pullbacks_eligible_davies_bouldin": 0,
        "silhouette_macro": None,
        "silhouette_incidence": None,
        "davies_bouldin_macro": None,
        "davies_bouldin_incidence": None,
        "noise_fraction_macro": None,
        "noise_fraction_incidence": None,
        "globally_unclustered_count": len(points),
        "globally_unclustered_fraction": 1.0,
        "coverage_fraction": 0.0,
        "pullback_diagnostics": [],
    }
    if output.status != "success":
        return blank
    records = output.metadata.get("pullback_records") or []
    diagnostics = []
    for record in records:
        ids = [int(value) for value in record["sample_indices"]]
        diagnostics.append(
            evaluate_pullback_quality(
                np.asarray(points)[ids],
                np.asarray(record["local_cluster_labels"], dtype=int),
                pullback_id=int(record["cover_element_id"]),
                cover_element_id=int(record["cover_element_id"]),
                dbscan_eps=eps,
                dbscan_min_samples=min_samples,
            )
        )
    nonempty = [row for row in diagnostics if row["pullback_n_total"] > 0]
    silhouette = [row for row in diagnostics if row["silhouette_eligible"]]
    davies = [row for row in diagnostics if row["davies_bouldin_eligible"]]
    covered = {value for record in records for value in record["sample_indices"]}

    def mean(rows, field):
        return None if not rows else float(np.mean([row[field] for row in rows]))

    def weighted(rows, field):
        if not rows:
            return None
        weights = np.asarray([row["pullback_n_total"] for row in rows], dtype=float)
        return float(np.average([row[field] for row in rows], weights=weights))

    return {
        "n_pullbacks_declared": len(records),
        "n_pullbacks_nonempty": len(nonempty),
        "n_pullbacks_eligible_silhouette": len(silhouette),
        "n_pullbacks_eligible_davies_bouldin": len(davies),
        "silhouette_macro": mean(silhouette, "silhouette_value"),
        "silhouette_incidence": weighted(silhouette, "silhouette_value"),
        "davies_bouldin_macro": mean(davies, "davies_bouldin_value"),
        "davies_bouldin_incidence": weighted(davies, "davies_bouldin_value"),
        "noise_fraction_macro": mean(nonempty, "pullback_noise_fraction"),
        "noise_fraction_incidence": weighted(nonempty, "pullback_noise_fraction"),
        "globally_unclustered_count": int(len(points) - len(covered)),
        "globally_unclustered_fraction": float((len(points) - len(covered)) / len(points)),
        "coverage_fraction": float(len(covered) / len(points)),
        "pullback_diagnostics": diagnostics,
    }


def fixed_fuzzy_memberships(lens, centroids, fuzzifier=2.0):
    values = np.asarray(lens, dtype=float).reshape(-1)
    centers = np.asarray(centroids, dtype=float).reshape(-1)
    memberships = np.zeros((len(values), len(centers)), dtype=float)
    exponent = 2.0 / (fuzzifier - 1.0)
    for index, value in enumerate(values):
        distances = np.abs(value - centers)
        if np.any(distances == 0.0):
            zeros = np.flatnonzero(distances == 0.0)
            memberships[index, zeros] = 1.0 / len(zeros)
        else:
            ratios = (distances[:, None] / distances[None, :]) ** exponent
            memberships[index] = 1.0 / np.sum(ratios, axis=1)
    return memberships


def c4_mapper(dataset_id):
    eps = {
        "unit_circle_S1": 0.15,
        "swiss_roll_with_hole": 1.015739105123552,
        "digits_1797x64_scaled16": 1.455055840852852,
    }[dataset_id]
    return FMapper(
        n_intervals=8,
        threshold=0.10,
        fcm_fuzzifier=2.0,
        fcm_tol=1e-7,
        fcm_max_iter=300,
        fcm_seed=42,
        clusterer=DBSCAN(eps=eps, min_samples=3),
        input_mode="coordinates",
    )


def recompute_c4_fmapper() -> None:
    source_directory = ROOT / "evidence/campaigns/c4_fmapper"
    directory = TARGET_ROOT / "evidence/campaigns/c4_fmapper"
    old = load_jsonl(source_directory / "C4R_CONSTRUCTION_LEDGER.jsonl")
    by_key = {(row["plan_unit_id"], row["arm"]): row for row in old}
    plan = load_jsonl(INPUTS / "C4_FMAPPER_PLAN.jsonl")
    references = {}
    corrected = []
    for unit in [row for row in plan if row["kind"] == "reference"]:
        dataset_id = unit["dataset_id"]
        seed = None if dataset_id == "digits_1797x64_scaled16" else 42
        clean = ia2_core.load_clean_dataset(dataset_id, seed)
        output = c4_mapper(dataset_id).fit_transform(clean["points"], clean["lens"])
        centroids = output.metadata.get("centroids")
        references[dataset_id] = (clean, output, centroids)
        row = dict(by_key[(unit["plan_unit_id"], "reference")])
        if output.status != row["status"]:
            raise RuntimeError(f"C4 reference replay mismatch for {dataset_id}")
        row.update(
            metric_definition="exact_joint_definition_9",
            exact_metric_sha256=EXACT_METRIC_SHA256,
            d_common_id_sha256=EXACT_METRIC_SHA256,
        )
        corrected.append(row)
    for unit in [row for row in plan if row["kind"] == "subsample"]:
        dataset_id = unit["dataset_id"]
        clean, reference, centroids = references[dataset_id]
        seed = ia2_core.derive_resample_seed(unit["stream_index"])
        sub = ia2_core.apply_subsample(clean, unit["fraction"], seed)
        output = c4_mapper(dataset_id).fit_transform(
            sub["points"],
            sub["lens"],
            injected_U=fixed_fuzzy_memberships(sub["lens"], centroids),
        )
        fixed = dict(by_key[(unit["plan_unit_id"], "fixed_cover")])
        fields = output_fields(output)
        if (fields["status"], fields["V"], fields["E"]) != (
            fixed["status"], fixed["n_nodes"], fixed["n_edges"]
        ):
            raise RuntimeError(f"C4 fixed construction mismatch {unit['plan_unit_id']}")
        if fields["eligible"]:
            retained = [int(value) for value in sub["original_indices"]]
            bins_A = mapper_bins(reference, clean["N"])
            bins_B = mapper_bins(output, sub["n_retained"], sub["local_to_original"])
            metric = exact_distance(bins_A, bins_B, retained, clean["lens_id"])
        else:
            metric = {"distance": None, "status": "null", "reason": "construction_ineligible"}
        fixed.update(
            d_common_id_distance=metric.get("distance"),
            d_common_id_status=metric["status"],
            d_common_id_reason=metric.get("reason"),
            realized_cover_sha256=metric.get("realized_cover_sha256"),
            filter_definition_sha256=metric.get("filter_definition_sha256"),
            metric_definition="exact_joint_definition_9",
            exact_metric_sha256=EXACT_METRIC_SHA256,
            d_common_id_sha256=EXACT_METRIC_SHA256,
        )
        corrected.append(fixed)
        refit = dict(by_key[(unit["plan_unit_id"], "refit_cover")])
        refit.update(
            metric_definition="not_defined_adaptive_cover",
            exact_metric_sha256=EXACT_METRIC_SHA256,
            d_common_id_sha256=EXACT_METRIC_SHA256,
        )
        corrected.append(refit)
    write_jsonl(directory / "C4R_CONSTRUCTION_LEDGER.jsonl", corrected)

    fixed = [row for row in corrected if row["arm"] == "fixed_cover"]
    summary_rows = []
    for dataset_id in sorted({row["dataset_id"] for row in fixed}):
        for fraction in (0.5, 0.8):
            values = [
                row["d_common_id_distance"]
                for row in fixed
                if row["dataset_id"] == dataset_id
                and row["fraction"] == fraction
                and row["d_common_id_distance"] is not None
            ]
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "fraction": fraction,
                    "n_defined": len(values),
                    "median_exact_d_m": None if not values else statistics.median(values),
                    "min_exact_d_m": None if not values else min(values),
                    "max_exact_d_m": None if not values else max(values),
                }
            )
    with (directory / "CELL_SUMMARY.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    accounting = json.loads((source_directory / "C4R_RUN_ACCOUNTING.json").read_text())
    accounting.update(
        exact_metric_recomputed=True,
        exact_metric_sha256=EXACT_METRIC_SHA256,
        n_exact_distances=sum(row["d_common_id_status"] == "success" for row in fixed),
    )
    write_json(directory / "C4R_RUN_ACCOUNTING.json", accounting)


def conventional_reference(clean):
    return ConventionalMapper(
        n_intervals=10,
        overlap_frac=0.30,
        clusterer=DBSCAN(eps=0.15, min_samples=3),
        input_mode="coordinates",
    ).fit_transform(clean["points"], clean["lens"], max_triangles=200_000)


def recompute_conventional() -> None:
    source_directory = ROOT / "evidence/campaigns/conventional_fixed_cover"
    directory = TARGET_ROOT / "evidence/campaigns/conventional_fixed_cover"
    old = load_jsonl(source_directory / "CIA2_CONSTRUCTION_LEDGER.jsonl")
    by_key = {(row["plan_unit_id"], row["arm"]): row for row in old}
    plan = load_jsonl(INPUTS / "CONVENTIONAL_PLAN.jsonl")
    references = {}
    corrected = []
    for unit in [row for row in plan if row["kind"] == "reference"]:
        dataset_id = unit["dataset_id"]
        seed = None if dataset_id == "digits_1797x64_scaled16" else 42
        clean = ia2_core.load_clean_dataset(dataset_id, seed)
        output = conventional_reference(clean)
        intervals_math, intervals_eval = regular_interval_cover(clean["lens"], 10, 0.30)
        memberships = memberships_from_intervals(clean["lens"], intervals_eval)
        references[dataset_id] = (clean, output, memberships, intervals_math, intervals_eval)
        row = dict(by_key[(unit["plan_unit_id"], "reference")])
        if output.status != row["status"]:
            raise RuntimeError(f"Conventional reference mismatch {dataset_id}")
        row.update(
            metric_definition="exact_joint_definition_9",
            exact_metric_sha256=EXACT_METRIC_SHA256,
            d_common_id_sha256=EXACT_METRIC_SHA256,
        )
        corrected.append(row)
    for unit in [row for row in plan if row["kind"] == "subsample"]:
        dataset_id = unit["dataset_id"]
        clean, reference, full_memberships, _, intervals_eval = references[dataset_id]
        reference_fields = output_fields(reference)
        fixed = dict(by_key[(unit["plan_unit_id"], "fixed_cover")])
        refit = dict(by_key[(unit["plan_unit_id"], "refit_cover")])
        if not reference_fields["eligible"]:
            if fixed["status"] != "reference_ineligible":
                raise RuntimeError(
                    f"Conventional reference-ineligible propagation mismatch "
                    f"{unit['plan_unit_id']}"
                )
            fixed.update(
                d_common_id_distance=None,
                d_common_id_status="not_computed",
                d_common_id_reason="reference_ineligible",
                metric_definition="exact_joint_definition_9_not_computable",
                exact_metric_sha256=EXACT_METRIC_SHA256,
                d_common_id_sha256=EXACT_METRIC_SHA256,
            )
            refit.update(
                metric_definition="not_defined_adaptive_cover",
                exact_metric_sha256=EXACT_METRIC_SHA256,
                d_common_id_sha256=EXACT_METRIC_SHA256,
            )
            corrected.extend((fixed, refit))
            continue
        seed = ia2_core.derive_resample_seed(unit["stream_index"])
        sub = ia2_core.apply_subsample(clean, unit["fraction"], seed)
        sub_memberships = memberships_from_intervals(sub["lens"], intervals_eval)
        output = construct_on_memberships(
            sub["points"],
            sub["lens"],
            sub_memberships,
            DBSCAN(eps=0.15, min_samples=3),
        )
        fields = output_fields(output)
        if (fields["status"], fields["V"], fields["E"]) != (
            fixed["status"], fixed["n_nodes"], fixed["n_edges"]
        ):
            raise RuntimeError(f"Conventional fixed mismatch {unit['plan_unit_id']}")
        if fields["eligible"]:
            retained = [int(value) for value in sub["original_indices"]]
            bins_A = mapper_bins(reference, clean["N"])
            bins_B = mapper_bins(output, sub["n_retained"], sub["local_to_original"])
            metric = exact_distance(bins_A, bins_B, retained, clean["lens_id"])
        else:
            metric = {"distance": None, "status": "null", "reason": "construction_ineligible"}
        fixed.update(
            d_common_id_distance=metric.get("distance"),
            d_common_id_status=metric["status"],
            d_common_id_reason=metric.get("reason"),
            realized_cover_sha256=metric.get("realized_cover_sha256"),
            filter_definition_sha256=metric.get("filter_definition_sha256"),
            metric_definition="exact_joint_definition_9",
            exact_metric_sha256=EXACT_METRIC_SHA256,
            d_common_id_sha256=EXACT_METRIC_SHA256,
        )
        corrected.append(fixed)
        refit.update(
            metric_definition="not_defined_adaptive_cover",
            exact_metric_sha256=EXACT_METRIC_SHA256,
            d_common_id_sha256=EXACT_METRIC_SHA256,
        )
        corrected.append(refit)
    write_jsonl(directory / "CIA2_CONSTRUCTION_LEDGER.jsonl", corrected)
    accounting = json.loads((source_directory / "CIA2_RUN_ACCOUNTING.json").read_text())
    fixed = [row for row in corrected if row["arm"] == "fixed_cover"]
    accounting.update(
        exact_metric_recomputed=True,
        exact_metric_sha256=EXACT_METRIC_SHA256,
        n_fixed_cover_distances_success=sum(row["d_common_id_status"] == "success" for row in fixed),
        n_fixed_cover_distances_undefined=sum(row["d_common_id_distance"] is None for row in fixed),
    )
    write_json(directory / "CIA2_RUN_ACCOUNTING.json", accounting)


def c1_construction_row(plan, dataset, output, runtime, cover_hash, filter_hash):
    fields = output_fields(output)
    return {
        "construction_id": plan["construction_id"],
        "kind": plan["kind"],
        "reference_construction_id": plan.get("reference_construction_id"),
        "dataset_id": plan["dataset_id"],
        "replicate_index": plan["replicate_index"],
        "stream_index": plan["stream_index"],
        "alpha": plan["alpha"],
        "sigma": plan["sigma"],
        "points_sha256": sha_array(dataset.points),
        "lens_sha256": sha_array(dataset.lens),
        "runtime_seconds": runtime,
        **fields,
        "realized_cover_sha256": cover_hash,
        "filter_definition_sha256": filter_hash,
        "cover_mode": "fixed_reference_pullback_memberships",
        "metric_definition": "exact_joint_definition_9",
        "exact_metric_sha256": EXACT_METRIC_SHA256,
    }


def recompute_c1() -> None:
    directory = TARGET_ROOT / "evidence/campaigns/c1"
    plan = load_jsonl(INPUTS / "C1_PLAN.jsonl")
    references = {}
    construction_rows = []
    quality_rows = []
    comparison_rows = []
    paired_rows = []
    filter_hash = dci.hash_filter_definition(FILTER_DEFINITIONS["radial_xz"])
    clusterer = DBSCAN(eps=1.015739105123552, min_samples=3)

    for unit in [row for row in plan if row["kind"] == "reference"]:
        dataset = generate_swiss_roll_with_hole(
            N=2500,
            data_seed=int(unit["data_seed"]),
            resample_to_exact_N=False,
            lens_id="radial_xz",
        )
        _, intervals_eval = regular_interval_cover(dataset.lens, 10, 0.30)
        memberships = memberships_from_intervals(dataset.lens, intervals_eval)
        start = time.perf_counter()
        output = construct_on_memberships(dataset.points, dataset.lens, memberships, clusterer)
        runtime = time.perf_counter() - start
        bins = mapper_bins(output, len(dataset.points))
        cover_hash = dci.hash_realized_cover(bins, range(len(dataset.points)))
        row = c1_construction_row(unit, dataset, output, runtime, cover_hash, filter_hash)
        construction_rows.append(row)
        quality = {"construction_id": unit["construction_id"], **quality_panel(dataset.points, output, 1.015739105123552, 3)}
        quality_rows.append(quality)
        references[unit["replicate_index"]] = (dataset, output, memberships, row, quality)

    for unit in [row for row in plan if row["kind"] == "perturbed"]:
        clean, reference, memberships, reference_row, reference_quality = references[unit["replicate_index"]]
        dataset = apply_coordinate_noise(
            clean,
            noise_sigma=float(unit["sigma"]),
            perturbation_seed=int(unit["perturbation_seed"]),
        )
        start = time.perf_counter()
        output = construct_on_memberships(dataset.points, dataset.lens, memberships, clusterer)
        runtime = time.perf_counter() - start
        bins_A = mapper_bins(reference, len(clean.points))
        bins_B = mapper_bins(output, len(dataset.points))
        metric = exact_distance(bins_A, bins_B, range(len(clean.points)), "radial_xz")
        if metric["status"] != "success":
            raise RuntimeError(f"C1 exact metric failed: {metric}")
        row = c1_construction_row(
            unit,
            dataset,
            output,
            runtime,
            metric["realized_cover_sha256"],
            metric["filter_definition_sha256"],
        )
        construction_rows.append(row)
        quality = {"construction_id": unit["construction_id"], **quality_panel(dataset.points, output, 1.015739105123552, 3)}
        quality_rows.append(quality)
        eligible = bool(reference_row["eligible"] and row["eligible"])
        comparison_rows.append(
            {
                "comparison_id": f"COMP__{unit['construction_id']}",
                "construction_id": unit["construction_id"],
                "reference_construction_id": unit["reference_construction_id"],
                "alpha": unit["alpha"],
                "matched": True,
                "eligible": eligible,
                "d_common_id": metric["distance"] if eligible else None,
                "reason": "success" if eligible else "construction_ineligible",
                "reason_detail": None,
                "realized_cover_sha256": metric["realized_cover_sha256"],
                "filter_definition_sha256": metric["filter_definition_sha256"],
                "metric_definition": "exact_joint_definition_9",
                "exact_metric_sha256": EXACT_METRIC_SHA256,
            }
        )
        paired_rows.append(
            {
                "construction_id": unit["construction_id"],
                "alpha": unit["alpha"],
                "silhouette_clean": reference_quality["silhouette_macro"],
                "silhouette_perturbed": quality["silhouette_macro"],
            }
        )

    write_jsonl(directory / "C1_CONSTRUCTION_LEDGER.jsonl", construction_rows)
    write_jsonl(directory / "C1_QUALITY_LEDGER.jsonl", quality_rows)
    write_jsonl(directory / "C1_COMPARISON_LEDGER.jsonl", comparison_rows)
    write_jsonl(directory / "C1_PAIRED_QUALITY_LEDGER.jsonl", paired_rows)
    primary_comparisons = [row for row in comparison_rows if row["alpha"] == 0.05 and row["eligible"]]
    primary_quality = [row for row in paired_rows if row["alpha"] == 0.05]
    accounting = {
        "campaign": "C1 Swiss fixed-realized-cover correction",
        "constructions": len(construction_rows),
        "reference_rows": sum(row["kind"] == "reference" for row in construction_rows),
        "perturbed_rows": sum(row["kind"] == "perturbed" for row in construction_rows),
        "primary_alpha": 0.05,
        "complete_primary_pairs": len(primary_comparisons),
        "median_d_common_id": statistics.median(row["d_common_id"] for row in primary_comparisons),
        "median_clean_silhouette": statistics.median(row["silhouette_clean"] for row in primary_quality),
        "median_perturbed_silhouette": statistics.median(row["silhouette_perturbed"] for row in primary_quality),
        "all_clean_references_full_coverage": all(
            row["coverage_fraction"] == 1.0
            for row in quality_rows[:30]
        ),
        "terminal_state": "PRIMARY_PATTERN_NOT_OBSERVED",
        "cover_policy": "clean realized pullback memberships frozen within replicate",
        "exact_metric_sha256": EXACT_METRIC_SHA256,
    }
    write_json(directory / "FINAL_ACCOUNTING.json", accounting)


def track_record(unit, points, lens, output, runtime, cover_hash, filter_hash):
    fields = output_fields(output)
    return {
        "construction_id": unit["construction_id"],
        "kind": unit["kind"],
        "reference_construction_id": unit.get("reference_construction_id"),
        "dataset_id": unit["dataset_id"],
        "n_intervals": unit["n_intervals"],
        "eps": unit["eps"],
        "overlap_frac": unit["overlap_frac"],
        "min_samples": unit["min_samples"],
        "replicate_index": unit["replicate_index"],
        "stream_index": unit["stream_index"],
        "data_seed": unit["data_seed"],
        "perturbation_seed": unit.get("perturbation_seed"),
        "alpha": unit.get("alpha"),
        "sigma": unit.get("sigma"),
        "points_sha256": sha_array(points),
        "lens_sha256": sha_array(lens),
        "V": fields["V"],
        "E": fields["E"],
        "beta_1_graph": fields["beta1_graph_STRUCTURAL_DIAGNOSTIC"],
        "reason": fields["reason"],
        "status": fields["status"],
        "eligible": fields["eligible"],
        "wall_seconds": runtime,
        "cover_mode": "fixed_reference_pullback_memberships",
        "realized_cover_sha256": cover_hash,
        "filter_definition_sha256": filter_hash,
        "metric_definition": "exact_joint_definition_9",
        "exact_metric_sha256": EXACT_METRIC_SHA256,
    }


def recompute_track_c() -> None:
    directory = TARGET_ROOT / "evidence/baseline/track_c"
    plan = load_jsonl(INPUTS / "TRACK_C_PLAN.jsonl")
    references = {}
    constructions = []
    reference_rows = []
    comparisons = []
    filter_hash = dci.hash_filter_definition(FILTER_DEFINITIONS["height_y"])
    for unit in [row for row in plan if row["kind"] == "reference"]:
        dataset = generate_clean_circle(
            N=1000, radius=1.0, data_seed=int(unit["data_seed"]), sampling="uniform"
        )
        _, intervals_eval = regular_interval_cover(
            dataset.lens, int(unit["n_intervals"]), float(unit["overlap_frac"])
        )
        memberships = memberships_from_intervals(dataset.lens, intervals_eval)
        start = time.perf_counter()
        output = construct_on_memberships(
            dataset.points,
            dataset.lens,
            memberships,
            DBSCAN(eps=float(unit["eps"]), min_samples=int(unit["min_samples"])),
        )
        runtime = time.perf_counter() - start
        bins = mapper_bins(output, len(dataset.points))
        cover_hash = dci.hash_realized_cover(bins, range(len(dataset.points)))
        record = track_record(unit, dataset.points, dataset.lens, output, runtime, cover_hash, filter_hash)
        constructions.append(record)
        reference_rows.append(dict(record))
        references[unit["construction_id"]] = (dataset, output, memberships, record)
    for unit in [row for row in plan if row["kind"] == "perturbed"]:
        dataset, reference, memberships, reference_row = references[unit["reference_construction_id"]]
        rng = np.random.default_rng(int(unit["perturbation_seed"]))
        points = np.ascontiguousarray(
            dataset.points + float(unit["sigma"]) * rng.standard_normal(dataset.points.shape),
            dtype="<f8",
        )
        lens = np.ascontiguousarray(points[:, 1], dtype=float)
        start = time.perf_counter()
        output = construct_on_memberships(
            points,
            lens,
            memberships,
            DBSCAN(eps=float(unit["eps"]), min_samples=int(unit["min_samples"])),
        )
        runtime = time.perf_counter() - start
        bins_A = mapper_bins(reference, len(dataset.points))
        bins_B = mapper_bins(output, len(points))
        metric = exact_distance(bins_A, bins_B, range(len(points)), "height_y")
        if metric["status"] != "success":
            raise RuntimeError(f"Track C exact metric failed: {metric}")
        record = track_record(
            unit,
            points,
            lens,
            output,
            runtime,
            metric["realized_cover_sha256"],
            metric["filter_definition_sha256"],
        )
        constructions.append(record)
        eligible = bool(reference_row["eligible"] and record["eligible"])
        comparisons.append(
            {
                "comparison_id": f"COMP__{unit['construction_id']}",
                "reference_construction_id": unit["reference_construction_id"],
                "perturbed_construction_id": unit["construction_id"],
                "dataset_id": unit["dataset_id"],
                "n_intervals": unit["n_intervals"],
                "eps": unit["eps"],
                "replicate_index": unit["replicate_index"],
                "alpha": unit["alpha"],
                "sigma": unit["sigma"],
                "d_common_id": metric["distance"] if eligible else None,
                "status": "success" if eligible else "construction_ineligible",
                "eligible": eligible,
                "realized_cover_sha256": metric["realized_cover_sha256"],
                "filter_definition_sha256": metric["filter_definition_sha256"],
                "metric_definition": "exact_joint_definition_9",
                "exact_metric_sha256": EXACT_METRIC_SHA256,
            }
        )
    write_jsonl(directory / "IB5_INTERACTIONS_CONSTRUCTION_LEDGER.jsonl", constructions)
    write_jsonl(directory / "IB5_INTERACTIONS_REFERENCE_LEDGER.jsonl", reference_rows)
    write_jsonl(directory / "IB5_INTERACTIONS_COMPARISON_LEDGER.jsonl", comparisons)
    accounting = {
        "campaign_id": "IB5_INTERACTIONS_FIXED_REALIZED_COVER_CORRECTION_V3",
        "planned_constructions": len(constructions),
        "completed_constructions": len(constructions),
        "comparison_rows": len(comparisons),
        "eligible": sum(row["eligible"] for row in comparisons),
        "ineligible": sum(not row["eligible"] for row in comparisons),
        "status_counts": dict(Counter(row["status"] for row in constructions)),
        "cover_policy": "clean realized pullback memberships frozen within configuration and replicate",
        "exact_metric_sha256": EXACT_METRIC_SHA256,
    }
    write_json(directory / "IB5_INTERACTIONS_RUN_ACCOUNTING.json", accounting)


def fit_within(X, y, groups):
    Xw, yw = X.copy(), y.copy()
    for group in np.unique(groups):
        mask = groups == group
        Xw[mask] -= Xw[mask].mean(axis=0)
        yw[mask] -= yw[mask].mean()
    if np.linalg.matrix_rank(Xw) < Xw.shape[1]:
        return np.full(Xw.shape[1], np.nan), False, Xw, yw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return beta, True, Xw, yw


def recompute_c7() -> None:
    track_root = TARGET_ROOT if (
        TARGET_ROOT / "evidence/baseline/track_c/IB5_INTERACTIONS_COMPARISON_LEDGER.jsonl"
    ).is_file() else ROOT
    source = track_root / "evidence/baseline/track_c/IB5_INTERACTIONS_COMPARISON_LEDGER.jsonl"
    accounting_path = track_root / "evidence/baseline/track_c/IB5_INTERACTIONS_RUN_ACCOUNTING.json"
    contract_path = ROOT / "contracts/C7_TRACKC_INTERACTION_ANALYSIS_V1.json"
    rows = [row for row in load_jsonl(source) if row["eligible"] and row["d_common_id"] is not None]
    alpha_levels = (0.10, 0.20, 0.30)
    X, y, groups = [], [], []
    for row in rows:
        nc = float(row["n_intervals"]) - 10.0
        ec = float(row["eps"]) - 0.15
        X.append([nc, ec, nc * ec] + [float(abs(row["alpha"] - value) < 1e-12) for value in alpha_levels])
        y.append(row["d_common_id"])
        groups.append(row["replicate_index"])
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    names = ["n_centered", "eps_centered", "interaction", "alpha_0.10", "alpha_0.20", "alpha_0.30"]
    beta, ok, Xw, yw = fit_within(X, y, groups)
    if not ok:
        raise RuntimeError("corrected C7 design is rank deficient")
    streams = np.unique(groups)
    where = {int(stream): np.flatnonzero(groups == stream) for stream in streams}
    rng = np.random.default_rng(20260829)
    boot = np.empty(10000)
    failures = 0
    for index in range(len(boot)):
        pick = rng.choice(streams, size=len(streams), replace=True)
        xb, yb, gb = [], [], []
        for copy_id, stream in enumerate(pick):
            selected = where[int(stream)]
            xb.append(X[selected]); yb.append(y[selected]); gb.append(np.full(len(selected), copy_id))
        estimate, success, _, _ = fit_within(np.vstack(xb), np.concatenate(yb), np.concatenate(gb))
        if success:
            boot[index] = estimate[2]
        else:
            boot[index] = np.nan
            failures += 1
    finite = boot[np.isfinite(boot)]
    lo, hi = (float(value) for value in np.percentile(finite, [2.5, 97.5]))

    residual = yw - Xw @ beta
    inv = np.linalg.pinv(Xw.T @ Xw)
    meat = np.zeros((Xw.shape[1], Xw.shape[1]))
    for group in streams:
        score = Xw[groups == group].T @ residual[groups == group]
        meat += np.outer(score, score)
    correction = (len(streams) / (len(streams) - 1)) * ((len(y) - 1) / (len(y) - Xw.shape[1] - len(streams)))
    standard_errors = np.sqrt(np.diag(correction * (inv @ meat @ inv)))
    tcrit = float(stats.t.ppf(0.975, df=len(streams) - 1))
    terminal = "T1" if lo > 0 else "T2" if hi < 0 else "T3"

    def cell_mean(n, eps, subset=rows):
        return float(np.mean([row["d_common_id"] for row in subset if row["n_intervals"] == n and row["eps"] == eps]))

    corner = (cell_mean(20, .25) - cell_mean(20, .08)) - (cell_mean(6, .25) - cell_mean(6, .08))
    by_stream = {stream: [row for row in rows if row["replicate_index"] == stream] for stream in streams}
    rng = np.random.default_rng(20260829)
    corner_boot = np.empty(10000)
    for index in range(len(corner_boot)):
        sample = [row for stream in rng.choice(streams, size=len(streams), replace=True) for row in by_stream[int(stream)]]
        corner_boot[index] = (
            (cell_mean(20, .25, sample) - cell_mean(20, .08, sample))
            - (cell_mean(6, .25, sample) - cell_mean(6, .08, sample))
        )
    corner_lo, corner_hi = (float(value) for value in np.percentile(corner_boot, [2.5, 97.5]))
    n_levels, eps_levels = (6, 10, 15, 20), (.08, .15, .25)
    grid = {f"n={n}": {f"eps={eps}": cell_mean(n, eps) for eps in eps_levels} for n in n_levels}
    slopes = {f"n={n}": cell_mean(n, .25) - cell_mean(n, .08) for n in n_levels}
    profile = {f"n={n}": float(np.mean([row["d_common_id"] for row in rows if row["n_intervals"] == n])) for n in n_levels}
    increments = {f"{n_levels[i]}->{n_levels[i+1]}": profile[f"n={n_levels[i+1]}"] - profile[f"n={n_levels[i]}"] for i in range(3)}

    result = {
        "campaign_id": "C7_TRACKC_FIXED_COVER_EXACT_REFIT_V3",
        "label": "reanalysis of the specified estimand after mathematical endpoint correction",
        "estimand": "beta_12, n_intervals x eps interaction",
        "input_ledger_sha256": sha256(source),
        "input_accounting_sha256": sha256(accounting_path),
        "analysis_contract_id": "C7_TRACKC_INTERACTION_ANALYSIS_V3_CORRECTED_ENDPOINT",
        "analysis_contract_sha256": sha256(contract_path),
        "mapper_constructions_generated": 0,
        "denominator": {"rows_in_ledger": len(rows), "rows_analysed": len(rows), "rows_excluded": 0},
        "design": {
            "n_intervals_levels": list(n_levels), "eps_levels": list(eps_levels),
            "alpha_levels": [0.05, 0.10, 0.20, 0.30], "replicate_streams": len(streams),
            "cover_policy": "fixed realized reference pullbacks", "metric": "joint common-cover distance (D_M^NA where special states occur)",
        },
        "beta_12": {
            "point_estimate": float(beta[2]),
            "primary_interval": {"method": "stream-level nonparametric percentile bootstrap", "B": 10000, "seed": 20260829, "clusters": len(streams), "bootstrap_failures": failures, "bootstrap_failure_rate": failures / 10000, "lo_2.5": lo, "hi_97.5": hi},
            "secondary_interval": {"method": "CR1 cluster-by-stream sandwich, t_{G-1}", "se": float(standard_errors[2]), "df": len(streams)-1, "t_crit": tcrit, "lo": float(beta[2]-tcrit*standard_errors[2]), "hi": float(beta[2]+tcrit*standard_errors[2])},
        },
        "other_coefficients": {name: float(beta[index]) for index, name in enumerate(names)},
        "terminal_state": terminal,
        "terminal_state_text": {"T1": "INTERACTION RESOLVED, POSITIVE", "T2": "INTERACTION RESOLVED, NEGATIVE", "T3": "INTERACTION NOT RESOLVED"}[terminal],
        "exact_metric_sha256": EXACT_METRIC_SHA256,
        "p_values_computed": False,
        "imputation_used": False,
    }
    # Post-hoc categorical-factor robustness.  The reference cell is (n=6, eps=.08).
    # Each interaction contrast is a difference-in-differences averaged over the
    # balanced alpha levels and replicate streams.  Stream bootstrap preserves the
    # independent replicate unit and reveals whether the bilinear trend is uniform
    # across factor levels.
    categorical_rows = []
    rng = np.random.default_rng(20260829)
    stream_picks = rng.choice(streams, size=(10000, len(streams)), replace=True)
    for n in (10, 15, 20):
        for eps in (.15, .25):
            per_stream = np.asarray([
                (cell_mean(n, eps, by_stream[int(stream)]) - cell_mean(n, .08, by_stream[int(stream)]))
                - (cell_mean(6, eps, by_stream[int(stream)]) - cell_mean(6, .08, by_stream[int(stream)]))
                for stream in streams
            ], dtype=float)
            contrast_boot = per_stream[stream_picks].mean(axis=1)
            c_lo, c_hi = (float(value) for value in np.percentile(contrast_boot, [2.5, 97.5]))
            categorical_rows.append({
                "n_intervals": n,
                "eps": eps,
                "point_estimate": float(per_stream.mean()),
                "stream_bootstrap_95": {"lo": c_lo, "hi": c_hi, "B": 10000, "seed": 20260829},
                "excludes_zero": bool(c_lo > 0 or c_hi < 0),
            })

    diagnostic = {
        "status": "POST-HOC ROBUSTNESS DIAGNOSTICS",
        "cell_means_d_common_id": grid,
        "eps_slope_at_each_n": slopes,
        "n_main_effect_profile": profile,
        "n_main_effect_successive_increments": increments,
        "model_free_corner_did": {"point_estimate": corner, "stream_bootstrap_95": {"lo": corner_lo, "hi": corner_hi, "B": 10000, "seed": 20260829}, "excludes_zero": bool(corner_lo > 0 or corner_hi < 0)},
        "categorical_factor_interaction_contrasts": {
            "reference_n_intervals": 6,
            "reference_eps": 0.08,
            "definition": "[mean D(n,eps)-mean D(n,0.08)]-[mean D(6,eps)-mean D(6,0.08)], averaged over balanced alpha levels; stream bootstrap resamples the 10 replicate streams",
            "role": "post-hoc robustness check of the numeric bilinear parameterization; not a replacement estimand",
            "rows": categorical_rows,
        },
    }
    directory = TARGET_ROOT / "evidence/campaigns/c7"
    write_json(directory / "C7_INTERACTION_FIT.json", result)
    write_json(directory / "C7_ROBUSTNESS_DIAGNOSTICS.json", diagnostic)


def verify_separate_reproduction() -> None:
    """Require deterministic manuscript-controlling products to reproduce bytewise."""

    relative_paths = (
        "evidence/campaigns/c4_fmapper/C4R_CONSTRUCTION_LEDGER.jsonl",
        "evidence/campaigns/c4_fmapper/CELL_SUMMARY.csv",
        "evidence/campaigns/c4_fmapper/C4R_RUN_ACCOUNTING.json",
        "evidence/campaigns/conventional_fixed_cover/CIA2_CONSTRUCTION_LEDGER.jsonl",
        "evidence/campaigns/conventional_fixed_cover/CIA2_RUN_ACCOUNTING.json",
        "evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl",
        "evidence/campaigns/c1/C1_QUALITY_LEDGER.jsonl",
        "evidence/campaigns/c1/C1_PAIRED_QUALITY_LEDGER.jsonl",
        "evidence/campaigns/c1/FINAL_ACCOUNTING.json",
        "evidence/baseline/track_c/IB5_INTERACTIONS_COMPARISON_LEDGER.jsonl",
        "evidence/baseline/track_c/IB5_INTERACTIONS_RUN_ACCOUNTING.json",
        "evidence/campaigns/c7/C7_INTERACTION_FIT.json",
        "evidence/campaigns/c7/C7_ROBUSTNESS_DIAGNOSTICS.json",
    )
    mismatches = [
        relative
        for relative in relative_paths
        if sha256(ROOT / relative) != sha256(TARGET_ROOT / relative)
    ]
    if mismatches:
        raise RuntimeError(
            "separate correction replay disagrees with released controlling products: "
            + ", ".join(mismatches)
        )
    print(
        f"[verification] {len(relative_paths)}/{len(relative_paths)} "
        "controlling products reproduced bytewise"
    )


def main() -> None:
    global TARGET_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("all", "c4", "conventional", "c1", "track-c", "c7"),
        default="all",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT,
        help="write corrected evidence under this root (defaults to the release root)",
    )
    args = parser.parse_args()
    TARGET_ROOT = args.output_root.resolve()
    actions = {
        "c4": recompute_c4_fmapper,
        "conventional": recompute_conventional,
        "c1": recompute_c1,
        "track-c": recompute_track_c,
        "c7": recompute_c7,
    }
    selected = actions if args.phase == "all" else {args.phase: actions[args.phase]}
    for name, action in selected.items():
        print(f"[{name}] starting", flush=True)
        action()
        print(f"[{name}] complete", flush=True)
    if args.phase == "all" and TARGET_ROOT != ROOT:
        verify_separate_reproduction()


if __name__ == "__main__":
    main()

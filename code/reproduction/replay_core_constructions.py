#!/usr/bin/env python3
"""Replay representative construction-level cases from frozen inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

from mapper_framework.ball_mapper import BallMapper
from mapper_framework.conventional import ConventionalMapper
from mapper_framework.dataset_generators import (
    generate_branching_tripod,
    generate_clean_circle,
    generate_swiss_roll_with_hole,
)
from mapper_framework.f_mapper import FMapper
from mapper_framework.support.d_common_id import (
    d_common_id,
    hash_filter_definition,
    hash_realized_cover,
)


ROOT = Path(__file__).resolve().parents[2]


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text())


def load_jsonl(relative: str):
    return [json.loads(line) for line in (ROOT / relative).read_text().splitlines() if line.strip()]


def conventional_replay(dataset_name: str) -> dict:
    if dataset_name == "circle":
        expected = load_json("evidence/baseline/axis3_iii1/CIRCLE_RESULT.json")
        dataset = generate_clean_circle(N=1000, radius=1.0, data_seed=42, sampling="uniform")
    else:
        expected = load_json("evidence/baseline/axis3_iii1/TRIPOD_RESULT.json")
        dataset = generate_branching_tripod(N=1000, data_seed=42)
    mapper = ConventionalMapper(
        n_intervals=10,
        overlap_frac=0.30,
        clusterer=DBSCAN(eps=0.15, min_samples=3, metric="euclidean"),
        input_mode="coordinates",
    )
    output = mapper.fit_transform(dataset.points, dataset.lens)
    observed = {
        "status": output.status,
        "n_nodes": output.graph.n_nodes if output.graph else None,
        "n_edges": output.graph.n_edges if output.graph else None,
        "n_components": output.graph.n_components if output.graph else None,
        "n_2_simplices": output.nerve.n_2 if output.nerve else None,
        "beta0_nerve": output.homology.beta_0_nerve if output.homology else None,
        "beta1_graph": output.graph.beta_1_graph if output.graph else None,
        "beta1_nerve": output.homology.beta_1_nerve if output.homology else None,
    }
    wanted = {
        "status": expected["status"],
        "n_nodes": expected["n_nodes"],
        "n_edges": expected["n_edges"],
        "n_components": expected["n_components"],
        "n_2_simplices": expected["n_2_simplices"],
        "beta0_nerve": expected["beta0_nerve_mapper"],
        "beta1_graph": expected["beta1_graph_mapper"],
        "beta1_nerve": expected["beta1_nerve_mapper"],
    }
    if observed != wanted:
        raise AssertionError(f"III-1 {dataset_name} replay mismatch: {observed} != {wanted}")
    if dataset_name == "tripod":
        degrees = {node: 0 for node in output.graph.nodes}
        for u, v in output.graph.edges:
            degrees[u] += 1
            degrees[v] += 1
        signature = sorted(d for d in degrees.values() if d != 2)
        if signature != expected["non_degree2_signature_mapper"]:
            raise AssertionError(f"Tripod signature mismatch: {signature}")
        observed["non_degree2_signature"] = signature
    return observed


def c8_replay(status: str) -> dict:
    rows = load_jsonl("evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl")
    target = next(row for row in rows if row["dataset_id"] == "unit_circle"
                  and row["n_intervals"] == 10 and row["threshold"] == 0.10
                  and row["status"] == status)
    dataset = generate_clean_circle(N=target["n_observations"], radius=1.0,
                                    data_seed=target["data_seed"], sampling="uniform")
    mapper = FMapper(
        n_intervals=target["n_intervals"], threshold=target["threshold"],
        fcm_fuzzifier=target["fcm_fuzzifier"], fcm_tol=target["fcm_tol"],
        fcm_max_iter=target["fcm_max_iter"], fcm_seed=target["fcm_seed"],
        clusterer=DBSCAN(eps=target["clusterer_spec"]["eps"],
                         min_samples=target["clusterer_spec"]["min_samples"]),
    )
    output = mapper.fit_transform(dataset.points, dataset.lens)
    if output.status != target["status"]:
        raise AssertionError(f"C8 {status} replay status {output.status}")
    observed = {"unit_id": target["unit_id"], "status": output.status}
    if status == "success":
        expected = (target["n_nodes"], target["n_edges"], target["beta1_graph_mapper"],
                    target["beta0_nerve_mapper"], target["beta1_nerve_mapper"])
        got = (output.graph.n_nodes, output.graph.n_edges, output.graph.beta_1_graph,
               output.homology.beta_0_nerve, output.homology.beta_1_nerve)
        if got != expected:
            raise AssertionError(f"C8 success replay mismatch: {got} != {expected}")
        observed["invariants"] = got
    return observed


def ball_swiss_replay() -> dict:
    expected = next(row for row in load_jsonl(
        "evidence/baseline/ball_parameter/BALL_CONSTRUCTION_LEDGER.jsonl")
        if row["construction_id"] == "BALL_IB4_III5_CONS_SWISS_M03_K00")
    orders = load_json("replay_inputs/BALL_ORDER_MANIFEST.json")["records"]
    order = next(row["permutation"] for row in orders if row["order_id"] == expected["order_id"])
    dataset = generate_swiss_roll_with_hole(N=2500, data_seed=42,
                                            resample_to_exact_N=False, lens_id="radial_xz")
    output = BallMapper(epsilon=expected["epsilon"]).fit_transform(
        dataset.points, point_order=order, max_triangles=200_000)
    got = (output.status, output.graph.n_nodes, output.graph.n_edges,
           output.graph.beta_1_graph, output.nerve.n_2,
           output.homology.beta_0_nerve, output.homology.beta_1_nerve)
    wanted = (expected["status"], expected["n_centers"], expected["n_edges"],
              expected["beta1_graph"], expected["nerve_triangles"],
              expected["beta0_nerve"], expected["beta1_nerve"])
    if got != wanted:
        raise AssertionError(f"Ball Swiss replay mismatch: {got} != {wanted}")
    return {"construction_id": expected["construction_id"], "observed": got}


def d_common_gate_replay() -> dict:
    bins = {0: {0: 0, 1: 1}, 1: {0: 2, 1: 3}}
    cover_hash = hash_realized_cover(bins, {0, 1})
    filter_hash = hash_filter_definition("f(x,y) = y")
    accepted = d_common_id(bins, bins, common_ids={0, 1}, cover_id_A="fixed",
                           cover_id_B="fixed", filter_id_A="height", filter_id_B="height",
                           cover_hash_A=cover_hash, cover_hash_B=cover_hash,
                           filter_hash_A=filter_hash, filter_hash_B=filter_hash)
    rejected = d_common_id(bins, bins, common_ids={0, 1}, cover_id_A="adaptive-a",
                           cover_id_B="adaptive-b", filter_id_A="height", filter_id_B="height",
                           cover_hash_A=cover_hash, cover_hash_B=cover_hash,
                           filter_hash_A=filter_hash, filter_hash_B=filter_hash)
    if accepted["distance"] != 0.0 or rejected["reason"] != "cover_mismatch":
        raise AssertionError("D_COMMON_ID fixed/adaptive cover gates failed")
    return {"fixed_cover_distance": accepted["distance"],
            "adaptive_cover_status": rejected["status"],
            "adaptive_cover_reason": rejected["reason"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "conventional_circle": conventional_replay("circle"),
        "conventional_tripod": conventional_replay("tripod"),
        "c8_success": c8_replay("success"),
        "c8_non_convergence": c8_replay("fcm_non_convergence"),
        "ball_swiss_dual_homology": ball_swiss_replay(),
        "d_common_id_gates": d_common_gate_replay(),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

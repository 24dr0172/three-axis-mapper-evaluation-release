"""Fixed realized-cover construction helpers for common-cover comparisons."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone

from mapper_framework.exceptions import HomologyConsistencyError, ResourceLimitError
from mapper_framework.homology_gf2 import compute_1skeleton_components, compute_dual_homology
from mapper_framework.nerve import build_membership_nerve_2d
from mapper_framework.types import (
    DualHomologyResult,
    MapperGraph,
    MapperNode,
    MapperOutput,
    SimplicialNerve2D,
)


def regular_interval_cover(
    lens: Sequence[float], n_intervals: int, overlap_frac: float
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Build the same regular interval cover used by ``ConventionalMapper``."""

    values = np.asarray(lens, dtype=float).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("lens must be a non-empty finite vector")
    if n_intervals < 1 or not 0.0 <= overlap_frac < 1.0:
        raise ValueError("invalid interval-cover parameters")
    f_min, f_max = float(np.min(values)), float(np.max(values))
    if f_min == f_max:
        math_intervals = [(f_min, f_min)]
        eval_intervals = [
            (np.nextafter(f_min, -np.inf), np.nextafter(f_min, np.inf))
        ]
        return math_intervals, eval_intervals
    width = (f_max - f_min) / (
        n_intervals - overlap_frac * (n_intervals - 1)
    )
    stride = width * (1.0 - overlap_frac)
    math_intervals = []
    eval_intervals = []
    for index in range(n_intervals):
        left = f_min + index * stride
        right = f_max if index == n_intervals - 1 else left + width
        math_intervals.append((float(left), float(right)))
        eval_intervals.append(
            (float(np.nextafter(left, -np.inf)), float(np.nextafter(right, np.inf)))
        )
    return math_intervals, eval_intervals


def memberships_from_intervals(
    lens: Sequence[float], intervals_eval: Sequence[Tuple[float, float]]
) -> Dict[int, Tuple[int, ...]]:
    """Materialize indexed pullback membership once for later reuse."""

    values = np.asarray(lens, dtype=float).reshape(-1)
    return {
        index: tuple(
            int(x)
            for x in np.flatnonzero((values >= left) & (values <= right))
        )
        for index, (left, right) in enumerate(intervals_eval)
    }


def construct_on_memberships(
    X: np.ndarray,
    lens: Sequence[float],
    memberships: Dict[int, Iterable[int]],
    clusterer,
    *,
    cover_mode: str = "fixed_realized_cover",
    max_triangles: Optional[int] = 200_000,
) -> MapperOutput:
    """Cluster two data realizations on exactly the same indexed cover sets."""

    points = np.asarray(X, dtype=float)
    values = np.asarray(lens, dtype=float).reshape(-1)
    if points.ndim != 2 or len(points) != len(values):
        return MapperOutput(
            status="configuration_invalid",
            reason="points/lens shape mismatch",
            graph=None,
            nerve=None,
            homology=None,
            metadata={},
        )
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        return MapperOutput(
            status="configuration_invalid",
            reason="non-finite points or lens",
            graph=None,
            nerve=None,
            homology=None,
            metadata={},
        )

    normalized = {
        int(bin_id): np.asarray(sorted(set(int(x) for x in ids)), dtype=int)
        for bin_id, ids in memberships.items()
    }
    expected_bins = list(range(len(normalized)))
    if sorted(normalized) != expected_bins:
        raise ValueError("memberships must use contiguous bin identifiers from zero")
    for bin_id, ids in normalized.items():
        if np.any(ids < 0) or np.any(ids >= len(points)):
            raise ValueError(f"membership index outside data domain in bin {bin_id}")

    metadata = {
        "N": int(len(points)),
        "effective_n_intervals": len(normalized),
        "requested_n_intervals": len(normalized),
        "cover_mode": cover_mode,
        "pullback_records": [],
    }
    nodes: Dict[int, MapperNode] = {}
    next_node_id = 0
    for bin_id in expected_bins:
        indices = normalized[bin_id]
        if len(indices) == 0:
            metadata["pullback_records"].append(
                {"cover_element_id": bin_id, "sample_indices": [], "local_cluster_labels": []}
            )
            continue
        try:
            labels = clone(clusterer).fit_predict(points[indices])
        except Exception as error:
            return MapperOutput(
                status="clusterer_failure",
                reason=f"clusterer failed in fixed cover element {bin_id}: {error}",
                graph=None,
                nerve=None,
                homology=None,
                metadata=metadata,
            )
        labels = np.asarray(labels, dtype=int)
        if len(labels) != len(indices):
            return MapperOutput(
                status="clusterer_failure",
                reason=f"malformed labels in fixed cover element {bin_id}",
                graph=None,
                nerve=None,
                homology=None,
                metadata=metadata,
            )
        metadata["pullback_records"].append(
            {
                "cover_element_id": bin_id,
                "sample_indices": [int(x) for x in indices],
                "local_cluster_labels": [int(x) for x in labels],
            }
        )
        for label in sorted(int(x) for x in set(labels) if x >= 0):
            members = frozenset(int(x) for x in indices[labels == label])
            nodes[next_node_id] = MapperNode(
                node_id=next_node_id,
                interval_idx=bin_id,
                cluster_label=label,
                members=members,
                size=len(members),
                mean_filter=float(np.mean(values[list(members)])),
            )
            next_node_id += 1

    if not nodes:
        graph = MapperGraph({}, [], {}, 0, 0, 0, 0)
        nerve = SimplicialNerve2D([], [], [], [], 0, 0, 0, {})
        homology = DualHomologyResult(
            0, 0, 0, 0, 0, "empty_complex", True, {}
        )
        return MapperOutput(
            status="degenerate_output",
            reason="all_points_unassigned",
            graph=graph,
            nerve=nerve,
            homology=homology,
            metadata=metadata,
        )

    edges = []
    edge_weights = {}
    node_ids = sorted(nodes)
    for position, left in enumerate(node_ids):
        for right in node_ids[position + 1 :]:
            weight = len(nodes[left].members & nodes[right].members)
            if weight:
                edges.append((left, right))
                edge_weights[(left, right)] = weight
    components = compute_1skeleton_components(len(nodes), edges)
    graph = MapperGraph(
        nodes=nodes,
        edges=edges,
        edge_weights=edge_weights,
        n_nodes=len(nodes),
        n_edges=len(edges),
        n_components=components,
        beta_1_graph=len(edges) - len(nodes) + components,
    )
    try:
        nerve = build_membership_nerve_2d(
            {node_id: node.members for node_id, node in nodes.items()},
            max_triangles=max_triangles,
        )
        homology = compute_dual_homology(nerve)
    except ResourceLimitError as error:
        return MapperOutput(
            status="resource_failure",
            reason=str(error),
            graph=graph,
            nerve=None,
            homology=None,
            metadata=metadata,
        )
    except HomologyConsistencyError:
        raise
    except Exception as error:
        return MapperOutput(
            status="homology_failure",
            reason=str(error),
            graph=graph,
            nerve=nerve if "nerve" in locals() else None,
            homology=None,
            metadata=metadata,
        )
    return MapperOutput("success", None, graph, nerve, homology, metadata)

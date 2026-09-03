"""Conventional Mapper core algorithm and pipeline implementation."""

from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union
import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.cluster import DBSCAN

from mapper_framework.exceptions import (
    ClustererFailureError,
    ConfigurationInvalidError,
    HomologyConsistencyError,
    ResourceLimitError,
)
from mapper_framework.homology_gf2 import compute_1skeleton_components, compute_dual_homology
from mapper_framework.nerve import build_membership_nerve_2d
from mapper_framework.types import (
    DualHomologyResult,
    MapperGraph,
    MapperNode,
    MapperOutput,
    SimplicialNerve2D,
)


class ConventionalMapper:
    """Conventional Mapper implementation conforming strictly to SPEC-CM v1.2.0."""

    def __init__(
        self,
        n_intervals: int = 10,
        overlap_frac: float = 0.5,
        clusterer: Optional[Any] = None,
        input_mode: str = "coordinates",
    ):
        if n_intervals < 1:
            raise ValueError(f"n_intervals must be >= 1, got {n_intervals}")
        if not (0.0 <= overlap_frac < 1.0):
            raise ValueError(f"overlap_frac must be in [0.0, 1.0), got {overlap_frac}")
        if input_mode not in ("coordinates", "precomputed_distance"):
            raise ValueError(f"input_mode must be 'coordinates' or 'precomputed_distance', got {input_mode}")

        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.clusterer = clusterer if clusterer is not None else DBSCAN(eps=0.5, min_samples=1)
        self.input_mode = input_mode

    def _clone_clusterer(self) -> Any:
        """Create a fresh independent clone of the clusterer configuration."""
        try:
            return clone(self.clusterer)
        except Exception:
            # Fallback if clusterer is not a standard scikit-learn estimator instance
            import copy
            return copy.deepcopy(self.clusterer)

    def _validate_inputs(
        self, X: np.ndarray, lens: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate input arrays according to SPEC-CM §4.1."""
        X_arr = np.asarray(X)
        lens_arr = np.asarray(lens, dtype=float)

        N = len(X_arr)
        if N == 0:
            raise ValueError("Input dataset X is empty (N=0).")
        if len(lens_arr) != N:
            raise ValueError(f"X length ({N}) and lens length ({len(lens_arr)}) must match.")

        if not np.all(np.isfinite(lens_arr)):
            raise ValueError("Lens contains non-finite values (NaN/Inf).")

        if self.input_mode == "coordinates":
            if X_arr.ndim != 2:
                raise ValueError(f"Coordinates X must be 2D array, got shape {X_arr.shape}")
            if not np.all(np.isfinite(X_arr)):
                raise ValueError("Coordinate data X contains non-finite values (NaN/Inf).")

        elif self.input_mode == "precomputed_distance":
            if X_arr.ndim != 2 or X_arr.shape[0] != X_arr.shape[1] or X_arr.shape[0] != N:
                raise ValueError(f"Distance matrix must be square (N, N), got {X_arr.shape} with N={N}")
            if not np.all(np.isfinite(X_arr)):
                raise ValueError("Distance matrix contains non-finite values (NaN/Inf).")

            # Check non-negativity
            if np.any(X_arr < -1e-12):
                raise ValueError("Distance matrix contains negative values.")

            # Check zero diagonal
            diag_max = np.max(np.abs(np.diag(X_arr)))
            if diag_max > 1e-12:
                raise ValueError(f"Distance matrix diagonal must be zero within tolerance, max |D_ii|={diag_max}")

            # Check symmetry
            sym_diff = np.max(np.abs(X_arr - X_arr.T))
            max_val = max(1.0, float(np.max(X_arr)))
            if sym_diff > 1e-12 * max_val:
                raise ValueError(f"Distance matrix is not symmetric within tolerance, diff={sym_diff}")

            # Enforce precomputed metric contract on clusterer
            if hasattr(self.clusterer, "metric") and getattr(self.clusterer, "metric") != "precomputed":
                raise ValueError(
                    f"In 'precomputed_distance' mode, clusterer metric must be 'precomputed', "
                    f"got '{getattr(self.clusterer, 'metric')}'."
                )

        return X_arr, lens_arr


    def fit_transform(
        self,
        X: np.ndarray,
        lens: np.ndarray,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
        max_triangles: Optional[int] = None,
    ) -> MapperOutput:
        """Execute Conventional Mapper construction."""
        try:
            X_arr, lens_arr = self._validate_inputs(X, lens)
        except ValueError as e:
            return MapperOutput(
                status="configuration_invalid",
                reason=str(e),
                graph=None,
                nerve=None,
                homology=None,
                metadata={"error": str(e)}
            )

        N = len(X_arr)
        f_min = float(np.min(lens_arr))
        f_max = float(np.max(lens_arr))

        # Check exact constant lens branch
        is_exact_constant = bool(f_max == f_min)
        metadata: Dict[str, Any] = {
            "N": N,
            "f_min": f_min,
            "f_max": f_max,
            "constant_lens": is_exact_constant,
            "requested_n_intervals": self.n_intervals,
            "overlap_frac": self.overlap_frac,
            "input_mode": self.input_mode,
        }

        # Build regular cover intervals
        intervals_math: List[Tuple[float, float]] = []
        intervals_eval: List[Tuple[float, float]] = []

        if is_exact_constant:
            metadata["effective_n_intervals"] = 1
            c = f_min
            intervals_math.append((c, c))
            intervals_eval.append((
                np.nextafter(c, -np.inf),
                np.nextafter(c, np.inf)
            ))
        else:
            n = self.n_intervals
            p = self.overlap_frac
            metadata["effective_n_intervals"] = n

            R = f_max - f_min
            w = R / (n - p * (n - 1))
            s = w * (1.0 - p)

            metadata["interval_width"] = w
            metadata["interval_stride"] = s

            for k in range(n):
                a_k = f_min + k * s
                b_k = f_max if k == n - 1 else a_k + w
                intervals_math.append((a_k, b_k))
                intervals_eval.append((
                    np.nextafter(a_k, -np.inf),
                    np.nextafter(b_k, np.inf)
                ))

        metadata["intervals_math"] = intervals_math
        metadata["intervals_eval"] = intervals_eval

        # Pullback clustering and deterministic node creation
        nodes: Dict[int, MapperNode] = {}
        next_node_id = 0
        pullback_records: List[Dict[str, Any]] = []

        for k, (a_eval, b_eval) in enumerate(intervals_eval):
            # Pullback indices
            pullback_mask = (lens_arr >= a_eval) & (lens_arr <= b_eval)
            pullback_indices = np.where(pullback_mask)[0]

            if len(pullback_indices) == 0:
                pullback_records.append({
                    "cover_element_id": k,
                    "sample_indices": [],
                    "local_cluster_labels": [],
                })
                continue

            # Sub-dataset extraction
            if self.input_mode == "coordinates":
                X_sub = X_arr[pullback_indices]
            else:
                X_sub = X_arr[np.ix_(pullback_indices, pullback_indices)]

            # Independent fresh clusterer clone
            local_clusterer = self._clone_clusterer()

            try:
                labels = local_clusterer.fit_predict(X_sub)
            except Exception as e:
                metadata["pullback_records"] = pullback_records
                return MapperOutput(
                    status="clusterer_failure",
                    reason=f"Clusterer raised exception in pullback interval {k}: {str(e)}",
                    graph=None,
                    nerve=None,
                    homology=None,
                    metadata=metadata
                )

            # Validate returned labels
            if labels is None or len(labels) != len(pullback_indices):
                metadata["pullback_records"] = pullback_records
                return MapperOutput(
                    status="clusterer_failure",
                    reason=f"Clusterer returned malformed label length in interval {k}: expected {len(pullback_indices)}, got {len(labels) if labels is not None else None}",
                    graph=None,
                    nerve=None,
                    homology=None,
                    metadata=metadata
                )

            pullback_records.append({
                "cover_element_id": k,
                "sample_indices": [int(idx) for idx in pullback_indices],
                "local_cluster_labels": [int(lbl) for lbl in labels],
            })

            # Extract unique valid non-negative cluster labels deterministically
            unique_labels = sorted([int(lbl) for lbl in set(labels) if lbl >= 0])

            for j in unique_labels:
                cluster_sample_indices = pullback_indices[labels == j]
                members = frozenset(int(idx) for idx in cluster_sample_indices)

                if len(members) > 0:
                    node = MapperNode(
                        node_id=next_node_id,
                        interval_idx=k,
                        cluster_label=j,
                        members=members,
                        size=len(members),
                        mean_filter=float(np.mean(lens_arr[list(members)])),
                    )
                    nodes[next_node_id] = node
                    next_node_id += 1

        metadata["pullback_records"] = pullback_records

        V = len(nodes)

        # Handle all points noise / empty graph
        if V == 0:
            empty_graph = MapperGraph(
                nodes={},
                edges=[],
                edge_weights={},
                n_nodes=0,
                n_edges=0,
                n_components=0,
                beta_1_graph=0,
            )
            empty_nerve = SimplicialNerve2D(
                nodes=[],
                sigma_0=[],
                sigma_1=[],
                sigma_2=[],
                n_0=0,
                n_1=0,
                n_2=0,
                vertex_to_row={}
            )
            empty_homology = DualHomologyResult(
                beta_0_nerve=0,
                beta_1_nerve=0,
                beta_1_graph=0,
                rank_d1=0,
                rank_d2=0,
                status="empty_complex",
                d1_d2_zero_mod2=True,
                details={}
            )
            return MapperOutput(
                status="degenerate_output",
                reason="all_points_unassigned",
                graph=empty_graph,
                nerve=empty_nerve,
                homology=empty_homology,
                metadata=metadata
            )

        # 1-Skeleton Graph Construction (edges from non-empty shared sample intersections)
        edges: List[Tuple[int, int]] = []
        edge_weights: Dict[Tuple[int, int], int] = {}

        for u in range(V):
            for v in range(u + 1, V):
                inter_len = len(nodes[u].members & nodes[v].members)
                if inter_len > 0:
                    edges.append((u, v))
                    edge_weights[(u, v)] = inter_len

        C = compute_1skeleton_components(V, edges)
        beta_1_graph = len(edges) - V + C

        graph = MapperGraph(
            nodes=nodes,
            edges=edges,
            edge_weights=edge_weights,
            n_nodes=V,
            n_edges=len(edges),
            n_components=C,
            beta_1_graph=beta_1_graph,
        )

        # Construct Simplicial Nerve 2-Skeleton
        node_supports = {node.node_id: node.members for node in nodes.values()}
        try:
            nerve = build_membership_nerve_2d(
                node_supports=node_supports,
                max_nodes=max_nodes,
                max_edges=max_edges,
                max_triangles=max_triangles,
            )
        except ResourceLimitError as e:
            return MapperOutput(
                status="resource_failure",
                reason=f"Nerve construction tripped resource guardrail: {str(e)}",
                graph=graph,
                nerve=None,
                homology=None,
                metadata=metadata
            )

        # Compute Dual Homology
        try:
            homology = compute_dual_homology(nerve)
        except HomologyConsistencyError as e:
            if not hasattr(e, "partial_graph") or e.partial_graph is None:
                e.partial_graph = graph
            if not hasattr(e, "partial_nerve") or e.partial_nerve is None:
                e.partial_nerve = nerve
            raise e  # Propagate mathematical / algebraic violations with partial artifacts attached
        except Exception as e:
            return MapperOutput(
                status="homology_failure",
                reason=f"Homology reduction error: {str(e)}",
                graph=graph,
                nerve=nerve,
                homology=None,
                metadata=metadata
            )

        return MapperOutput(
            status="success",
            reason=None,
            graph=graph,
            nerve=nerve,
            homology=homology,
            metadata=metadata
        )

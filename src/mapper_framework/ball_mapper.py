"""Standard Ball Mapper implementation conforming to Dłotko (2019) Algorithms 1 & 3."""

from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist

from mapper_framework.exceptions import (
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


class BallMapper:
    """Standard Ball Mapper (Dłotko 2019 Algorithm 1 + Algorithm 3).

    Lens-free metric-cover and geometric connectivity summary based on greedy
    epsilon-net centre selection and observed-point common coverage.
    """

    def __init__(
        self,
        epsilon: float,
        input_mode: str = "coordinates",
    ):
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0.0, got {epsilon}")
        if input_mode not in ("coordinates", "precomputed_distance"):
            raise ValueError(f"input_mode must be 'coordinates' or 'precomputed_distance', got {input_mode}")

        self.epsilon = float(epsilon)
        self.input_mode = input_mode

    def _validate_inputs(
        self,
        X: np.ndarray,
        point_order: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate input data, distance matrix, and point_order permutation."""
        X_arr = np.asarray(X)
        N = len(X_arr)
        if N == 0:
            raise ValueError("Input dataset X is empty (N=0).")

        if self.input_mode == "coordinates":
            if X_arr.ndim != 2:
                raise ValueError(f"Coordinates X must be a 2D array, got shape {X_arr.shape}")
            if not np.all(np.isfinite(X_arr)):
                raise ValueError("Coordinate array X contains non-finite values (NaN/Inf).")
            # Compute Euclidean distance matrix
            D = cdist(X_arr, X_arr, metric="euclidean")
        elif self.input_mode == "precomputed_distance":
            if X_arr.ndim != 2 or X_arr.shape[0] != X_arr.shape[1] or X_arr.shape[0] != N:
                raise ValueError(f"Distance matrix must be square (N, N), got {X_arr.shape}")
            if not np.all(np.isfinite(X_arr)):
                raise ValueError("Distance matrix contains non-finite values (NaN/Inf).")
            if np.any(X_arr < -1e-12):
                raise ValueError("Distance matrix contains negative values.")
            diag_max = np.max(np.abs(np.diag(X_arr)))
            if diag_max > 1e-12:
                raise ValueError(f"Distance matrix diagonal must be zero within tolerance, max={diag_max}")
            sym_diff = np.max(np.abs(X_arr - X_arr.T))
            max_val = max(1.0, float(np.max(X_arr)))
            if sym_diff > 1e-12 * max_val:
                raise ValueError(f"Distance matrix is not symmetric within tolerance, diff={sym_diff}")
            D = X_arr
        else:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")

        # Validate point_order
        if point_order is None:
            point_order_arr = np.arange(N, dtype=int)
        else:
            point_order_arr = np.asarray(point_order, dtype=int)
            if len(point_order_arr) != N:
                raise ValueError(f"point_order length ({len(point_order_arr)}) must match dataset size ({N})")
            if set(point_order_arr.tolist()) != set(range(N)):
                raise ValueError("point_order must be a valid permutation of {0, ..., N-1}")

        return X_arr, D, point_order_arr

    def fit_transform(
        self,
        X: np.ndarray,
        point_order: Optional[Sequence[int]] = None,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
        max_triangles: Optional[int] = None,
    ) -> MapperOutput:
        """Execute Standard Ball Mapper algorithm.

        Parameters
        ----------
        X : np.ndarray
            Dataset coordinates (N, d) or precomputed distance matrix (N, N).
        point_order : Optional[Sequence[int]]
            Deterministic permutation of range(N) controlling greedy candidate ordering.
        max_nodes, max_edges, max_triangles : Optional[int]
            Resource ceiling guardrails for membership nerve construction.

        Returns
        -------
        MapperOutput
            Verified Ball Mapper output with 1-skeleton graph, 2-skeleton nerve,
            and dual homology invariants.
        """
        try:
            X_arr, D, point_order_arr = self._validate_inputs(X, point_order)
        except ValueError as e:
            return MapperOutput(
                status="configuration_invalid",
                reason=str(e),
                graph=None,
                nerve=None,
                homology=None,
                metadata={"error": str(e), "scientific_evidence_eligible": False},
            )

        N = len(X_arr)
        eps = self.epsilon

        # 1. Algorithm 1: Greedy epsilon-Net Centre Selection
        covered = np.zeros(N, dtype=bool)
        selected_centers: List[int] = []

        for p_idx in point_order_arr:
            p = int(p_idx)
            if not covered[p]:
                selected_centers.append(p)
                # Mark all points within distance epsilon as covered
                in_ball_mask = (D[p] <= eps)
                covered[in_ball_mask] = True
                if np.all(covered):
                    break

        m = len(selected_centers)

        # 2. Construct Full Ball Supports (Dłotko §3)
        # S(c_j) includes ALL observed points within distance epsilon,
        # including points already covered by earlier selected balls.
        nodes: Dict[int, MapperNode] = {}
        for v in range(m):
            c_p = selected_centers[v]
            in_ball_indices = np.where(D[c_p] <= eps)[0]
            members = frozenset(int(idx) for idx in in_ball_indices)
            node = MapperNode(
                node_id=v,
                interval_idx=v,
                cluster_label=0,
                members=members,
                size=len(members),
                mean_filter=0.0,
            )
            nodes[v] = node

        # 3. Algorithm 3: Standard Ball Mapper Graph Construction
        # Edge exists iff there is an observed sample in S(c_i) cap S(c_j).
        # Ambient d(c_i, c_j) <= 2*eps without observed witness creates NO edge.
        edges: List[Tuple[int, int]] = []
        edge_weights: Dict[Tuple[int, int], int] = {}

        for u in range(m):
            for v in range(u + 1, m):
                inter_len = len(nodes[u].members & nodes[v].members)
                if inter_len > 0:
                    edges.append((u, v))
                    edge_weights[(u, v)] = inter_len

        C = compute_1skeleton_components(m, edges)
        beta_1_graph = len(edges) - m + C

        graph = MapperGraph(
            nodes=nodes,
            edges=edges,
            edge_weights=edge_weights,
            n_nodes=m,
            n_edges=len(edges),
            n_components=C,
            beta_1_graph=beta_1_graph,
        )

        metadata: Dict[str, Any] = {
            "method": "standard_ball_mapper_algorithm1",
            "N": N,
            "epsilon": eps,
            "input_mode": self.input_mode,
            "point_order": point_order_arr.tolist(),
            "selected_center_sample_indices": selected_centers,
            "n_centers": m,
            "coverage_fraction": 1.0,
            "reference_role": "metric_cover_only",
            "lens_used": False,
            "pullback_clustering_used": False,
            "centre_selection": "greedy_epsilon_net_algorithm1",
            "intersection_semantics": "observed_point_common_coverage",
            "ambient_geometric_overlap_used": False,
            "scientific_evidence_eligible": True,
            "description": "Lens-free metric-cover / geometric-connectivity summary; not a general Reeb-graph or manifold-topology recovery method.",
        }

        # 4. Simplicial Nerve 2-Skeleton Construction (M0 Reuse)
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
                metadata={**metadata, "scientific_evidence_eligible": False},
            )

        # 5. Dual Homology Computation (M0 Reuse)
        try:
            homology = compute_dual_homology(nerve)
        except HomologyConsistencyError as e:
            raise e
        except Exception as e:
            return MapperOutput(
                status="homology_failure",
                reason=f"Homology reduction error: {str(e)}",
                graph=graph,
                nerve=nerve,
                homology=None,
                metadata={**metadata, "scientific_evidence_eligible": False},
            )

        return MapperOutput(
            status="success",
            reason=None,
            graph=graph,
            nerve=nerve,
            homology=homology,
            metadata=metadata,
        )

"""F-Mapper core algorithm and pipeline implementation."""

from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union
import numpy as np
from sklearn.base import clone
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


def run_fcm_1d(
    y: np.ndarray,
    c: int,
    alpha: float = 2.0,
    tol: float = 1e-7,
    max_iter: int = 300,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Execute 1-D Fuzzy c-Means clustering on scalar lens values.

    Parameters
    ----------
    y : np.ndarray
        1-D array of shape (N_samples,) containing scalar filter values.
    c : int
        Number of fuzzy clusters / cover elements (c >= 1).
    alpha : float
        Weighting exponent / fuzzifier (default 2.0 per Bui et al.).
    tol : float
        Convergence tolerance on membership matrix difference.
    max_iter : int
        Maximum number of iterations.
    seed : Optional[int]
        Retained for interface compatibility only. Centroid initialization is deterministic
        by construction (linear spacing across observed lens range) and does not use random draws.

    Returns
    -------
    U_canon : np.ndarray
        Canonical membership matrix of shape (N_samples, c), with columns
        ordered by ascending 1-D centroid.
    centroids_canon : np.ndarray
        Sorted 1-D cluster centroids of shape (c,).
    converged : bool
        True if FCM converged within max_iter, False otherwise.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    N = len(y_arr)

    if c < 1:
        raise ValueError(f"Number of clusters c must be >= 1, got {c}")
    if N == 0:
        raise ValueError("Input lens array y is empty (N=0).")

    # Degenerate case: c = 1
    if c == 1:
        U = np.ones((N, 1), dtype=float)
        centroids = np.array([float(np.mean(y_arr))])
        return U, centroids, True

    # Check for degenerate constant lens
    if float(np.max(y_arr)) == float(np.min(y_arr)):
        # Cannot construct distinct c > 1 clusters on identical points
        return np.zeros((N, c)), np.zeros(c), False

    rng = np.random.default_rng(seed)

    # Initialize centroids deterministically using quantiles or seeded random choice
    min_val, max_val = float(np.min(y_arr)), float(np.max(y_arr))
    initial_centroids = np.linspace(min_val, max_val, c)

    # Compute initial membership matrix U
    U = np.zeros((N, c), dtype=float)
    exp = 2.0 / (alpha - 1.0)

    for i in range(N):
        dists = np.abs(y_arr[i] - initial_centroids)
        if np.any(dists == 0.0):
            zero_idx = np.where(dists == 0.0)[0]
            U[i, zero_idx] = 1.0 / len(zero_idx)
        else:
            denom_terms = (dists[:, None] / dists[None, :]) ** exp
            U[i, :] = 1.0 / np.sum(denom_terms, axis=1)

    converged = False
    centroids = initial_centroids.copy()

    for it in range(max_iter):
        U_prev = U.copy()

        # Update centroids
        u_alpha = U ** alpha
        denom = np.sum(u_alpha, axis=0)
        # Avoid division by zero in empty cluster
        denom = np.where(denom == 0.0, 1e-12, denom)
        centroids = np.sum(u_alpha * y_arr[:, None], axis=0) / denom

        # Update memberships
        for i in range(N):
            dists = np.abs(y_arr[i] - centroids)
            if np.any(dists == 0.0):
                zero_idx = np.where(dists == 0.0)[0]
                U[i, :] = 0.0
                U[i, zero_idx] = 1.0 / len(zero_idx)
            else:
                denom_terms = (dists[:, None] / dists[None, :]) ** exp
                U[i, :] = 1.0 / np.sum(denom_terms, axis=1)

        # Check convergence
        diff = np.max(np.abs(U - U_prev))
        if diff < tol:
            converged = True
            break

    # Canonical Centroid Sorting (ascending order, tie-break by column index)
    sort_keys = [(centroids[j], j) for j in range(c)]
    sort_indices = [idx for _, idx in sorted(sort_keys)]

    U_canon = U[:, sort_indices]
    centroids_canon = centroids[sort_indices]

    # Validate membership matrix
    if not np.all(np.isfinite(U_canon)):
        return U_canon, centroids_canon, False

    row_sums = np.sum(U_canon, axis=1)
    if not np.all(np.abs(row_sums - 1.0) <= 1e-4):
        return U_canon, centroids_canon, False

    return U_canon, centroids_canon, converged


class FMapper:
    """F-Mapper core implementation conforming strictly to SPEC-FM v1.0.0."""

    def __init__(
        self,
        n_intervals: int = 5,
        threshold: float = 0.2,
        fcm_fuzzifier: float = 2.0,
        fcm_tol: float = 1e-7,
        fcm_max_iter: int = 300,
        fcm_seed: Optional[int] = 42,
        clusterer: Optional[Any] = None,
        input_mode: str = "coordinates",
    ):
        if n_intervals < 1:
            raise ValueError(f"n_intervals must be >= 1, got {n_intervals}")
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0.0, 1.0], got {threshold}")
        if input_mode not in ("coordinates", "precomputed_distance"):
            raise ValueError(f"input_mode must be 'coordinates' or 'precomputed_distance', got {input_mode}")

        self.n_intervals = n_intervals
        self.threshold = threshold
        self.fcm_fuzzifier = fcm_fuzzifier
        self.fcm_tol = fcm_tol
        self.fcm_max_iter = fcm_max_iter
        self.fcm_seed = fcm_seed
        self.clusterer = clusterer if clusterer is not None else DBSCAN(eps=0.5, min_samples=1)
        self.input_mode = input_mode

    def _clone_clusterer(self) -> Any:
        """Create an independent fresh clusterer clone."""
        try:
            return clone(self.clusterer)
        except Exception:
            import copy
            return copy.deepcopy(self.clusterer)

    def _validate_inputs(
        self, X: np.ndarray, lens: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate input arrays."""
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
        injected_U: Optional[np.ndarray] = None,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
        max_triangles: Optional[int] = None,
    ) -> MapperOutput:
        """Execute F-Mapper construction."""
        try:
            X_arr, lens_arr = self._validate_inputs(X, lens)
        except ValueError as e:
            return MapperOutput(
                status="configuration_invalid",
                reason=str(e),
                graph=None,
                nerve=None,
                homology=None,
                metadata={"error": str(e), "scientific_evidence_eligible": False}
            )

        N = len(X_arr)
        f_min = float(np.min(lens_arr))
        f_max = float(np.max(lens_arr))

        metadata: Dict[str, Any] = {
            "N": N,
            "f_min": f_min,
            "f_max": f_max,
            "requested_n_intervals": self.n_intervals,
            "threshold": self.threshold,
            "input_mode": self.input_mode,
            "fcm_fuzzifier": self.fcm_fuzzifier,
            "fcm_seed": self.fcm_seed,
        }

        # Check degenerate constant lens
        if f_max == f_min and self.n_intervals > 1 and injected_U is None:
            return MapperOutput(
                status="fcm_degenerate_input",
                reason="Constant lens with n_intervals > 1 is degenerate for FCM cover construction.",
                graph=None,
                nerve=None,
                homology=None,
                metadata={**metadata, "scientific_evidence_eligible": False}
            )

        # 1. Obtain Fuzzy Membership Matrix U
        if injected_U is not None:
            U_canon = np.asarray(injected_U, dtype=float)
            if U_canon.shape[0] != N:
                raise ValueError(f"Injected U row count ({U_canon.shape[0]}) must match sample count N ({N})")
            c = U_canon.shape[1]
            centroids_canon = np.zeros(c)
            fcm_converged = True
        else:
            c = self.n_intervals
            U_canon, centroids_canon, fcm_converged = run_fcm_1d(
                y=lens_arr,
                c=c,
                alpha=self.fcm_fuzzifier,
                tol=self.fcm_tol,
                max_iter=self.fcm_max_iter,
                seed=self.fcm_seed,
            )

        metadata["fcm_converged"] = bool(fcm_converged)
        metadata["centroids"] = centroids_canon.tolist()

        if not fcm_converged:
            return MapperOutput(
                status="fcm_non_convergence",
                reason="FCM failed to converge within maximum iterations or produced invalid memberships.",
                graph=None,
                nerve=None,
                homology=None,
                metadata={**metadata, "scientific_evidence_eligible": False}
            )

        # 2. Strict Thresholding (u_ij > tau)
        tau = self.threshold
        membership_mask = (U_canon > tau)

        # 3. Check for Threshold Coverage Gaps
        cover_counts_per_sample = np.sum(membership_mask, axis=1)
        uncovered_indices = np.where(cover_counts_per_sample == 0)[0].tolist()
        has_coverage_gap = len(uncovered_indices) > 0
        coverage_fraction = float(np.count_nonzero(cover_counts_per_sample > 0) / N)

        metadata["uncovered_sample_indices"] = uncovered_indices
        metadata["threshold_coverage_fraction"] = coverage_fraction
        metadata["scientific_evidence_eligible"] = not has_coverage_gap

        # 4. Pullback Clustering & Deterministic Node Creation
        nodes: Dict[int, MapperNode] = {}
        next_node_id = 0
        pullback_records: List[Dict[str, Any]] = []

        for j in range(c):
            pullback_indices = np.where(membership_mask[:, j])[0]
            if len(pullback_indices) == 0:
                pullback_records.append({
                    "cover_element_id": j,
                    "sample_indices": [],
                    "local_cluster_labels": [],
                })
                continue

            if self.input_mode == "coordinates":
                X_sub = X_arr[pullback_indices]
            else:
                X_sub = X_arr[np.ix_(pullback_indices, pullback_indices)]

            local_clusterer = self._clone_clusterer()
            try:
                labels = local_clusterer.fit_predict(X_sub)
            except Exception as e:
                metadata["pullback_records"] = pullback_records
                return MapperOutput(
                    status="clusterer_failure",
                    reason=f"Clusterer raised exception in fuzzy cover element {j}: {str(e)}",
                    graph=None,
                    nerve=None,
                    homology=None,
                    metadata={**metadata, "scientific_evidence_eligible": False}
                )

            if labels is None or len(labels) != len(pullback_indices):
                metadata["pullback_records"] = pullback_records
                return MapperOutput(
                    status="clusterer_failure",
                    reason=f"Clusterer returned malformed label length in fuzzy cover {j}",
                    graph=None,
                    nerve=None,
                    homology=None,
                    metadata={**metadata, "scientific_evidence_eligible": False}
                )

            pullback_records.append({
                "cover_element_id": j,
                "sample_indices": [int(idx) for idx in pullback_indices],
                "local_cluster_labels": [int(lbl) for lbl in labels],
            })

            unique_labels = sorted([int(lbl) for lbl in set(labels) if lbl >= 0])
            for lbl_idx in unique_labels:
                cluster_sample_indices = pullback_indices[labels == lbl_idx]
                members = frozenset(int(idx) for idx in cluster_sample_indices)

                if len(members) > 0:
                    node = MapperNode(
                        node_id=next_node_id,
                        interval_idx=j,
                        cluster_label=lbl_idx,
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
                nodes={}, edges=[], edge_weights={}, n_nodes=0, n_edges=0, n_components=0, beta_1_graph=0
            )
            empty_nerve = SimplicialNerve2D(
                nodes=[], sigma_0=[], sigma_1=[], sigma_2=[], n_0=0, n_1=0, n_2=0, vertex_to_row={}
            )
            empty_homology = DualHomologyResult(
                beta_0_nerve=0, beta_1_nerve=0, beta_1_graph=0, rank_d1=0, rank_d2=0,
                status="empty_complex", d1_d2_zero_mod2=True, details={}
            )
            status = "coverage_gap" if has_coverage_gap else "degenerate_output"
            reason = "coverage_gap" if has_coverage_gap else "all_points_unassigned"
            return MapperOutput(
                status=status,
                reason=reason,
                graph=empty_graph,
                nerve=empty_nerve,
                homology=empty_homology,
                metadata=metadata
            )

        # 5. 1-Skeleton Graph Construction (Abridged Form)
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

        # 6. Simplicial Nerve 2-Skeleton Construction (General Form)
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
                metadata={**metadata, "scientific_evidence_eligible": False}
            )

        # 7. Dual Homology Computation
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
                metadata={**metadata, "scientific_evidence_eligible": False}
            )

        status = "coverage_gap" if has_coverage_gap else "success"
        reason = f"Coverage gap: {len(uncovered_indices)} uncovered samples" if has_coverage_gap else None

        return MapperOutput(
            status=status,
            reason=reason,
            graph=graph,
            nerve=nerve,
            homology=homology,
            metadata=metadata
        )

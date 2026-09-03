"""Source-faithful Kang-Lim Ensemble Mapper (2021) with Project-Specified Metric-Medoid Silhouette Hardening (SPEC-EM v1.1.0)."""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from mapper_framework.conventional import ConventionalMapper
from mapper_framework.exceptions import (
    ClustererFailureError,
    ConfigurationInvalidError,
    HomologyConsistencyError,
    ResourceLimitError,
)
from mapper_framework.experiment_records import serialize_for_canonical_hash
from mapper_framework.homology_gf2 import compute_1skeleton_components, compute_dual_homology
from mapper_framework.nerve import build_membership_nerve_2d
from mapper_framework.types import (
    CandidateSelectionScore,
    DualHomologyResult,
    MapperGraph,
    MapperNode,
    MapperOutput,
    SimplicialNerve2D,
)

# Unified frozen numerical tolerance constant
DEFAULT_NUMERICAL_TOL: float = 1e-12


def resolve_effective_clusterer(clusterer: Optional[Any]) -> Any:
    """Resolve effective clusterer configuration prior to canonical serialization.

    If clusterer is None, ConventionalMapper defaults to DBSCAN(eps=0.5, min_samples=1).
    This function ensures canonical candidate keys serialize the concrete effective estimator
    rather than a silent None.
    """
    if clusterer is None:
        return DBSCAN(eps=0.5, min_samples=1)
    return clusterer


def compute_semantic_node_key(node: MapperNode) -> Tuple[Tuple[int, ...], int, int]:
    """Compute the immutable semantic node key independent of graph node IDs or insertion order.

    nodekey(C) = (tuple(sorted(members)), interval_idx, cluster_label)
    """
    return (
        tuple(sorted(list(node.members))),
        int(node.interval_idx),
        int(node.cluster_label),
    )


def compute_canonical_candidate_key(candidate_def: Dict[str, Any], strict: bool = False) -> str:
    """Compute deterministic canonical candidate key from effective candidate-defining parameters.

    In strict mode, validates that all parameter values can be canonically serialized without
    unstable callables or memory address fallbacks.
    """
    cand_dict = dict(candidate_def)
    if "clusterer" in cand_dict:
        cand_dict["clusterer"] = resolve_effective_clusterer(cand_dict["clusterer"])

    serialized = serialize_for_canonical_hash(cand_dict, strict=strict)
    return json.dumps(serialized, sort_keys=True)


def validate_distance_matrix(
    D: np.ndarray,
    N: int,
    tol: float = DEFAULT_NUMERICAL_TOL,
) -> np.ndarray:
    """Validate distance matrix and return a canonical working copy."""
    return validate_and_canonicalize_distance_matrix(D, N, tol=tol)


def validate_and_canonicalize_distance_matrix(
    D: np.ndarray,
    N: int,
    tol: float = DEFAULT_NUMERICAL_TOL,
) -> np.ndarray:
    """Validate that D is a valid pairwise dissimilarity matrix and construct a canonical working copy.

    Validation conditions:
    1. Square (N, N) matching N.
    2. Finite values (no NaN/Inf).
    3. No entry D_ij < -tol.
    4. Diagonal magnitude max |D_ii| <= tol.
    5. Symmetry max |D_ij - D_ji| <= tol * max(1.0, max(D)).

    Canonical working copy construction:
    D* = (D + D.T) / 2
    diag(D*) = 0.0
    D*[D* < 0.0] = 0.0

    The caller's raw matrix D is never mutated in place.

    Parameters
    ----------
    D : np.ndarray
        Raw pairwise distance matrix.
    N : int
        Expected number of observations.
    tol : float
        Numerical tolerance.

    Returns
    -------
    D_star : np.ndarray
        Canonical working copy of shape (N, N).
    """
    D_arr = np.asarray(D, dtype=float)
    if D_arr.ndim != 2 or D_arr.shape[0] != N or D_arr.shape[1] != N:
        raise ValueError(
            f"Distance matrix must be square (N, N) with N={N}, got shape {D_arr.shape}"
        )
    if not np.all(np.isfinite(D_arr)):
        raise ValueError("Distance matrix contains non-finite values (NaN/Inf).")

    # Non-negativity check
    if np.any(D_arr < -tol):
        min_val = float(np.min(D_arr))
        raise ValueError(f"Distance matrix contains entries < -{tol}: min value is {min_val}")

    # Zero diagonal check
    diag_max = np.max(np.abs(np.diag(D_arr)))
    if diag_max > tol:
        raise ValueError(
            f"Distance matrix diagonal must be zero within tolerance, max |D_ii|={diag_max}"
        )

    # Symmetry check
    max_val = max(1.0, float(np.max(D_arr)))
    sym_diff = np.max(np.abs(D_arr - D_arr.T))
    if sym_diff > tol * max_val:
        raise ValueError(
            f"Distance matrix is not symmetric within tolerance, max |D_ij - D_ji|={sym_diff}"
        )

    # Construct internal canonical working copy without mutating caller's array
    D_star = (D_arr + D_arr.T) / 2.0
    np.fill_diagonal(D_star, 0.0)
    D_star[D_star < 0.0] = 0.0
    return D_star


def compute_node_medoid(
    members: FrozenSet[int],
    X: np.ndarray,
    input_mode: str = "coordinates",
    tol: float = DEFAULT_NUMERICAL_TOL,
) -> int:
    """Compute the metric medoid of a Mapper node support.

    The medoid is defined as m_j = argmin_{z in C_j} sum_{y in C_j} d_X(z, y).
    Ties are broken deterministically by choosing the candidate with the
    smallest integer sample index min z.

    Parameters
    ----------
    members : FrozenSet[int]
        Set of observation indices contained in the node.
    X : np.ndarray
        Coordinate array (shape N x d) or canonical pairwise distance matrix (shape N x N).
    input_mode : str
        'coordinates' or 'precomputed_distance'.
    tol : float
        Numerical tolerance for near-tie detection.

    Returns
    -------
    medoid_sample_idx : int
        Observation index of the metric medoid.
    """
    members_list = sorted(list(members))
    if len(members_list) == 0:
        raise ValueError("Cannot compute medoid of an empty node.")
    if len(members_list) == 1:
        return members_list[0]

    if input_mode == "coordinates":
        sub_X = X[members_list]
        dist_mat = cdist(sub_X, sub_X, metric="euclidean")
    elif input_mode == "precomputed_distance":
        dist_mat = X[np.ix_(members_list, members_list)]
    else:
        raise ValueError(f"Invalid input_mode '{input_mode}'. Must be 'coordinates' or 'precomputed_distance'.")

    # Sum of distances to all other members in the node
    dist_sums = np.sum(dist_mat, axis=1)
    min_dist = float(np.min(dist_sums))

    # Identify all sample indices attaining minimum distance sum within unified numerical tolerance
    candidates = [
        members_list[i]
        for i, val in enumerate(dist_sums)
        if (val - min_dist) <= tol or np.isclose(val, min_dist, atol=tol, rtol=0.0)
    ]

    # Deterministic tie rule: choose smallest sample index
    return min(candidates)


def metric_medoid_hardening(
    nodes: Sequence[MapperNode],
    N: int,
    X: np.ndarray,
    input_mode: str = "coordinates",
    tol: float = DEFAULT_NUMERICAL_TOL,
) -> Tuple[np.ndarray, Dict[Tuple[Tuple[int, ...], int, int], int]]:
    """Convert overlapping candidate Mapper node memberships into a crisp partition.

    For each observation x_n, the set of containing nodes is J_n = {j : x_n in C_j}.
    The hardened selection label is h(x_n) = argmin_{j in J_n} d_X(x_n, m_j).
    Ties among containing nodes are broken deterministically by the canonical semantic node key:
    nodekey(C) = (tuple(sorted(members)), interval_idx, cluster_label).

    Parameters
    ----------
    nodes : Sequence[MapperNode]
        List of candidate Mapper nodes.
    N : int
        Total number of observations in X.
    X : np.ndarray
        Coordinate array or precomputed distance matrix.
    input_mode : str
        'coordinates' or 'precomputed_distance'.
    tol : float
        Numerical tolerance for near-tie comparisons.

    Returns
    -------
    labels : np.ndarray
        1D integer array of shape (N,) containing hardened cluster labels in 0..K_hard-1,
        or -1 for uncovered observations.
    node_medoids : Dict[Tuple[Tuple[int, ...], int, int], int]
        Mapping from canonical node key to sample index of its metric medoid.
    """
    if len(nodes) == 0:
        return np.full(N, -1, dtype=int), {}

    if input_mode == "precomputed_distance":
        X_work = validate_and_canonicalize_distance_matrix(X, N, tol=tol)
    else:
        X_work = X

    # Sort nodes by canonical semantic node key (strictly independent of node_id or insertion order)
    canonical_nodes = sorted(
        nodes,
        key=compute_semantic_node_key,
    )

    # Compute metric medoid for each node in canonical list
    node_medoids: Dict[Tuple[Tuple[int, ...], int, int], int] = {}
    for node in canonical_nodes:
        nkey = compute_semantic_node_key(node)
        node_medoids[nkey] = compute_node_medoid(node.members, X_work, input_mode=input_mode, tol=tol)

    raw_labels = np.full(N, -1, dtype=int)
    for n in range(N):
        # Identify containing nodes
        containing = [
            node for node in canonical_nodes if n in node.members
        ]
        if len(containing) == 0:
            raw_labels[n] = -1
        elif len(containing) == 1:
            raw_labels[n] = canonical_nodes.index(containing[0])
        else:
            # Multiply-covered observation: find containing node with closest medoid
            dists: List[Tuple[float, Tuple[Tuple[int, ...], int, int], int]] = []
            for node in containing:
                nkey = compute_semantic_node_key(node)
                m_j = node_medoids[nkey]
                if input_mode == "coordinates":
                    d = float(np.linalg.norm(X_work[n] - X_work[m_j]))
                else:
                    d = float(X_work[n, m_j])
                dists.append((d, nkey, canonical_nodes.index(node)))

            # Unified tolerance comparison for assignment ties
            min_d = min(item[0] for item in dists)
            near_min_candidates = [
                item for item in dists
                if (item[0] - min_d) <= tol or np.isclose(item[0], min_d, atol=tol, rtol=0.0)
            ]
            # Tie-break deterministically by smallest semantic node key
            near_min_candidates.sort(key=lambda item: item[1])
            raw_labels[n] = near_min_candidates[0][2]

    # Map assigned canonical node indices to contiguous 0..K_hard-1 labels in canonical nodekey order
    assigned_mask = raw_labels >= 0
    assigned_indices = np.where(assigned_mask)[0]
    if len(assigned_indices) == 0:
        return raw_labels, node_medoids

    unique_assigned = sorted(list(set(raw_labels[assigned_indices])))
    c_to_label = {c_idx: lbl for lbl, c_idx in enumerate(unique_assigned)}

    final_labels = np.full(N, -1, dtype=int)
    for idx in assigned_indices:
        final_labels[idx] = c_to_label[raw_labels[idx]]

    return final_labels, node_medoids


def compute_candidate_selection_score(
    X: np.ndarray,
    mapper_output: MapperOutput,
    input_mode: str = "coordinates",
    tol: float = DEFAULT_NUMERICAL_TOL,
) -> CandidateSelectionScore:
    """Compute internal selection Silhouette score for a candidate Conventional Mapper (SPEC-EM v1.1.0).

    Evaluates full observation coverage and clustering validity prior to score computation.
    Uses metric-medoid hardening (Decision Z).

    Parameters
    ----------
    X : np.ndarray
        Point cloud coordinates or pairwise distance matrix.
    mapper_output : MapperOutput
        Result of candidate Conventional Mapper execution.
    input_mode : str
        'coordinates' or 'precomputed_distance'.
    tol : float
        Numerical tolerance.

    Returns
    -------
    result : CandidateSelectionScore
        Structured score container with eligibility, numeric score, status, and hardened labels.
    """
    if mapper_output.status != "success" or mapper_output.graph is None:
        return CandidateSelectionScore(
            eligible=False,
            score=None,
            status="invalid_candidate_output",
            reason=f"Candidate execution was unsuccessful or graph is None: status='{mapper_output.status}'",
            labels=None,
        )

    nodes = list(mapper_output.graph.nodes.values())
    if len(nodes) < 2:
        return CandidateSelectionScore(
            eligible=False,
            score=None,
            status="insufficient_candidate_nodes",
            reason=f"Candidate graph has {len(nodes)} nodes; minimum 2 required for Silhouette selection.",
            labels=None,
        )

    N = len(X)
    if input_mode == "precomputed_distance":
        try:
            X_canonical = validate_and_canonicalize_distance_matrix(X, N, tol=tol)
        except ValueError as e:
            return CandidateSelectionScore(
                eligible=False,
                score=None,
                status="invalid_distance_matrix",
                reason=str(e),
                labels=None,
            )
    else:
        X_canonical = X

    # 1. Pre-Ranking Full-Coverage Eligibility Gate (Kang-Lim §3 premise: Union C_j = X)
    for node in nodes:
        for idx in node.members:
            if not isinstance(idx, (int, np.integer)) or idx < 0 or idx >= N:
                return CandidateSelectionScore(
                    eligible=False,
                    score=None,
                    status="invalid_member_index",
                    reason=f"Node {node.node_id} contains invalid member index {idx} on universe N={N}.",
                    labels=None,
                )

    covered_set: Set[int] = set()
    for node in nodes:
        covered_set.update(node.members)

    if covered_set != set(range(N)):
        return CandidateSelectionScore(
            eligible=False,
            score=None,
            status="partial_coverage_ineligible",
            reason=f"Candidate covers {len(covered_set)}/{N} observations; exact universe coverage {{0..{N-1}}} is required.",
            labels=None,
        )

    # 2. Metric-Medoid Hardening (Decision Z)
    hard_labels, _ = metric_medoid_hardening(nodes, N, X_canonical, input_mode=input_mode, tol=tol)

    # 3. Hardened Label Vector Eligibility Checks
    unique_labels = set(hard_labels)
    k_hard = len(unique_labels)

    if k_hard < 2:
        return CandidateSelectionScore(
            eligible=False,
            score=None,
            status="single_hardened_cluster",
            reason="All observations collapsed into a single hardened cluster; Silhouette is mathematically undefined.",
            labels=hard_labels,
        )
    if k_hard == N:
        return CandidateSelectionScore(
            eligible=False,
            score=None,
            status="all_singleton_clusters",
            reason="Every observation forms an isolated singleton cluster; Silhouette is mathematically undefined.",
            labels=hard_labels,
        )

    # 4. Compute Classical Rousseeuw (1987) Silhouette Score
    try:
        if input_mode == "coordinates":
            score_val = float(
                silhouette_score(X_canonical, hard_labels, metric="euclidean")
            )
        else:
            score_val = float(
                silhouette_score(X_canonical, hard_labels, metric="precomputed")
            )
        return CandidateSelectionScore(
            eligible=True,
            score=score_val,
            status="success",
            reason=None,
            labels=hard_labels,
        )
    except Exception as e:
        return CandidateSelectionScore(
            eligible=False,
            score=None,
            status="silhouette_calculation_error",
            reason=f"Scikit-learn silhouette_score failed: {str(e)}",
            labels=hard_labels,
        )


def compute_candidate_silhouette_score(
    X: np.ndarray,
    mapper_output: MapperOutput,
    input_mode: str = "coordinates",
    raise_on_ineligible: bool = False,
) -> float:
    """Diagnostic/convenience float interface for candidate Silhouette scoring.

    NOTE: Production pipelines and candidate ranking must use compute_candidate_selection_score
    to preserve structured eligibility status and failure semantics.
    """
    res = compute_candidate_selection_score(X, mapper_output, input_mode=input_mode)
    if not res.eligible or res.score is None:
        if raise_on_ineligible:
            raise ValueError(f"Candidate is ineligible for Silhouette scoring: {res.status} ({res.reason})")
        return float("nan")
    return res.score


def compute_interset_correlation(
    node_supports: List[FrozenSet[int]],
    N: int,
) -> np.ndarray:
    """Compute Kang-Lim cluster-cluster interset correlation matrix S_c (Equation 2).

    Parameters
    ----------
    node_supports : List[FrozenSet[int]]
        List of sample-support sets for pooled base Mapper nodes.
    N : int
        Total number of observations in common universe X.

    Returns
    -------
    S_c : np.ndarray
        Symmetric matrix of shape (a_total, a_total) with values in [-1, 1].
    """
    a_total = len(node_supports)
    if a_total == 0:
        return np.empty((0, 0), dtype=float)

    S_c = np.ones((a_total, a_total), dtype=float)

    # Check for degenerate base nodes (size 0 or N)
    for i, supp in enumerate(node_supports):
        sz = len(supp)
        if sz == 0 or sz == N:
            raise ValueError(
                f"Base node {i} has degenerate support size ({sz}) on universe N={N}; "
                f"Equation (2) denominator is zero."
            )

    for i in range(a_total):
        A = node_supports[i]
        a = len(A)
        denom_a = a * (N - a)

        for j in range(i + 1, a_total):
            B = node_supports[j]
            b = len(B)
            q = len(A & B)

            numer = N * q - a * b
            denom = np.sqrt(denom_a * b * (N - b))

            corr = float(numer / denom)
            # Clip for numerical precision
            corr = max(-1.0, min(1.0, corr))

            S_c[i, j] = corr
            S_c[j, i] = corr

    return S_c


def exact_k_dendrogram_cut(
    Z: np.ndarray,
    a: int,
    K: int,
) -> Dict[int, List[int]]:
    """Execute an exact cluster-count dendrogram cut after exactly a - K merges.

    Guarantees exactly K meta-clusters (for 1 <= K <= a) independent of tied linkage heights.

    Parameters
    ----------
    Z : np.ndarray
        Linkage matrix of shape (a-1, 4) from hierarchical clustering.
    a : int
        Number of leaf nodes (pooled base nodes).
    K : int
        Requested number of meta-clusters.

    Returns
    -------
    groups : Dict[int, List[int]]
        Mapping {0: [node_indices], ..., K-1: [node_indices]}.
    """
    if K < 1 or K > a:
        raise ValueError(f"Requested cluster count K ({K}) must satisfy 1 <= K <= {a}")
    if K == a:
        return {i: [i] for i in range(a)}
    if K == 1:
        return {0: list(range(a))}

    parent = list(range(2 * a))

    def find(i: int) -> int:
        path = []
        curr = i
        while parent[curr] != curr:
            path.append(curr)
            curr = parent[curr]
        for node in path:
            parent[node] = curr
        return curr

    def union(i: int, j: int, new_id: int) -> None:
        ri = find(i)
        rj = find(j)
        parent[ri] = new_id
        parent[rj] = new_id
        parent[new_id] = new_id

    # Execute exactly a - K sequential merges
    n_merges = a - K
    for step in range(n_merges):
        c1 = int(Z[step, 0])
        c2 = int(Z[step, 1])
        new_node_id = a + step
        union(c1, c2, new_node_id)

    # Collect leaf members for each root
    cluster_buckets: Dict[int, List[int]] = {}
    for leaf in range(a):
        root = find(leaf)
        cluster_buckets.setdefault(root, []).append(leaf)

    # Initial grouping
    result: Dict[int, List[int]] = {}
    for k_idx, mems in enumerate(cluster_buckets.values()):
        result[k_idx] = sorted(mems)

    if len(result) != K:
        raise RuntimeError(
            f"Exact-K cut failed invariant: expected {K} clusters, got {len(result)}"
        )

    return result


def rcescc_consensus_clustering(
    node_supports: List[FrozenSet[int]],
    N: int,
    K: int,
    pooled_node_keys: Optional[List[Any]] = None,
) -> Tuple[Dict[int, FrozenSet[int]], np.ndarray, np.ndarray, np.ndarray]:
    """Execute RCESCC consensus clustering pipeline (Kang-Lim 2021 Equations 4-6, SPEC-EM v1.1.0).

    Parameters
    ----------
    node_supports : List[FrozenSet[int]]
        Pooled base-node supports from selected base Mappers (ordered canonically).
    N : int
        Number of observations in X.
    K : int
        Exact number of final meta-clusters.
    pooled_node_keys : Optional[List[Any]]
        Canonical pooled keys for each entry in node_supports.

    Returns
    -------
    final_supports : Dict[int, FrozenSet[int]]
        Sample supports for each final meta-cluster {0: S(omega_0), ..., K-1: S(omega_{K-1})}.
    S_c : np.ndarray
        Cluster-cluster similarity matrix.
    D_c : np.ndarray
        Distance matrix (Equation 4).
    O_c : np.ndarray
        Normalized object-cluster membership matrix (Equation 5) in canonical meta-cluster order.
    """
    a_total = len(node_supports)
    if K < 1:
        raise ValueError(f"n_meta_clusters K must be >= 1, got {K}")
    if a_total < K:
        raise ValueError(f"Total pooled base nodes ({a_total}) is less than meta-cluster count K ({K})")

    # Canonicalize leaf order if pooled_node_keys provided
    if pooled_node_keys is not None and len(pooled_node_keys) == a_total:
        combined = list(zip(pooled_node_keys, node_supports))
        combined.sort(key=lambda item: item[0])
        pooled_node_keys = [item[0] for item in combined]
        node_supports = [item[1] for item in combined]

    # 1. Interset correlation matrix S_c (Equation 2)
    S_c = compute_interset_correlation(node_supports, N)

    # 2. Distance matrix D_c = (1 - S_c) / 2 (Equation 4)
    D_c = (1.0 - S_c) / 2.0
    np.fill_diagonal(D_c, 0.0)
    D_c = np.clip(D_c, 0.0, 1.0)

    # 3. Hierarchical Average-Linkage Clustering with Exact-K Dendrogram Cut
    if a_total > 1:
        condensed_D = squareform(D_c, checks=False)
        Z = linkage(condensed_D, method="average")
        raw_meta_clusters = exact_k_dendrogram_cut(Z, a_total, K)
    else:
        raw_meta_clusters = {0: [0]}

    # 4. Canonicalize Meta-Cluster Identities by metakey
    # metakey(omega) = tuple(sorted([pooled_node_keys[idx] for idx in cluster_indices]))
    if pooled_node_keys is not None and len(pooled_node_keys) == a_total:
        meta_items = []
        for raw_k, base_indices in raw_meta_clusters.items():
            mkey = tuple(sorted([pooled_node_keys[idx] for idx in base_indices]))
            meta_items.append((mkey, sorted(base_indices)))
        meta_items.sort(key=lambda item: item[0])
        canonical_meta_clusters = {k_idx: mems for k_idx, (_, mems) in enumerate(meta_items)}
    else:
        # Fallback sorting by smallest base node index
        sorted_mems = sorted(raw_meta_clusters.values(), key=lambda mems: sorted(mems)[0])
        canonical_meta_clusters = {k_idx: mems for k_idx, mems in enumerate(sorted_mems)}

    # 5. Object-to-Cluster Support Score s(x_n, omega_k) (Equation 6)
    s_scores = np.zeros((N, K), dtype=float)
    for k in range(K):
        base_indices = canonical_meta_clusters[k]
        omega_size = len(base_indices)
        if omega_size == 0:
            continue

        for n in range(N):
            cnt = sum(1 for b_idx in base_indices if n in node_supports[b_idx])
            s_scores[n, k] = cnt / omega_size

    # 6. Normalized Object-Cluster Membership Matrix O_c (Equation 5)
    row_sums = np.sum(s_scores, axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("Encountered observation with zero total meta-cluster support.")

    O_c = s_scores / row_sums[:, None]

    # 7. Final Meta-Node Supports S(omega_k) = {n | O_c[n, k] > 0} (Preserving overlap)
    final_supports: Dict[int, FrozenSet[int]] = {}
    for k in range(K):
        in_meta = np.where(O_c[:, k] > 0.0)[0]
        final_supports[k] = frozenset(int(idx) for idx in in_meta)

    return final_supports, S_c, D_c, O_c


class KangLimEnsembleMapper:
    """Kang-Lim RCESCC Ensemble Mapper with Project-Specified Metric-Medoid Silhouette Hardening (SPEC-EM v1.1.0)."""

    def __init__(
        self,
        n_meta_clusters: Optional[int] = None,
        n_selected_base_mappers: int = 10,
        clusterer: Optional[Any] = None,
        input_mode: str = "coordinates",
        candidate_pool_id: Optional[str] = None,
        strict_scientific: bool = False,
    ):
        if n_meta_clusters is not None and n_meta_clusters < 1:
            raise ValueError(f"n_meta_clusters must be >= 1, got {n_meta_clusters}")
        if n_selected_base_mappers < 1:
            raise ValueError(f"n_selected_base_mappers must be >= 1, got {n_selected_base_mappers}")
        if input_mode not in ("coordinates", "precomputed_distance"):
            raise ValueError(f"input_mode must be 'coordinates' or 'precomputed_distance', got {input_mode}")

        self.n_meta_clusters = n_meta_clusters
        self.n_selected_base_mappers = n_selected_base_mappers
        self.clusterer = clusterer
        self.input_mode = input_mode
        self.candidate_pool_id = candidate_pool_id
        self.strict_scientific = strict_scientific

    def fit_transform(
        self,
        X: np.ndarray,
        lens: np.ndarray,
        n_meta_clusters: Optional[int] = None,
        candidate_params: Optional[List[Tuple[int, float]]] = None,
        injected_candidates: Optional[List[MapperOutput]] = None,
        injected_scores: Optional[List[float]] = None,
        lens_id: Optional[str] = None,
        data_seed: Optional[int] = None,
        construction_seed: Optional[int] = None,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
        max_triangles: Optional[int] = None,
    ) -> MapperOutput:
        """Execute Kang-Lim Ensemble Mapper pipeline with metric-medoid selection hardening."""
        # Resolve K parameter (no implicit hidden default allowed)
        K = n_meta_clusters if n_meta_clusters is not None else self.n_meta_clusters
        if K is None:
            raise ConfigurationInvalidError(
                "n_meta_clusters (K) must be explicitly specified; no implicit scientific default is permitted under SPEC-EM v1.1.0."
            )

        # ------------------------------------------------------------------
        # Execution-Mode and Firewall Invariants
        # ------------------------------------------------------------------
        is_diagnostic_mode = False
        diagnostic_reasons: List[str] = []

        if not self.strict_scientific:
            is_diagnostic_mode = True
            diagnostic_reasons.append("non_strict_mode")

        # 1. Injected Scores Firewall
        if injected_scores is not None:
            if self.strict_scientific:
                raise ConfigurationInvalidError(
                    "injected_scores bypass is strictly prohibited in scientific/confirmatory execution mode."
                )
            is_diagnostic_mode = True
            diagnostic_reasons.append("injected_scores_bypass_used")

        # 2. Injected Candidates Firewall
        if injected_candidates is not None:
            if self.strict_scientific:
                raise ConfigurationInvalidError(
                    "injected_candidates is strictly prohibited in scientific/confirmatory execution mode."
                )
            is_diagnostic_mode = True
            diagnostic_reasons.append("injected_candidates_used")

        # 3. Canonical Base-Mapper Count Firewall
        if self.n_selected_base_mappers != 10:
            if self.strict_scientific:
                raise ConfigurationInvalidError(
                    f"Canonical scientific execution requires exactly n_selected_base_mappers=10, got {self.n_selected_base_mappers}."
                )
            is_diagnostic_mode = True
            diagnostic_reasons.append(f"non_canonical_n_selected_{self.n_selected_base_mappers}")

        # 4. Lens Identifier Validation
        lens_id_str = str(lens_id).strip() if lens_id is not None else ""
        if self.strict_scientific:
            if not lens_id_str or lens_id_str == "unspecified_lens":
                raise ConfigurationInvalidError(
                    "lens_id must be a non-empty, non-placeholder identifier in strict scientific mode."
                )
        else:
            if not lens_id_str or lens_id_str == "unspecified_lens":
                is_diagnostic_mode = True
                diagnostic_reasons.append("unspecified_lens_id")
                lens_id_str = "unspecified_lens"

        # 5. Candidate Pool Identifier Validation
        pool_id_str = str(self.candidate_pool_id).strip() if self.candidate_pool_id is not None else ""
        if self.strict_scientific:
            if not pool_id_str or pool_id_str == "unspecified_pool":
                raise ConfigurationInvalidError(
                    "candidate_pool_id must be a non-empty, non-placeholder identifier in strict scientific mode."
                )
        else:
            if not pool_id_str or pool_id_str == "unspecified_pool":
                is_diagnostic_mode = True
                diagnostic_reasons.append("unspecified_candidate_pool_id")
                pool_id_str = "unspecified_pool"

        # 6. Candidate Grid Firewall
        if injected_candidates is None:
            if candidate_params is None:
                if self.strict_scientific:
                    raise ConfigurationInvalidError(
                        "candidate_params must be explicitly specified in scientific/confirmatory mode; hidden fallback candidate grid is prohibited."
                    )
                is_diagnostic_mode = True
                diagnostic_reasons.append("exploratory_fallback_candidate_grid")
                candidate_params = [
                    (l_val, p_val)
                    for l_val in range(4, 13)
                    for p_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                ]

        X_arr = np.asarray(X)
        lens_arr = np.asarray(lens, dtype=float)
        N = len(X_arr)

        if N == 0:
            return MapperOutput(
                status="configuration_invalid",
                reason="Input dataset X is empty (N=0).",
                graph=None,
                nerve=None,
                homology=None,
                metadata={"scientific_evidence_eligible": False},
            )

        if self.input_mode == "precomputed_distance":
            try:
                X_arr = validate_and_canonicalize_distance_matrix(X_arr, N)
            except ValueError as e:
                return MapperOutput(
                    status="configuration_invalid",
                    reason=f"Invalid distance matrix: {str(e)}",
                    graph=None,
                    nerve=None,
                    homology=None,
                    metadata={"scientific_evidence_eligible": False},
                )

        # ------------------------------------------------------------------
        # Candidate Generation, Scoring, and Pre-Ranking Eligibility Filtering
        # ------------------------------------------------------------------
        selected_candidates: List[MapperOutput] = []
        selected_scores: List[float] = []
        selected_candidate_keys: List[str] = []
        candidate_provenance_records: List[Dict[str, Any]] = []

        if injected_candidates is not None:
            seen_injected_keys: Set[str] = set()
            eligible_pool: List[Tuple[float, str, MapperOutput]] = []

            for i, cand in enumerate(injected_candidates):
                cand_key = cand.metadata.get("candidate_key") if cand.metadata else None
                if cand_key is None:
                    if self.strict_scientific:
                        raise ConfigurationInvalidError(
                            "Injected candidate lacks canonical candidate_key / configuration metadata in strict scientific mode."
                        )
                    # For diagnostic runs, generate a structural candidate key strictly without using enumeration index
                    cand_structure = {
                        "type": "diagnostic_injected_candidate",
                        "n_nodes": cand.graph.n_nodes if cand.graph else 0,
                        "n_edges": cand.graph.n_edges if cand.graph else 0,
                        "node_keys": [
                            compute_semantic_node_key(n) for n in cand.graph.nodes.values()
                        ] if cand.graph else [],
                    }
                    cand_key = compute_canonical_candidate_key(cand_structure, strict=False)
                    is_diagnostic_mode = True
                    diagnostic_reasons.append("unidentified_injected_candidate")

                if cand_key in seen_injected_keys:
                    raise ConfigurationInvalidError(
                        f"Duplicate candidate configuration detected in candidate pool: '{cand_key}'"
                    )
                seen_injected_keys.add(cand_key)

                if injected_scores is not None and i < len(injected_scores):
                    score_val = injected_scores[i]
                    is_elig = cand.status == "success" and cand.graph is not None and cand.graph.n_nodes >= 2
                    status_str = "injected_score" if is_elig else "ineligible_injected_candidate"
                    reason_str = None if is_elig else "Injected candidate status was not success or graph was None/degenerate."
                else:
                    score_res = compute_candidate_selection_score(X_arr, cand, input_mode=self.input_mode)
                    score_val = score_res.score if (score_res.eligible and score_res.score is not None) else None
                    is_elig = score_res.eligible
                    status_str = score_res.status
                    reason_str = score_res.reason

                candidate_provenance_records.append({
                    "candidate_id": i,
                    "candidate_key": cand_key,
                    "status": status_str,
                    "eligible": is_elig,
                    "score": score_val,
                    "reason": reason_str,
                })

                if is_elig and score_val is not None:
                    eligible_pool.append((score_val, cand_key, cand))

            if len(eligible_pool) < self.n_selected_base_mappers:
                return MapperOutput(
                    status="insufficient_valid_candidates",
                    reason=f"Eligible injected candidate count ({len(eligible_pool)}) is less than required ({self.n_selected_base_mappers})",
                    graph=None,
                    nerve=None,
                    homology=None,
                    metadata={
                        "candidate_records": candidate_provenance_records,
                        "scientific_evidence_eligible": False,
                    },
                )

            # Deterministic selection by (-score, candidate_key)
            eligible_pool.sort(key=lambda x: (-x[0], x[1]))
            top_pairs = eligible_pool[: self.n_selected_base_mappers]
            selected_candidates = [pair[2] for pair in top_pairs]
            selected_scores = [pair[0] for pair in top_pairs]
            selected_candidate_keys = [pair[1] for pair in top_pairs]

        else:
            seen_grid_keys: Set[str] = set()
            eligible_pool: List[Tuple[float, str, MapperOutput]] = []

            for c_id, (l_val, p_val) in enumerate(candidate_params):
                eff_clusterer = resolve_effective_clusterer(self.clusterer)
                cand_def = {
                    "method": "conventional_mapper",
                    "n_intervals": int(l_val),
                    "overlap_frac": float(p_val),
                    "clusterer": eff_clusterer,
                    "input_mode": str(self.input_mode),
                    "lens_id": str(lens_id_str),
                    "candidate_pool_id": str(pool_id_str),
                    "data_seed": data_seed,
                    "construction_seed": construction_seed,
                }
                cand_key = compute_canonical_candidate_key(cand_def, strict=self.strict_scientific)

                if cand_key in seen_grid_keys:
                    raise ConfigurationInvalidError(
                        f"Duplicate candidate configuration detected in candidate grid: '{cand_key}'"
                    )
                seen_grid_keys.add(cand_key)

                cm = ConventionalMapper(
                    n_intervals=l_val,
                    overlap_frac=p_val,
                    clusterer=self.clusterer,
                    input_mode=self.input_mode,
                )
                c_out = cm.fit_transform(X_arr, lens_arr)
                score_res = compute_candidate_selection_score(X_arr, c_out, input_mode=self.input_mode)

                candidate_provenance_records.append({
                    "candidate_id": c_id,
                    "candidate_key": cand_key,
                    "params": (l_val, p_val),
                    "status": score_res.status,
                    "eligible": score_res.eligible,
                    "score": score_res.score,
                    "reason": score_res.reason,
                })

                if score_res.eligible and score_res.score is not None:
                    eligible_pool.append((score_res.score, cand_key, c_out))

            if len(eligible_pool) < self.n_selected_base_mappers:
                return MapperOutput(
                    status="insufficient_valid_candidates",
                    reason=f"Eligible candidate count ({len(eligible_pool)}) is less than required ({self.n_selected_base_mappers})",
                    graph=None,
                    nerve=None,
                    homology=None,
                    metadata={
                        "candidate_records": candidate_provenance_records,
                        "scientific_evidence_eligible": False,
                    },
                )

            # Select top N with deterministic ranking by (-score, candidate_key)
            eligible_pool.sort(key=lambda x: (-x[0], x[1]))
            top_selected = eligible_pool[: self.n_selected_base_mappers]
            selected_candidates = [item[2] for item in top_selected]
            selected_scores = [item[0] for item in top_selected]
            selected_candidate_keys = [item[1] for item in top_selected]

        # ------------------------------------------------------------------
        # Pool Base Nodes with Canonical Pooled-Node Keys
        # ------------------------------------------------------------------
        pooled_entries: List[Tuple[Tuple[str, Tuple[Tuple[int, ...], int, int]], FrozenSet[int]]] = []
        for c_idx, cand in enumerate(selected_candidates):
            cand_key = selected_candidate_keys[c_idx]
            if cand.graph is not None:
                for node in cand.graph.nodes.values():
                    nkey = compute_semantic_node_key(node)
                    pkey = (cand_key, nkey)
                    pooled_entries.append((pkey, node.members))

        # Sort all pooled base nodes lexicographically by pooledkey before S_c and D_c construction
        pooled_entries.sort(key=lambda item: item[0])
        pooled_node_keys = [item[0] for item in pooled_entries]
        pooled_node_supports = [item[1] for item in pooled_entries]

        # ------------------------------------------------------------------
        # RCESCC Consensus Clustering with Canonical Meta-Cluster Ordering
        # ------------------------------------------------------------------
        try:
            final_supports, S_c, D_c, O_c = rcescc_consensus_clustering(
                node_supports=pooled_node_supports,
                N=N,
                K=K,
                pooled_node_keys=pooled_node_keys,
            )
        except ValueError as e:
            return MapperOutput(
                status="consensus_clustering_failure",
                reason=str(e),
                graph=None,
                nerve=None,
                homology=None,
                metadata={"error": str(e), "scientific_evidence_eligible": False},
            )

        # ------------------------------------------------------------------
        # Build Final Ensemble Mapper Graph in Canonical Meta-Cluster Order
        # ------------------------------------------------------------------
        final_nodes: Dict[int, MapperNode] = {}
        for k in range(K):
            members = final_supports[k]
            mean_f = float(np.mean(lens_arr[list(members)])) if len(members) > 0 else 0.0
            final_nodes[k] = MapperNode(
                node_id=k,
                interval_idx=k,
                cluster_label=0,
                members=members,
                size=len(members),
                mean_filter=mean_f,
            )

        edges: List[Tuple[int, int]] = []
        edge_weights: Dict[Tuple[int, int], int] = {}
        for u in range(K):
            for v in range(u + 1, K):
                inter_len = len(final_nodes[u].members & final_nodes[v].members)
                if inter_len > 0:
                    edges.append((u, v))
                    edge_weights[(u, v)] = inter_len

        C = compute_1skeleton_components(K, edges)
        beta_1_graph = len(edges) - K + C

        graph = MapperGraph(
            nodes=final_nodes,
            edges=edges,
            edge_weights=edge_weights,
            n_nodes=K,
            n_edges=len(edges),
            n_components=C,
            beta_1_graph=beta_1_graph,
        )

        # ------------------------------------------------------------------
        # Scientific Evidence Eligibility Derivation
        # ------------------------------------------------------------------
        is_evidence_eligible = (
            self.strict_scientific is True
            and not is_diagnostic_mode
            and len(diagnostic_reasons) == 0
            and self.n_selected_base_mappers == 10
            and injected_scores is None
            and injected_candidates is None
            and candidate_params is not None
            and bool(pool_id_str and pool_id_str != "unspecified_pool")
            and bool(lens_id_str and lens_id_str != "unspecified_lens")
            and len(selected_candidates) == 10
            and all(r["status"] != "injected_score" for r in candidate_provenance_records if r["eligible"])
        )

        metadata: Dict[str, Any] = {
            "method": "kang_lim_ensemble_mapper_metric_medoid_2021",
            "method_identity": "Kang-Lim RCESCC Ensemble Mapper with project-specified metric-medoid Silhouette hardening",
            "execution_mode": "canonical_scientific" if is_evidence_eligible else ("diagnostic:" + ",".join(diagnostic_reasons) if diagnostic_reasons else "diagnostic:non_strict_mode"),
            "N": N,
            "n_selected_base_mappers": self.n_selected_base_mappers,
            "n_total_pooled_base_nodes": len(pooled_node_supports),
            "n_meta_clusters": K,
            "linkage": "average",
            "cluster_similarity": "kang_lim_interset_correlation",
            "distance_transform": "(1-S_c)/2",
            "object_meta_membership": "KangLim_Eq5_Eq6",
            "final_membership_rule": "O_c > 0",
            "selection_hardening": "metric_medoid",
            "numerical_tolerance": DEFAULT_NUMERICAL_TOL,
            "point_cooccurrence_used": False,
            "threshold_graph_used": False,
            "fitzpatrick_hybrid_used": False,
            "selection_statistic_is_evaluation_metric": False,
            "selected_scores": selected_scores,
            "selected_candidate_keys": selected_candidate_keys,
            "candidate_provenance": candidate_provenance_records,
            "scientific_evidence_eligible": is_evidence_eligible,
        }

        # ------------------------------------------------------------------
        # Simplicial Nerve 2-Skeleton Construction (M0 Reuse)
        # ------------------------------------------------------------------
        node_supports_dict = {node.node_id: node.members for node in final_nodes.values()}
        try:
            nerve = build_membership_nerve_2d(
                node_supports=node_supports_dict,
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

        # ------------------------------------------------------------------
        # Dual Homology Computation (M0 Reuse)
        # ------------------------------------------------------------------
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

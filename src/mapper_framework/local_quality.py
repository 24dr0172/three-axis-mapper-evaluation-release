"""Reusable Part-II Local-Quality Evaluation Layer, External-Label Diagnostics,

and Membership Persistence Infrastructure.

Governing Authorities:
- N1_FINAL_HUMAN_DECISION_RECORD.md (Decisions 1A-1D, 2A-2F, 3, 5, 7)
- AXIS2_LOCAL_QUALITY_SPEC.md (as amended Part-II-wide by Amendment A2)
- N1_PHASE_MATERIALIZATION_PLAN_v2.md (Amendments A1-A4)
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import sklearn.metrics


# =============================================================================
# 1. Part-II-Wide Frozen Common Metric Panel (Amendment A2)
# =============================================================================

@dataclass(frozen=True)
class Axis2MetricSpecification:
    metric_id: str
    name: str
    direction: str  # "higher_is_better" | "lower_is_better"
    geometry: str   # "euclidean"
    scope: str      # "local_pullback_nonnoise"
    description: str


FROZEN_AXIS2_METRIC_PANEL: Dict[str, Axis2MetricSpecification] = {
    "silhouette_euclidean": Axis2MetricSpecification(
        metric_id="silhouette_euclidean",
        name="Euclidean Sample Silhouette",
        direction="higher_is_better",
        geometry="euclidean",
        scope="local_pullback_nonnoise",
        description="Local compactness-separation via sklearn.metrics.silhouette_score on non-noise observations.",
    ),
    "davies_bouldin_euclidean": Axis2MetricSpecification(
        metric_id="davies_bouldin_euclidean",
        name="Euclidean Davies-Bouldin Index",
        direction="lower_is_better",
        geometry="euclidean",
        scope="local_pullback_nonnoise",
        description="Local compactness-separation via sklearn.metrics.davies_bouldin_score on non-noise observations with explicit degeneracy guards.",
    ),
}


# =============================================================================
# 2. Local Pullback Quality Evaluator (Decisions 1B, 1C)
# =============================================================================

def evaluate_pullback_quality(
    X_pullback: np.ndarray,
    labels_pullback: np.ndarray,
    external_labels: Optional[np.ndarray] = None,
    pullback_id: Optional[Union[str, int]] = None,
    cover_element_id: Optional[Union[str, int]] = None,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate local clustering quality, noise retention, and external-label diagnostics for a single pullback.
    
    Parameters
    ----------
    X_pullback : np.ndarray
        Array of coordinates for observations in this pullback (shape: n_obs, n_features).
    labels_pullback : np.ndarray
        Cluster labels assigned by the local clusterer (noise labeled as -1).
    external_labels : Optional[np.ndarray]
        Ground-truth or external partition labels for observations in this pullback.
    """
    n_total = int(len(X_pullback))
    labels_arr = np.asarray(labels_pullback, dtype=int)
    
    if n_total == 0:
        return {
            "pullback_id": pullback_id,
            "cover_element_id": cover_element_id,
            "pullback_status": "empty_pullback",
            "pullback_n_total": 0,
            "pullback_n_noise": 0,
            "pullback_noise_fraction": 0.0,
            "pullback_n_nonnoise": 0,
            "dbscan_eps": dbscan_eps,
            "dbscan_min_samples": dbscan_min_samples,
            "dbscan_noise_structurally_possible": (dbscan_min_samples > 1) if dbscan_min_samples is not None else None,
            "local_cluster_count": 0,
            "silhouette_eligible": False,
            "silhouette_value": None,
            "silhouette_reason": "empty_pullback",
            "davies_bouldin_eligible": False,
            "davies_bouldin_value": None,
            "davies_bouldin_reason": "empty_pullback",
            "n_coincident_centroid_pairs": 0,
            "local_ari": None,
            "local_nmi": None,
            "n_true_classes": 0,
            "n_predicted_clusters": 0,
            "external_label_informative": False,
        }

    is_noise = (labels_arr == -1)
    n_noise = int(np.sum(is_noise))
    n_nonnoise = n_total - n_noise
    noise_fraction = float(n_noise / n_total)
    noise_possible = (dbscan_min_samples > 1) if dbscan_min_samples is not None else None

    if n_nonnoise == 0:
        return {
            "pullback_id": pullback_id,
            "cover_element_id": cover_element_id,
            "pullback_status": "all_noise",
            "pullback_n_total": n_total,
            "pullback_n_noise": n_noise,
            "pullback_noise_fraction": 1.0,
            "pullback_n_nonnoise": 0,
            "dbscan_eps": dbscan_eps,
            "dbscan_min_samples": dbscan_min_samples,
            "dbscan_noise_structurally_possible": noise_possible,
            "local_cluster_count": 0,
            "silhouette_eligible": False,
            "silhouette_value": None,
            "silhouette_reason": "all_noise",
            "davies_bouldin_eligible": False,
            "davies_bouldin_value": None,
            "davies_bouldin_reason": "all_noise",
            "n_coincident_centroid_pairs": 0,
            "local_ari": None,
            "local_nmi": None,
            "n_true_classes": 0,
            "n_predicted_clusters": 0,
            "external_label_informative": False,
        }

    X_nonnoise = X_pullback[~is_noise]
    labels_nonnoise = labels_arr[~is_noise]
    unique_clusters = np.unique(labels_nonnoise)
    k = int(len(unique_clusters))

    # 1. Silhouette Evaluation
    sil_eligible = False
    sil_val = None
    sil_reason = None

    if n_nonnoise < 2:
        sil_reason = "too_few_nonnoise"
    elif k < 2:
        sil_reason = "single_cluster"
    elif k >= n_nonnoise:
        sil_reason = "all_singletons_or_k_equals_n"
    elif not np.all(np.isfinite(X_nonnoise)):
        sil_reason = "nonfinite_input"
    else:
        try:
            sil_score = sklearn.metrics.silhouette_score(X_nonnoise, labels_nonnoise, metric="euclidean")
            if np.isfinite(sil_score):
                sil_val = float(sil_score)
                sil_eligible = True
            else:
                sil_reason = "metric_failure"
        except Exception:
            sil_reason = "metric_failure"

    # 2. Davies-Bouldin Evaluation with Degeneracy Guards
    db_eligible = False
    db_val = None
    db_reason = None
    n_coincident = 0

    if n_nonnoise < 2:
        db_reason = "too_few_nonnoise"
    elif k < 2:
        db_reason = "single_cluster"
    elif k >= n_nonnoise:
        db_reason = "all_singletons_or_k_equals_n"
    elif not np.all(np.isfinite(X_nonnoise)):
        db_reason = "nonfinite_input"
    else:
        # Compute cluster centroids and dispersions
        centroids = []
        dispersions = []
        for c in unique_clusters:
            c_pts = X_nonnoise[labels_nonnoise == c]
            c_mean = np.mean(c_pts, axis=0)
            centroids.append(c_mean)
            # Average distance of points in cluster to centroid
            if len(c_pts) > 0:
                dists = np.linalg.norm(c_pts - c_mean, axis=1)
                dispersions.append(float(np.mean(dists)))
            else:
                dispersions.append(0.0)
        
        centroids_arr = np.array(centroids)
        # Check pairwise centroid distances
        for i in range(k):
            for j in range(i + 1, k):
                if np.linalg.norm(centroids_arr[i] - centroids_arr[j]) < 1e-12:
                    n_coincident += 1

        all_zero_dispersion = all(d < 1e-12 for d in dispersions)
        all_coincident = (n_coincident == (k * (k - 1)) // 2)

        if all_zero_dispersion or all_coincident or n_coincident > 0:
            db_eligible = False
            db_val = None
            db_reason = "degenerate_partition"
        else:
            try:
                db_score = sklearn.metrics.davies_bouldin_score(X_nonnoise, labels_nonnoise)
                if np.isfinite(db_score) and db_score >= 0.0:
                    # Sanity check: a returned 0.0 cannot be accepted blindly
                    if db_score < 1e-12 and (all_zero_dispersion or all_coincident):
                        db_eligible = False
                        db_val = None
                        db_reason = "degenerate_partition"
                    else:
                        db_val = float(db_score)
                        db_eligible = True
                else:
                    db_reason = "metric_failure"
            except Exception:
                db_reason = "metric_failure"

    # 3. External Label Diagnostics (ARI / NMI)
    local_ari = None
    local_nmi = None
    n_true = 0
    ext_informative = False

    if external_labels is not None:
        ext_arr = np.asarray(external_labels)[~is_noise]
        n_true = int(len(np.unique(ext_arr)))
        ext_informative = (n_true >= 2)
        if n_nonnoise >= 2:
            try:
                local_ari = float(sklearn.metrics.adjusted_rand_score(ext_arr, labels_nonnoise))
            except Exception:
                local_ari = None
            try:
                local_nmi = float(sklearn.metrics.normalized_mutual_info_score(ext_arr, labels_nonnoise))
            except Exception:
                local_nmi = None

    return {
        "pullback_id": pullback_id,
        "cover_element_id": cover_element_id,
        "pullback_status": "success",
        "pullback_n_total": n_total,
        "pullback_n_noise": n_noise,
        "pullback_noise_fraction": noise_fraction,
        "pullback_n_nonnoise": n_nonnoise,
        "dbscan_eps": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
        "dbscan_noise_structurally_possible": noise_possible,
        "local_cluster_count": k,
        "silhouette_eligible": sil_eligible,
        "silhouette_value": sil_val,
        "silhouette_reason": sil_reason,
        "davies_bouldin_eligible": db_eligible,
        "davies_bouldin_value": db_val,
        "davies_bouldin_reason": db_reason,
        "n_coincident_centroid_pairs": n_coincident,
        "local_ari": local_ari,
        "local_nmi": local_nmi,
        "n_true_classes": n_true,
        "n_predicted_clusters": k,
        "external_label_informative": ext_informative,
    }


# =============================================================================
# 3. Node-Level External-Label Diagnostics (Decision 1B, Spec §9)
# =============================================================================

def compute_node_purity(
    node_member_indices: Sequence[int],
    external_labels: Sequence[Any],
) -> float:
    """Compute node purity: purity(v) = max_c |{i in S(v) : y_i = c}| / |S(v)|."""
    if not node_member_indices:
        return 0.0
    labels = [external_labels[i] for i in node_member_indices]
    counts: Dict[Any, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    max_count = max(counts.values()) if counts else 0
    return float(max_count / len(node_member_indices))


def compute_normalized_node_entropy(
    node_member_indices: Sequence[int],
    external_labels: Sequence[Any],
    n_classes: int = 10,
) -> float:
    """Compute normalized node label entropy: H_norm(v) = - sum_c p_c log(p_c) / log(n_classes)."""
    if not node_member_indices or n_classes <= 1:
        return 0.0
    labels = [external_labels[i] for i in node_member_indices]
    counts: Dict[Any, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    n_total = len(node_member_indices)
    
    h = 0.0
    for cnt in counts.values():
        p = cnt / n_total
        if p > 0.0:
            h -= p * math.log(p)
    
    h_norm = h / math.log(n_classes)
    return float(np.clip(h_norm, 0.0, 1.0))


def compute_class_fragmentation(
    node_members_dict: Dict[Any, Sequence[int]],
    external_labels: Sequence[Any],
    n_classes: int = 10,
) -> Tuple[Dict[int, int], float]:
    """Compute class fragmentation F_c = number of Mapper nodes containing at least one member of class c.
    
    Returns
    -------
    class_node_counts : Dict[int, int]
        Mapping from class label c to node count F_c.
    macro_mean_fragmentation : float
        Arithmetic mean of F_c across all classes.
    """
    class_node_counts: Dict[int, int] = {c: 0 for c in range(n_classes)}
    for node_id, member_indices in node_members_dict.items():
        node_classes = {external_labels[i] for i in member_indices}
        for c in node_classes:
            if c in class_node_counts:
                class_node_counts[c] += 1
            else:
                class_node_counts[c] = 1

    macro_mean = float(np.mean(list(class_node_counts.values()))) if class_node_counts else 0.0
    return class_node_counts, macro_mean


# =============================================================================
# 4. Construction-Level Summary Aggregator (Decision 1D)
# =============================================================================

def aggregate_axis2_construction_metrics(
    pullback_records: List[Dict[str, Any]],
    n_total_observations: int,
    node_members_dict: Optional[Dict[Any, Sequence[int]]] = None,
    external_labels: Optional[Sequence[Any]] = None,
    n_classes: int = 10,
) -> Dict[str, Any]:
    """Aggregate pullback-level records into construction-level macro and incidence-weighted summaries."""
    n_declared = len(pullback_records)
    n_nonempty = sum(1 for p in pullback_records if p.get("pullback_n_total", 0) > 0)

    # 1. Silhouette Aggregation
    sil_eligible_pullbacks = [p for p in pullback_records if p.get("silhouette_eligible") is True]
    n_sil_eligible = len(sil_eligible_pullbacks)
    e_all_sil = float(n_sil_eligible / n_declared) if n_declared > 0 else 0.0
    e_nonempty_sil = float(n_sil_eligible / n_nonempty) if n_nonempty > 0 else 0.0

    if n_sil_eligible > 0:
        sil_macro = float(np.mean([p["silhouette_value"] for p in sil_eligible_pullbacks]))
        sil_weights = [p["pullback_n_nonnoise"] for p in sil_eligible_pullbacks]
        sum_w = sum(sil_weights)
        if sum_w > 0:
            sil_incidence = float(np.average([p["silhouette_value"] for p in sil_eligible_pullbacks], weights=sil_weights))
        else:
            sil_incidence = None
        sil_agg_reason = None
    else:
        sil_macro = None
        sil_incidence = None
        sil_agg_reason = "no_eligible_pullbacks"

    # 2. Davies-Bouldin Aggregation
    db_eligible_pullbacks = [p for p in pullback_records if p.get("davies_bouldin_eligible") is True]
    n_db_eligible = len(db_eligible_pullbacks)
    e_all_db = float(n_db_eligible / n_declared) if n_declared > 0 else 0.0
    e_nonempty_db = float(n_db_eligible / n_nonempty) if n_nonempty > 0 else 0.0

    if n_db_eligible > 0:
        db_macro = float(np.mean([p["davies_bouldin_value"] for p in db_eligible_pullbacks]))
        db_weights = [p["pullback_n_nonnoise"] for p in db_eligible_pullbacks]
        sum_w = sum(db_weights)
        if sum_w > 0:
            db_incidence = float(np.average([p["davies_bouldin_value"] for p in db_eligible_pullbacks], weights=db_weights))
        else:
            db_incidence = None
        db_agg_reason = None
    else:
        db_macro = None
        db_incidence = None
        db_agg_reason = "no_eligible_pullbacks"

    # 3. Noise Summaries
    nonempty_pullbacks = [p for p in pullback_records if p.get("pullback_n_total", 0) > 0]
    if nonempty_pullbacks:
        noise_macro = float(np.mean([p["pullback_noise_fraction"] for p in nonempty_pullbacks]))
        total_noise_count = sum(p["pullback_n_noise"] for p in nonempty_pullbacks)
        total_pullback_count = sum(p["pullback_n_total"] for p in nonempty_pullbacks)
        noise_incidence = float(total_noise_count / total_pullback_count) if total_pullback_count > 0 else 0.0
    else:
        noise_macro = 0.0
        noise_incidence = 0.0

    # 4. Global Retention Tracking (Observation-level, no overlap double-counting)
    if node_members_dict:
        clustered_obs: Set[int] = set()
        for members in node_members_dict.values():
            clustered_obs.update(members)
        globally_unclustered_count = int(n_total_observations - len(clustered_obs))
        globally_unclustered_fraction = float(globally_unclustered_count / n_total_observations) if n_total_observations > 0 else 0.0
    else:
        globally_unclustered_count = int(n_total_observations)
        globally_unclustered_fraction = 1.0

    # 5. Node-level External Diagnostics
    purity_macro = None
    purity_incidence = None
    entropy_macro = None
    entropy_incidence = None
    frag_counts = None
    frag_macro = None

    if node_members_dict and external_labels is not None:
        purities = []
        entropies = []
        node_sizes = []
        for members in node_members_dict.values():
            if members:
                purities.append(compute_node_purity(members, external_labels))
                entropies.append(compute_normalized_node_entropy(members, external_labels, n_classes=n_classes))
                node_sizes.append(len(members))
        
        if purities:
            purity_macro = float(np.mean(purities))
            entropy_macro = float(np.mean(entropies))
            sum_sizes = sum(node_sizes)
            if sum_sizes > 0:
                purity_incidence = float(np.average(purities, weights=node_sizes))
                entropy_incidence = float(np.average(entropies, weights=node_sizes))
        
        frag_counts, frag_macro = compute_class_fragmentation(node_members_dict, external_labels, n_classes=n_classes)

    return {
        "n_pullbacks_declared": n_declared,
        "n_pullbacks_nonempty": n_nonempty,
        "n_pullbacks_eligible_silhouette": n_sil_eligible,
        "n_pullbacks_eligible_davies_bouldin": n_db_eligible,
        "E_all_silhouette": e_all_sil,
        "E_nonempty_silhouette": e_nonempty_sil,
        "E_all_davies_bouldin": e_all_db,
        "E_nonempty_davies_bouldin": e_nonempty_db,
        "silhouette_macro": sil_macro,
        "silhouette_incidence": sil_incidence,
        "silhouette_aggregation_reason": sil_agg_reason,
        "davies_bouldin_macro": db_macro,
        "davies_bouldin_incidence": db_incidence,
        "davies_bouldin_aggregation_reason": db_agg_reason,
        "noise_fraction_macro": noise_macro,
        "noise_fraction_incidence": noise_incidence,
        "globally_unclustered_count": globally_unclustered_count,
        "globally_unclustered_fraction": globally_unclustered_fraction,
        "node_purity_macro": purity_macro,
        "node_purity_incidence": purity_incidence,
        "normalized_node_entropy_macro": entropy_macro,
        "normalized_node_entropy_incidence": entropy_incidence,
        "class_fragmentation_counts": frag_counts,
        "class_fragmentation_macro": frag_macro,
    }


# =============================================================================
# 5. Usage Role Derivation (Decision 1A)
# =============================================================================

def derive_axis2_usage_role(graph: Any, construction_evidence_eligible: Optional[bool]) -> str:
    """Derive axis2_usage_role from record-level construction_evidence_eligible and graph existence.
    
    Semantics:
    - graph is None -> "unavailable"
    - graph exists (even if empty V=0) and construction_evidence_eligible is True -> "confirmatory"
    - graph exists and construction_evidence_eligible is False -> "diagnostic_only"
    - unknown/malformed -> raise ValueError
    """
    if graph is None:
        return "unavailable"
    if construction_evidence_eligible is True:
        return "confirmatory"
    if construction_evidence_eligible is False:
        return "diagnostic_only"
    raise ValueError(f"Malformed state: graph={type(graph)}, construction_evidence_eligible={construction_evidence_eligible}")


# =============================================================================
# 6. Shared Construction ID Generator (Decision 5, Spec §5)
# =============================================================================

def compute_shared_construction_id(
    schema_version: str,
    dataset_id: str,
    dataset_replication_id: int,
    clean_dataset_hash: str,
    perturbed_dataset_hash: str,
    noise_condition_id: str,
    perturbation_id: str,
    method_id: str,
    method_config_hash: str,
    lens_definition_hash: str,
    preprocessing_hash: str,
    construction_seed: int,
    output_variant_semantics: str,
) -> str:
    """Compute deterministic SHA-256 cross-axis shared_construction_id from the 13 scientific identity fields.
    
    Contains NO metric values.
    """
    identity_dict = {
        "clean_dataset_hash": str(clean_dataset_hash),
        "construction_seed": int(construction_seed),
        "dataset_id": str(dataset_id),
        "dataset_replication_id": int(dataset_replication_id),
        "lens_definition_hash": str(lens_definition_hash),
        "method_config_hash": str(method_config_hash),
        "method_id": str(method_id),
        "noise_condition_id": str(noise_condition_id),
        "output_variant_semantics": str(output_variant_semantics),
        "perturbation_id": str(perturbation_id),
        "perturbed_dataset_hash": str(perturbed_dataset_hash),
        "preprocessing_hash": str(preprocessing_hash),
        "schema_version": str(schema_version),
    }
    canonical_json_bytes = json.dumps(identity_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_json_bytes).hexdigest()


# =============================================================================
# 7. Membership Payload Serialization & Hashing (Decision 5, Spec §9)
# =============================================================================

def serialize_membership_payload(
    pullback_memberships: List[Dict[str, Any]],
    node_memberships: Dict[Any, Sequence[int]],
) -> Tuple[str, str, str, str]:
    """Serialize pullback and node membership structures to deterministic JSON and compute SHA-256 hashes.
    
    Returns
    -------
    pullback_payload_str : str
    pullback_payload_sha256 : str
    node_payload_str : str
    node_payload_sha256 : str
    """
    # Canonicalize pullback memberships
    canon_pullbacks = []
    for pb in sorted(pullback_memberships, key=lambda x: str(x.get("pullback_id", ""))):
        canon_pullbacks.append({
            "pullback_id": pb.get("pullback_id"),
            "cover_element_id": pb.get("cover_element_id"),
            "sample_indices": sorted([int(i) for i in pb.get("sample_indices", [])]),
            "local_cluster_labels": [int(l) for l in pb.get("local_cluster_labels", [])],
        })
    
    pullback_str = json.dumps(canon_pullbacks, sort_keys=True, separators=(",", ":"))
    pullback_sha = hashlib.sha256(pullback_str.encode("utf-8")).hexdigest()

    # Canonicalize node memberships
    canon_nodes = []
    for node_id in sorted(node_memberships.keys(), key=lambda x: str(x)):
        canon_nodes.append({
            "node_id": str(node_id),
            "member_sample_ids": sorted([int(i) for i in node_memberships[node_id]]),
        })

    node_str = json.dumps(canon_nodes, sort_keys=True, separators=(",", ":"))
    node_sha = hashlib.sha256(node_str.encode("utf-8")).hexdigest()

    return pullback_str, pullback_sha, node_str, node_sha

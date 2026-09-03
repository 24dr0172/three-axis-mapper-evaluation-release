"""Candidate Stability and Topological Fidelity Metric Primitives with Compatibility Contracts."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from mapper_framework.extended_persistence import (
    ExtendedPersistenceDiagram,
    compute_diagram_bottleneck_distance,
    compute_diagram_wasserstein_distance,
    compute_extended_diagram_bottleneck_distance,
    compute_extended_diagram_wasserstein_distance,
    compute_graph_extended_persistence,
)
from mapper_framework.reeb_references import ReebReference
from mapper_framework.types import MapperOutput
from mapper_framework.filter_perturbations import (
    PerturbedFilter,
    compute_filter_definition_hash,
)


@dataclass(frozen=True)
class MetricCompatibilityContract:
    """Immutable metadata contract defining valid applicability of an evaluation metric."""

    metric_id: str
    input_type: str
    requires_same_sample: bool
    requires_same_filter: bool
    applicable_methods: List[str]
    axis_suitability: List[str]
    theoretical_interpretation: str


# Candidate Metric Contracts Registry (Open Decision: No Universal Metric Frozen)
METRIC_REGISTRY: Dict[str, MetricCompatibilityContract] = {
    "bottleneck_extended_pd": MetricCompatibilityContract(
        metric_id="bottleneck_extended_pd",
        input_type="graph_function_pair",
        requires_same_sample=False,
        requires_same_filter=True,
        applicable_methods=["conventional", "f_mapper", "ensemble_kang_lim"],
        axis_suitability=["Axis_I_Stability", "Axis_III_Fidelity"],
        theoretical_interpretation="Maximum L_infinity bottleneck distance between corresponding extended-persistence subdiagram types.",
    ),
    "wasserstein_extended_pd": MetricCompatibilityContract(
        metric_id="wasserstein_extended_pd",
        input_type="graph_function_pair",
        requires_same_sample=False,
        requires_same_filter=True,
        applicable_methods=["conventional", "f_mapper", "ensemble_kang_lim"],
        axis_suitability=["Axis_I_Stability", "Axis_III_Fidelity"],
        theoretical_interpretation="L_1 Wasserstein transportation distance on extended persistence diagrams.",
    ),
    "graph_cycle_rank_diff": MetricCompatibilityContract(
        metric_id="graph_cycle_rank_diff",
        input_type="mapper_graph_1skeleton",
        requires_same_sample=False,
        requires_same_filter=False,
        applicable_methods=["conventional", "f_mapper", "ball_mapper", "ensemble_kang_lim"],
        axis_suitability=["Axis_I_Stability", "Axis_III_Diagnostic"],
        theoretical_interpretation="Absolute difference in 1-skeleton cycle rank |beta_1^graph(M1) - beta_1^graph(M2)|.",
    ),
    "nerve_betti_diff": MetricCompatibilityContract(
        metric_id="nerve_betti_diff",
        input_type="dual_homology_result",
        requires_same_sample=False,
        requires_same_filter=False,
        applicable_methods=["conventional", "f_mapper", "ball_mapper", "ensemble_kang_lim"],
        axis_suitability=["Axis_I_Stability", "Axis_III_Diagnostic"],
        theoretical_interpretation="Absolute difference in simplicial membership nerve 2-skeleton Betti-1 |beta_1^nerve(M1) - beta_1^nerve(M2)|.",
    ),
    "branch_feature_diff": MetricCompatibilityContract(
        metric_id="branch_feature_diff",
        input_type="graph_degree_sequence",
        requires_same_sample=False,
        requires_same_filter=True,
        applicable_methods=["conventional", "f_mapper", "ensemble_kang_lim"],
        axis_suitability=["Axis_III_Fidelity_Tier_B"],
        theoretical_interpretation="Difference in leaf count and branch-vertex count between Mapper and Reeb reference.",
    ),
}


def compute_mapper_extended_pd(
    mapper_out: MapperOutput,
) -> Tuple[List[Tuple[float, float]], ExtendedPersistenceDiagram]:
    """Compute extended persistence diagram for a Mapper output using node mean filters."""
    if mapper_out.graph is None:
        empty_dgm = ExtendedPersistenceDiagram()
        return [], empty_dgm
    node_values = {nid: node.mean_filter for nid, node in mapper_out.graph.nodes.items()}
    dgm = compute_graph_extended_persistence(mapper_out.graph, node_values)
    return dgm.all_points, dgm


def compute_reeb_extended_pd(
    reeb_ref: ReebReference,
) -> Tuple[List[Tuple[float, float]], ExtendedPersistenceDiagram]:
    """Compute extended persistence diagram for a continuous/PL Reeb reference."""
    dgm = compute_graph_extended_persistence(reeb_ref.graph, reeb_ref.node_values)
    return dgm.all_points, dgm


# Canonical Filter Definitions Registry
CANONICAL_FILTER_REGISTRY: Dict[str, Dict[str, str]] = {
    "circle_height_y": {
        "filter_id": "circle_height_y",
        "definition": "f(x,y) = y",
        "domain_id": "unit_circle_R2",
        "codomain": "R",
        "orientation": "increasing_y",
        "definition_version": "1.0.0",
        "sha256": "d784b8d86fb471649c90278087fec52270cf927c76bf862758e183e20f42b78d",
    },
    "tripod_height_y": {
        "filter_id": "tripod_height_y",
        "definition": "f(x,y) = y",
        "domain_id": "branching_tripod_Y",
        "codomain": "R",
        "orientation": "increasing_y",
        "definition_version": "1.0.0",
        "sha256": "685f2da6e8571fb12140d8971dfbd5d3ef26b072a38a9c15a9cd1c68a955c3f4",
    },
    "digits_pca1_frozen": {
        "filter_id": "digits_pca1_frozen",
        "definition": "f(x) = (x - mu0)^T w1, mu0/w1 frozen from clean X0",
        "domain_id": "digits_1797x64_scaled16",
        "codomain": "R",
        "definition_version": "1.0.0",
        "sha256": "a0ec90cf1c45dc7d4860692eecc8c48ede0f61525a3a2b0b4fd2af45719d867a",
    },
}

# Supported filter aliases
FILTER_ALIASES: Dict[str, str] = {
    "height": "circle_height_y",  # default legacy alias for circle height
    "frozen_pca1": "digits_pca1_frozen",
    "frozen_pca1_clean_digits": "digits_pca1_frozen",
}


def resolve_canonical_filter(
    lens_id: Optional[str],
    filter_definition_hash: Optional[str] = None,
    space_name: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a lens identifier and optional hash to canonical filter ID and SHA-256.

    If filter_definition_hash is supplied, it MUST match the canonical registry hash.
    If mismatched, resolution fails hard and returns (None, None).
    Never returns a non-canonical supplied hash as canonical.
    """
    if not lens_id:
        return None, None

    canon_id = lens_id
    if lens_id in FILTER_ALIASES:
        if space_name == "branching_tripod_Y":
            canon_id = "tripod_height_y"
        else:
            canon_id = FILTER_ALIASES[lens_id]

    entry = CANONICAL_FILTER_REGISTRY.get(canon_id)
    if entry is not None:
        reg_hash = entry["sha256"]
        if filter_definition_hash is not None:
            if filter_definition_hash != reg_hash:
                return None, None
            return canon_id, reg_hash
        return canon_id, reg_hash

    return canon_id, filter_definition_hash


def evaluate_same_filter_fidelity(
    mapper_out: MapperOutput,
    reeb_ref: ReebReference,
    metric_id: str = "bottleneck_extended_pd",
    mapper_lens_id: Optional[str] = None,
    mapper_filter_definition_hash: Optional[str] = None,
    reference_lens_id: Optional[str] = None,
    reference_filter_definition_hash: Optional[str] = None,
    strict_confirmatory: bool = False,
) -> Dict[str, Any]:
    """Evaluate structural fidelity between Mapper output and continuous Reeb reference under the Same-Filter Reeb Contract."""
    if mapper_out.graph is None:
        return {
            "status": "mapper_construction_failure",
            "distance": None,
            "metric_id": metric_id,
            "reason": "MapperOutput has no valid graph.",
        }

    # Extract explicit mapper lens identity without inventing "height" fallback
    if mapper_lens_id is not None:
        raw_mapper_lens = mapper_lens_id
    elif "lens_id" in mapper_out.metadata and mapper_out.metadata["lens_id"] is not None:
        raw_mapper_lens = mapper_out.metadata["lens_id"]
    else:
        raw_mapper_lens = None

    if reference_lens_id is not None:
        raw_ref_lens = reference_lens_id
    elif reeb_ref is not None and reeb_ref.lens_id is not None:
        raw_ref_lens = reeb_ref.lens_id
    else:
        raw_ref_lens = None

    raw_m_hash = mapper_filter_definition_hash or (mapper_out.metadata.get("filter_definition_hash") if mapper_out.metadata else None)
    raw_r_hash = reference_filter_definition_hash or (reeb_ref.filter_definition_hash if reeb_ref else None)

    # In strict confirmatory mode: explicit hash is required (registry lookup may verify, but cannot invent missing confirmatory hash)
    if strict_confirmatory and (raw_m_hash is None or raw_r_hash is None):
        eff_m_id, _ = resolve_canonical_filter(raw_mapper_lens, space_name=reeb_ref.space_name if reeb_ref else None)
        eff_r_id, _ = resolve_canonical_filter(raw_ref_lens, space_name=reeb_ref.space_name if reeb_ref else None)
        return {
            "metric_id": metric_id,
            "reference_filter_id": eff_r_id,
            "mapper_filter_id": eff_m_id,
            "reference_filter_definition_hash": raw_r_hash,
            "mapper_filter_definition_hash": raw_m_hash,
            "beta0_graph_mapper": mapper_out.graph.n_components if mapper_out.graph else None,
            "beta1_graph_mapper": mapper_out.graph.beta_1_graph if mapper_out.graph else None,
            "beta0_nerve_mapper": mapper_out.homology.beta_0_nerve if mapper_out.homology else None,
            "beta1_nerve_mapper": mapper_out.homology.beta_1_nerve if mapper_out.homology else None,
            "beta0_reeb_reference": reeb_ref.invariants.get("beta_0") if (reeb_ref and reeb_ref.invariants) else None,
            "beta1_reeb_reference": reeb_ref.invariants.get("beta_1") if (reeb_ref and reeb_ref.invariants) else None,
            "delta_beta1_graph": None,
            "delta_beta1_nerve_to_reference": None,
            "status": "same_filter_contract_unverifiable",
            "distance": None,
            "same_filter_contract": False,
            "reason": "Missing explicit filter definition hash in strict confirmatory mode; registry lookup may not invent missing confirmatory hash.",
        }

    # Resolve canonical filter IDs and definition hashes
    eff_m_id, eff_m_hash = resolve_canonical_filter(
        raw_mapper_lens,
        filter_definition_hash=raw_m_hash,
        space_name=reeb_ref.space_name if reeb_ref else None,
    )
    eff_r_id, eff_r_hash = resolve_canonical_filter(
        raw_ref_lens,
        filter_definition_hash=raw_r_hash,
        space_name=reeb_ref.space_name if reeb_ref else None,
    )

    # Invariant extraction from reference
    reeb_b0 = reeb_ref.invariants.get("beta_0") if (reeb_ref and reeb_ref.invariants) else None
    reeb_b1 = reeb_ref.invariants.get("beta_1") if (reeb_ref and reeb_ref.invariants) else None

    # Invariant extraction from Mapper
    m_b0_graph = mapper_out.graph.n_components if mapper_out.graph else None
    m_b1_graph = mapper_out.graph.beta_1_graph if mapper_out.graph else None
    m_b0_nerve = mapper_out.homology.beta_0_nerve if mapper_out.homology else None
    m_b1_nerve = mapper_out.homology.beta_1_nerve if mapper_out.homology else None

    delta_b1_graph = abs(m_b1_graph - reeb_b1) if (m_b1_graph is not None and reeb_b1 is not None) else None
    delta_b1_nerve = (
        abs(m_b1_nerve - reeb_b1)
        if (m_b1_nerve is not None and reeb_b1 is not None)
        else None
    )

    # Base payload structure with unambiguous Tier-A notation
    base_res = {
        "metric_id": metric_id,
        "reference_filter_id": eff_r_id,
        "mapper_filter_id": eff_m_id,
        "reference_filter_definition_hash": eff_r_hash,
        "mapper_filter_definition_hash": eff_m_hash,
        "beta0_graph_mapper": m_b0_graph,
        "beta1_graph_mapper": m_b1_graph,
        "beta0_nerve_mapper": m_b0_nerve,
        "beta1_nerve_mapper": m_b1_nerve,
        "beta0_reeb_reference": reeb_b0,
        "beta1_reeb_reference": reeb_b1,
        "delta_beta1_graph": delta_b1_graph,
        "delta_beta1_nerve_to_reference": delta_b1_nerve,
    }

    # 1. Enforce filter ID presence & equality
    if eff_m_id is None or eff_r_id is None:
        if (raw_mapper_lens is None or raw_ref_lens is None or raw_m_hash is None or raw_r_hash is None) and strict_confirmatory:
            status_code = "same_filter_contract_unverifiable"
        else:
            status_code = "same_filter_contract_violation"
        return {
            **base_res,
            "status": status_code,
            "distance": None,
            "same_filter_contract": False,
            "reason": f"Missing or unresolvable filter identity: mapper_filter_id={eff_m_id}, reference_filter_id={eff_r_id}",
        }

    if eff_m_id != eff_r_id:
        return {
            **base_res,
            "status": "same_filter_contract_violation",
            "distance": None,
            "same_filter_contract": False,
            "reason": f"Same-filter violation: Mapper filter ID '{eff_m_id}' != Reference filter ID '{eff_r_id}'",
        }

    # 2. Enforce filter definition hash presence
    if eff_m_hash is None or eff_r_hash is None:
        return {
            **base_res,
            "status": "same_filter_contract_unverifiable",
            "distance": None,
            "same_filter_contract": False,
            "reason": "Missing filter definition hash in strict confirmatory mode.",
        }

    # 3. Canonical Registry Verification (B1): supplied hash must equal authoritative registry hash
    m_reg = CANONICAL_FILTER_REGISTRY.get(eff_m_id)
    r_reg = CANONICAL_FILTER_REGISTRY.get(eff_r_id)
    if m_reg is not None and eff_m_hash != m_reg["sha256"]:
        return {
            **base_res,
            "status": "same_filter_contract_violation",
            "distance": None,
            "same_filter_contract": False,
            "reason": f"Supplied mapper filter hash '{eff_m_hash}' does not match canonical registry hash '{m_reg['sha256']}' for '{eff_m_id}'.",
        }
    if r_reg is not None and eff_r_hash != r_reg["sha256"]:
        return {
            **base_res,
            "status": "same_filter_contract_violation",
            "distance": None,
            "same_filter_contract": False,
            "reason": f"Supplied reference filter hash '{eff_r_hash}' does not match canonical registry hash '{r_reg['sha256']}' for '{eff_r_id}'.",
        }

    # 4. Enforce filter definition hash equality
    if eff_m_hash != eff_r_hash:
        return {
            **base_res,
            "status": "same_filter_contract_violation",
            "distance": None,
            "same_filter_contract": False,
            "reason": f"Same-filter violation: Mapper filter hash '{eff_m_hash}' != Reference filter hash '{eff_r_hash}'",
        }

    same_filter_passed = (eff_m_id == eff_r_id) and (eff_m_hash == eff_r_hash)
    base_res["same_filter_contract"] = same_filter_passed

    contract = METRIC_REGISTRY.get(metric_id)
    if contract is None:
        return {
            **base_res,
            "status": "metric_undefined",
            "distance": None,
            "reason": f"Unknown metric ID: {metric_id}",
        }

    if metric_id == "bottleneck_extended_pd":
        m_pts, m_dgm = compute_mapper_extended_pd(mapper_out)
        r_pts, r_dgm = compute_reeb_extended_pd(reeb_ref)
        dist = compute_extended_diagram_bottleneck_distance(m_dgm, r_dgm)
        return {
            **base_res,
            "status": "success",
            "distance": dist,
            "mapper_points_count": len(m_pts),
            "reeb_points_count": len(r_pts),
        }
    elif metric_id == "wasserstein_extended_pd":
        m_pts, m_dgm = compute_mapper_extended_pd(mapper_out)
        r_pts, r_dgm = compute_reeb_extended_pd(reeb_ref)
        dist = compute_extended_diagram_wasserstein_distance(m_dgm, r_dgm, p=1)
        return {
            **base_res,
            "status": "success",
            "distance": dist,
            "mapper_points_count": len(m_pts),
            "reeb_points_count": len(r_pts),
        }

    elif metric_id == "graph_cycle_rank_diff":
        if reeb_b1 is None or mapper_out.graph.beta_1_graph is None:
            return {
                **base_res,
                "status": "metric_undefined",
                "distance": None,
                "reason": "Missing beta_1 in Reeb reference or Mapper graph.",
            }
        dist = abs(mapper_out.graph.beta_1_graph - reeb_b1)
        return {
            **base_res,
            "status": "success",
            "distance": float(dist),
        }
    elif metric_id == "nerve_betti_diff":
        if mapper_out.homology is None or mapper_out.homology.beta_1_nerve is None:
            status_code = "homology_resource_failure" if mapper_out.status == "resource_failure" else "homology_computation_failure"
            return {
                **base_res,
                "status": status_code,
                "distance": None,
                "reason": f"Simplicial membership nerve homology is unavailable ({status_code}); zero-imputation is strictly prohibited.",
            }
        if reeb_b1 is None:
            return {
                **base_res,
                "status": "reference_failure",
                "distance": None,
                "reason": "Reeb reference invariants missing beta_1.",
            }
        nerve_b1 = mapper_out.homology.beta_1_nerve
        dist = abs(nerve_b1 - reeb_b1)
        return {
            **base_res,
            "status": "success",
            "distance": float(dist),
        }
    elif metric_id == "branch_feature_diff":
        degs = {nid: 0 for nid in mapper_out.graph.nodes}
        for u, v in mapper_out.graph.edges:
            degs[u] += 1
            degs[v] += 1
        m_leaves = sum(1 for d in degs.values() if d == 1)
        m_branch = sum(1 for d in degs.values() if d >= 3)

        if "n_leaves" not in reeb_ref.invariants or "n_branch_vertices" not in reeb_ref.invariants:
            return {
                **base_res,
                "status": "reference_failure",
                "distance": None,
                "reason": "Reeb reference invariants missing leaf or branch vertex counts.",
            }
        r_leaves = reeb_ref.invariants["n_leaves"]
        r_branch = reeb_ref.invariants["n_branch_vertices"]

        leaf_diff = abs(m_leaves - r_leaves)
        branch_diff = abs(m_branch - r_branch)
        return {
            **base_res,
            "status": "success",
            "distance": float(leaf_diff + branch_diff),
            "leaf_diff": leaf_diff,
            "branch_diff": branch_diff,
        }

    return {
        **base_res,
        "status": "metric_undefined",
        "distance": None,
        "reason": f"Execution logic not implemented for metric: {metric_id} (unsupported_metric_execution)",
    }


# =============================================================================
# Cross-Filter Perturbation Stability Mode (Joint I-B1 / III-4, Phase B)
# =============================================================================
#
# Evaluates D_R(delta, r) = d(R_f(S), R_{f_delta,r}(S)) and D_M(delta, r) = d(M(X,f), M(X,f_delta,r)):
# comparisons between objects generated under DISTINCT filters (baseline f vs perturbed f_delta,r).
# This is deliberately a NARROW, separate contract from same_filter_fidelity above; it never
# relaxes or bypasses the same-filter requirement used by evaluate_same_filter_fidelity / E_M.

def _verify_cross_filter_lineage(
    benchmark_id: str,
    baseline_filter: PerturbedFilter,
    perturbed_filter: PerturbedFilter,
) -> Optional[str]:
    """Return None if baseline-to-perturbed lineage is valid under the frozen registry, else a reason string."""
    if not isinstance(baseline_filter, PerturbedFilter) or not isinstance(perturbed_filter, PerturbedFilter):
        return "baseline_filter and perturbed_filter must be PerturbedFilter instances bound to the frozen registry."
    if not baseline_filter.is_baseline:
        return f"baseline_filter must be the current-campaign baseline (delta=0.0); got delta={baseline_filter.delta}"
    if perturbed_filter.is_baseline:
        return "perturbed_filter must be a nonbaseline perturbed filter (delta > 0)."
    if baseline_filter.benchmark_id != benchmark_id or perturbed_filter.benchmark_id != benchmark_id:
        return (
            f"Filter benchmark_id mismatch against declared benchmark_id='{benchmark_id}' "
            f"(baseline={baseline_filter.benchmark_id}, perturbed={perturbed_filter.benchmark_id})."
        )
    if perturbed_filter.baseline_filter_id != baseline_filter.filter_id:
        return (
            f"Declared baseline-to-perturbed lineage violation: perturbed_filter.baseline_filter_id="
            f"'{perturbed_filter.baseline_filter_id}' != baseline_filter.filter_id='{baseline_filter.filter_id}'."
        )
    expected_hash = compute_filter_definition_hash(
        perturbed_filter.benchmark_id, perturbed_filter.direction_id, perturbed_filter.delta
    )
    if perturbed_filter.filter_definition_hash != expected_hash:
        return (
            "Perturbed filter_definition_hash does not match frozen perturbation-family registry "
            f"recomputation (got {perturbed_filter.filter_definition_hash}, expected {expected_hash})."
        )
    if not (perturbed_filter.delta > 0.0):
        return f"cross_filter_perturbation_stability requires a nonzero confirmatory delta; got {perturbed_filter.delta}."
    return None


def evaluate_cross_filter_perturbation_stability(
    estimand: str,
    benchmark_id: str,
    baseline_filter: PerturbedFilter,
    perturbed_filter: PerturbedFilter,
    baseline_object: Any,
    perturbed_object: Any,
    metric_id: str = "bottleneck_extended_pd",
) -> Dict[str, Any]:
    """Evaluate D_R or D_M under the cross_filter_perturbation_stability semantic mode.

    Parameters
    ----------
    estimand : 'D_R' (ReebReference vs ReebReference) or 'D_M' (MapperOutput vs MapperOutput).
    baseline_filter, perturbed_filter : PerturbedFilter
        Must satisfy declared baseline-to-perturbed lineage bound to the frozen filter registry.
    baseline_object, perturbed_object : ReebReference pair (D_R) or MapperOutput pair (D_M).
    """
    base_res: Dict[str, Any] = {
        "metric_id": metric_id,
        "estimand": estimand,
        "benchmark_id": benchmark_id,
        "baseline_filter_id": getattr(baseline_filter, "filter_id", None),
        "baseline_filter_definition_hash": getattr(baseline_filter, "filter_definition_hash", None),
        "perturbed_filter_id": getattr(perturbed_filter, "filter_id", None),
        "perturbed_filter_definition_hash": getattr(perturbed_filter, "filter_definition_hash", None),
        "delta": getattr(perturbed_filter, "delta", None),
        "direction_id": getattr(perturbed_filter, "direction_id", None),
    }

    if estimand not in ("D_R", "D_M"):
        return {**base_res, "status": "metric_undefined", "distance": None, "cross_filter_contract": False,
                "reason": f"Unknown estimand '{estimand}'; expected 'D_R' or 'D_M'."}

    lineage_error = _verify_cross_filter_lineage(benchmark_id, baseline_filter, perturbed_filter)
    if lineage_error is not None:
        return {**base_res, "status": "cross_filter_lineage_violation", "distance": None,
                "cross_filter_contract": False, "reason": lineage_error}

    if estimand == "D_R":
        if not isinstance(baseline_object, ReebReference) or not isinstance(perturbed_object, ReebReference):
            return {**base_res, "status": "configuration_invalid", "distance": None,
                    "reason": "D_R requires ReebReference baseline_object and perturbed_object."}
        if baseline_object.space_name != benchmark_id or perturbed_object.space_name != benchmark_id:
            return {**base_res, "status": "cross_filter_lineage_violation", "distance": None,
                    "cross_filter_contract": False,
                    "reason": "Reference space_name does not match declared benchmark_id."}
        _, b_dgm = compute_reeb_extended_pd(baseline_object)
        _, p_dgm = compute_reeb_extended_pd(perturbed_object)
    else:
        if not isinstance(baseline_object, MapperOutput) or not isinstance(perturbed_object, MapperOutput):
            return {**base_res, "status": "configuration_invalid", "distance": None,
                    "reason": "D_M requires MapperOutput baseline_object and perturbed_object."}
        if baseline_object.graph is None or perturbed_object.graph is None:
            return {**base_res, "status": "mapper_construction_failure", "distance": None,
                    "reason": "One or both Mapper outputs have no valid graph."}
        method_b = str((baseline_object.metadata or {}).get("method", "conventional")).lower()
        method_p = str((perturbed_object.metadata or {}).get("method", "conventional")).lower()
        if method_b != method_p:
            return {**base_res, "status": "cross_filter_lineage_violation", "distance": None,
                    "cross_filter_contract": False,
                    "reason": f"Mapper method/configuration mismatch: '{method_b}' != '{method_p}'."}
        _, b_dgm = compute_mapper_extended_pd(baseline_object)
        _, p_dgm = compute_mapper_extended_pd(perturbed_object)

    if metric_id == "bottleneck_extended_pd":
        dist = compute_extended_diagram_bottleneck_distance(b_dgm, p_dgm)
    elif metric_id == "wasserstein_extended_pd":
        dist = compute_extended_diagram_wasserstein_distance(b_dgm, p_dgm, p=1)
    else:
        return {**base_res, "status": "metric_undefined", "distance": None,
                "reason": f"Unsupported metric_id '{metric_id}' for cross_filter_perturbation_stability."}

    return {**base_res, "status": "success", "distance": dist, "cross_filter_contract": True}


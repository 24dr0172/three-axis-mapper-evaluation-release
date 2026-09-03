"""Safe Metric Dispatcher with Compatibility Verification."""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from mapper_framework.extended_persistence import (
    compute_diagram_bottleneck_distance,
    compute_diagram_wasserstein_distance,
)
from mapper_framework.stability_metrics import (
    METRIC_REGISTRY,
    MetricCompatibilityContract,
    compute_mapper_extended_pd,
    evaluate_cross_filter_perturbation_stability,
    resolve_canonical_filter,
)
from mapper_framework.types import MapperOutput


class MetricDispatcher:
    """Safe dispatcher for stability and structural fidelity metrics."""

    @staticmethod
    def evaluate_output_distance(
        m_ref: MapperOutput,
        m_perturbed: MapperOutput,
        metric_id: str = "bottleneck_extended_pd",
        **kwargs: Any,
    ) -> Tuple[Optional[float], str, Optional[str]]:
        """Evaluate output distance between clean reference Mapper and perturbed Mapper.

        Returns
        -------
        distance : Optional[float]
            Computed distance, or None if incompatible/failed.
        compatibility_status : str
            'compatible' | 'incompatible' | 'failed'
        reason : Optional[str]
            Explanation of incompatibility or failure.
        """
        if m_ref.graph is None or m_perturbed.graph is None:
            return None, "failed", "One or both Mapper outputs have no valid graph."

        contract = METRIC_REGISTRY.get(metric_id)
        if contract is None:
            return None, "incompatible", f"Unknown metric ID: '{metric_id}'."

        method_ref = m_ref.metadata.get("method", "conventional").lower()
        method_pert = m_perturbed.metadata.get("method", "conventional").lower()

        # Check Ball Mapper compatibility: Ball Mapper is metric_cover_only and has no scalar lens extended PD
        role_ref = str(m_ref.metadata.get("reference_role", "")).lower()
        role_pert = str(m_perturbed.metadata.get("reference_role", "")).lower()
        if "ball_mapper" in method_ref or "ball_mapper" in method_pert or "metric_cover_only" in (role_ref, role_pert):
            if metric_id in ("bottleneck_extended_pd", "wasserstein_extended_pd"):
                return (
                    None,
                    "incompatible",
                    f"Metric '{metric_id}' requires a continuous scalar lens function; "
                    f"Ball Mapper is strictly 'metric_cover_only' and does not support Reeb/lens extended persistence.",
                )

        # Enforce requires_same_filter contract (Amendment A1 / Obligation I14)
        if contract.requires_same_filter:
            lens_ref = m_ref.metadata.get("lens_id") if m_ref.metadata else None
            hash_ref = m_ref.metadata.get("filter_definition_hash") if m_ref.metadata else None
            lens_pert = m_perturbed.metadata.get("lens_id") if m_perturbed.metadata else None
            hash_pert = m_perturbed.metadata.get("filter_definition_hash") if m_perturbed.metadata else None

            # Fallback to explicit kwargs if supplied
            if lens_ref is None:
                lens_ref = kwargs.get("lens_id_ref", kwargs.get("lens_id"))
            if lens_pert is None:
                lens_pert = kwargs.get("lens_id_pert", kwargs.get("lens_id"))
            if hash_ref is None:
                hash_ref = kwargs.get("filter_definition_hash_ref", kwargs.get("filter_definition_hash"))
            if hash_pert is None:
                hash_pert = kwargs.get("filter_definition_hash_pert", kwargs.get("filter_definition_hash"))

            # If lens_id is supplied without explicit hash, try resolving canonical hash
            if lens_ref is not None and hash_ref is None:
                _, hash_ref = resolve_canonical_filter(lens_id=lens_ref)
            if lens_pert is not None and hash_pert is None:
                _, hash_pert = resolve_canonical_filter(lens_id=lens_pert)

            if lens_ref is None or lens_pert is None or hash_ref is None or hash_pert is None:
                return (
                    None,
                    "incompatible",
                    f"Metric '{metric_id}' requires explicit filter identity (lens_id and filter_definition_hash) on both reference and perturbed outputs; missing identity (ref: lens={lens_ref}, hash={hash_ref}; pert: lens={lens_pert}, hash={hash_pert}).",
                )

            if lens_ref != lens_pert or hash_ref != hash_pert:
                return (
                    None,
                    "incompatible",
                    f"Metric '{metric_id}' requires identical filter lens and definition hash between reference ('{lens_ref}', '{hash_ref}') and perturbed ('{lens_pert}', '{hash_pert}').",
                )

        if metric_id == "bottleneck_extended_pd":
            _, dgm_ref = compute_mapper_extended_pd(m_ref)
            _, dgm_pert = compute_mapper_extended_pd(m_perturbed)
            dist = compute_diagram_bottleneck_distance(dgm_ref, dgm_pert)
            return dist, "compatible", None

        elif metric_id == "wasserstein_extended_pd":
            _, dgm_ref = compute_mapper_extended_pd(m_ref)
            _, dgm_pert = compute_mapper_extended_pd(m_perturbed)
            dist = compute_diagram_wasserstein_distance(dgm_ref, dgm_pert, p=1)
            return dist, "compatible", None


        elif metric_id == "graph_cycle_rank_diff":
            dist = float(abs(m_ref.graph.beta_1_graph - m_perturbed.graph.beta_1_graph))
            return dist, "compatible", None

        elif metric_id == "nerve_betti_diff":
            b1_ref = m_ref.homology.beta_1_nerve if m_ref.homology else None
            b1_pert = m_perturbed.homology.beta_1_nerve if m_perturbed.homology else None
            if b1_ref is None or b1_pert is None:
                return None, "failed", "Dual homology nerve beta_1 is unavailable on one or both outputs."
            dist = float(abs(b1_ref - b1_pert))
            return dist, "compatible", None

        return None, "incompatible", f"Execution handler not implemented for metric '{metric_id}'."

    @staticmethod
    def evaluate_cross_filter_stability(
        estimand: str,
        benchmark_id: str,
        baseline_filter: Any,
        perturbed_filter: Any,
        baseline_object: Any,
        perturbed_object: Any,
        metric_id: str = "bottleneck_extended_pd",
    ) -> Dict[str, Any]:
        """Dispatch D_R / D_M evaluation under the cross_filter_perturbation_stability mode.

        Deliberately separate from evaluate_output_distance's requires_same_filter contract
        above; never relaxes or bypasses that same-filter enforcement.
        """
        return evaluate_cross_filter_perturbation_stability(
            estimand=estimand,
            benchmark_id=benchmark_id,
            baseline_filter=baseline_filter,
            perturbed_filter=perturbed_filter,
            baseline_object=baseline_object,
            perturbed_object=perturbed_object,
            metric_id=metric_id,
        )

    @staticmethod
    def evaluate_secondary_diagnostics(
        m_ref: MapperOutput,
        m_perturbed: MapperOutput,
    ) -> Dict[str, Optional[float]]:
        """Compute all valid secondary structural diagnostics between reference and perturbed outputs."""
        if m_ref.graph is None or m_perturbed.graph is None:
            return {}

        diag: Dict[str, Optional[float]] = {}

        # 0. Typed Extended-Persistence 1-Wasserstein Distance
        try:
            _, dgm_ref = compute_mapper_extended_pd(m_ref)
            _, dgm_pert = compute_mapper_extended_pd(m_perturbed)
            diag["wasserstein_extended_pd"] = compute_diagram_wasserstein_distance(dgm_ref, dgm_pert, p=1)
        except Exception:
            diag["wasserstein_extended_pd"] = None

        # 1. Vertex count delta
        diag["delta_V"] = float(abs(m_perturbed.graph.n_nodes - m_ref.graph.n_nodes))

        # 2. Edge count delta
        diag["delta_E"] = float(abs(m_perturbed.graph.n_edges - m_ref.graph.n_edges))

        # 3. Component count delta
        diag["delta_C"] = float(abs(m_perturbed.graph.n_components - m_ref.graph.n_components))

        # 4. Graph cycle rank delta (secondary diagnostic)
        diag["graph_cycle_rank_diff"] = float(abs(m_perturbed.graph.beta_1_graph - m_ref.graph.beta_1_graph))

        # 5. Nerve Betti-1 delta (if available)
        if (
            m_ref.homology is not None
            and m_perturbed.homology is not None
            and m_ref.homology.beta_1_nerve is not None
            and m_perturbed.homology.beta_1_nerve is not None
        ):
            diag["nerve_betti_diff"] = float(abs(m_perturbed.homology.beta_1_nerve - m_ref.homology.beta_1_nerve))
        else:
            diag["nerve_betti_diff"] = None

        return diag


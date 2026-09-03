"""Acceptance Rules and Protocol Decision Registers (No Arbitrary Binary Thresholds).

Contains the authoritative frozen protocol specifications established in Milestones M5-H/M6
alongside the historical Phase-0 open register for archival provenance.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ProtocolDecision:
    """Immutable representation of an experimental protocol decision status."""

    decision_key: str
    status: str  # 'OPEN' | 'APPROVED' | 'FROZEN' | 'DEFERRED'
    candidate_options: List[Any]
    frozen_value: Optional[Any] = None
    rationale: str = ""
    phase_gate: str = "Milestone_M5_H_Freeze"


# Authoritative Frozen Protocol Decision Register (M5-H / M6)
FROZEN_PROTOCOL_REGISTER: Dict[str, ProtocolDecision] = {
    "final_sample_sizes": ProtocolDecision(
        decision_key="final_sample_sizes",
        status="FROZEN",
        candidate_options=[100, 300, 600, 1000, 2500],
        frozen_value={"unit_circle_S1": 1000, "swiss_roll_base": 2500, "swiss_roll_realized": 2241},
        rationale="Locked in M5-H freeze sheet and verified in confirmatory execution.",
        phase_gate="Milestone_M5_H_Freeze",
    ),
    "coordinate_noise_grid": ProtocolDecision(
        decision_key="coordinate_noise_grid",
        status="FROZEN",
        candidate_options=[[0.0, 0.05, 0.10, 0.20], [0.00, 0.01, 0.03, 0.05, 0.08, 0.10]],
        frozen_value=[0.00, 0.01, 0.03, 0.05, 0.08, 0.10],
        rationale="Fine-grained coordinate noise grid spanning linear response and structural distortion regimes.",
        phase_gate="Milestone_M5_H_Freeze",
    ),
    "replication_count_R": ProtocolDecision(
        decision_key="replication_count_R",
        status="FROZEN",
        candidate_options=[10, 30, 50],
        frozen_value=30,
        rationale="30 independent perturbation realizations per noise condition for 95% bootstrap CIs.",
        phase_gate="Milestone_M5_H_Freeze",
    ),
    "primary_axis1_metric": ProtocolDecision(
        decision_key="primary_axis1_metric",
        status="FROZEN",
        candidate_options=["bottleneck_extended_pd", "wasserstein_extended_pd", "graph_cycle_rank_diff"],
        frozen_value="bottleneck_extended_pd",
        rationale="Maximum L_infinity bottleneck distance between corresponding subdiagram types in the stated four-type extended-persistence construction.",
        phase_gate="Milestone_M5_H_Freeze",
    ),
    "secondary_axis1_metric": ProtocolDecision(
        decision_key="secondary_axis1_metric",
        status="FROZEN",
        candidate_options=["wasserstein_extended_pd"],
        frozen_value="wasserstein_extended_pd",
        rationale="L_1 Wasserstein transportation distance across matching subdiagrams.",
        phase_gate="Milestone_M5_H_Freeze",
    ),
    "acceptance_policy": ProtocolDecision(
        decision_key="acceptance_policy",
        status="FROZEN",
        candidate_options=["arbitrary_binary_cutoff", "comparative_empirical_stability_curves"],
        frozen_value="comparative_empirical_stability_curves",
        rationale="Evaluation is comparative via full empirical stability curves and 5,000 bootstrap CIs (no invented binary cutoff).",
        phase_gate="Milestone_M5_H_Freeze",
    ),
}

# ==============================================================================
# DEPRECATED / QUARANTINED HISTORICAL REGISTER
# WARNING: The following register contains obsolete Phase-0 OPEN draft states.
# It is preserved STRICTLY for historical provenance and must NEVER be consumed
# by active experimental runners, confirmatory drivers, or evaluation pipelines.
# Active pipelines MUST consume FROZEN_PROTOCOL_REGISTER exclusively.
# ==============================================================================
DEPRECATED_HISTORICAL_PHASE0_REGISTER: Dict[str, ProtocolDecision] = {
    "final_sample_sizes": ProtocolDecision(
        decision_key="final_sample_sizes",
        status="OPEN",
        candidate_options=[100, 300, 600, 1200],
        frozen_value=None,
        rationale="[DEPRECATED] Historical draft candidate.",
    ),
    "coordinate_noise_grid": ProtocolDecision(
        decision_key="coordinate_noise_grid",
        status="OPEN",
        candidate_options=[0.0, 0.05, 0.10, 0.20],
        frozen_value=None,
        rationale="[DEPRECATED] Historical draft candidate.",
    ),
    "replication_count_R": ProtocolDecision(
        decision_key="replication_count_R",
        status="OPEN",
        candidate_options=[10, 30, 50],
        frozen_value=None,
        rationale="[DEPRECATED] Historical draft candidate.",
    ),
    "primary_axis1_metric": ProtocolDecision(
        decision_key="primary_axis1_metric",
        status="OPEN",
        candidate_options=["bottleneck_extended_pd", "wasserstein_extended_pd", "graph_cycle_rank_diff"],
        frozen_value=None,
        rationale="[DEPRECATED] Historical draft candidate.",
    ),
    "acceptance_thresholds": ProtocolDecision(
        decision_key="acceptance_thresholds",
        status="OPEN",
        candidate_options=["empirical_sd_relative_bound", "absolute_distance_cutoff"],
        frozen_value=None,
        rationale="[DEPRECATED] Historical draft candidate.",
    ),
}

# Backward compatibility alias - DO NOT USE IN PRODUCTION
PHASE0_DECISION_REGISTER = DEPRECATED_HISTORICAL_PHASE0_REGISTER



@dataclass(frozen=True)
class AcceptanceRuleResult:
    """Result of an evaluation rule check against a declared protocol threshold."""

    rule_id: str
    target_metric: str
    observed_value: float
    threshold_value: Optional[float]
    passed: Optional[bool]
    decision_status: str  # 'OPEN' | 'EVALUATED' | 'COMPARATIVE_ONLY'
    metadata: Dict[str, Any] = field(default_factory=dict)


def evaluate_acceptance_rule(
    rule_id: str,
    target_metric: str,
    observed_value: float,
    threshold: Optional[float] = None,
) -> AcceptanceRuleResult:
    """Evaluate an acceptance rule under comparative protocol guidelines."""
    if threshold is None:
        return AcceptanceRuleResult(
            rule_id=rule_id,
            target_metric=target_metric,
            observed_value=observed_value,
            threshold_value=None,
            passed=None,
            decision_status="OPEN",
            metadata={"note": "Evaluated comparatively via empirical stability curve; threshold is OPEN without arbitrary binary cutoff."},
        )


    passed = bool(observed_value <= threshold)
    return AcceptanceRuleResult(
        rule_id=rule_id,
        target_metric=target_metric,
        observed_value=observed_value,
        threshold_value=threshold,
        passed=passed,
        decision_status="EVALUATED",
    )

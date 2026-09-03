"""Summary Utilities, Field-Specific Eligibility, and Automated Ledger Reconciliation.

Enforces:
1. Field-specific scientific eligibility (no blanket exclusion for localized NA fields).
2. No silent zero imputation (NA never imputed as 0; mean/std computed over defined values).
3. Conditional denominator reporting (reports N_intended, N_defined, N_excluded, exclusion reasons).
4. Bootstrap operating strictly over defined values (N_entering_bootstrap == N_defined).
5. Automated hard ledger reconciliation invariants:
   N_intended == N_success + N_fcm_nonconvergence + N_degenerate + N_other_failure.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from mapper_framework.experiment_records import ExperimentRunRecord


@dataclass(frozen=True)
class FieldSummary:
    """Field-specific statistical summary with explicit denominator and exclusion accounting."""

    field_name: str
    N_intended: int
    N_defined: int
    N_excluded: int
    exclusion_reasons: Dict[str, int]
    values: List[float]
    mean: Optional[float]
    std: Optional[float]
    is_conditional: bool
    condition_label: Optional[str] = None


@dataclass(frozen=True)
class CellSummaryResult:
    """Cell-level status breakdown and field summaries across a dataset x method x noise cell."""

    dataset_id: str
    method_id: str
    noise_condition_id: str
    N_intended: int
    N_success: int
    N_fcm_nonconvergence: int
    N_degenerate: int
    N_other_failure: int
    fcm_nonconvergence_rate: float
    degenerate_rate: float
    field_summaries: Dict[str, FieldSummary] = field(default_factory=dict)

    def verify_status_invariant(self) -> bool:
        """Verify hard invariant: N_intended == N_success + N_fcm_nonconv + N_degenerate + N_other."""
        accounted = self.N_success + self.N_fcm_nonconvergence + self.N_degenerate + self.N_other_failure
        return self.N_intended == accounted


def compute_field_summary(
    records: List[ExperimentRunRecord],
    field_accessor: Callable[[ExperimentRunRecord], Optional[Any]],
    field_name: str,
    eligibility_predicate: Optional[Callable[[ExperimentRunRecord], bool]] = None,
) -> FieldSummary:
    """Compute summary statistics for a specific field with exact denominator and exclusion breakdown.

    NA values are NEVER imputed as zero.
    """
    n_intended = len(records)
    defined_values: List[float] = []
    exclusion_reasons: Dict[str, int] = {}

    for r in records:
        # Check eligibility predicate if provided
        if eligibility_predicate is not None and not eligibility_predicate(r):
            reason = r.failure_reason or r.status or "ineligible"
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
            continue

        raw_val = field_accessor(r)
        if raw_val is None:
            reason = r.failure_reason or r.status or "undefined_null"
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
        elif isinstance(raw_val, (int, float)) and math.isnan(raw_val):
            reason = "nan_value"
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
        else:
            try:
                val_float = float(raw_val)
                defined_values.append(val_float)
            except (ValueError, TypeError):
                reason = "non_numeric"
                exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

    n_defined = len(defined_values)
    n_excluded = n_intended - n_defined
    is_conditional = (n_defined < n_intended)
    cond_label = "conditional on metric-defined outputs" if is_conditional else None

    mean_val = float(np.mean(defined_values)) if n_defined > 0 else None
    std_val = float(np.std(defined_values, ddof=1)) if n_defined > 1 else (0.0 if n_defined == 1 else None)

    return FieldSummary(
        field_name=field_name,
        N_intended=n_intended,
        N_defined=n_defined,
        N_excluded=n_excluded,
        exclusion_reasons=exclusion_reasons,
        values=defined_values,
        mean=mean_val,
        std=std_val,
        is_conditional=is_conditional,
        condition_label=cond_label,
    )


def compute_cell_summary(
    records: List[ExperimentRunRecord],
    dataset_id: Optional[str] = None,
    method_id: Optional[str] = None,
    noise_condition_id: Optional[str] = None,
) -> CellSummaryResult:
    """Compute complete cell-level summary with status accounting and field-specific statistics."""
    n_intended = len(records)
    if n_intended == 0:
        return CellSummaryResult(
            dataset_id=dataset_id or "unknown",
            method_id=method_id or "unknown",
            noise_condition_id=noise_condition_id or "unknown",
            N_intended=0,
            N_success=0,
            N_fcm_nonconvergence=0,
            N_degenerate=0,
            N_other_failure=0,
            fcm_nonconvergence_rate=0.0,
            degenerate_rate=0.0,
        )

    ds_id = dataset_id or records[0].identity.dataset_id
    m_id = method_id or records[0].identity.method_id
    noise_id = noise_condition_id or records[0].identity.noise_condition_id

    n_success = sum(1 for r in records if r.status == "success")
    n_fcm_nonconv = sum(1 for r in records if r.status == "fcm_non_convergence")
    n_degenerate = sum(1 for r in records if r.status == "degenerate_output")
    n_other = sum(1 for r in records if r.status not in ("success", "fcm_non_convergence", "degenerate_output"))

    # Hard invariant verification
    assert n_intended == (n_success + n_fcm_nonconv + n_degenerate + n_other), (
        f"Cell status invariant failed: N_intended ({n_intended}) != "
        f"N_success ({n_success}) + N_fcm ({n_fcm_nonconv}) + N_deg ({n_degenerate}) + N_other ({n_other})"
    )

    fcm_rate = n_fcm_nonconv / n_intended if n_intended > 0 else 0.0
    deg_rate = n_degenerate / n_intended if n_intended > 0 else 0.0

    # Field summaries
    fields: Dict[str, FieldSummary] = {}

    # 1. Primary stability distance (only defined for eligible runs with valid distance)
    fields["primary_stability_distance"] = compute_field_summary(
        records,
        field_accessor=lambda r: r.primary_stability_distance,
        field_name="primary_stability_distance",
        eligibility_predicate=lambda r: r.scientific_evidence_eligible and r.primary_stability_distance is not None,
    )

    # 2. Graph Betti 1 (defined for successful and degenerate runs where graph was constructed)
    fields["beta_1_graph"] = compute_field_summary(
        records,
        field_accessor=lambda r: r.beta_1_graph,
        field_name="beta_1_graph",
        eligibility_predicate=lambda r: r.beta_1_graph is not None,
    )

    # 3. Nerve Betti 1 (defined only where nerve homology was computed)
    fields["beta_1_nerve"] = compute_field_summary(
        records,
        field_accessor=lambda r: r.beta_1_nerve,
        field_name="beta_1_nerve",
        eligibility_predicate=lambda r: r.beta_1_nerve is not None,
    )

    # 4. Wall runtime (defined for all attempted runs)
    fields["wall_time_seconds"] = compute_field_summary(
        records,
        field_accessor=lambda r: r.wall_time_seconds,
        field_name="wall_time_seconds",
    )

    return CellSummaryResult(
        dataset_id=ds_id,
        method_id=m_id,
        noise_condition_id=noise_id,
        N_intended=n_intended,
        N_success=n_success,
        N_fcm_nonconvergence=n_fcm_nonconv,
        N_degenerate=n_degenerate,
        N_other_failure=n_other,
        fcm_nonconvergence_rate=fcm_rate,
        degenerate_rate=deg_rate,
        field_summaries=fields,
    )


def compute_conditional_bootstrap_ci(
    values: List[float],
    n_resamples: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute bootstrap confidence interval operating strictly over defined values.

    Parameters
    ----------
    values : List[float]
        The defined values entering the statistic.
    n_resamples : int
        Number of bootstrap resamples.
    seed : int
        Deterministic evaluation seed.
    alpha : float
        Significance level (e.g. 0.05 for 95% CI).

    Returns
    -------
    Dict[str, Any]
        Dictionary with mean, median, ci_lower, ci_upper, and n_entering_bootstrap.
    """
    n = len(values)
    if n == 0:
        return {
            "mean": None,
            "median": None,
            "ci_lower": None,
            "ci_upper": None,
            "n_entering_bootstrap": 0,
        }
    if n == 1:
        v = float(values[0])
        return {
            "mean": v,
            "median": v,
            "ci_lower": v,
            "ci_upper": v,
            "n_entering_bootstrap": 1,
        }

    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    boot_means = np.zeros(n_resamples, dtype=float)

    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_low = float(np.percentile(boot_means, lower_pct))
    ci_high = float(np.percentile(boot_means, upper_pct))
    mean_val = float(np.mean(arr))
    med_val = float(np.median(arr))

    return {
        "mean": mean_val,
        "median": med_val,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "n_entering_bootstrap": n,
    }


def reconcile_ledger_and_summary_counts(
    raw_records: List[ExperimentRunRecord],
    cell_summaries: List[Dict[str, Any]],
    failure_records: Optional[List[Any]] = None,
) -> Tuple[bool, List[str]]:
    """Reconcile raw execution records against derived completion-report status counts.

    Returns (is_reconciled, list_of_errors).
    """
    errors: List[str] = []

    # 1. Group raw records by (dataset_id, method_id, noise_condition_id)
    cell_groups: Dict[Tuple[str, str, str], List[ExperimentRunRecord]] = {}
    for r in raw_records:
        k = (r.identity.dataset_id, r.identity.method_id, r.identity.noise_condition_id)
        cell_groups.setdefault(k, []).append(r)

    # 2. Recompute exact status counts from raw records
    recomputed_counts: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    total_recomputed = {"success": 0, "fcm_non_convergence": 0, "degenerate_output": 0, "other": 0}

    for k, recs in cell_groups.items():
        n_succ = sum(1 for r in recs if r.status == "success")
        n_fcm = sum(1 for r in recs if r.status == "fcm_non_convergence")
        n_deg = sum(1 for r in recs if r.status == "degenerate_output")
        n_oth = sum(1 for r in recs if r.status not in ("success", "fcm_non_convergence", "degenerate_output"))
        recomputed_counts[k] = {
            "N_intended": len(recs),
            "N_success": n_succ,
            "N_fcm_nonconvergence": n_fcm,
            "N_degenerate": n_deg,
            "N_other_failure": n_oth,
        }
        total_recomputed["success"] += n_succ
        total_recomputed["fcm_non_convergence"] += n_fcm
        total_recomputed["degenerate_output"] += n_deg
        total_recomputed["other"] += n_oth

    # 3. Check each summary row against recomputed counts
    for srow in cell_summaries:
        ds = srow.get("dataset_id")
        m = srow.get("method_id")
        noise = srow.get("noise_condition_id")
        if noise is None and "noise_sigma" in srow:
            sigma_val = float(srow["noise_sigma"])
            noise = f"sigma_{sigma_val:.2f}"

        k = (ds, m, noise)
        if k not in recomputed_counts:
            errors.append(f"Summary row cell {k} not present in raw ledger.")
            continue

        expected = recomputed_counts[k]
        field_alias_map = {
            "N_intended": ["N_intended", "n_intended"],
            "N_success": ["N_success", "n_success"],
            "N_fcm_nonconvergence": ["N_fcm_nonconvergence", "n_fcm_nonconvergence"],
            "N_degenerate": ["N_degenerate", "n_degenerate"],
        }
        for std_name, aliases in field_alias_map.items():
            for alias in aliases:
                if alias in srow:
                    reported_val = int(srow[alias])
                    expected_val = expected[std_name]
                    if reported_val != expected_val:
                        errors.append(
                            f"Mismatch in cell {k} for {alias}: summary reports {reported_val}, "
                            f"raw ledger recomputation has {expected_val}."
                        )
                    break

    # 4. Check failure records reconciliation if provided
    if failure_records is not None:
        total_non_success = (
            total_recomputed["fcm_non_convergence"]
            + total_recomputed["degenerate_output"]
            + total_recomputed["other"]
        )
        if len(failure_records) != total_non_success:
            errors.append(
                f"Failure ledger count mismatch: failure ledger has {len(failure_records)} rows, "
                f"raw ledger has {total_non_success} non-success records."
            )

    return (len(errors) == 0, errors)


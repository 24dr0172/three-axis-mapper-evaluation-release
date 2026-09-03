"""Experiment Run Records, Identity Schemas, and Failure Taxonomy."""

import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from mapper_framework.exceptions import ConfigurationInvalidError


@dataclass(frozen=True)
class ExperimentRunIdentity:
    """Immutable identity identifying an exact experimental unit."""

    experiment_id: str
    dataset_id: str
    dataset_replication_id: int
    sample_size: int
    noise_condition_id: str
    method_id: str  # 'conventional' | 'f_mapper' | 'ball_mapper' | 'ensemble_kang_lim'
    output_variant: str
    lens_id: str
    parameter_configuration_id: str
    perturbation_id: str
    data_seed: Optional[int]
    perturbation_seed: Optional[int]
    resample_seed: Optional[int]
    construction_seed: Optional[int]
    selection_seed: Optional[int]
    evaluation_seed: Optional[int]
    candidate_pool_id: Optional[str] = None
    selection_split_id: Optional[str] = None
    evaluation_split_id: Optional[str] = None

    def compute_run_key(
        self,
        method_parameters: Optional[Dict[str, Any]] = None,
        primary_metric_id: Optional[str] = None,
    ) -> str:
        """Compute deterministic canonical run_id cryptographically bound to identity, parameters, and metric."""
        run_id, _ = compute_canonical_run_id(self, method_parameters, primary_metric_id)
        return run_id


def serialize_for_canonical_hash(obj: Any, strict: bool = False) -> Any:
    """Canonicalize Python objects into lossless, typed JSON-serializable primitives for deterministic hashing.

    Preserves exact type and value:
    - Floats are encoded via IEEE-754 hexadecimal notation (float.hex()), guaranteeing zero precision loss.
    - Integers, booleans, strings, and nulls are explicitly tagged with their concrete type.
    - Scikit-learn estimators are canonicalized with their fully qualified module and class name plus deep=False parameters.
    - Stable top-level callables are serialized via module and __qualname__.
    - Unstable callables (lambdas, local/nested functions) or objects containing memory addresses are rejected in strict mode.
    """
    if hasattr(obj, "get_params"):
        # Scikit-learn estimator instance
        mod = getattr(obj.__class__, "__module__", "")
        cname = obj.__class__.__name__
        # Normalize private module names (e.g. sklearn.cluster._dbscan -> sklearn.cluster)
        if mod.startswith("sklearn.cluster._"):
            mod = "sklearn.cluster"
        full_cls = f"{mod}.{cname}" if mod else cname
        params = obj.get_params(deep=False)
        return {
            "type": "estimator",
            "class": full_cls,
            "params": {str(k): serialize_for_canonical_hash(v, strict=strict) for k, v in sorted(params.items()) if not str(k).startswith("_")}
        }
    elif isinstance(obj, dict):
        # Check if this is an already serialized or reloaded estimator dictionary
        if obj.get("type") == "estimator" or ("estimator_class" in obj and "params" in obj):
            cls_name = obj.get("class") or obj.get("estimator_class", "")
            if cls_name == "DBSCAN" or cls_name.startswith("sklearn.cluster._dbscan."):
                cls_name = "sklearn.cluster.DBSCAN"
            raw_params = obj.get("params", {})
            return {
                "type": "estimator",
                "class": cls_name,
                "params": {str(k): serialize_for_canonical_hash(v, strict=strict) for k, v in sorted(raw_params.items()) if not str(k).startswith("_")}
            }
        return {str(k): serialize_for_canonical_hash(v, strict=strict) for k, v in sorted(obj.items())}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_canonical_hash(v, strict=strict) for v in obj]
    elif isinstance(obj, (bool, np.bool_)):
        return {"type": "bool", "value": bool(obj)}
    elif isinstance(obj, (int, np.integer)):
        return {"type": "int", "value": int(obj)}
    elif isinstance(obj, (float, np.floating)):
        f_val = float(obj)
        if np.isnan(f_val):
            return {"type": "float64", "value": "NaN"}
        if np.isinf(f_val):
            return {"type": "float64", "value": "Infinity" if f_val > 0 else "-Infinity"}
        return {"type": "float64", "hex": f_val.hex()}
    elif isinstance(obj, str):
        return {"type": "str", "value": obj}
    elif obj is None:
        return {"type": "null", "value": None}
    elif callable(obj):
        mod = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", None)
        is_lambda = qualname is not None and "<lambda>" in qualname
        is_local = qualname is not None and "<locals>" in qualname
        if mod and qualname and not is_lambda and not is_local:
            return {"type": "function", "module": mod, "qualname": qualname}
        if strict:
            raise ConfigurationInvalidError(
                f"Unstable or anonymous callable cannot be canonically serialized for scientific identity: {obj}"
            )
        return {"type": "unstable_callable", "name": getattr(obj, "__name__", "anonymous")}
    else:
        if strict:
            raise ConfigurationInvalidError(
                f"Object of type {type(obj).__name__} cannot be canonically serialized for scientific identity: {obj}"
            )
        s_val = str(obj)
        if re.search(r"at 0x[0-9a-fA-F]+", s_val):
            if strict:
                raise ConfigurationInvalidError(f"Memory address detected in serialized representation: {s_val}")
            s_val = re.sub(r" at 0x[0-9a-fA-F]+", "", s_val)
        return {"type": "str", "value": s_val}


def compute_canonical_run_id(
    identity: ExperimentRunIdentity,
    method_parameters: Optional[Dict[str, Any]] = None,
    primary_metric_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Compute a collision-free human-readable run_id and exact SHA-256 configuration hash.

    Parameters
    ----------
    identity : ExperimentRunIdentity
        The 16-field experimental unit identity.
    method_parameters : dict, optional
        The actual frozen method parameters passed to the constructor.
    primary_metric_id : str, optional
        The primary stability metric identifier.

    Returns
    -------
    run_id : str
        Structured human-readable identifier ending with a 16-character SHA-256 configuration fingerprint.
    config_hash : str
        Full 64-character SHA-256 hash over the lossless canonical JSON payload.
    """
    ident_dict = {str(k): serialize_for_canonical_hash(v) for k, v in sorted(asdict(identity).items())}
    params_dict = {str(k): serialize_for_canonical_hash(v) for k, v in sorted((method_parameters or {}).items())}
    payload = {
        "identity": ident_dict,
        "method_parameters": params_dict,
        "primary_metric_id": serialize_for_canonical_hash(primary_metric_id or "bottleneck_extended_pd"),
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    full_sha = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    prefix = (
        f"{identity.experiment_id}__{identity.dataset_id}__"
        f"{identity.method_id}__{identity.noise_condition_id}__"
        f"rep{identity.dataset_replication_id}"
    )
    run_id = f"{prefix}__{full_sha[:16]}"
    return run_id, full_sha


def classify_run_eligibility(
    status: str,
    construction_valid: bool = True,
    method_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> Dict[str, bool]:
    """Classify field-specific eligibility under the frozen failure and degeneracy taxonomy.

    Fail-Closed Semantics:
    - Any algorithmic failure (fcm_non_convergence), configuration invalidity,
      execution exception, resource failure, or invalid construction has ALL eligibility flags False.
    - Valid degenerate constructions (e.g. empty Mapper from pullback unassignment)
      are eligible as construction/degeneracy evidence and manuscript descriptive analysis,
      but are NOT eligible for primary persistence-stability claims.
    - Successful non-degenerate constructions are eligible across all fields.
    """
    if status == "success" and construction_valid:
        return {
            "scientific_evidence_eligible": True,
            "selection_eligible": True,
            "evaluation_eligible": True,
            "manuscript_eligible": True,
            "pilot_values_promotable_to_science": True,
            "construction_evidence_eligible": True,
            "primary_stability_eligible": True,
            "topology_eligible": True,
            "manuscript_descriptive_eligible": True,
        }
    elif status == "degenerate_output" and construction_valid:
        # Valid degenerate construction: legitimate degeneracy evidence, but NOT primary stability claim
        return {
            "scientific_evidence_eligible": False,  # Inadmissible for primary stability claim
            "selection_eligible": False,
            "evaluation_eligible": False,
            "manuscript_eligible": False,
            "pilot_values_promotable_to_science": False,
            "construction_evidence_eligible": True,   # Valid as degeneracy evidence
            "primary_stability_eligible": False,      # Inadmissible for primary stability claim
            "topology_eligible": True,                # Empty complex has well-defined Betti 0=0, 1=0
            "manuscript_descriptive_eligible": True,  # Reportable as degeneracy diagnostic
        }
    else:
        # Algorithmic failure / non-convergence / configuration invalid / exception -> FAIL CLOSED
        return {
            "scientific_evidence_eligible": False,
            "selection_eligible": False,
            "evaluation_eligible": False,
            "manuscript_eligible": False,
            "pilot_values_promotable_to_science": False,
            "construction_evidence_eligible": False,
            "primary_stability_eligible": False,
            "topology_eligible": False,
            "manuscript_descriptive_eligible": False,
        }



@dataclass(frozen=True)
class ExperimentRunRecord:
    """Comprehensive, machine-readable record of an attempted experimental run."""

    run_id: str
    timestamp: str
    run_class: str  # 'debug_only' | 'pilot' | 'confirmatory'
    identity: ExperimentRunIdentity
    method_parameters: Dict[str, Any]
    status: str  # 'success' | 'fcm_non_convergence' | 'degenerate_output' | 'resource_failure' | etc.
    failure_reason: Optional[str]
    scientific_evidence_eligible: bool
    selection_eligible: bool
    evaluation_eligible: bool
    manuscript_eligible: bool
    pilot_values_promotable_to_science: bool
    construction_evidence_eligible: bool = True
    primary_stability_eligible: bool = True
    topology_eligible: bool = True
    manuscript_descriptive_eligible: bool = True
    sample_size_requested: Optional[int] = None
    sample_size_realized: Optional[int] = None
    coverage_fraction: Optional[float] = None
    noise_fraction: Optional[float] = None
    n_nodes: Optional[int] = None
    n_edges: Optional[int] = None
    n_components: Optional[int] = None
    n_2_simplices: Optional[int] = None
    beta_0_graph: Optional[int] = None
    beta_1_graph: Optional[int] = None
    beta_0_nerve: Optional[int] = None
    beta_1_nerve: Optional[int] = None
    primary_stability_metric_id: Optional[str] = None
    primary_stability_distance: Optional[float] = None
    secondary_stability_metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    metric_compatibility_status: str = "unattempted"
    metric_undefined_reason: Optional[str] = None
    wall_time_seconds: float = 0.0
    cpu_time_seconds: float = 0.0
    peak_memory_bytes: Optional[int] = None
    construction_time_seconds: float = 0.0
    homology_time_seconds: float = 0.0
    metric_time_seconds: float = 0.0
    source_commit: Optional[str] = None
    construction_valid: bool = True
    source_hashes: Dict[str, str] = field(default_factory=dict)
    clean_method_metadata: Optional[Dict[str, Any]] = None
    perturbed_method_metadata: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        data = asdict(self)
        return _sanitize_for_json(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRunRecord":
        """Reconstruct record from dictionary with backwards compatibility for field-specific eligibility and sample size."""
        data_copy = dict(data)
        identity_data = data_copy.pop("identity")
        identity = ExperimentRunIdentity(**identity_data)
        if "construction_valid" not in data_copy:
            data_copy["construction_valid"] = (data_copy.get("status") != "fcm_non_convergence")
        
        status_val = data_copy.get("status", "success")
        c_valid = data_copy.get("construction_valid", True)
        if "construction_evidence_eligible" not in data_copy:
            elig = classify_run_eligibility(status_val, construction_valid=c_valid)
            data_copy["construction_evidence_eligible"] = elig["construction_evidence_eligible"]
            data_copy["primary_stability_eligible"] = elig["primary_stability_eligible"]
            data_copy["topology_eligible"] = elig["topology_eligible"]
            data_copy["manuscript_descriptive_eligible"] = elig["manuscript_descriptive_eligible"]

        if "sample_size_requested" not in data_copy:
            data_copy["sample_size_requested"] = identity.sample_size

        return cls(identity=identity, **data_copy)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize complex Python objects for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set, frozenset)):
        return [_sanitize_for_json(v) for v in obj]
    elif hasattr(obj, "get_params"):  # scikit-learn estimators
        return {"estimator_class": obj.__class__.__name__, "params": _sanitize_for_json(obj.get_params())}
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return repr(obj)


def save_records_json(records: List[ExperimentRunRecord], output_path: Union[str, Path]) -> None:
    """Save a list of experiment run records as structured JSON."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    dict_list = [r.to_dict() for r in records]
    p.write_text(json.dumps(dict_list, indent=2))


def load_records_json(input_path: Union[str, Path]) -> List[ExperimentRunRecord]:
    """Load experiment run records from JSON."""
    p = Path(input_path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return [ExperimentRunRecord.from_dict(d) for d in data]


def save_records_csv(records: List[ExperimentRunRecord], output_path: Union[str, Path]) -> None:
    """Save flattened experiment run records to CSV for tabular analysis."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if len(records) == 0:
        return

    flat_rows = []
    for r in records:
        row = {
            "run_id": r.run_id,
            "timestamp": r.timestamp,
            "run_class": r.run_class,
            "experiment_id": r.identity.experiment_id,
            "dataset_id": r.identity.dataset_id,
            "sample_size": r.identity.sample_size,
            "sample_size_requested": r.sample_size_requested,
            "sample_size_realized": r.sample_size_realized,
            "noise_condition_id": r.identity.noise_condition_id,
            "method_id": r.identity.method_id,
            "parameter_configuration_id": r.identity.parameter_configuration_id,
            "dataset_replication_id": r.identity.dataset_replication_id,
            "data_seed": r.identity.data_seed,
            "perturbation_seed": r.identity.perturbation_seed,
            "construction_seed": r.identity.construction_seed,
            "selection_seed": r.identity.selection_seed,
            "evaluation_seed": r.identity.evaluation_seed,
            "status": r.status,
            "failure_reason": r.failure_reason,
            "scientific_evidence_eligible": r.scientific_evidence_eligible,
            "selection_eligible": r.selection_eligible,
            "construction_evidence_eligible": r.construction_evidence_eligible,
            "primary_stability_eligible": r.primary_stability_eligible,
            "topology_eligible": r.topology_eligible,
            "manuscript_descriptive_eligible": r.manuscript_descriptive_eligible,
            "construction_valid": r.construction_valid,
            "coverage_fraction": r.coverage_fraction,
            "noise_fraction": r.noise_fraction,
            "n_nodes": r.n_nodes,
            "n_edges": r.n_edges,
            "n_components": r.n_components,
            "n_2_simplices": r.n_2_simplices,
            "beta_0_graph": r.beta_0_graph,
            "beta_1_graph": r.beta_1_graph,
            "beta_0_nerve": r.beta_0_nerve,
            "beta_1_nerve": r.beta_1_nerve,
            "primary_stability_metric_id": r.primary_stability_metric_id,
            "primary_stability_distance": r.primary_stability_distance,
            "metric_compatibility_status": r.metric_compatibility_status,
            "wall_time_seconds": r.wall_time_seconds,
            "cpu_time_seconds": r.cpu_time_seconds,
            "peak_memory_bytes": r.peak_memory_bytes,
        }
        flat_rows.append(row)

    fieldnames = list(flat_rows[0].keys())
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


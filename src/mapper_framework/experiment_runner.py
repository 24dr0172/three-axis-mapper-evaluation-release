"""Experiment Runner Orchestrator with Ledger Logging, Resource Monitoring, and Failure Preservation."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from mapper_framework.ball_mapper import BallMapper
from mapper_framework.conventional import ConventionalMapper
from mapper_framework.dataset_generators import (
    SyntheticDataset,
    apply_coordinate_noise,
    generate_branching_tripod,
    generate_clean_circle,
    generate_swiss_roll_with_hole,
)
from mapper_framework.ensemble_kang_lim import KangLimEnsembleMapper
from mapper_framework.exceptions import ConfigurationInvalidError, MapperError
from mapper_framework.experiment_records import (
    ExperimentRunIdentity,
    ExperimentRunRecord,
    classify_run_eligibility,
    compute_canonical_run_id,
    load_records_json,
    save_records_csv,
    save_records_json,
)
from mapper_framework.f_mapper import FMapper
from mapper_framework.metric_dispatch import MetricDispatcher
from mapper_framework.resource_monitor import ResourceMonitor
from mapper_framework.types import MapperOutput


def _get_git_commit_sha(source_root: Path) -> Optional[str]:
    """Retrieve the exact HEAD commit hash from git."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(source_root),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


class ExperimentRunner:
    """Orchestrates single experimental units, resource accounting, ledger serialization, and provenance tracking."""

    def __init__(
        self,
        ledger_dir: Optional[Union[str, Path]] = None,
        source_root: Optional[Union[str, Path]] = None,
    ):
        self.ledger_dir = Path(ledger_dir) if ledger_dir else Path("evidence/M5")
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.source_root = Path(source_root) if source_root else Path(".")
        self.ledger_json = self.ledger_dir / "experiment_run_ledger.json"
        self.ledger_csv = self.ledger_dir / "experiment_run_ledger.csv"
        self._source_hashes = self._compute_source_hashes()
        self._git_commit_sha = _get_git_commit_sha(self.source_root)

    def _compute_source_hashes(self) -> Dict[str, str]:
        """Compute SHA-256 hashes of core framework source files."""
        hashes = {}
        for sdir in ["src", "specs", "docs"]:
            p = self.source_root / sdir
            if p.exists():
                for f in sorted(p.rglob("*.py")) + sorted(p.rglob("*.md")):
                    if f.is_file() and not f.name.startswith("."):
                        try:
                            h = hashlib.sha256(f.read_bytes()).hexdigest()
                            hashes[str(f.relative_to(self.source_root))] = h
                        except Exception:
                            pass
        return hashes

    def run_unit(
        self,
        identity: ExperimentRunIdentity,
        method_parameters: Dict[str, Any],
        run_class: str = "debug_only",
        primary_metric_id: str = "bottleneck_extended_pd",
        resume: bool = True,
    ) -> ExperimentRunRecord:
        """Execute a single experimental unit under declared identity and method parameters."""
        run_id, config_sha = compute_canonical_run_id(identity, method_parameters, primary_metric_id)

        # Check ledger for existing record if resuming
        if resume and self.ledger_json.exists():
            existing_records = load_records_json(self.ledger_json)
            for r in existing_records:
                if r.run_id == run_id:
                    _, r_sha = compute_canonical_run_id(r.identity, r.method_parameters, r.primary_stability_metric_id)
                    if r_sha == config_sha:
                        return r

        is_confirmatory = (run_class == "confirmatory")
        scientific_eligible = is_confirmatory
        selection_eligible = is_confirmatory
        evaluation_eligible = is_confirmatory
        manuscript_eligible = is_confirmatory
        promotable = is_confirmatory

        timestamp_str = datetime.now(timezone.utc).isoformat()

        # Execute under resource monitor
        with ResourceMonitor() as monitor:
            status = "success"
            failure_reason = None
            construction_valid = True
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
            primary_dist: Optional[float] = None
            compat_status: str = "unattempted"
            undefined_reason: Optional[str] = None
            secondary_metrics: Dict[str, Optional[float]] = {}
            construction_time = 0.0
            homology_time = 0.0
            metric_time = 0.0
            m_ref = None
            m_pert = None

            try:
                # 1. Generate Clean Base Dataset X_0
                if identity.dataset_id == "clean_circle" or identity.dataset_id == "unit_circle_S1":
                    ds_clean = generate_clean_circle(
                        N=identity.sample_size,
                        data_seed=identity.data_seed,
                    )
                elif identity.dataset_id == "branching_tripod":
                    ds_clean = generate_branching_tripod(
                        N=identity.sample_size,
                        data_seed=identity.data_seed,
                    )
                elif identity.dataset_id == "swiss_roll_with_hole":
                    is_m6a2 = "M6A2" in identity.experiment_id.upper().replace("-", "") or "SWISS-REVISED" in identity.experiment_id.upper()
                    if is_m6a2:
                        if identity.lens_id != "radial_xz":
                            raise ConfigurationInvalidError(
                                f"M6-A2 Swiss Roll protocol strictly requires lens_id='radial_xz', got '{identity.lens_id}'"
                            )
                    else:
                        if identity.lens_id not in ("radial_xz", "unrolled_u", "height"):
                            raise ConfigurationInvalidError(
                                f"Unsupported lens_id '{identity.lens_id}' for Swiss Roll with hole. Supported: 'radial_xz', 'unrolled_u', 'height'"
                            )
                    try:
                        ds_clean = generate_swiss_roll_with_hole(
                            N=identity.sample_size,
                            data_seed=identity.data_seed,
                            lens_id=identity.lens_id,
                        )
                    except ValueError as e:
                        raise ConfigurationInvalidError(str(e))

                    if identity.data_seed == 42 and identity.sample_size == 2500 and len(ds_clean.points) != 2241:
                        raise ConfigurationInvalidError(
                            f"Canonical Swiss Roll under seed 42 must realize 2241 points, got {len(ds_clean.points)}"
                        )
                else:
                    raise ValueError(f"Unrecognized dataset_id '{identity.dataset_id}'")

                sample_size_req = ds_clean.metadata.get("sample_size_requested", identity.sample_size)
                sample_size_real = ds_clean.metadata.get("sample_size_realized", len(ds_clean.points))

                # Validate lens contract (Ball Mapper is metric_cover_only and lens_id is 'none')
                if ds_clean.lens_id != identity.lens_id and not (identity.method_id == "ball_mapper" and identity.lens_id == "none"):
                    raise ConfigurationInvalidError(
                        f"Lens mismatch: Dataset lens_id '{ds_clean.lens_id}' does not match identity.lens_id '{identity.lens_id}'"
                    )


                # 2. Apply Perturbation if noise > 0
                noise_str = identity.noise_condition_id.replace("sigma_", "").replace("noise_", "")
                noise_sigma = float(noise_str)
                if noise_sigma > 0.0:
                    if identity.perturbation_seed is None:
                        raise ConfigurationInvalidError(
                            f"Perturbation seed is required for noise condition {identity.noise_condition_id} > 0."
                        )
                    ds_perturbed = apply_coordinate_noise(
                        ds_clean,
                        noise_sigma=noise_sigma,
                        perturbation_seed=identity.perturbation_seed,
                    )
                else:
                    ds_perturbed = ds_clean

                # 3. Construct Clean Reference Mapper with actual timing
                t0 = time.perf_counter()
                m_ref = self._build_mapper_output(identity.method_id, method_parameters, ds_clean, identity=identity, is_confirmatory=is_confirmatory)
                t1 = time.perf_counter()

                # 4. Construct Perturbed Mapper with actual timing
                t2 = time.perf_counter()
                m_pert = self._build_mapper_output(identity.method_id, method_parameters, ds_perturbed, identity=identity, is_confirmatory=is_confirmatory)
                t3 = time.perf_counter()
                construction_time = (t1 - t0) + (t3 - t2)

                status = m_pert.status
                if status == "fcm_non_convergence":
                    status = "fcm_non_convergence"
                    failure_reason = m_pert.reason
                    scientific_eligible = False
                    selection_eligible = False
                    evaluation_eligible = False
                    manuscript_eligible = False
                    construction_valid = False
                    compat_status = "incompatible_construction_failure"
                    undefined_reason = f"FCM optimization did not converge: {m_pert.reason}"
                elif status == "degenerate_output":
                    status = "degenerate_output"
                    failure_reason = m_pert.reason
                    scientific_eligible = is_confirmatory
                    construction_valid = True
                    if m_pert.graph is not None:
                        g = m_pert.graph
                        n_nodes = g.n_nodes
                        n_edges = g.n_edges
                        n_components = g.n_components
                        beta_0_graph = g.n_components
                        beta_1_graph = g.beta_1_graph
                    else:
                        n_nodes = 0
                        n_edges = 0
                        n_components = 0
                        beta_0_graph = 0
                        beta_1_graph = 0

                    if m_pert.nerve is not None:
                        n_2_simplices = m_pert.nerve.n_2
                    else:
                        n_2_simplices = 0

                    if m_pert.homology is not None:
                        beta_0_nerve = m_pert.homology.beta_0_nerve
                        beta_1_nerve = m_pert.homology.beta_1_nerve
                    else:
                        beta_0_nerve = 0
                        beta_1_nerve = 0

                    coverage_fraction = 0.0
                    noise_fraction = 1.0
                    primary_dist = None
                    compat_status = "incompatible_degenerate_output"
                    undefined_reason = failure_reason or "Degenerate output"

                elif status == "success":
                    if m_pert.graph is None or m_pert.graph.n_nodes == 0:
                        status = "degenerate_output"
                        failure_reason = "Empty Mapper graph"
                        scientific_eligible = is_confirmatory
                        construction_valid = True
                        n_nodes = 0
                        n_edges = 0
                        n_components = 0
                        beta_0_graph = 0
                        beta_1_graph = 0
                        n_2_simplices = 0
                        beta_0_nerve = 0
                        beta_1_nerve = 0
                        coverage_fraction = 0.0
                        noise_fraction = 1.0
                        primary_dist = None
                        compat_status = "incompatible_degenerate_output"
                        undefined_reason = "Empty Mapper graph"
                    else:
                        g = m_pert.graph
                        n_nodes = g.n_nodes
                        n_edges = g.n_edges
                        n_components = g.n_components
                        beta_0_graph = g.n_components
                        beta_1_graph = g.beta_1_graph
                        covered = set()
                        for node in g.nodes.values():
                            covered.update(node.members)
                        coverage_fraction = len(covered) / len(ds_perturbed.points) if len(ds_perturbed.points) > 0 else 0.0
                        noise_fraction = 1.0 - coverage_fraction

                        t4 = time.perf_counter()
                        if m_pert.nerve:
                            n_2_simplices = m_pert.nerve.n_2
                        else:
                            n_2_simplices = 0

                        if m_pert.homology:
                            beta_0_nerve = m_pert.homology.beta_0_nerve
                            beta_1_nerve = m_pert.homology.beta_1_nerve
                        else:
                            beta_0_nerve = None
                            beta_1_nerve = None
                            if is_confirmatory and identity.output_variant in ("nerve", "dual_homology"):
                                raise ConfigurationInvalidError("Missing dual homology output in confirmatory run.")
                        t5 = time.perf_counter()
                        homology_time = t5 - t4

                        # 5. Stability Metric Calculation
                        t6 = time.perf_counter()
                        primary_dist, compat_status, undefined_reason = MetricDispatcher.evaluate_output_distance(
                            m_ref=m_ref,
                            m_perturbed=m_pert,
                            metric_id=primary_metric_id,
                        )
                        secondary_metrics = MetricDispatcher.evaluate_secondary_diagnostics(m_ref, m_pert)
                        t7 = time.perf_counter()
                        metric_time = t7 - t6
                else:
                    failure_reason = m_pert.reason or "Unknown construction failure"
                    scientific_eligible = False
                    construction_valid = False
                    compat_status = "failed"
                    undefined_reason = failure_reason

            except ConfigurationInvalidError as e:
                status = "configuration_invalid"
                failure_reason = str(e)
                scientific_eligible = False
                selection_eligible = False
                evaluation_eligible = False
                manuscript_eligible = False
                construction_valid = False
                compat_status = "failed"
                undefined_reason = str(e)
            except Exception as e:
                status = "execution_exception"
                failure_reason = str(e)
                scientific_eligible = False
                selection_eligible = False
                evaluation_eligible = False
                manuscript_eligible = False
                construction_valid = False
                compat_status = "failed"
                undefined_reason = str(e)


        clean_meta = m_ref.metadata if (m_ref and hasattr(m_ref, "metadata")) else None
        pert_meta = m_pert.metadata if (m_pert and hasattr(m_pert, "metadata")) else None

        # Determine eligibility using field-specific taxonomy and method-level evidence eligibility
        eligibility = classify_run_eligibility(
            status=status,
            construction_valid=construction_valid,
            method_id=identity.method_id,
            dataset_id=identity.dataset_id,
        )

        method_eligible = True
        if identity.method_id == "ensemble_kang_lim":
            clean_elig = clean_meta.get("scientific_evidence_eligible", False) if clean_meta else False
            pert_elig = pert_meta.get("scientific_evidence_eligible", False) if pert_meta else False
            method_eligible = clean_elig and pert_elig

        scientific_eligible = eligibility["scientific_evidence_eligible"] and is_confirmatory and method_eligible
        selection_eligible = eligibility["selection_eligible"] and is_confirmatory and method_eligible
        evaluation_eligible = eligibility["evaluation_eligible"] and is_confirmatory and method_eligible
        manuscript_eligible = eligibility["manuscript_eligible"] and is_confirmatory and method_eligible
        promotable = eligibility["pilot_values_promotable_to_science"] and is_confirmatory and method_eligible
        construction_evidence_eligible = eligibility["construction_evidence_eligible"] and is_confirmatory
        primary_stability_eligible = eligibility["primary_stability_eligible"] and is_confirmatory and method_eligible
        topology_eligible = eligibility["topology_eligible"] and is_confirmatory and method_eligible
        manuscript_descriptive_eligible = eligibility["manuscript_descriptive_eligible"] and is_confirmatory

        record = ExperimentRunRecord(
            run_id=run_id,
            timestamp=timestamp_str,
            run_class=run_class,
            identity=identity,
            method_parameters=method_parameters,
            status=status,
            failure_reason=failure_reason,
            scientific_evidence_eligible=scientific_eligible,
            selection_eligible=selection_eligible,
            evaluation_eligible=evaluation_eligible,
            manuscript_eligible=manuscript_eligible,
            pilot_values_promotable_to_science=promotable,
            construction_evidence_eligible=construction_evidence_eligible,
            primary_stability_eligible=primary_stability_eligible,
            topology_eligible=topology_eligible,
            manuscript_descriptive_eligible=manuscript_descriptive_eligible,
            sample_size_requested=sample_size_req if 'sample_size_req' in locals() else identity.sample_size,
            sample_size_realized=sample_size_real if 'sample_size_real' in locals() else identity.sample_size,
            coverage_fraction=coverage_fraction,
            noise_fraction=noise_fraction,
            n_nodes=n_nodes,
            n_edges=n_edges,
            n_components=n_components,
            n_2_simplices=n_2_simplices,
            beta_0_graph=beta_0_graph,
            beta_1_graph=beta_1_graph,
            beta_0_nerve=beta_0_nerve,
            beta_1_nerve=beta_1_nerve,
            primary_stability_metric_id=primary_metric_id,
            primary_stability_distance=primary_dist,
            secondary_stability_metrics=secondary_metrics,
            metric_compatibility_status=compat_status,
            metric_undefined_reason=undefined_reason,
            wall_time_seconds=monitor.wall_time_seconds,
            cpu_time_seconds=monitor.cpu_time_seconds,
            peak_memory_bytes=monitor.peak_memory_bytes,
            construction_time_seconds=construction_time,
            homology_time_seconds=homology_time,
            metric_time_seconds=metric_time,
            source_commit=self._git_commit_sha,
            construction_valid=construction_valid,
            source_hashes=self._source_hashes,
            clean_method_metadata=clean_meta,
            perturbed_method_metadata=pert_meta,
            notes=None,
        )

        # Append to ledger
        self._append_to_ledger(record)
        return record

    def _build_mapper_output(
        self,
        method_id: str,
        params: Dict[str, Any],
        dataset: SyntheticDataset,
        identity: Optional[ExperimentRunIdentity] = None,
        is_confirmatory: bool = False,
    ) -> MapperOutput:
        """Construct MapperOutput for declared method and dataset with strict parameter validation."""
        if method_id == "conventional":
            if is_confirmatory:
                for req in ["n_intervals", "overlap_frac", "clusterer", "input_mode"]:
                    if req not in params or params[req] is None:
                        raise ConfigurationInvalidError(f"Missing required parameter '{req}' for Conventional Mapper in confirmatory mode.")
                cm = ConventionalMapper(
                    n_intervals=params["n_intervals"],
                    overlap_frac=params["overlap_frac"],
                    clusterer=params["clusterer"],
                    input_mode=params["input_mode"],
                )
            else:
                cm = ConventionalMapper(
                    n_intervals=params.get("n_intervals", 10),
                    overlap_frac=params.get("overlap_frac", 0.3),
                    clusterer=params.get("clusterer", None),
                    input_mode=params.get("input_mode", "coordinates"),
                )
            return cm.fit_transform(dataset.points, dataset.lens)

        elif method_id == "f_mapper":
            if is_confirmatory:
                if "fcm_m" in params:
                    raise ConfigurationInvalidError("Alias 'fcm_m' is forbidden in confirmatory mode; specify 'fcm_fuzzifier' explicitly.")
                for req in ["n_intervals", "threshold", "clusterer", "fcm_fuzzifier", "fcm_max_iter", "fcm_tol", "fcm_seed", "input_mode"]:
                    if req not in params or params[req] is None:
                        raise ConfigurationInvalidError(f"Missing required parameter '{req}' for F-Mapper in confirmatory mode.")
                fm = FMapper(
                    n_intervals=params["n_intervals"],
                    threshold=params["threshold"],
                    fcm_fuzzifier=params["fcm_fuzzifier"],
                    fcm_max_iter=params["fcm_max_iter"],
                    fcm_tol=params["fcm_tol"],
                    fcm_seed=params["fcm_seed"],
                    clusterer=params["clusterer"],
                    input_mode=params["input_mode"],
                )
            else:
                fm = FMapper(
                    n_intervals=params.get("n_intervals", 10),
                    threshold=params.get("threshold", 0.2),
                    fcm_fuzzifier=params.get("fcm_fuzzifier", params.get("fcm_m", 2.0)),
                    fcm_max_iter=params.get("fcm_max_iter", 300),
                    fcm_seed=params.get("fcm_seed", 42),
                    clusterer=params.get("clusterer", None),
                    input_mode=params.get("input_mode", "coordinates"),
                )
            return fm.fit_transform(dataset.points, dataset.lens)

        elif method_id == "ball_mapper":
            if is_confirmatory:
                for req in ["epsilon", "point_order", "input_mode"]:
                    if req not in params or params[req] is None:
                        raise ConfigurationInvalidError(f"Missing required parameter '{req}' for Ball Mapper in confirmatory mode.")
                bm = BallMapper(
                    epsilon=params["epsilon"],
                    input_mode=params["input_mode"],
                )
                return bm.fit_transform(dataset.points, point_order=params["point_order"])
            else:
                bm = BallMapper(
                    epsilon=params.get("epsilon", 0.5),
                    input_mode=params.get("input_mode", "coordinates"),
                )
                return bm.fit_transform(dataset.points, point_order=params.get("point_order", None))

        elif method_id == "ensemble_kang_lim":
            lens_id_val = identity.lens_id if identity is not None else params.get("lens_id", "height")
            data_seed_val = identity.data_seed if identity is not None else params.get("data_seed", None)
            const_seed_val = identity.construction_seed if identity is not None else params.get("construction_seed", None)
            if is_confirmatory:
                for req in ["candidate_pool_id", "input_mode", "n_meta_clusters", "candidate_params", "clusterer"]:
                    if req not in params or params[req] is None:
                        raise ConfigurationInvalidError(f"Missing required parameter '{req}' for Ensemble Mapper in confirmatory mode.")
                if params.get("n_selected_base_mappers", 10) != 10:
                    raise ConfigurationInvalidError(f"n_selected_base_mappers must be exactly 10 in confirmatory mode, got {params.get('n_selected_base_mappers')}")
                em = KangLimEnsembleMapper(
                    n_meta_clusters=params["n_meta_clusters"],
                    n_selected_base_mappers=10,
                    clusterer=params["clusterer"],
                    input_mode=params["input_mode"],
                    candidate_pool_id=params["candidate_pool_id"],
                    strict_scientific=True,
                )
                return em.fit_transform(
                    dataset.points,
                    dataset.lens,
                    n_meta_clusters=params["n_meta_clusters"],
                    candidate_params=params["candidate_params"],
                    lens_id=lens_id_val,
                    data_seed=data_seed_val,
                    construction_seed=const_seed_val,
                )
            else:
                em = KangLimEnsembleMapper(
                    n_meta_clusters=params.get("n_meta_clusters", None),
                    n_selected_base_mappers=params.get("n_selected_base_mappers", 10),
                    clusterer=params.get("clusterer", None),
                    input_mode=params.get("input_mode", "coordinates"),
                    candidate_pool_id=params.get("candidate_pool_id", None),
                    strict_scientific=False,
                )
                return em.fit_transform(
                    dataset.points,
                    dataset.lens,
                    n_meta_clusters=params.get("n_meta_clusters", None),
                    candidate_params=params.get("candidate_params", None),
                    injected_candidates=params.get("injected_candidates", None),
                    injected_scores=params.get("injected_scores", None),
                    lens_id=lens_id_val,
                    data_seed=data_seed_val,
                    construction_seed=const_seed_val,
                )

        raise ConfigurationInvalidError(f"Unrecognized method_id: '{method_id}'")

    def _append_to_ledger(self, record: ExperimentRunRecord) -> None:
        """Append record to persistent JSON and CSV ledgers."""
        records = load_records_json(self.ledger_json) if self.ledger_json.exists() else []
        records.append(record)
        save_records_json(records, self.ledger_json)
        save_records_csv(records, self.ledger_csv)

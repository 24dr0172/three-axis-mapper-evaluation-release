"""I-A2 shared sealed-pipeline helpers.

Uses only:
  - sealed ConventionalMapper, apply_subsampling, SeedManager, dataset generators,
    extended-persistence bottleneck, local_quality (optional secondary);
  - the PI-frozen additive D_COMMON_ID candidate (hash-bound copy).

The release copy resolves all inputs relative to the release root. Its
scientific logic is unchanged from the executed source; only the historical
workspace/bootstrap paths were replaced for portability.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[2]
SEALED_SRC = ROOT / "src/mapper_framework"
PCA_PATH = ROOT / "data/N1_DIGITS_PCA_ARTIFACT.npz"

DATASETS = ("unit_circle_S1", "swiss_roll_with_hole", "digits_1797x64_scaled16")
FRACTIONS = (0.50, 0.65, 0.80)
DS_IDX = {d: i for i, d in enumerate(DATASETS)}
FRAC_IDX = {f: i for i, f in enumerate(FRACTIONS)}

N_INTERVALS = 10
OVERLAP_FRAC = 0.30
EPS = 0.15
MIN_SAMPLES = 3
MASTER_SEED = 42
COVER_CONFIG_ID = "cm_n10_p0.30"
PROD_DATA_SEED = 42
CAL_DATA_SEED = 868686
PROD_RESAMPLE_BLOCK = 9600
CAL_RESAMPLE_BLOCK = 9700
B_PROD = 10
B_CAL = 3
CIRCLE_N = 1000
SWISS_N_BASE = 2500
FIXED_DIGITS = "digits_1797x64_scaled16"

LENS_IDS = {
    "unit_circle_S1": "circle_height_y",
    "swiss_roll_with_hole": "radial_xz",
    "digits_1797x64_scaled16": "digits_pca1_frozen",
}

EXPECTED_DIGITS_POINTS_SHA256 = "23a53393488b04f92efc45ebc92767e2dcfdf206ba37a928f497d261e42267d2"
EXPECTED_DIGITS_PCA_SHA256 = "f74fb09f00f11a3420dfffd0d5292cfe59b5df8e38eedb9dc688e27217622e34"
EXPECTED_D_COMMON_ID_SHA256 = "5ddeac40a45bd2e32b670b51359d95ad8bcf1c4ebbf96a55cb24be062a4ab864"

# Eligibility: a Mapper construction is a scientific success only if the sealed
# pipeline returned status==success and the graph has at least two nodes.
# V==0 is typed degenerate_output/all_points_unassigned (R10 taxonomy).
# D_COMMON_ID is eligible only when BOTH constructions succeeded.


def install_sealed_imports() -> None:
    if "mapper_framework" not in sys.modules:
        pkg = types.ModuleType("mapper_framework")
        pkg.__path__ = [str(SEALED_SRC)]
        sys.modules["mapper_framework"] = pkg


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha_arr(a) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes(order="C")).hexdigest()


def sha_int_list(ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(i) for i in ids).encode("utf-8")).hexdigest()


def bind_d_common_id_copy() -> None:
    got = sha256_file(THIS_DIR / "d_common_id.py")
    if got != EXPECTED_D_COMMON_ID_SHA256:
        raise RuntimeError(
            f"d_common_id.py hash mismatch: {got} != {EXPECTED_D_COMMON_ID_SHA256}"
        )


def resample_stream_index(block: int, dataset_id: str, fraction: float, replicate: int) -> int:
    return int(block + DS_IDX[dataset_id] * 100 + FRAC_IDX[fraction] * 10 + replicate)


def derive_resample_seed(stream_index: int) -> int:
    install_sealed_imports()
    from mapper_framework.seed_manager import SeedManager
    return SeedManager(MASTER_SEED).derive_seed("resample", stream_index=stream_index)


def sample_size_for(n: int, fraction: float) -> int:
    """Frozen rounding rule: nearest integer, Python round (banker's), clamped to [1, N-1] when N>1."""
    n = int(n)
    if n <= 1:
        return n
    k = int(round(float(fraction) * n))
    return min(max(k, 1), n - 1)


def load_digits_points():
    install_sealed_imports()
    import numpy as np
    from sklearn.datasets import load_digits
    bunch = load_digits(return_X_y=False, as_frame=False)
    X = np.ascontiguousarray(bunch.data.astype("<f8", copy=True) / 16.0)
    got = sha_arr(X)
    if got != EXPECTED_DIGITS_POINTS_SHA256:
        raise RuntimeError(f"digits points digest mismatch: {got}")
    return X


def load_pca():
    install_sealed_imports()
    import numpy as np
    got = sha256_file(PCA_PATH)
    if got != EXPECTED_DIGITS_PCA_SHA256:
        raise RuntimeError(f"digits PCA digest mismatch: {got}")
    return np.load(PCA_PATH)


def digits_lens(X, pca) -> Any:
    import numpy as np
    return np.asarray((X - pca["pca_mean"]) @ pca["signed_pc1_component"], dtype=float)


def load_clean_dataset(dataset_id: str, data_seed: Optional[int]) -> Dict[str, Any]:
    """Return points, lens, N, space metadata. Digits ignores data_seed (fixed array)."""
    install_sealed_imports()
    import numpy as np
    from mapper_framework.dataset_generators import generate_clean_circle, generate_swiss_roll_with_hole

    if dataset_id == "unit_circle_S1":
        ds = generate_clean_circle(N=CIRCLE_N, radius=1.0, data_seed=data_seed, sampling="uniform")
        return {
            "dataset_id": dataset_id,
            "points": np.ascontiguousarray(ds.points, dtype="<f8"),
            "lens": np.ascontiguousarray(ds.lens, dtype=float),
            "lens_id": LENS_IDS[dataset_id],
            "data_seed": data_seed,
            "N": int(len(ds.points)),
            "synthetic": ds,
        }
    if dataset_id == "swiss_roll_with_hole":
        ds = generate_swiss_roll_with_hole(
            N=SWISS_N_BASE, data_seed=data_seed, resample_to_exact_N=False, lens_id="radial_xz"
        )
        return {
            "dataset_id": dataset_id,
            "points": np.ascontiguousarray(ds.points, dtype="<f8"),
            "lens": np.ascontiguousarray(ds.lens, dtype=float),
            "lens_id": LENS_IDS[dataset_id],
            "data_seed": data_seed,
            "N": int(len(ds.points)),
            "synthetic": ds,
        }
    if dataset_id == FIXED_DIGITS:
        X = load_digits_points()
        pca = load_pca()
        lens = digits_lens(X, pca)
        from mapper_framework.dataset_generators import SyntheticDataset
        syn = SyntheticDataset(
            dataset_id="digits_1797x64_scaled16",
            space_name="digits_1797x64_scaled16",
            points=X,
            lens=lens,
            lens_id=LENS_IDS[dataset_id],
            data_seed=None,
            perturbation_seed=None,
            resample_seed=None,
            point_order=tuple(range(len(X))),
            noise_sigma=0.0,
            is_perturbed=False,
            metadata={"pca_artifact": str(PCA_PATH)},
        )
        return {
            "dataset_id": dataset_id,
            "points": X,
            "lens": np.ascontiguousarray(lens, dtype=float),
            "lens_id": LENS_IDS[dataset_id],
            "data_seed": None,
            "N": int(len(X)),
            "synthetic": syn,
        }
    raise ValueError(f"unknown dataset_id {dataset_id}")


def apply_subsample(clean: Dict[str, Any], fraction: float, resample_seed: int) -> Dict[str, Any]:
    install_sealed_imports()
    from mapper_framework.dataset_generators import apply_subsampling
    n = clean["N"]
    k = sample_size_for(n, fraction)
    sub = apply_subsampling(clean["synthetic"], sample_size=k, resample_seed=resample_seed)
    idx = [int(i) for i in sub.metadata["subsample_indices"]]
    # apply_subsampling sorts indices; local i <-> original idx[i]
    local_to_original = {i: idx[i] for i in range(len(idx))}
    return {
        "points": sub.points,
        "lens": sub.lens,
        "n_retained": int(len(idx)),
        "fraction": float(fraction),
        "resample_seed": int(resample_seed),
        "original_indices": idx,
        "original_indices_sha256": sha_int_list(idx),
        "local_to_original": local_to_original,
        "synthetic": sub,
    }


def construct_conventional(X, lens):
    install_sealed_imports()
    from sklearn.cluster import DBSCAN
    from mapper_framework.conventional import ConventionalMapper
    t0 = time.perf_counter()
    mapper = ConventionalMapper(
        n_intervals=N_INTERVALS,
        overlap_frac=OVERLAP_FRAC,
        clusterer=DBSCAN(eps=EPS, min_samples=MIN_SAMPLES),
        input_mode="coordinates",
    )
    out = mapper.fit_transform(X, lens, max_triangles=200_000)
    dt = time.perf_counter() - t0
    return out, float(dt)


def construction_fields(out) -> Dict[str, Any]:
    graph = out.graph
    hom = out.homology
    nerve = out.nerve
    V = graph.n_nodes if graph is not None else None
    E = graph.n_edges if graph is not None else None
    b1g = graph.beta_1_graph if graph is not None else None
    tri = nerve.n_2 if nerve is not None else None
    b0n = hom.beta_0_nerve if hom is not None else None
    b1n = hom.beta_1_nerve if hom is not None else None
    status = out.status
    reason = out.reason
    if status == "success" and (V is None or V == 0):
        status = "degenerate_output"
        reason = "all_points_unassigned"
    eligible = bool(status == "success" and V is not None and V >= 2)
    return {
        "status": status,
        "reason": None if reason is None else str(reason)[:300],
        "eligible": eligible,
        "V": V,
        "E": E,
        "beta1_graph_STRUCTURAL_DIAGNOSTIC": b1g,
        "nerve_triangles_STRUCTURAL_DIAGNOSTIC": tri,
        "beta0_nerve_STRUCTURAL_DIAGNOSTIC": b0n,
        "beta1_nerve_STRUCTURAL_DIAGNOSTIC": b1n,
    }


def _import_d_common_id():
    bind_d_common_id_copy()
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    import d_common_id as mod
    return mod


def mapper_function_bins(out, n_obs: int, index_map: Optional[Dict[int, int]] = None) -> Dict[int, Dict[int, Any]]:
    """Build Belchí Mapper-function bins from sealed pullback_records.

    index_map: local index -> original observation id. None => identity (full array).
    Unassigned = outside_cover; DBSCAN label < 0 = noise_unassigned.
    """
    dci = _import_d_common_id()
    OUTSIDE, NOISE = dci.OUTSIDE, dci.NOISE

    meta = out.metadata or {}
    n_bins = int(meta.get("effective_n_intervals") or meta.get("requested_n_intervals") or N_INTERVALS)
    orig_ids = [index_map[i] if index_map is not None else i for i in range(n_obs)]
    bins: Dict[int, Dict[int, Any]] = {k: {oid: OUTSIDE for oid in orig_ids} for k in range(n_bins)}
    for rec in meta.get("pullback_records") or []:
        k = int(rec["cover_element_id"])
        if k not in bins:
            bins[k] = {oid: OUTSIDE for oid in orig_ids}
        for local_i, label in zip(rec["sample_indices"], rec["local_cluster_labels"]):
            oid = index_map[int(local_i)] if index_map is not None else int(local_i)
            if oid not in bins[k]:
                bins[k][oid] = OUTSIDE
            if int(label) < 0:
                bins[k][oid] = NOISE
            else:
                bins[k][oid] = int(label)
    return bins


def primary_d_common_id(ref_out, sub_out, common_ids: List[int], n_ref: int, n_sub: int,
                        local_to_original: Dict[int, int], filter_id: str,
                        filter_definition_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Restricted comparison: D_COMMON_ID(Mapper(X), Mapper(S_r); I = original retained ids)."""
    dci = _import_d_common_id()

    if ref_out.status != "success" or (ref_out.graph is None) or (ref_out.graph.n_nodes == 0):
        return {"distance": None, "status": "null", "reason": f"invalid_output_A={ref_out.status}",
                "n_common": len(common_ids), "common_digest": sha_int_list(sorted(common_ids))}
    if sub_out.status != "success" or (sub_out.graph is None) or (sub_out.graph.n_nodes == 0):
        return {"distance": None, "status": "null", "reason": f"invalid_output_B={sub_out.status}",
                "n_common": len(common_ids), "common_digest": sha_int_list(sorted(common_ids))}

    bins_A = mapper_function_bins(ref_out, n_ref, index_map=None)
    bins_B = mapper_function_bins(sub_out, n_sub, index_map=local_to_original)
    cover_hash_A = dci.hash_realized_cover(bins_A, common_ids)
    cover_hash_B = dci.hash_realized_cover(bins_B, common_ids)
    result = dci.d_common_id(
        bins_A, bins_B,
        common_ids=set(common_ids),
        cover_id_A=f"realized_cover:{cover_hash_A}",
        cover_id_B=f"realized_cover:{cover_hash_B}",
        filter_id_A=filter_id,
        filter_id_B=filter_id,
        cover_hash_A=cover_hash_A,
        cover_hash_B=cover_hash_B,
        filter_hash_A=filter_definition_sha256,
        filter_hash_B=filter_definition_sha256,
        comparison_scope="same_filter",
        status_A="success",
        status_B="success",
    )
    return result


def secondary_ep_bottleneck(ref_out, sub_out) -> Dict[str, Any]:
    """Typed extended-persistence bottleneck. SECONDARY diagnostic, not restricted D-COMMON.

    Import of gudhi is inside the typed-failure path: absence of gudhi is
    `extended_persistence_failure`, never an uncaught crash and never a
    reason to null the primary D_COMMON_ID.
    """
    payload = {
        "metric": "typed_extended_persistence_bottleneck_SIGNATURE_DIAGNOSTIC",
        "not_restricted_d_common": True,
        "distance": None,
        "status": "not_computed",
        "reason": None,
    }
    if ref_out is None or sub_out is None:
        payload["status"] = "null"
        payload["reason"] = "construction_missing"
        return payload
    if ref_out.status != "success" or sub_out.status != "success":
        payload["status"] = "null"
        payload["reason"] = "construction_not_success"
        return payload
    if ref_out.graph is None or sub_out.graph is None:
        payload["status"] = "null"
        payload["reason"] = "missing_graph"
        return payload
    try:
        install_sealed_imports()
        from mapper_framework.stability_metrics import compute_mapper_extended_pd
        from mapper_framework.extended_persistence import compute_extended_diagram_bottleneck_distance
        _, dgm_ref = compute_mapper_extended_pd(ref_out)
        _, dgm_sub = compute_mapper_extended_pd(sub_out)
        dist = compute_extended_diagram_bottleneck_distance(dgm_ref, dgm_sub)
        if dist != dist or dist == float("inf"):
            payload["status"] = "metric_undefined"
            payload["reason"] = "nonfinite_bottleneck"
            return payload
        payload["distance"] = float(dist)
        payload["status"] = "success"
        return payload
    except Exception as e:
        payload["status"] = "extended_persistence_failure"
        payload["reason"] = f"{type(e).__name__}:{e}"[:200]
        return payload


def comparison_eligibility(ref_fields: Dict[str, Any], sub_fields: Dict[str, Any],
                           d_common: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
    """Scientific unit eligibility for the restricted D_COMMON_ID estimand."""
    if not ref_fields["eligible"]:
        return False, "reference_ineligible", ref_fields.get("reason")
    if not sub_fields["eligible"]:
        return False, "subsample_ineligible", sub_fields.get("reason")
    if d_common.get("status") != "success" or d_common.get("distance") is None:
        return False, "d_common_id_null", d_common.get("reason")
    dist = d_common["distance"]
    if dist != dist or dist < 0:
        return False, "d_common_id_nonfinite", "nonfinite_or_negative"
    return True, "success", None


def seed_table_rows(block: int, n_rep: int, used_registry: Optional[set] = None) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for ds in DATASETS:
        for frac in FRACTIONS:
            for r in range(n_rep):
                si = resample_stream_index(block, ds, frac, r)
                seed = derive_resample_seed(si)
                if seed in seen:
                    raise RuntimeError(f"internal resample seed collision {seed}")
                seen.add(seed)
                if used_registry is not None and seed in used_registry:
                    raise RuntimeError(f"resample seed {seed} intersects programme registry (si={si})")
                rows.append({
                    "dataset_id": ds,
                    "fraction": frac,
                    "replicate_index": r,
                    "stream_index": si,
                    "resample_seed": seed,
                    "resample_seed_hex": hex(seed),
                    "namespace": "resample",
                    "master_seed": MASTER_SEED,
                })
    return rows


def construction_id(campaign_id: str, kind: str, dataset_id: str,
                    fraction: Optional[float] = None, replicate: Optional[int] = None) -> str:
    if kind == "reference":
        return f"{campaign_id}__REF__{dataset_id}"
    return f"{campaign_id}__SUB__{dataset_id}__f{fraction:.2f}__r{replicate}"


def plan_rows(campaign_id: str, n_rep: int) -> List[Dict[str, Any]]:
    rows = []
    for ds in DATASETS:
        rows.append({
            "kind": "reference",
            "construction_id": construction_id(campaign_id, "reference", ds),
            "dataset_id": ds,
            "method": "conventional",
            "fraction": None,
            "replicate_index": None,
        })
    for ds in DATASETS:
        for frac in FRACTIONS:
            for r in range(n_rep):
                rows.append({
                    "kind": "subsample",
                    "construction_id": construction_id(campaign_id, "subsample", ds, frac, r),
                    "dataset_id": ds,
                    "method": "conventional",
                    "fraction": frac,
                    "replicate_index": r,
                    "reference_construction_id": construction_id(campaign_id, "reference", ds),
                })
    return rows

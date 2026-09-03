"""Typed, Immutable Filter Perturbation Definitions for Joint I-B1 / III-4 Campaign.

Implements the controlled filter perturbation family

    f_{delta, r}(x) = f(x) + delta * h_r(x),   ||h_r||_inf = 1.0

on the two frozen Axis-III benchmark domains (unit_circle_S1, branching_tripod_Y),
governed by:
    final/scientific_contracts/IB1_III4_FILTER_PERTURBATION_CONTRACT.md
    final/scientific_contracts/configs/frozen/IB1_III4_FILTER_REGISTRY.json

h_r is defined globally on the continuous/PL reference domain S, never invented
point-by-point at discrete sample observations: `evaluate_perturbed_lens` is the
single shared code path used both for exact reference-domain evaluation and for
matched finite-sample restriction f_{delta,r}(x_i).
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from mapper_framework.exceptions import ConfigurationInvalidError
from mapper_framework.reeb_references import CIRCLE_HEIGHT_Y_HASH, TRIPOD_HEIGHT_Y_HASH

# =============================================================================
# 1. Canonical Identifiers
# =============================================================================

CIRCLE_BENCHMARK_ID = "unit_circle_S1"
TRIPOD_BENCHMARK_ID = "branching_tripod_Y"

CIRCLE_BASELINE_FILTER_ID = "circle_height_y"
TRIPOD_BASELINE_FILTER_ID = "tripod_height_y"

BASELINE_DIRECTION_ID = "baseline_none"

CIRCLE_DIRECTIONS: Tuple[str, ...] = ("harm_sin2", "harm_cos2")
TRIPOD_DIRECTIONS: Tuple[str, ...] = ("tripod_hat_junction", "tripod_hat_lower_leaf")

VALID_DIRECTIONS_BY_BENCHMARK: Dict[str, Tuple[str, ...]] = {
    CIRCLE_BENCHMARK_ID: CIRCLE_DIRECTIONS,
    TRIPOD_BENCHMARK_ID: TRIPOD_DIRECTIONS,
}

BASELINE_FILTER_ID_BY_BENCHMARK: Dict[str, str] = {
    CIRCLE_BENCHMARK_ID: CIRCLE_BASELINE_FILTER_ID,
    TRIPOD_BENCHMARK_ID: TRIPOD_BASELINE_FILTER_ID,
}

BASELINE_FILTER_HASH_BY_BENCHMARK: Dict[str, str] = {
    CIRCLE_BENCHMARK_ID: CIRCLE_HEIGHT_Y_HASH,
    TRIPOD_BENCHMARK_ID: TRIPOD_HEIGHT_Y_HASH,
}

DOMAIN_ID_BY_BENCHMARK: Dict[str, str] = {
    CIRCLE_BENCHMARK_ID: "unit_circle_R2",
    TRIPOD_BENCHMARK_ID: "branching_tripod_Y",
}

REFERENCE_CONSTRUCTOR_ID_BY_BENCHMARK: Dict[str, str] = {
    CIRCLE_BENCHMARK_ID: "circle_perturbed_analytic_reeb",
    TRIPOD_BENCHMARK_ID: "tripod_perturbed_pl_reeb",
}

PERTURBATION_FORMULA_TEXT: Dict[Tuple[str, str], str] = {
    (CIRCLE_BENCHMARK_ID, "harm_sin2"): "f_delta(theta) = sin(theta) + delta*sin(2*theta)",
    (CIRCLE_BENCHMARK_ID, "harm_cos2"): "f_delta(theta) = sin(theta) + delta*cos(2*theta)",
    (TRIPOD_BENCHMARK_ID, "tripod_hat_junction"): "h(O)=1.0, h(v0)=h(v2)=h(v3)=0.0, linear PL interpolation on each edge",
    (TRIPOD_BENCHMARK_ID, "tripod_hat_lower_leaf"): "h(v0)=1.0, h(O)=h(v2)=h(v3)=0.0, linear PL interpolation on each edge",
}

# Exact analytic/PL critical transition delta for each (benchmark, direction).
# Reference/test metadata only; never used to gate the confirmatory campaign grid.
CRITICAL_TRANSITION_DELTA: Dict[Tuple[str, str], float] = {
    (CIRCLE_BENCHMARK_ID, "harm_sin2"): 0.5,
    (CIRCLE_BENCHMARK_ID, "harm_cos2"): 0.25,
    (TRIPOD_BENCHMARK_ID, "tripod_hat_junction"): 1.0,
    (TRIPOD_BENCHMARK_ID, "tripod_hat_lower_leaf"): 1.0,
}

CONFIRMATORY_DELTA_GRID: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.45, 0.55, 0.80, 1.10)
VALIDATION_ONLY_DELTAS: Tuple[float, ...] = (0.25, 0.50, 1.00)

CIRCLE_DOMAIN_TOL = 1e-9
TRIPOD_ARM_TOL = 1e-9
FILTER_REGISTRY_SPEC_VERSION = "1.0.0"


def delta_code(delta: float) -> str:
    """Deterministic zero-padded delta code, e.g. 0.05 -> 'd005', 1.10 -> 'd110'."""
    if not math.isfinite(delta) or delta < 0:
        raise ConfigurationInvalidError(f"delta must be finite and >= 0, got {delta}")
    return f"d{int(round(delta * 100)):03d}"


# =============================================================================
# 2. Immutable Perturbed Filter Definition
# =============================================================================

@dataclass(frozen=True)
class PerturbedFilter:
    """Immutable, canonically identified perturbed filter f_{delta,r} = f + delta*h_r."""

    benchmark_id: str
    baseline_filter_id: str
    direction_id: str
    delta: float
    filter_id: str
    filter_definition_hash: str
    is_baseline: bool


def _canonical_perturbation_payload(benchmark_id: str, direction_id: str, delta: float) -> Dict[str, object]:
    """Deterministic, locale-independent, round-trip-safe canonical serialization basis for hashing."""
    return {
        "benchmark_id": benchmark_id,
        "baseline_filter_id": BASELINE_FILTER_ID_BY_BENCHMARK[benchmark_id],
        "direction_id": direction_id,
        "delta_hex": float(delta).hex(),
        "formula": PERTURBATION_FORMULA_TEXT[(benchmark_id, direction_id)],
        "domain_id": DOMAIN_ID_BY_BENCHMARK[benchmark_id],
        "reference_constructor_id": REFERENCE_CONSTRUCTOR_ID_BY_BENCHMARK[benchmark_id],
        "l_infinity_norm": 1.0,
        "spec_version": FILTER_REGISTRY_SPEC_VERSION,
    }


def compute_filter_definition_hash(benchmark_id: str, direction_id: str, delta: float) -> str:
    """Deterministic SHA-256 filter_definition_hash over the canonical perturbation payload."""
    payload = _canonical_perturbation_payload(benchmark_id, direction_id, delta)
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_perturbed_filter(benchmark_id: str, direction_id: str, delta: float) -> PerturbedFilter:
    """Construct and validate an immutable perturbed filter definition. No hidden defaults.

    Rejects: unknown benchmark_id, unknown/incompatible direction_id, non-finite or negative
    delta, and any implicit baseline/nonbaseline ambiguity.
    """
    if benchmark_id not in VALID_DIRECTIONS_BY_BENCHMARK:
        raise ConfigurationInvalidError(f"Unknown benchmark_id '{benchmark_id}'.")

    try:
        delta_f = float(delta)
    except (TypeError, ValueError):
        raise ConfigurationInvalidError(f"delta must be a real number, got {delta!r}.")
    if not math.isfinite(delta_f):
        raise ConfigurationInvalidError(f"delta must be finite, got {delta!r}.")
    if delta_f < 0.0:
        raise ConfigurationInvalidError(f"delta must be >= 0, got {delta_f}.")

    baseline_id = BASELINE_FILTER_ID_BY_BENCHMARK[benchmark_id]

    if delta_f == 0.0:
        if direction_id != BASELINE_DIRECTION_ID:
            raise ConfigurationInvalidError(
                f"delta=0.0 requires explicit direction_id='{BASELINE_DIRECTION_ID}'; got '{direction_id}'."
            )
        return PerturbedFilter(
            benchmark_id=benchmark_id,
            baseline_filter_id=baseline_id,
            direction_id=BASELINE_DIRECTION_ID,
            delta=0.0,
            filter_id=baseline_id,
            filter_definition_hash=BASELINE_FILTER_HASH_BY_BENCHMARK[benchmark_id],
            is_baseline=True,
        )

    if direction_id == BASELINE_DIRECTION_ID:
        raise ConfigurationInvalidError("direction_id='baseline_none' requires delta=0.0.")

    valid_dirs = VALID_DIRECTIONS_BY_BENCHMARK[benchmark_id]
    if direction_id not in valid_dirs:
        raise ConfigurationInvalidError(
            f"direction_id '{direction_id}' is invalid or incompatible for benchmark '{benchmark_id}'; "
            f"valid directions: {valid_dirs}."
        )

    filter_id = f"{baseline_id}__{direction_id}__{delta_code(delta_f)}"
    f_hash = compute_filter_definition_hash(benchmark_id, direction_id, delta_f)
    return PerturbedFilter(
        benchmark_id=benchmark_id,
        baseline_filter_id=baseline_id,
        direction_id=direction_id,
        delta=delta_f,
        filter_id=filter_id,
        filter_definition_hash=f_hash,
        is_baseline=False,
    )


# =============================================================================
# 3. Circle Evaluation: theta = atan2(y, x); f(theta) = sin(theta) = y
# =============================================================================

def circle_theta(coordinates: np.ndarray, radius: float = 1.0, tol: float = CIRCLE_DOMAIN_TOL) -> np.ndarray:
    """Compute theta = atan2(y, x) in [0, 2*pi), validating strict Circle-domain membership."""
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ConfigurationInvalidError(f"Circle coordinates must be shape (N, 2), got {coords.shape}")
    x = coords[:, 0]
    y = coords[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    off_domain = np.abs(r - radius) > tol
    if np.any(off_domain):
        bad = np.where(off_domain)[0]
        raise ConfigurationInvalidError(
            f"{len(bad)} point(s) off Circle domain (radius={radius}) beyond tol={tol}; "
            f"first bad index {int(bad[0])}, r={float(r[bad[0]])}."
        )
    return np.arctan2(y, x) % (2.0 * np.pi)


def evaluate_circle_h(direction_id: str, theta: np.ndarray) -> np.ndarray:
    """Evaluate h_r(theta) for a Circle perturbation direction. Never invented per-sample."""
    theta_arr = np.asarray(theta, dtype=float)
    if direction_id == "harm_sin2":
        return np.sin(2.0 * theta_arr)
    elif direction_id == "harm_cos2":
        return np.cos(2.0 * theta_arr)
    raise ConfigurationInvalidError(f"Unknown Circle perturbation direction_id '{direction_id}'.")


def max_abs_h_circle(direction_id: str, n_grid: int = 200_000) -> float:
    """Dense-grid numerical verification of ||h_r||_inf on the Circle (exact analytic value is 1.0)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    return float(np.max(np.abs(evaluate_circle_h(direction_id, theta))))


# =============================================================================
# 4. Tripod Evaluation: exact 4-vertex complex, verified arm identity + affine coordinate
# =============================================================================

# Sealed 4-vertex complex (must match mapper_framework.reeb_references.build_tripod_exact_pl_reeb).
TRIPOD_V0 = (0.0, -1.0)   # lower leaf
TRIPOD_O = (0.0, 0.0)     # junction
TRIPOD_V2 = (-1.0, 1.0)   # upper-left leaf
TRIPOD_V3 = (1.0, 1.0)    # upper-right leaf


def tripod_arm_and_t(coordinates: np.ndarray, tol: float = TRIPOD_ARM_TOL) -> Tuple[List[str], np.ndarray]:
    """Classify each observation by verified arm identity and affine coordinate t in [0, 1] (t=0 at O).

    Geometric inference (no per-sample invented perturbation values): a point lies on arm e0
    (O->v0) iff x==0 and y<=0; on arm e2 (O->v2) iff x==-y and x<=0; on arm e3 (O->v3) iff
    x==y and x>=0. Ambiguous or off-domain points are rejected.
    """
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ConfigurationInvalidError(f"Tripod coordinates must be shape (N, 2), got {coords.shape}")
    x = coords[:, 0]
    y = coords[:, 1]
    n = len(x)

    on_v0 = (np.abs(x) <= tol) & (y <= tol)
    on_v2 = (np.abs(x + y) <= tol) & (x <= tol)
    on_v3 = (np.abs(x - y) <= tol) & (x >= -tol)

    arm: List[str] = [""] * n
    t = np.full(n, np.nan, dtype=float)
    assigned = np.zeros(n, dtype=bool)

    for mask, arm_id, t_expr in (
        (on_v0, "v0", -y),
        (on_v2, "v2", y),
        (on_v3, "v3", y),
    ):
        sel = mask & (~assigned)
        idxs = np.where(sel)[0]
        for i in idxs:
            arm[i] = arm_id
            t[i] = t_expr[i]
        assigned |= sel

    if not np.all(assigned):
        bad = np.where(~assigned)[0]
        raise ConfigurationInvalidError(
            f"{len(bad)} point(s) off Tripod domain / ambiguous arm classification beyond tol={tol}; "
            f"first bad index {int(bad[0])}, point=({float(x[bad[0]])}, {float(y[bad[0]])})."
        )

    out_of_range = (t < -tol) | (t > 1.0 + tol)
    if np.any(out_of_range):
        bad = np.where(out_of_range)[0]
        raise ConfigurationInvalidError(
            f"{len(bad)} point(s) outside affine arm range [0, 1] beyond tol={tol}; first bad index {int(bad[0])}, t={float(t[bad[0]])}."
        )

    return arm, np.clip(t, 0.0, 1.0)


def evaluate_tripod_h(direction_id: str, coordinates: np.ndarray) -> np.ndarray:
    """Evaluate h_r at Tripod observations by verified arm identity + affine coordinate. Never invented per-sample."""
    arm, t = tripod_arm_and_t(coordinates)
    n = len(t)
    h = np.zeros(n, dtype=float)
    if direction_id == "tripod_hat_junction":
        # h(O)=1, h(leaf)=0 on every arm -> h(t) = 1 - t uniformly.
        h = 1.0 - t
    elif direction_id == "tripod_hat_lower_leaf":
        # h(v0)=1, h(O)=h(v2)=h(v3)=0 -> h(t)=t on arm v0, h==0 on arms v2/v3.
        for i in range(n):
            if arm[i] == "v0":
                h[i] = t[i]
    else:
        raise ConfigurationInvalidError(f"Unknown Tripod perturbation direction_id '{direction_id}'.")
    return h


def max_abs_h_tripod(direction_id: str, n_per_arm: int = 50_000) -> float:
    """Dense-grid numerical verification of ||h_r||_inf on the Tripod (exact analytic value is 1.0)."""
    t = np.linspace(0.0, 1.0, n_per_arm)
    pts = np.vstack([
        np.column_stack([np.zeros_like(t), -t]),        # arm v0
        np.column_stack([-t, t]),                        # arm v2
        np.column_stack([t, t]),                         # arm v3
    ])
    h = evaluate_tripod_h(direction_id, pts)
    return float(np.max(np.abs(h)))


# =============================================================================
# 5. Unified Matched-Sample / Reference-Domain Evaluation (single shared code path)
# =============================================================================

def evaluate_perturbed_lens(benchmark_id: str, direction_id: str, delta: float, coordinates: np.ndarray) -> np.ndarray:
    """Evaluate f_{delta,r}(x) = f(x) + delta*h_r(x) at the given coordinates.

    f(x, y) = y (height lens) for both benchmarks. Used identically for exact reference-domain
    evaluation and for restriction to a finite matched observation sample.
    """
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ConfigurationInvalidError(f"coordinates must be shape (N, 2), got {coords.shape}")
    baseline = coords[:, 1].copy()

    if delta == 0.0:
        return baseline

    if benchmark_id == CIRCLE_BENCHMARK_ID:
        theta = circle_theta(coords)
        h = evaluate_circle_h(direction_id, theta)
    elif benchmark_id == TRIPOD_BENCHMARK_ID:
        h = evaluate_tripod_h(direction_id, coords)
    else:
        raise ConfigurationInvalidError(f"Unknown benchmark_id '{benchmark_id}'.")

    return baseline + delta * h


def evaluate_perturbed_lens_for_filter(pf: PerturbedFilter, coordinates: np.ndarray) -> np.ndarray:
    """Convenience wrapper evaluating a PerturbedFilter at the given coordinates."""
    return evaluate_perturbed_lens(pf.benchmark_id, pf.direction_id, pf.delta, coordinates)


# =============================================================================
# 6. Registry Enumeration (used to materialize the frozen filter-registry config)
# =============================================================================

def all_perturbation_directions() -> List[Dict[str, object]]:
    """Enumerate canonical direction definitions across both benchmarks."""
    out: List[Dict[str, object]] = []
    for benchmark_id, dirs in VALID_DIRECTIONS_BY_BENCHMARK.items():
        for direction_id in dirs:
            out.append({
                "benchmark_id": benchmark_id,
                "direction_id": direction_id,
                "baseline_filter_id": BASELINE_FILTER_ID_BY_BENCHMARK[benchmark_id],
                "formula": PERTURBATION_FORMULA_TEXT[(benchmark_id, direction_id)],
                "l_infinity_norm": 1.0,
                "critical_transition_delta": CRITICAL_TRANSITION_DELTA[(benchmark_id, direction_id)],
                "domain_id": DOMAIN_ID_BY_BENCHMARK[benchmark_id],
                "reference_constructor_id": REFERENCE_CONSTRUCTOR_ID_BY_BENCHMARK[benchmark_id],
            })
    return out

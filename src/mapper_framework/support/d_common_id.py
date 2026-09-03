"""Exact common-cover Mapper-function distance.

This module implements the joint optimization in Belchi et al. Definition 9:

    D_M(f, g) = min_pi |{x : f(x) != pi g(x)}| / |I|.

The binary mismatch indicator is optimized once per observation over all cover
elements simultaneously. It is not the cheaper construction obtained by
optimizing every cover element independently; Belchi et al. describe that
construction only as an upper estimate.

The released extension keeps ``outside_cover`` and DBSCAN noise as immutable
states. Ordinary cluster labels may be permuted independently within each
cover element. Every successful call is bound to the actual realized cover on
the compared identities and to an explicit filter-definition digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, Optional, Set


OUTSIDE = "outside_cover"
NOISE = "noise_unassigned"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _hash_sorted_ids(ids: Iterable[Any]) -> str:
    return hashlib.sha256(",".join(sorted(map(str, ids))).encode()).hexdigest()


def _is_special(value: Any) -> bool:
    return value in (OUTSIDE, NOISE)


def _normalize(value: Any) -> Any:
    if isinstance(value, (set, frozenset, list, tuple)):
        return frozenset(value)
    return value


def _is_failed_status(status: Optional[str]) -> bool:
    if status is None:
        return False
    return str(status).lower() not in (
        "success", "valid_empty", "empty_valid", "success_empty"
    )


def hash_realized_cover(
    bins: Dict[Any, Dict[Any, Any]], common_ids: Iterable[Any]
) -> str:
    """Hash indexed point-to-cover-element incidence on ``common_ids``.

    Cluster labels are excluded. Noise is still membership in a pullback; only
    ``outside_cover`` denotes non-membership. Restriction to ``common_ids``
    makes full-versus-subsample comparisons well typed.
    """

    ids = set(common_ids)
    payload = []
    for bin_id in sorted(bins, key=lambda value: str(value)):
        members = sorted(
            str(x)
            for x in ids
            if bins.get(bin_id, {}).get(x, OUTSIDE) != OUTSIDE
        )
        payload.append([str(bin_id), members])
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def hash_filter_definition(definition: str) -> str:
    """Return the canonical UTF-8 SHA-256 of an explicit filter definition."""

    if not isinstance(definition, str) or not definition.strip():
        raise ValueError("filter definition must be a non-empty string")
    return hashlib.sha256(definition.strip().encode("utf-8")).hexdigest()


def _rejected(reason: str) -> Dict[str, Any]:
    return {
        "distance": None,
        "common_ids": None,
        "reason": reason,
        "status": "rejected",
    }


def _require_digest(name: str, value: Optional[str]) -> Optional[Dict[str, Any]]:
    if value is None:
        return _rejected(f"missing_{name}")
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        return _rejected(f"invalid_{name}")
    return None


def d_common_id(
    bins_A: Dict[Any, Dict[Any, Any]],
    bins_B: Dict[Any, Dict[Any, Any]],
    common_ids: Optional[Set[Any]] = None,
    cover_id_A: Optional[str] = None,
    cover_id_B: Optional[str] = None,
    filter_id_A: Optional[str] = None,
    filter_id_B: Optional[str] = None,
    cover_hash_A: Optional[str] = None,
    cover_hash_B: Optional[str] = None,
    filter_hash_A: Optional[str] = None,
    filter_hash_B: Optional[str] = None,
    comparison_scope: str = "same_filter",
    cover_transport: Optional[Any] = None,
    status_A: Optional[str] = "success",
    status_B: Optional[str] = "success",
) -> Dict[str, Any]:
    """Compute the exact joint Definition-9 objective on common identities.

    The binary linear program has a permutation matrix for the ordinary labels
    in each cover element and one mismatch variable per observation. Those
    observation variables couple all cover elements and distinguish exact
    ``D_M`` from the per-bin upper estimate.
    """

    if _is_failed_status(status_A) or _is_failed_status(status_B):
        if _is_failed_status(status_A) and _is_failed_status(status_B):
            reason = f"invalid_output_A={status_A}_B={status_B}"
        elif _is_failed_status(status_A):
            reason = f"invalid_output_A={status_A}"
        else:
            reason = f"invalid_output_B={status_B}"
        return {"distance": None, "common_ids": None, "reason": reason, "status": "null"}

    if common_ids is None:
        return {
            "distance": None,
            "common_ids": None,
            "reason": "missing_explicit_common_identity_set",
            "status": "null",
        }
    common_ids = set(common_ids)
    if not common_ids:
        return {
            "distance": None,
            "common_ids": [],
            "common_digest": _hash_sorted_ids([]),
            "reason": "empty_common_identity_set",
            "status": "null",
        }

    for name, value in (
        ("cover_hash_A", cover_hash_A),
        ("cover_hash_B", cover_hash_B),
        ("filter_hash_A", filter_hash_A),
        ("filter_hash_B", filter_hash_B),
    ):
        bad = _require_digest(name, value)
        if bad is not None:
            return bad
    if not cover_id_A or not cover_id_B:
        return _rejected("missing_cover_identity")
    if not filter_id_A or not filter_id_B:
        return _rejected("missing_filter_identity")

    if comparison_scope != "same_filter":
        return _rejected("only_same_filter_common_cover_scope_supported")
    if cover_transport is not None:
        return _rejected("cover_transport_not_supported_by_exact_common_cover_metric")
    if filter_id_A != filter_id_B or filter_hash_A != filter_hash_B:
        return _rejected("filter_mismatch")
    if cover_id_A != cover_id_B or cover_hash_A != cover_hash_B:
        return _rejected("cover_mismatch")

    computed_cover_A = hash_realized_cover(bins_A, common_ids)
    computed_cover_B = hash_realized_cover(bins_B, common_ids)
    if cover_hash_A != computed_cover_A:
        return _rejected("cover_hash_A_does_not_bind_realized_cover")
    if cover_hash_B != computed_cover_B:
        return _rejected("cover_hash_B_does_not_bind_realized_cover")

    all_bins = sorted(set(bins_A) | set(bins_B), key=lambda value: str(value))
    ids = sorted(common_ids, key=lambda value: str(value))
    vectors_A: Dict[Any, tuple] = {}
    vectors_B: Dict[Any, tuple] = {}
    for x in ids:
        vectors_A[x] = tuple(
            _normalize(bins_A.get(bin_id, {}).get(x, OUTSIDE)) for bin_id in all_bins
        )
        vectors_B[x] = tuple(
            _normalize(bins_B.get(bin_id, {}).get(x, OUTSIDE)) for bin_id in all_bins
        )

    # Released constructions use scalar DBSCAN labels. Set-valued labels need a
    # different exact linearization and therefore fail closed.
    if any(
        isinstance(value, frozenset)
        for x in ids
        for value in vectors_A[x] + vectors_B[x]
    ):
        return {
            "distance": None,
            "common_ids": ids,
            "reason": "unsupported_set_valued_label_for_exact_optimizer",
            "status": "null",
        }

    assignment_indices = []
    next_var = 0
    for bin_index, _ in enumerate(all_bins):
        labels_A = sorted(
            {
                vectors_A[x][bin_index]
                for x in ids
                if not _is_special(vectors_A[x][bin_index])
            },
            key=lambda value: str(value),
        )
        labels_B = sorted(
            {
                vectors_B[x][bin_index]
                for x in ids
                if not _is_special(vectors_B[x][bin_index])
            },
            key=lambda value: str(value),
        )
        k = max(len(labels_A), len(labels_B))
        padded_A = labels_A + [
            ("__dummy_A__", bin_index, j) for j in range(k - len(labels_A))
        ]
        padded_B = labels_B + [
            ("__dummy_B__", bin_index, j) for j in range(k - len(labels_B))
        ]
        index = {}
        for label_A in padded_A:
            for label_B in padded_B:
                index[(label_A, label_B)] = next_var
                next_var += 1
        assignment_indices.append((padded_A, padded_B, index))

    y_index = {x: next_var + position for position, x in enumerate(ids)}
    n_variables = next_var + len(ids)

    if next_var == 0:
        mismatched = sum(vectors_A[x] != vectors_B[x] for x in ids)
        return {
            "distance": float(mismatched / len(ids)),
            "common_ids": ids,
            "common_digest": _hash_sorted_ids(ids),
            "cover_hash": cover_hash_A,
            "filter_hash": filter_hash_A,
            "n_common": len(ids),
            "mismatched": mismatched,
            "status": "success",
            "optimizer": "exact_joint_trivial",
        }

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception:
        return {
            "distance": None,
            "common_ids": ids,
            "reason": "resource_failure_exact_optimizer_dependencies_unavailable",
            "status": "null",
        }

    rows = []
    lower = []
    upper = []

    for padded_A, padded_B, index in assignment_indices:
        for label_A in padded_A:
            rows.append([(index[(label_A, label_B)], 1.0) for label_B in padded_B])
            lower.append(1.0)
            upper.append(1.0)
        for label_B in padded_B:
            rows.append([(index[(label_A, label_B)], 1.0) for label_A in padded_A])
            lower.append(1.0)
            upper.append(1.0)

    forced_mismatch = set()
    for x in ids:
        for bin_index, _ in enumerate(all_bins):
            value_A = vectors_A[x][bin_index]
            value_B = vectors_B[x][bin_index]
            if _is_special(value_A) or _is_special(value_B):
                if value_A != value_B:
                    forced_mismatch.add(x)
                continue
            _, _, index = assignment_indices[bin_index]
            # y_x + z_(a,b) >= 1.
            rows.append([(y_index[x], 1.0), (index[(value_A, value_B)], 1.0)])
            lower.append(1.0)
            upper.append(np.inf)
    for x in forced_mismatch:
        rows.append([(y_index[x], 1.0)])
        lower.append(1.0)
        upper.append(1.0)

    matrix = lil_matrix((len(rows), n_variables), dtype=float)
    for row_index, entries in enumerate(rows):
        for variable, coefficient in entries:
            matrix[row_index, variable] = coefficient

    objective = np.zeros(n_variables, dtype=float)
    for variable in y_index.values():
        objective[variable] = 1.0
    result = milp(
        c=objective,
        integrality=np.ones(n_variables, dtype=int),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=LinearConstraint(matrix.tocsr(), np.asarray(lower), np.asarray(upper)),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        return {
            "distance": None,
            "common_ids": ids,
            "reason": f"exact_joint_optimizer_failure:{result.message}",
            "status": "null",
        }

    y_values = np.rint(result.x[[y_index[x] for x in ids]]).astype(int)
    mismatched = int(y_values.sum())
    return {
        "distance": float(mismatched / len(ids)),
        "common_ids": ids,
        "common_digest": _hash_sorted_ids(ids),
        "cover_hash": cover_hash_A,
        "filter_hash": filter_hash_A,
        "n_common": len(ids),
        "mismatched": mismatched,
        "status": "success",
        "optimizer": "scipy_milp_exact_joint",
        "optimizer_objective": float(result.fun),
    }


def is_diagram_informative(points: Any):
    if not points:
        return False, "empty_diagram"
    off_diag = [point for point in points if abs(point[0] - point[1]) > 1e-12]
    if not off_diag:
        return False, "diagonal_only"
    return True, "informative"


def circle_representation_metadata():
    return {
        "canonical_representation_type": "multigraph_2v2e",
        "note": "compatibility helper retained for historical callers",
    }

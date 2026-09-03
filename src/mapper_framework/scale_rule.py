"""Deterministic Scale-Aware Clustering Radius Rule for Swiss Roll (M6-A2).

Implements the deterministic clean-sample geometric scale contract:
  epsilon_scale = Q_0.90({r_i}_{i=1}^N)
where r_i is the Euclidean distance from observation x_i to its SECOND-NEAREST
DISTINCT observation in clean X_0 (self-distance is excluded).

This rule is motivated by the local-neighbour requirement associated with
DBSCAN min_samples=3 and provides a deterministic, scale-equivariant clean-sample radius.
Pullback-level adequacy is verified descriptively during clean preflight.
"""

import hashlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.spatial.distance as dist


def compute_second_nearest_neighbor_distances(X: np.ndarray) -> np.ndarray:
    """Compute distance from each observation x_i in X to its second-nearest distinct observation.

    Parameters
    ----------
    X : np.ndarray
        Point cloud array of shape (N, d).

    Returns
    -------
    r : np.ndarray
        1-D array of shape (N,) where r[i] is the Euclidean distance from x_i to its
        second-nearest distinct neighbor in X. Self-distance (j=i) is strictly excluded.
    """
    X_arr = np.asarray(X, dtype=float)
    N = len(X_arr)
    if N < 3:
        raise ValueError(f"Point cloud must contain at least 3 points, got {N}")

    # Compute pairwise Euclidean distance matrix
    D = dist.cdist(X_arr, X_arr, metric="euclidean")

    # Set self-distance diagonal to infinity so self is never included
    np.fill_diagonal(D, np.inf)

    # Sort each row in ascending order
    D_sorted = np.sort(D, axis=1)

    # 2nd nearest distinct neighbor is index 1 (index 0 is 1st NN, index 1 is 2nd NN)
    r = D_sorted[:, 1]
    return r


def compute_scale_aware_dbscan_eps(
    X_clean: np.ndarray,
    quantile: float = 0.90,
    quantile_method: str = "linear",
) -> float:
    """Compute the deterministic scale-aware DBSCAN epsilon from clean reference X_0.

    Parameters
    ----------
    X_clean : np.ndarray
        Clean reference point cloud X_0 of shape (N, d).
    quantile : float
        Quantile level (frozen at 0.90 for M6-A2).
    quantile_method : str
        Quantile interpolation method (frozen at 'linear').

    Returns
    -------
    eps_scale : float
        Deterministic numeric epsilon value.
    """
    r = compute_second_nearest_neighbor_distances(X_clean)
    eps_scale = float(np.quantile(r, quantile, method=quantile_method))
    return eps_scale


def compute_scale_rule_diagnostics(
    X_clean: np.ndarray,
    quantile: float = 0.90,
    old_eps: float = 0.15,
) -> Dict[str, Any]:
    """Compute complete design-stage scale diagnostics on clean point cloud X_0."""
    X_arr = np.asarray(X_clean, dtype=float)
    N, d = X_arr.shape
    r = compute_second_nearest_neighbor_distances(X_arr)

    eps_m6a2 = float(np.quantile(r, quantile, method="linear"))
    hex_eps = float.hex(eps_m6a2)

    # Compute SHA-256 hashes for cryptographic provenance
    x_bytes = np.ascontiguousarray(X_arr).tobytes()
    x_hash = hashlib.sha256(x_bytes).hexdigest()

    r_sorted = np.sort(r)
    r_bytes = np.ascontiguousarray(r_sorted).tobytes()
    r_hash = hashlib.sha256(r_bytes).hexdigest()

    coord_min = [float(np.min(X_arr[:, j])) for j in range(d)]
    coord_max = [float(np.max(X_arr[:, j])) for j in range(d)]

    return {
        "N": N,
        "d": d,
        "coordinate_system": "raw_ambient",
        "distance_metric": "euclidean",
        "min_samples": 3,
        "statistic_definition": "distance to second-nearest distinct neighbor (self-distance excluded)",
        "scale_quantile": quantile,
        "quantile_method": "linear",
        "coordinate_minima": coord_min,
        "coordinate_maxima": coord_max,
        "min_r": float(np.min(r)),
        "Q_0.25_r": float(np.quantile(r, 0.25, method="linear")),
        "median_r": float(np.median(r)),
        "Q_0.75_r": float(np.quantile(r, 0.75, method="linear")),
        "Q_0.90_r": eps_m6a2,
        "Q_0.95_r": float(np.quantile(r, 0.95, method="linear")),
        "max_r": float(np.max(r)),
        "old_M6A_eps": old_eps,
        "new_M6A2_eps": eps_m6a2,
        "new_M6A2_eps_hex": hex_eps,
        "ratio_new_to_old": float(eps_m6a2 / old_eps),
        "clean_dataset_sha256": x_hash,
        "ordered_2nn_vector_sha256": r_hash,
        "numpy_version": np.__version__,
    }


def compute_pullback_scale_diagnostics(
    pullback_points_list: List[np.ndarray],
    frozen_eps: float,
) -> List[Dict[str, Any]]:
    """Compute descriptive pullback-level scale diagnostics for a list of pullback point clouds.

    These values are diagnostic only and must NOT trigger automatic epsilon adjustment.
    """
    diagnostics = []
    for idx, pts_pb in enumerate(pullback_points_list):
        n_pb = len(pts_pb)
        if n_pb < 3:
            diagnostics.append({
                "pullback_index": idx,
                "N_pullback": n_pb,
                "is_trivial": True,
                "median_2NN": None,
                "Q90_2NN": None,
                "ratio_frozen_eps_to_Q90": None,
            })
            continue

        D = dist.cdist(pts_pb, pts_pb, metric="euclidean")
        np.fill_diagonal(D, np.inf)
        D_sorted = np.sort(D, axis=1)
        r_pb = D_sorted[:, 1]
        med_r = float(np.median(r_pb))
        q90_r = float(np.quantile(r_pb, 0.90, method="linear"))
        ratio = float(frozen_eps / q90_r) if q90_r > 0 else None

        diagnostics.append({
            "pullback_index": idx,
            "N_pullback": n_pb,
            "is_trivial": False,
            "median_2NN": med_r,
            "Q90_2NN": q90_r,
            "ratio_frozen_eps_to_Q90": ratio,
        })
    return diagnostics

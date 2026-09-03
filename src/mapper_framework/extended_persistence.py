"""Exact Extended Persistence and Type-Restricted Diagram Distance Matching.

Conforms to the mathematical framework of Carrière, Michel, & Oudot (2018):
Extended persistence of a graph/scalar function pair decomposes into 4 distinct
subdiagrams by persistence type and homological dimension:
  - Ord_0: Ordinary 0-dimensional persistence (b <= d)
  - Rel_1: Relative 1-dimensional persistence (b >= d)
  - Ext_0+: Extended 0-dimensional persistence (b <= d)
  - Ext_1-: Extended 1-dimensional persistence (b >= d)

Matching between two extended persistence diagrams is strictly type- and dimension-restricted:
  d_B(D1, D2) = max_{k in {Ord_0, Rel_1, Ext_0+, Ext_1-}} d_B(D1[k], D2[k])
  W_1(D1, D2) = sum_{k in {Ord_0, Rel_1, Ext_0+, Ext_1-}} W_1(D1[k], D2[k])

Diagonal projection distance for any off-diagonal point p = (b, d) is strictly |d - b| / 2,
regardless of whether p is above or below the diagonal.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import gudhi
import gudhi.bottleneck
from scipy.optimize import linear_sum_assignment

from mapper_framework.types import MapperGraph


@dataclass(frozen=True)
class ExtendedPersistenceDiagram:
    """Exact 4-subdiagram Extended Persistence Diagram representation."""

    Ord_0: Tuple[Tuple[float, float], ...] = ()
    Rel_1: Tuple[Tuple[float, float], ...] = ()
    Ext_0_plus: Tuple[Tuple[float, float], ...] = ()
    Ext_1_minus: Tuple[Tuple[float, float], ...] = ()

    @property
    def all_points(self) -> List[Tuple[float, float]]:
        """Flattened list of all finite off-diagonal persistence pairs across subdiagrams."""
        return list(self.Ord_0) + list(self.Rel_1) + list(self.Ext_0_plus) + list(self.Ext_1_minus)

    @property
    def total_points(self) -> int:
        return len(self.Ord_0) + len(self.Rel_1) + len(self.Ext_0_plus) + len(self.Ext_1_minus)

    def to_dict(self) -> Dict[str, List[Tuple[float, float]]]:
        return {
            "Ord_0": list(self.Ord_0),
            "Rel_1": list(self.Rel_1),
            "Ext_0+": list(self.Ext_0_plus),
            "Ext_1-": list(self.Ext_1_minus),
        }

    def subdiagram_counts(self) -> Dict[str, int]:
        return {
            "Ord_0": len(self.Ord_0),
            "Rel_1": len(self.Rel_1),
            "Ext_0+": len(self.Ext_0_plus),
            "Ext_1-": len(self.Ext_1_minus),
        }


def compute_graph_extended_persistence(
    graph: MapperGraph,
    node_values: Dict[int, float],
) -> ExtendedPersistenceDiagram:
    """Compute exact extended persistence diagram for a 1D graph with node scalar values.

    Parameters
    ----------
    graph : MapperGraph
        Input 1-skeleton graph.
    node_values : Dict[int, float]
        Scalar filter values assigned to each node. Every node in graph.nodes MUST have an entry.

    Returns
    -------
    ext_diagram : ExtendedPersistenceDiagram
        Structured extended persistence diagram with 4 subdiagrams.
    """
    if len(graph.nodes) == 0:
        return ExtendedPersistenceDiagram()

    st = gudhi.SimplexTree()

    # 1. Insert 0-simplices (vertices) - strict check, no silent default to 0.0
    for nid in graph.nodes:
        if nid not in node_values:
            raise ValueError(f"Missing scalar filter value for graph node {nid}.")
        val = float(node_values[nid])
        if not np.isfinite(val):
            raise ValueError(f"Non-finite filter value ({val}) for graph node {nid}.")
        st.insert([nid], filtration=val)

    # 2. Insert 1-simplices (edges) with lower-star filtration
    for u, v in graph.edges:
        if u not in node_values or v not in node_values:
            raise ValueError(f"Missing scalar filter value for edge ({u}, {v}) endpoints.")
        val_u = float(node_values[u])
        val_v = float(node_values[v])
        edge_val = max(val_u, val_v)
        st.insert([u, v], filtration=edge_val)

    # 3. Compute extended persistence via GUDHI
    st.extend_filtration()
    raw_ext_dgm = st.extended_persistence()

    # GUDHI returns 4 subdiagram lists: [Ord_0, Rel_1, Ext_0+, Ext_1-]
    # Parse each subdiagram filtering infinite and diagonal points
    sub_lists: List[List[Tuple[float, float]]] = [[], [], [], []]
    for sub_idx, sub in enumerate(raw_ext_dgm):
        for dim, (b, d) in sub:
            if not np.isinf(b) and not np.isinf(d) and b != d:
                sub_lists[sub_idx].append((float(b), float(d)))

    return ExtendedPersistenceDiagram(
        Ord_0=tuple(sub_lists[0]),
        Rel_1=tuple(sub_lists[1]),
        Ext_0_plus=tuple(sub_lists[2]),
        Ext_1_minus=tuple(sub_lists[3]),
    )


def compute_subdiagram_bottleneck_distance(
    dgm1: Sequence[Tuple[float, float]],
    dgm2: Sequence[Tuple[float, float]],
) -> float:
    """Compute exact Bottleneck distance between two 2D persistence subdiagrams.

    Correctly accounts for diagonal distance |d - b| / 2 for points above and below diagonal.
    """
    pts1 = np.asarray(dgm1, dtype=float).reshape(-1, 2) if len(dgm1) > 0 else np.empty((0, 2))
    pts2 = np.asarray(dgm2, dtype=float).reshape(-1, 2) if len(dgm2) > 0 else np.empty((0, 2))

    n1 = len(pts1)
    n2 = len(pts2)

    if n1 == 0 and n2 == 0:
        return 0.0
    if n1 == 0:
        return float(np.max(np.abs(pts2[:, 1] - pts2[:, 0]) / 2.0))
    if n2 == 0:
        return float(np.max(np.abs(pts1[:, 1] - pts1[:, 0]) / 2.0))

    # Orient to canonical upper half-plane (min(b, d), max(b, d)) so GUDHI's C++ solver
    # correctly evaluates diagonal distances for lower-half-plane points
    c1 = np.column_stack([np.minimum(pts1[:, 0], pts1[:, 1]), np.maximum(pts1[:, 0], pts1[:, 1])])
    c2 = np.column_stack([np.minimum(pts2[:, 0], pts2[:, 1]), np.maximum(pts2[:, 0], pts2[:, 1])])

    d = float(gudhi.bottleneck.bottleneck_distance(c1, c2))
    return 0.0 if abs(d) < 1e-14 else d


def compute_subdiagram_wasserstein_distance(
    dgm1: Sequence[Tuple[float, float]],
    dgm2: Sequence[Tuple[float, float]],
    p: int = 1,
) -> float:
    """Compute exact p-Wasserstein distance between two 2D persistence subdiagrams.

    Uses Hungarian matching with exact L_infinity point matching cost and
    diagonal projection cost (|d - b| / 2)^p.
    """
    if p < 1:
        raise ValueError(f"Wasserstein order p must be >= 1, got {p}")

    pts1 = np.asarray(dgm1, dtype=float).reshape(-1, 2) if len(dgm1) > 0 else np.empty((0, 2))
    pts2 = np.asarray(dgm2, dtype=float).reshape(-1, 2) if len(dgm2) > 0 else np.empty((0, 2))

    n = len(pts1)
    m = len(pts2)

    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        diag_costs = [(abs(pts2[j, 1] - pts2[j, 0]) / 2.0) ** p for j in range(m)]
        return float((sum(diag_costs)) ** (1.0 / p))
    if m == 0:
        diag_costs = [(abs(pts1[i, 1] - pts1[i, 0]) / 2.0) ** p for i in range(n)]
        return float((sum(diag_costs)) ** (1.0 / p))

    dim = n + m
    LARGE_COST = 1e12
    C = np.full((dim, dim), LARGE_COST, dtype=float)

    # 1. Off-diagonal point matching cost: ||p - q||_infinity^p
    for i in range(n):
        for j in range(m):
            C[i, j] = max(abs(pts1[i, 0] - pts2[j, 0]), abs(pts1[i, 1] - pts2[j, 1])) ** p

    # 2. Diagonal projection cost for pts1: (|death - birth| / 2)^p
    for i in range(n):
        diag_dist = abs(pts1[i, 1] - pts1[i, 0]) / 2.0
        C[i, m + i] = diag_dist ** p

    # 3. Diagonal projection cost for pts2: (|death - birth| / 2)^p
    for j in range(m):
        diag_dist = abs(pts2[j, 1] - pts2[j, 0]) / 2.0
        C[n + j, j] = diag_dist ** p

    # 4. Diagonal-to-diagonal zero cost matching
    for j in range(m):
        for i in range(n):
            C[n + j, m + i] = 0.0

    row_ind, col_ind = linear_sum_assignment(C)
    total_cost = float(C[row_ind, col_ind].sum())
    return float(total_cost ** (1.0 / p))


def compute_extended_diagram_bottleneck_distance(
    dgm1: ExtendedPersistenceDiagram,
    dgm2: ExtendedPersistenceDiagram,
) -> float:
    """Compute exact type-restricted Bottleneck distance between two ExtendedPersistenceDiagrams.

    d_B(D1, D2) = max_{k in {Ord_0, Rel_1, Ext_0+, Ext_1-}} d_B(D1[k], D2[k])
    """
    d_ord0 = compute_subdiagram_bottleneck_distance(dgm1.Ord_0, dgm2.Ord_0)
    d_rel1 = compute_subdiagram_bottleneck_distance(dgm1.Rel_1, dgm2.Rel_1)
    d_ext0 = compute_subdiagram_bottleneck_distance(dgm1.Ext_0_plus, dgm2.Ext_0_plus)
    d_ext1 = compute_subdiagram_bottleneck_distance(dgm1.Ext_1_minus, dgm2.Ext_1_minus)

    return float(max(d_ord0, d_rel1, d_ext0, d_ext1))


def compute_extended_diagram_wasserstein_distance(
    dgm1: ExtendedPersistenceDiagram,
    dgm2: ExtendedPersistenceDiagram,
    p: int = 1,
) -> float:
    """Compute exact type-restricted p-Wasserstein distance between two ExtendedPersistenceDiagrams.

    For p=1: W_1(D1, D2) = sum_{k in {Ord_0, Rel_1, Ext_0+, Ext_1-}} W_1(D1[k], D2[k])
    For general p: W_p(D1, D2) = (sum_k W_p(D1[k], D2[k])^p)^(1/p)
    """
    w_ord0 = compute_subdiagram_wasserstein_distance(dgm1.Ord_0, dgm2.Ord_0, p=p)
    w_rel1 = compute_subdiagram_wasserstein_distance(dgm1.Rel_1, dgm2.Rel_1, p=p)
    w_ext0 = compute_subdiagram_wasserstein_distance(dgm1.Ext_0_plus, dgm2.Ext_0_plus, p=p)
    w_ext1 = compute_subdiagram_wasserstein_distance(dgm1.Ext_1_minus, dgm2.Ext_1_minus, p=p)

    total_p = (w_ord0 ** p) + (w_rel1 ** p) + (w_ext0 ** p) + (w_ext1 ** p)
    return float(total_p ** (1.0 / p))


# Compatibility aliases for legacy interfaces
def compute_diagram_bottleneck_distance(
    dgm1: Union[ExtendedPersistenceDiagram, Sequence[Tuple[float, float]]],
    dgm2: Union[ExtendedPersistenceDiagram, Sequence[Tuple[float, float]]],
) -> float:
    """Unified Bottleneck distance supporting both ExtendedPersistenceDiagram and single subdiagram sequences."""
    if isinstance(dgm1, ExtendedPersistenceDiagram) and isinstance(dgm2, ExtendedPersistenceDiagram):
        return compute_extended_diagram_bottleneck_distance(dgm1, dgm2)
    elif isinstance(dgm1, ExtendedPersistenceDiagram) or isinstance(dgm2, ExtendedPersistenceDiagram):
        raise TypeError("Both diagrams must be ExtendedPersistenceDiagram or both sequences of (b, d) tuples.")
    return compute_subdiagram_bottleneck_distance(dgm1, dgm2)


def compute_diagram_wasserstein_distance(
    dgm1: Union[ExtendedPersistenceDiagram, Sequence[Tuple[float, float]]],
    dgm2: Union[ExtendedPersistenceDiagram, Sequence[Tuple[float, float]]],
    p: int = 1,
) -> float:
    """Unified Wasserstein distance supporting both ExtendedPersistenceDiagram and single subdiagram sequences."""
    if isinstance(dgm1, ExtendedPersistenceDiagram) and isinstance(dgm2, ExtendedPersistenceDiagram):
        return compute_extended_diagram_wasserstein_distance(dgm1, dgm2, p=p)
    elif isinstance(dgm1, ExtendedPersistenceDiagram) or isinstance(dgm2, ExtendedPersistenceDiagram):
        raise TypeError("Both diagrams must be ExtendedPersistenceDiagram or both sequences of (b, d) tuples.")
    return compute_subdiagram_wasserstein_distance(dgm1, dgm2, p=p)

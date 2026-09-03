"""Membership-defined simplicial nerve 2-skeleton construction."""

from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import itertools

from mapper_framework.exceptions import ConfigurationInvalidError, ResourceLimitError
from mapper_framework.types import SimplicialNerve2D


def build_membership_nerve_2d(
    node_supports: Dict[int, FrozenSet[int]],
    max_nodes: Optional[int] = None,
    max_edges: Optional[int] = None,
    max_triangles: Optional[int] = None,
) -> SimplicialNerve2D:
    """Construct the membership-defined simplicial nerve 2-skeleton.

    Parameters
    ----------
    node_supports : Dict[int, FrozenSet[int]]
        Mapping of external integer node IDs to their non-empty observed sample sets.
    max_nodes, max_edges, max_triangles : Optional[int]
        Provisional resource guardrails. If exceeded, raises ResourceLimitError.

    Returns
    -------
    SimplicialNerve2D
        The verified 2-skeleton with simplices Sigma_0, Sigma_1, Sigma_2.
    """
    # 1. Validate external node identifier domain (Integers only in Phase 0)
    for node_id, support in node_supports.items():
        if not isinstance(node_id, (int,)):
            raise ValueError(f"External node identifier must be an integer in Phase 0, got {type(node_id)}: {node_id}")
        if len(support) == 0:
            raise ValueError(f"Node with empty support provided to nerve construction: node_id={node_id}")

    sorted_nodes = sorted(node_supports.keys())
    n_0 = len(sorted_nodes)

    if max_nodes is not None and n_0 > max_nodes:
        raise ResourceLimitError(f"Node count {n_0} exceeds limit {max_nodes}")

    if n_0 == 0:
        return SimplicialNerve2D(
            nodes=[],
            sigma_0=[],
            sigma_1=[],
            sigma_2=[],
            n_0=0,
            n_1=0,
            n_2=0,
            vertex_to_row={}
        )

    # Deterministic internal mapping
    vertex_to_row: Dict[int, int] = {node_id: i for i, node_id in enumerate(sorted_nodes)}

    # 2. Build sample-to-node inverted index
    sample_to_nodes: Dict[int, Set[int]] = {}
    for node_id, support in node_supports.items():
        row = vertex_to_row[node_id]
        for sample_idx in support:
            if sample_idx not in sample_to_nodes:
                sample_to_nodes[sample_idx] = set()
            sample_to_nodes[sample_idx].add(row)

    # 3. 0-simplices
    sigma_0: List[Tuple[int]] = [(i,) for i in range(n_0)]

    # 4. 1-simplices and 2-simplices via sample incidence
    edge_set: Set[Tuple[int, int]] = set()
    triangle_set: Set[Tuple[int, int, int]] = set()

    for sample_idx, inc_nodes in sample_to_nodes.items():
        if len(inc_nodes) >= 2:
            # All pairs in inc_nodes
            for u, v in itertools.combinations(sorted(inc_nodes), 2):
                edge_set.add((u, v))
                if max_edges is not None and len(edge_set) > max_edges:
                    raise ResourceLimitError(f"Edge count exceeded limit {max_edges}")

        if len(inc_nodes) >= 3:
            # All triples in inc_nodes
            for u, v, w in itertools.combinations(sorted(inc_nodes), 3):
                triangle_set.add((u, v, w))
                if max_triangles is not None and len(triangle_set) > max_triangles:
                    raise ResourceLimitError(f"Triangle count exceeded limit {max_triangles}")

    sigma_1: List[Tuple[int, int]] = sorted(edge_set)
    sigma_2: List[Tuple[int, int, int]] = sorted(triangle_set)

    return SimplicialNerve2D(
        nodes=sorted_nodes,
        sigma_0=sigma_0,
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        n_0=n_0,
        n_1=len(sigma_1),
        n_2=len(sigma_2),
        vertex_to_row=vertex_to_row
    )

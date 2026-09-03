"""Continuous and Exact-PL Reeb Reference Constructors for Topological Fidelity."""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
import math
import numpy as np

from mapper_framework.exceptions import ConfigurationInvalidError
from mapper_framework.types import MapperGraph, MapperNode, MapperOutput


@dataclass(frozen=True)
class ReebReference:
    """Immutable representation of a continuous or exact PL Reeb reference graph."""

    name: str
    reference_type: str  # 'exact_continuous' | 'exact_PL' | 'numerical_reference' | 'mapper_proxy_sanity_check'
    space_name: str
    lens_id: str
    graph: MapperGraph
    node_values: Dict[int, float]
    critical_values: List[float]
    invariants: Dict[str, Any]
    is_ground_truth: bool
    description: str
    filter_definition_hash: Optional[str] = None


# Canonical filter hashes
CIRCLE_HEIGHT_Y_HASH = "d784b8d86fb471649c90278087fec52270cf927c76bf862758e183e20f42b78d"
TRIPOD_HEIGHT_Y_HASH = "685f2da6e8571fb12140d8971dfbd5d3ef26b072a38a9c15a9cd1c68a955c3f4"


def compute_canonical_reeb_hash(ref: ReebReference, spec_version: str = "1.1.0") -> str:
    """Compute deterministic SHA-256 hash of a ReebReference canonical scientific representation."""
    import hashlib
    import json

    data = {
        "reference_object_id": ref.name,
        "reference_type": ref.reference_type,
        "domain_identity": ref.space_name,
        "lens_id": ref.lens_id,
        "filter_definition_hash": ref.filter_definition_hash or "",
        "node_values": {str(k): v for k, v in sorted(ref.node_values.items())},
        "edges": sorted([list(e) for e in ref.graph.edges]),
        "critical_values": sorted(ref.critical_values),
        "invariants": {k: v for k, v in sorted(ref.invariants.items())},
        "reference_spec_version": spec_version,
    }
    s = json.dumps(data, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_circle_analytic_reeb(
    radius: float = 1.0,
    lens_id: str = "circle_height_y",
) -> ReebReference:
    """Construct the authoritative analytic Reeb graph for Unit Circle S^1 with height lens f(x, y) = y.

    Reeb Graph Structure:
    - 4 nodes forming a 1-skeleton simple cycle:
        v0: (0, -1) [f = -radius, min]
        v1: (1, 0)  [f = 0.0, intermediate]
        v2: (0, 1)  [f = radius, max]
        v3: (-1, 0) [f = 0.0, intermediate]
    - 4 edges: (0, 1), (1, 2), (2, 3), (0, 3).
    - beta_0 = 1, beta_1 = 1, leaves = 0, branch vertices = 0, degree sequence = (2, 2, 2, 2).
    """
    nodes_4 = {
        0: MapperNode(node_id=0, interval_idx=0, cluster_label=0, members=frozenset(), size=0, mean_filter=-radius),
        1: MapperNode(node_id=1, interval_idx=1, cluster_label=0, members=frozenset(), size=0, mean_filter=0.0),
        2: MapperNode(node_id=2, interval_idx=2, cluster_label=0, members=frozenset(), size=0, mean_filter=radius),
        3: MapperNode(node_id=3, interval_idx=1, cluster_label=1, members=frozenset(), size=0, mean_filter=0.0),
    }
    edges_4 = [(0, 1), (1, 2), (2, 3), (0, 3)]
    edge_weights_4 = {e: 1 for e in edges_4}
    graph_4 = MapperGraph(
        nodes=nodes_4,
        edges=edges_4,
        edge_weights=edge_weights_4,
        n_nodes=4,
        n_edges=4,
        n_components=1,
        beta_1_graph=1,
    )
    node_values = {0: -radius, 1: 0.0, 2: radius, 3: 0.0}
    critical_values = [-radius, radius]

    invariants = {
        "n_nodes": 4,
        "n_edges": 4,
        "n_components": 1,
        "beta_0": 1,
        "beta_1": 1,
        "n_leaves": 0,
        "n_branch_vertices": 0,
        "degree_sequence": [2, 2, 2, 2],
        "critical_points_count": 2,
    }

    # Normalize lens_id if legacy 'height' passed
    eff_lens = "circle_height_y" if lens_id in ("height", "circle_height_y") else lens_id
    f_hash = CIRCLE_HEIGHT_Y_HASH if eff_lens == "circle_height_y" else None

    return ReebReference(
        name="circle_analytic_reeb",
        reference_type="exact_continuous",
        space_name="unit_circle_S1",
        lens_id=eff_lens,
        graph=graph_4,
        node_values=node_values,
        critical_values=critical_values,
        invariants=invariants,
        is_ground_truth=True,
        description="Authoritative analytic Reeb graph for Unit Circle S^1 with height lens f(x, y)=y.",
        filter_definition_hash=f_hash,
    )


def build_tripod_exact_pl_reeb(
    lens_id: str = "tripod_height_y",
) -> ReebReference:
    """Construct the exact piecewise-linear Reeb graph for Branching Y-Graph (Tripod) with height lens f(x, y) = y.

    Structure:
    - 4 vertices:
        v0: (0, -1) [f = -1.0, leaf, deg 1]
        v1: (0, 0)  [f =  0.0, branch vertex, deg 3]
        v2: (-1, 1) [f =  1.0, leaf, deg 1]
        v3: (1, 1)  [f =  1.0, leaf, deg 1]
    - 3 edges: (0, 1), (1, 2), (1, 3).
    - beta_0 = 1, beta_1 = 0, leaves = 3, branch vertices = 1, degree sequence = (1, 3, 1, 1).
    """
    nodes = {
        0: MapperNode(node_id=0, interval_idx=0, cluster_label=0, members=frozenset(), size=0, mean_filter=-1.0),
        1: MapperNode(node_id=1, interval_idx=1, cluster_label=0, members=frozenset(), size=0, mean_filter=0.0),
        2: MapperNode(node_id=2, interval_idx=2, cluster_label=0, members=frozenset(), size=0, mean_filter=1.0),
        3: MapperNode(node_id=3, interval_idx=2, cluster_label=1, members=frozenset(), size=0, mean_filter=1.0),
    }
    edges = [(0, 1), (1, 2), (1, 3)]
    edge_weights = {e: 1 for e in edges}
    graph = MapperGraph(
        nodes=nodes,
        edges=edges,
        edge_weights=edge_weights,
        n_nodes=4,
        n_edges=3,
        n_components=1,
        beta_1_graph=0,
    )
    node_values = {0: -1.0, 1: 0.0, 2: 1.0, 3: 1.0}
    critical_values = [-1.0, 0.0, 1.0]

    invariants = {
        "n_nodes": 4,
        "n_edges": 3,
        "n_components": 1,
        "beta_0": 1,
        "beta_1": 0,
        "n_leaves": 3,
        "n_branch_vertices": 1,
        "degree_sequence": [1, 3, 1, 1],
        "critical_points_count": 4,
    }

    eff_lens = "tripod_height_y" if lens_id in ("height", "tripod_height_y") else lens_id
    f_hash = TRIPOD_HEIGHT_Y_HASH if eff_lens == "tripod_height_y" else None

    return ReebReference(
        name="tripod_exact_pl_reeb",
        reference_type="exact_PL",
        space_name="branching_tripod_Y",
        lens_id=eff_lens,
        graph=graph,
        node_values=node_values,
        critical_values=critical_values,
        invariants=invariants,
        is_ground_truth=True,
        description="Exact piecewise-linear Reeb graph for Tripod branching space with height lens f(x, y)=y.",
        filter_definition_hash=f_hash,
    )


def _dedupe_angles(thetas: List[float], tol: float) -> List[float]:
    """Sort ascending and merge angles within tol (exact-transition degenerate coincidences)."""
    if not thetas:
        return []
    s = sorted(float(t) % (2.0 * np.pi) for t in thetas)
    out: List[float] = [s[0]]
    for th in s[1:]:
        if abs(th - out[-1]) > tol:
            out.append(th)
    # Wrap check: first and last within tol of each other (mod 2*pi) means they're the same point
    if len(out) > 1 and (2.0 * np.pi - abs(out[-1] - out[0])) <= tol:
        out.pop()
    return out


def _circle_perturbed_critical_thetas(direction_id: str, delta: float, tol: float = 1e-9) -> List[float]:
    """Exact analytic critical theta values for f_delta = sin(theta) + delta*h_r(theta)."""
    if delta < 0.0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    if delta == 0.0:
        return [np.pi / 2.0, 3.0 * np.pi / 2.0]

    if direction_id == "harm_sin2":
        # f' = cos(theta) + 2*delta*cos(2*theta) = 0  =>  4*delta*u^2 + u - 2*delta = 0, u = cos(theta)
        disc = 1.0 + 32.0 * delta ** 2
        sqrt_disc = np.sqrt(disc)
        u_plus = (-1.0 + sqrt_disc) / (8.0 * delta)
        u_plus_c = min(1.0, max(-1.0, u_plus))
        theta1 = float(np.arccos(u_plus_c))
        theta2 = (2.0 * np.pi - theta1) % (2.0 * np.pi)
        thetas = [theta1, theta2]

        u_minus = (-1.0 - sqrt_disc) / (8.0 * delta)
        if u_minus >= -1.0 - tol:
            u_minus_c = min(1.0, max(-1.0, u_minus))
            theta3 = float(np.arccos(u_minus_c))
            theta4 = (2.0 * np.pi - theta3) % (2.0 * np.pi)
            thetas += [theta3, theta4]
        return _dedupe_angles(thetas, tol)

    elif direction_id == "harm_cos2":
        # f' = cos(theta)*(1 - 4*delta*sin(theta)) = 0
        thetas = [np.pi / 2.0, 3.0 * np.pi / 2.0]
        s = 1.0 / (4.0 * delta)
        if s <= 1.0 + tol:
            s_c = min(1.0, s)
            theta3 = float(np.arcsin(s_c))
            theta4 = np.pi - theta3
            thetas += [theta3, theta4]
        return _dedupe_angles(thetas, tol)

    raise ValueError(f"Unknown Circle perturbation direction_id '{direction_id}'.")


def build_circle_perturbed_analytic_reeb(
    delta: float,
    direction_id: str,
    radius: float = 1.0,
    lens_id: Optional[str] = None,
    filter_definition_hash: Optional[str] = None,
) -> ReebReference:
    """Construct the exact continuous perturbed Reeb graph for f_delta = sin(theta) + delta*h_r(theta).

    The Reeb graph of any Morse function on S^1 is homeomorphic to a simple cycle: vertices are
    the cyclically-ordered critical points; regular degree-2 subdivision vertices are inserted
    only when exactly two critical points would otherwise require a duplicate (parallel) edge
    pair. Preserves beta_0=1, beta_1=1, leaves=0, branch=0 for every delta >= 0.
    """
    if radius != 1.0:
        raise ValueError("Only unit radius (1.0) is supported for perturbed Circle references.")
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError(f"delta must be finite and >= 0, got {delta}")

    if direction_id == "baseline_none":
        if delta != 0.0:
            raise ValueError("direction_id='baseline_none' requires delta=0.0")
        # Delegate to the unchanged, byte-identical baseline constructor rather than reproducing
        # it via the delta=0 perturbed-formula path (which would carry different name/hash metadata).
        return build_circle_analytic_reeb(radius=radius, lens_id=(lens_id if lens_id is not None else "circle_height_y"))
    if direction_id not in ("harm_sin2", "harm_cos2"):
        raise ValueError(f"Unknown Circle perturbation direction_id '{direction_id}'.")

    critical_thetas = _circle_perturbed_critical_thetas(direction_id, delta)
    n_critical = len(critical_thetas)

    if n_critical == 2:
        a, b = critical_thetas
        mid1 = ((a + b) / 2.0) if b > a else ((a + b + 2.0 * np.pi) / 2.0) % (2.0 * np.pi)
        mid2 = (mid1 + np.pi) % (2.0 * np.pi)
        all_thetas = _dedupe_angles([a, b, mid1, mid2], tol=1e-9)
        subdivision_set = {mid1, mid2}
    else:
        all_thetas = sorted(critical_thetas)
        subdivision_set = set()

    n_nodes = len(all_thetas)
    if n_nodes < 2:
        raise ValueError(f"Degenerate perturbed Circle critical structure: only {n_nodes} distinct node(s).")

    def f_delta(theta: float) -> float:
        h = np.sin(2.0 * theta) if direction_id == "harm_sin2" else np.cos(2.0 * theta)
        return float(np.sin(theta) + delta * h)

    nodes: Dict[int, MapperNode] = {}
    node_values: Dict[int, float] = {}
    is_subdivision: Dict[int, bool] = {}
    for i, th in enumerate(all_thetas):
        val = f_delta(th)
        nodes[i] = MapperNode(node_id=i, interval_idx=i, cluster_label=0, members=frozenset(), size=0, mean_filter=val)
        node_values[i] = val
        is_subdivision[i] = any(abs(th - s) < 1e-9 for s in subdivision_set)

    edges: List[Tuple[int, int]] = []
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        e = (min(i, j), max(i, j))
        edges.append(e)
    edge_weights = {e: 1 for e in edges}

    graph = MapperGraph(
        nodes=nodes,
        edges=edges,
        edge_weights=edge_weights,
        n_nodes=n_nodes,
        n_edges=n_nodes,
        n_components=1,
        beta_1_graph=1,
    )

    critical_values = sorted(node_values[i] for i in range(n_nodes) if not is_subdivision[i])
    n_critical_actual = len(critical_values)

    invariants = {
        "n_nodes": n_nodes,
        "n_edges": n_nodes,
        "n_components": 1,
        "beta_0": 1,
        "beta_1": 1,
        "n_leaves": 0,
        "n_branch_vertices": 0,
        "degree_sequence": [2] * n_nodes,
        "critical_points_count": n_critical_actual,
        "delta": delta,
        "direction_id": direction_id,
    }

    eff_lens = lens_id if lens_id is not None else f"circle_height_y__{direction_id}__d{int(round(delta * 100)):03d}" if delta > 0 else "circle_height_y"

    return ReebReference(
        name=f"circle_perturbed_analytic_reeb__{direction_id}__delta{delta}",
        reference_type="exact_continuous",
        space_name="unit_circle_S1",
        lens_id=eff_lens,
        graph=graph,
        node_values=node_values,
        critical_values=critical_values,
        invariants=invariants,
        is_ground_truth=True,
        description=f"Exact analytic perturbed Reeb graph for Unit Circle S^1, direction={direction_id}, delta={delta}.",
        filter_definition_hash=filter_definition_hash,
    )


def _tripod_perturbed_node_values(direction_id: str, delta: float) -> Dict[str, float]:
    if delta < 0.0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    if direction_id == "tripod_hat_junction":
        return {"v0": -1.0, "O": delta, "v2": 1.0, "v3": 1.0}
    elif direction_id == "tripod_hat_lower_leaf":
        return {"v0": -1.0 + delta, "O": 0.0, "v2": 1.0, "v3": 1.0}
    raise ValueError(f"Unknown Tripod perturbation direction_id '{direction_id}'.")


def build_tripod_perturbed_pl_reeb(
    delta: float,
    direction_id: str,
    lens_id: Optional[str] = None,
    filter_definition_hash: Optional[str] = None,
    flat_tol: float = 1e-12,
) -> ReebReference:
    """Construct the exact PL perturbed Reeb graph on the sealed 4-vertex Tripod complex.

    Builds the 4-vertex, 3-edge tree with node values from the exact PL perturbation formula,
    then contracts any edge whose endpoints carry equal values (a flat PL edge collapses its
    level-set connected component to a single Reeb-quotient point). For every delta != the
    exact flat-edge transition this is a no-op and the result is the standard 3-leaf, 1-branch
    tree; at the exact transition it yields the correct collapsed quotient.
    """
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError(f"delta must be finite and >= 0, got {delta}")

    if direction_id == "baseline_none":
        if delta != 0.0:
            raise ValueError("direction_id='baseline_none' requires delta=0.0")
        # Delegate to the unchanged, byte-identical baseline constructor rather than reproducing
        # it via the delta=0 perturbed-formula path (which would carry different name/hash metadata).
        return build_tripod_exact_pl_reeb(lens_id=(lens_id if lens_id is not None else "tripod_height_y"))
    if direction_id not in ("tripod_hat_junction", "tripod_hat_lower_leaf"):
        raise ValueError(f"Unknown Tripod perturbation direction_id '{direction_id}'.")

    base_ids = ["v0", "O", "v2", "v3"]
    base_edges = [("v0", "O"), ("O", "v2"), ("O", "v3")]
    values = _tripod_perturbed_node_values(direction_id, delta)

    # Union-Find contraction of flat (equal-valued) adjacent edges.
    parent: Dict[str, str] = {vid: vid for vid in base_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in base_edges:
        if abs(values[u] - values[v]) <= flat_tol:
            union(u, v)

    groups: Dict[str, List[str]] = {}
    for vid in base_ids:
        groups.setdefault(find(vid), []).append(vid)

    group_reps = sorted(groups.keys(), key=lambda r: base_ids.index(r))
    rep_to_node_id = {rep: i for i, rep in enumerate(group_reps)}

    node_values: Dict[int, float] = {}
    for rep in group_reps:
        members = groups[rep]
        vals = [values[m] for m in members]
        assert max(vals) - min(vals) <= flat_tol, "Contracted group must share a common value."
        node_values[rep_to_node_id[rep]] = float(np.mean(vals))

    contracted_edges_set = set()
    for u, v in base_edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            a, b = rep_to_node_id[ru], rep_to_node_id[rv]
            contracted_edges_set.add((min(a, b), max(a, b)))
    edges = sorted(contracted_edges_set)

    n_nodes = len(group_reps)
    n_edges = len(edges)

    degs = {i: 0 for i in range(n_nodes)}
    for u, v in edges:
        degs[u] += 1
        degs[v] += 1
    leaves = sum(1 for d in degs.values() if d == 1)
    branch = sum(1 for d in degs.values() if d >= 3)
    deg_seq = sorted(degs.values())

    n_components = 1  # contraction of a connected tree remains connected
    beta_1 = n_edges - n_nodes + n_components

    nodes: Dict[int, MapperNode] = {}
    for i in range(n_nodes):
        nodes[i] = MapperNode(node_id=i, interval_idx=i, cluster_label=0, members=frozenset(), size=0, mean_filter=node_values[i])

    edge_weights = {e: 1 for e in edges}
    graph = MapperGraph(
        nodes=nodes,
        edges=edges,
        edge_weights=edge_weights,
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_components=n_components,
        beta_1_graph=beta_1,
    )

    critical_values = sorted(node_values.values())
    is_collapsed = n_nodes < 4

    invariants = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_components": n_components,
        "beta_0": n_components,
        "beta_1": beta_1,
        "n_leaves": leaves,
        "n_branch_vertices": branch,
        "degree_sequence": deg_seq,
        "critical_points_count": n_nodes,
        "delta": delta,
        "direction_id": direction_id,
        "is_flat_edge_collapsed": is_collapsed,
    }

    eff_lens = lens_id if lens_id is not None else f"tripod_height_y__{direction_id}__d{int(round(delta * 100)):03d}" if delta > 0 else "tripod_height_y"

    return ReebReference(
        name=f"tripod_perturbed_pl_reeb__{direction_id}__delta{delta}",
        reference_type="exact_PL",
        space_name="branching_tripod_Y",
        lens_id=eff_lens,
        graph=graph,
        node_values=node_values,
        critical_values=critical_values,
        invariants=invariants,
        is_ground_truth=True,
        description=f"Exact PL perturbed Reeb graph for Tripod branching space, direction={direction_id}, delta={delta}.",
        filter_definition_hash=filter_definition_hash,
    )


def build_mapper_proxy_reeb(
    space_name: str,
    lens_id: str,
    mapper_output: MapperOutput,
) -> ReebReference:
    """Construct a high-resolution Mapper proxy sanity check (NEVER ground-truth)."""
    if mapper_output.graph is None:
        raise ValueError("Cannot construct mapper proxy Reeb from failed MapperOutput.")

    g = mapper_output.graph
    node_values = {nid: node.mean_filter for nid, node in g.nodes.items()}

    # Calculate degrees
    degs = {nid: 0 for nid in g.nodes}
    for u, v in g.edges:
        degs[u] += 1
        degs[v] += 1

    deg_seq = sorted(list(degs.values()))
    leaves = sum(1 for d in deg_seq if d == 1)
    branch = sum(1 for d in deg_seq if d >= 3)

    invariants = {
        "n_nodes": g.n_nodes,
        "n_edges": g.n_edges,
        "n_components": g.n_components,
        "beta_0": g.n_components,
        "beta_1": g.beta_1_graph,
        "n_leaves": leaves,
        "n_branch_vertices": branch,
        "degree_sequence": deg_seq,
    }

    return ReebReference(
        name=f"mapper_proxy_{space_name}_{lens_id}",
        reference_type="mapper_proxy_sanity_check",
        space_name=space_name,
        lens_id=lens_id,
        graph=g,
        node_values=node_values,
        critical_values=sorted(list(node_values.values())),
        invariants=invariants,
        is_ground_truth=False,
        description="Numerical high-resolution Mapper proxy for convergence sanity check ONLY.",
        filter_definition_hash=None,
    )


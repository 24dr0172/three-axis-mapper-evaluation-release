"""Data structures and types for the Mapper evaluation framework."""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass(frozen=True)
class MapperNode:
    """Represents a single node in a Mapper graph."""
    node_id: int
    interval_idx: int
    cluster_label: int
    members: FrozenSet[int]
    size: int
    mean_filter: float


@dataclass(frozen=True)
class MapperGraph:
    """Represents the 1-skeleton graph of a Mapper complex."""
    nodes: Dict[int, MapperNode]
    edges: List[Tuple[int, int]]
    edge_weights: Dict[Tuple[int, int], int]
    n_nodes: int
    n_edges: int
    n_components: int
    beta_1_graph: int


@dataclass(frozen=True)
class SimplicialNerve2D:
    """Represents the 2-skeleton of a membership-defined simplicial nerve."""
    nodes: List[int]
    sigma_0: List[Tuple[int]]
    sigma_1: List[Tuple[int, int]]
    sigma_2: List[Tuple[int, int, int]]
    n_0: int
    n_1: int
    n_2: int
    vertex_to_row: Dict[int, int]


@dataclass(frozen=True)
class DualHomologyResult:
    """Represents dual homology results (graph cyclomatic vs nerve Betti numbers)."""
    beta_0_nerve: Optional[int]
    beta_1_nerve: Optional[int]
    beta_1_graph: Optional[int]
    rank_d1: Optional[int]
    rank_d2: Optional[int]
    status: str
    d1_d2_zero_mod2: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MapperOutput:
    """Unified container for Mapper construction results."""
    status: str
    reason: Optional[str]
    graph: Optional[MapperGraph]
    nerve: Optional[SimplicialNerve2D]
    homology: Optional[DualHomologyResult]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSelectionScore:
    """Structured result of internal candidate selection scoring."""
    eligible: bool
    score: Optional[float]
    status: str
    reason: Optional[str]
    labels: Optional[Any] = None


"""Exact GF(2) linear algebra, boundary reduction, and dual homology solver."""

from typing import Dict, List, Optional, Tuple
import numpy as np

from mapper_framework.exceptions import HomologyConsistencyError
from mapper_framework.types import DualHomologyResult, SimplicialNerve2D


def compute_gf2_matrix_rank(matrix: np.ndarray) -> int:
    """Compute the exact rank of a binary matrix over Galois field GF(2) (F_2).

    Uses Gaussian elimination with bitwise XOR arithmetic (modulo 2).
    Never uses real-valued SVD / numpy.linalg.matrix_rank.
    """
    if matrix.size == 0:
        return 0

    # Ensure binary copy with uint8
    A = (matrix.copy() % 2).astype(np.uint8)
    nrows, ncols = A.shape

    rank = 0
    current_row = 0

    for col in range(ncols):
        if current_row >= nrows:
            break

        # Find pivot in this column
        pivot_rows = np.where(A[current_row:, col] == 1)[0]
        if len(pivot_rows) == 0:
            continue

        pivot_row = current_row + pivot_rows[0]

        # Swap current row with pivot row
        if pivot_row != current_row:
            A[[current_row, pivot_row]] = A[[pivot_row, current_row]]

        # Eliminate all other 1s in this column
        other_rows = np.where(A[:, col] == 1)[0]
        for r in other_rows:
            if r != current_row:
                A[r] ^= A[current_row]

        rank += 1
        current_row += 1

    return rank


def compute_1skeleton_components(
    n_0: int,
    sigma_1: List[Tuple[int, int]],
    sigma_0: Optional[List[int]] = None,
) -> int:
    """Compute connected components of the 1-skeleton graph using BFS/DFS."""
    if n_0 == 0:
        return 0

    vertices = sigma_0 if sigma_0 is not None else list(range(n_0))
    adj: Dict[int, List[int]] = {v: [] for v in vertices}
    for u, v in sigma_1:
        if u in adj and v in adj:
            adj[u].append(v)
            adj[v].append(u)

    visited = set()
    components = 0

    for v in vertices:
        if v not in visited:
            components += 1
            queue = [v]
            visited.add(v)
            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

    return components


def compute_dual_homology(nerve: SimplicialNerve2D) -> DualHomologyResult:
    """Compute exact dual homology: beta_1^graph vs beta_1^nerve over F_2."""
    n_0 = nerve.n_0
    n_1 = nerve.n_1
    n_2 = nerve.n_2

    # Case: Empty complex
    if n_0 == 0:
        return DualHomologyResult(
            beta_0_nerve=0,
            beta_1_nerve=0,
            beta_1_graph=0,
            rank_d1=0,
            rank_d2=0,
            status="empty_complex",
            d1_d2_zero_mod2=True,
            details={"n_0": 0, "n_1": 0, "n_2": 0, "C": 0}
        )

    # Case: Zero edges
    if n_1 == 0:
        return DualHomologyResult(
            beta_0_nerve=n_0,
            beta_1_nerve=0,
            beta_1_graph=0,
            rank_d1=0,
            rank_d2=0,
            status="success",
            d1_d2_zero_mod2=True,
            details={"n_0": n_0, "n_1": 0, "n_2": 0, "C": n_0}
        )

    # 1. Build Boundary Matrix D1 (n_0 x n_1)
    D1 = np.zeros((n_0, n_1), dtype=np.uint8)
    edge_to_idx: Dict[Tuple[int, int], int] = {}

    for j, (u, v) in enumerate(nerve.sigma_1):
        if not (0 <= u < n_0 and 0 <= v < n_0):
            raise HomologyConsistencyError(
                f"1-simplex edge ({u}, {v}) references vertex index out of bounds [0, {n_0})."
            )
        D1[u, j] = 1
        D1[v, j] = 1
        edge_to_idx[(u, v)] = j


    # 2. Build Boundary Matrix D2 (n_1 x n_2)
    D2 = np.zeros((n_1, n_2), dtype=np.uint8)
    for k, (u, v, w) in enumerate(nerve.sigma_2):
        e_uv = (u, v) if u < v else (v, u)
        e_uw = (u, w) if u < w else (w, u)
        e_vw = (v, w) if v < w else (w, v)

        for e in (e_uv, e_uw, e_vw):
            if e not in edge_to_idx:
                raise HomologyConsistencyError(
                    f"2-simplex ({u}, {v}, {w}) boundary edge {e} is missing from 1-simplices."
                )
            D2[edge_to_idx[e], k] = 1


    # 3. Assert Mandatory Chain Condition: D1 @ D2 == 0 (mod 2)
    if n_2 > 0:
        D1_D2 = (D1.astype(int) @ D2.astype(int)) % 2
        d1_d2_zero = bool(np.all(D1_D2 == 0))
        if not d1_d2_zero:
            raise HomologyConsistencyError(
                f"Fundamental chain condition D1 * D2 != 0 (mod 2) violated! Non-zero entries: {np.count_nonzero(D1_D2)}"
            )
    else:
        d1_d2_zero = True

    # 4. Exact GF(2) Matrix Ranks
    r1 = compute_gf2_matrix_rank(D1)
    r2 = compute_gf2_matrix_rank(D2) if n_2 > 0 else 0

    # 5. Connected components of 1-skeleton
    C = compute_1skeleton_components(n_0, nerve.sigma_1)

    # 6. Algebraic consistency checks
    if r1 != n_0 - C:
        raise HomologyConsistencyError(
            f"Algebraic consistency check r1 == n_0 - C violated: r1={r1}, n_0={n_0}, C={C}"
        )

    beta_0_nerve = n_0 - r1  # == C
    beta_1_graph = n_1 - r1  # == E - V + C
    beta_1_nerve = beta_1_graph - r2

    if not (0 <= beta_1_nerve <= beta_1_graph):
        raise HomologyConsistencyError(
            f"Homology bound 0 <= beta_1_nerve <= beta_1_graph violated: "
            f"beta_1_nerve={beta_1_nerve}, beta_1_graph={beta_1_graph}, r2={r2}"
        )

    return DualHomologyResult(
        beta_0_nerve=int(beta_0_nerve),
        beta_1_nerve=int(beta_1_nerve),
        beta_1_graph=int(beta_1_graph),
        rank_d1=int(r1),
        rank_d2=int(r2),
        status="success",
        d1_d2_zero_mod2=d1_d2_zero,
        details={
            "n_0": n_0,
            "n_1": n_1,
            "n_2": n_2,
            "C": C,
            "r1": int(r1),
            "r2": int(r2),
        }
    )

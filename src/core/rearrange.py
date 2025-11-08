"""
rearrange.py

Core rearrangement API for Sydnify.

Provides:
- helpers to map between flat index <-> (x,y)
- a memory-aware `optimal` solver wrapper around SciPy's Hungarian method
- a practical `heuristic` (greedy) solver that produces a decent initial assignment
- a single `rearrange` function that selects the requested solver and returns assignments
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

# Use relative imports so module works both as package and as script-run.
from . import distance, utils


def idx_to_xy(idx: int, sidelen: int) -> Tuple[int, int]:
    return (idx % sidelen, idx // sidelen)


def xy_to_idx(x: int, y: int, sidelen: int) -> int:
    return y * sidelen + x


def build_positions(sidelen: int) -> np.ndarray:
    """
    Returns positions as an (N,2) int array where N = sidelen**2.
    pos[i] = (x, y)
    """
    N = sidelen * sidelen
    pos = np.empty((N, 2), dtype=np.int32)
    for i in range(N):
        pos[i, 0] = i % sidelen
        pos[i, 1] = i // sidelen
    return pos


def pair_cost(
    src_pixel: np.ndarray,
    tgt_pixel: np.ndarray,
    src_pos: Tuple[int, int],
    tgt_pos: Tuple[int, int],
    alpha: float = 1.0,
    beta: float = 0.02,
    metric: str = "rgb",
) -> float:
    """
    Cost between a single source pixel and a target cell.
    """
    return distance.combined_distance(src_pixel, tgt_pixel, src_pos, tgt_pos, alpha=alpha, beta=beta, metric=metric)


def build_full_cost_matrix(
    src_pixels: np.ndarray,
    tgt_pixels: np.ndarray,
    src_positions: np.ndarray,
    tgt_positions: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.02,
    metric: str = "rgb",
) -> np.ndarray:
    """
    Build NxN cost matrix where element (i,j) is cost of assigning source i to target j.

    WARNING: This allocates an N x N matrix. For sidelen >= ~120 this will be very large.
    Use the `rearrange` heuristic fallback for large N.
    """
    N = src_pixels.shape[0]
    if N != tgt_pixels.shape[0]:
        raise ValueError("src and tgt must have same number of pixels (flattened).")

    # Attempt to be memory conscious: check N*N size
    est_bytes = N * N * 8  # float64 bytes estimate
    # If matrix > ~1.5 GB, refuse by default
    if est_bytes > 1.5 * (1024 ** 3):
        raise MemoryError(f"Requested cost matrix is large: ~{est_bytes/(1024**3):.2f} GiB. Use heuristic mode or reduce sidelen.")

    # Build cost matrix (rows: source, cols: target)
    mat = np.empty((N, N), dtype=np.float64)
    for j in range(N):
        # compute cost vector for all sources to target j
        tgt_px = tgt_pixels[j]
        tgt_pos = (int(tgt_positions[j, 0]), int(tgt_positions[j, 1]))
        # vectorized color diff:
        # Using distance.batch_color_distance for color part, then add proximity penalty
        color_diffs = distance.batch_color_distance(src_pixels, np.tile(tgt_px, (N, 1)), metric=metric).astype(np.float64)
        prox = ((src_positions[:, 0] - tgt_pos[0]) ** 2 + (src_positions[:, 1] - tgt_pos[1]) ** 2).astype(np.float64)
        mat[:, j] = alpha * color_diffs + beta * np.sqrt(prox)  # note: sqrt to match Euclidean spatial distance
    return mat


def rearrange_optimal(
    src_pixels: np.ndarray,
    tgt_pixels: np.ndarray,
    sidelen: int,
    alpha: float = 1.0,
    beta: float = 0.02,
    metric: str = "rgb",
) -> np.ndarray:
    """
    Optimal assignment via Hungarian algorithm. Returns assignments array of length N
    where assignments[target_index] = source_index.

    This wrapper is memory-aware and will raise MemoryError if the full cost matrix
    would be too large.
    """
    N = sidelen * sidelen
    assert src_pixels.shape[0] == N and tgt_pixels.shape[0] == N

    src_pos = build_positions(sidelen)
    tgt_pos = src_pos.copy()  # target cells are grid positions 0..N-1

    # build expensive matrix
    cost = build_full_cost_matrix(src_pixels, tgt_pixels, src_pos, tgt_pos, alpha=alpha, beta=beta, metric=metric)

    # import inside function to avoid requirement unless user calls optimal
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(cost)  # row_ind: source idx, col_ind: target idx
    assignments = np.empty(N, dtype=np.int32)
    assignments[col_ind] = row_ind
    return assignments


def rearrange_heuristic_greedy(
    src_pixels: np.ndarray,
    tgt_pixels: np.ndarray,
    sidelen: int,
    alpha: float = 1.0,
    beta: float = 0.02,
    metric: str = "rgb",
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Simple greedy heuristic:
      - For each target cell (in raster order), pick the *closest available* source pixel by color+pos heuristic.
      - Remove picked source from pool.

    Complexity: O(N^2) time, O(N) memory. Works well as a fast initial assignment or for medium sizes.
    """
    rng = np.random.default_rng(seed)
    N = sidelen * sidelen
    assert src_pixels.shape[0] == N and tgt_pixels.shape[0] == N

    src_pos = build_positions(sidelen)
    tgt_pos = src_pos.copy()

    available = np.ones(N, dtype=bool)
    assignments = np.empty(N, dtype=np.int32)

    # For efficiency: precompute color distances incrementally per target
    for t in range(N):
        tgt_px = tgt_pixels[t]
        tx, ty = int(tgt_pos[t, 0]), int(tgt_pos[t, 1])

        # Compute cost for all available sources (vectorized)
        # color part:
        color_costs = distance.batch_color_distance(src_pixels, np.tile(tgt_px, (N, 1)), metric=metric).astype(np.float64)
        # spatial part:
        prox = np.sqrt(((src_pos[:, 0] - tx) ** 2 + (src_pos[:, 1] - ty) ** 2).astype(np.float64))
        total_cost = alpha * color_costs + beta * prox

        # mask out-unavailable sources
        masked_cost = np.where(available, total_cost, np.inf)

        # pick best
        best_src = int(np.argmin(masked_cost))
        assignments[t] = best_src
        available[best_src] = False

        # small randomization: occasionally break ties with a tiny jitter to avoid deterministic artifacts
        if (t % max(1, N // 1000)) == 0:
            # shuffle a few available indices to diversify slightly for similar pixels
            # (cheap and avoids worst-case deterministic behavior)
            ix = np.flatnonzero(available)
            if ix.size > 1:
                # swap two random available indices (no change to assignments yet)
                i1, i2 = rng.choice(ix, size=2, replace=False)
                # swap their order in the available array (doesn't change correctness)
                # This is only to introduce a tiny nondeterminism for near-equal costs.
                pass

    print("DEBUG: returning assignments, shape =", assignments.shape)
    return assignments



def rearrange(
    source_img: np.ndarray,
    target_img: np.ndarray,
    sidelen: int,
    mode: str = "heuristic",
    alpha: float = 1.0,
    beta: float = 0.02,
    metric: str = "rgb",
    **kwargs,
) -> np.ndarray:
    """
    Unified entrypoint.

    Parameters
    ----------
    source_img, target_img : np.ndarray
        HxW x 3 arrays or flattened arrays. If HxW provided, they will be resized/flatted
        to sidelen x sidelen inside caller. This function expects flattened shape (N,3).
    mode : 'heuristic' or 'optimal'
        Which solver to run.
    alpha, beta : floats
        Color vs spatial weights.
    metric : 'rgb', 'lab', or 'cosine'
    Returns
    -------
    assignments : np.ndarray (N,) int32
    """
        # Accept either flattened arrays or HxWx3 arrays:
    def _normalize(img):
        arr = np.asarray(img)
        if arr.ndim == 3:
            h, w = arr.shape[0], arr.shape[1]
            if h != sidelen or w != sidelen:
                raise ValueError("Source/target arrays must already be resized to sidelen x sidelen.")
            return arr.reshape(-1, 3).astype(np.int32)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr.astype(np.int32)
        raise ValueError("Unsupported image array shape; expected HxWx3 or (N,3).")

    src_flat = _normalize(source_img)
    tgt_flat = _normalize(target_img)

    if mode == "optimal":
        return rearrange_optimal(
            src_flat,
            tgt_flat,
            sidelen,
            alpha=alpha,
            beta=beta,
            metric=metric,
        )
    elif mode == "heuristic":
        return rearrange_heuristic_greedy(
            src_flat,
            tgt_flat,
            sidelen,
            alpha=alpha,
            beta=beta,
            metric=metric,
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'optimal' or 'heuristic'.")

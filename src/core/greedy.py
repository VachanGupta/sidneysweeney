"""
greedy.py

Initialization and baseline cost utilities for Sydnify.

Provides:
- random initialization utilities
- fast total-cost computation for an assignment
- a simple "color-sort" initializer for a better starting assignment
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

from . import distance, utils, rearrange


def init_random(N: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Return a random permutation of [0..N-1] as initial assignments.
    assignments[target_index] = source_index
    """
    rng = np.random.default_rng(seed)
    arr = np.arange(N, dtype=np.int32)
    rng.shuffle(arr)
    return arr


def init_color_sorted(src_pixels: np.ndarray, tgt_pixels: np.ndarray) -> np.ndarray:
    """
    A deterministic initializer that sorts source and target pixels by luminance
    (simple heuristic) and pairs them. Useful as a better-than-random start.

    Returns assignments: array of length N where assignments[target_idx] = source_idx
    """
    N = src_pixels.shape[0]
    assert tgt_pixels.shape[0] == N
    # compute luminance approx: 0.2126 R + 0.7152 G + 0.0722 B
    def lum(a):
        return 0.2126 * a[:, 0] + 0.7152 * a[:, 1] + 0.0722 * a[:, 2]

    src_l = lum(src_pixels.astype(np.float32))
    tgt_l = lum(tgt_pixels.astype(np.float32))

    src_order = np.argsort(src_l)
    tgt_order = np.argsort(tgt_l)

    # assignments such that tgt_order[i] gets src_order[i]
    assignments = np.empty(N, dtype=np.int32)
    # inverse of tgt_order gives mapping from target idx -> rank
    tgt_rank = np.empty(N, dtype=np.int32)
    tgt_rank[tgt_order] = np.arange(N, dtype=np.int32)
    # now for each target index t, find its rank r and map to src_order[r]
    for t in range(N):
        r = int(tgt_rank[t])
        assignments[t] = int(src_order[r])
    return assignments


def total_cost(
    assignments: np.ndarray,
    src_pixels: np.ndarray,
    tgt_pixels: np.ndarray,
    sidelen: int,
    alpha: float = 1.0,
    beta: float = 0.02,
    metric: str = "rgb",
) -> float:
    """
    Compute total heuristic cost for assignments:
      sum_over_t [ alpha * color_cost(src[assignments[t]], tgt[t]) + beta * spatial_cost(...) ]

    This is vectorized for p

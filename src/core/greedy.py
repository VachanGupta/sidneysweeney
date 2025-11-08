from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

from . import distance, utils

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


def build_positions(sidelen: int) -> np.ndarray:
    return utils.build_positions(sidelen)  # reuse utils implementation



def initialize_assignments(
    src_pixels: np.ndarray,
    tgt_pixels: np.ndarray,
    sidelen: int,
    mode: str = "random",
    seed: Optional[int] = None,
) -> np.ndarray:
    N = sidelen * sidelen
    if src_pixels.shape[0] != N or tgt_pixels.shape[0] != N:
        raise ValueError("src and tgt must be flattened arrays with shape (N,3) for sidelen.")

    rng = np.random.default_rng(seed)

    if mode == "random":
        perm = np.arange(N, dtype=np.int32)
        rng.shuffle(perm)
        return perm

    elif mode == "color_greedy":
        available = np.ones(N, dtype=bool)
        assignments = np.empty(N, dtype=np.int32)

        for t in range(N):
            tgt_px = tgt_pixels[t:t+1]  # shape (1,3)
            color_costs = distance.batch_color_distance(src_pixels, np.tile(tgt_px, (N, 1)))
            masked = np.where(available, color_costs, np.inf)
            best = int(np.argmin(masked))
            assignments[t] = best
            available[best] = False
        return assignments

    else:
        raise ValueError(f"Unknown init mode: {mode}")


def calc_total_heuristic(
    assignments: np.ndarray,
    src_pixels: np.ndarray,
    tgt_pixels: np.ndarray,
    sidelen: int,
    alpha: float = 1.0,
    beta: float = 0.02,
    metric: str = "rgb",
) -> float:
    N = sidelen * sidelen
    if assignments.shape[0] != N:
        raise ValueError("assignments length mismatch sidelen.")

    pos = build_positions(sidelen)
    total = 0.0

    assigned_src = src_pixels[assignments]

    color_costs = distance.batch_color_distance(assigned_src, tgt_pixels, metric=metric).astype(np.float64)

    src_pos = pos[assignments]  # (N,2)
    tgt_pos = pos  # (N,2)
    dx = src_pos[:, 0].astype(np.float64) - tgt_pos[:, 0].astype(np.float64)
    dy = src_pos[:, 1].astype(np.float64) - tgt_pos[:, 1].astype(np.float64)
    spatial = np.sqrt(dx * dx + dy * dy)
    total = float(np.sum(alpha * color_costs + beta * spatial))
    return total


def local_swap_delta(
    assignments: np.ndarray,
    src_pixels: np.ndarray,
    tgt_pixels: np.ndarray,
    sidelen: int,
    a: int,
    b: int,
    alpha: float = 1.0,
    beta: float = 0.02,
    metric: str = "rgb",
) -> float:
  
    if a == b:
        return 0.0
    N = sidelen * sidelen
    if not (0 <= a < N and 0 <= b < N):
        raise IndexError("a or b out of range")

    pos = build_positions(sidelen)
    sa = int(assignments[a])
    sb = int(assignments[b])


    old_color_a = distance.batch_color_distance(src_pixels[sa:sa+1], tgt_pixels[a:a+1], metric=metric)[0]
    old_color_b = distance.batch_color_distance(src_pixels[sb:sb+1], tgt_pixels[b:b+1], metric=metric)[0]


    src_pos_sa = tuple(int(x) for x in pos[sa])
    src_pos_sb = tuple(int(x) for x in pos[sb])
    tgt_pos_a = tuple(int(x) for x in pos[a])
    tgt_pos_b = tuple(int(x) for x in pos[b])
    old_sp_a = distance.spatial_distance(src_pos_sa, tgt_pos_a)
    old_sp_b = distance.spatial_distance(src_pos_sb, tgt_pos_b)

    old = alpha * old_color_a + beta * old_sp_a + alpha * old_color_b + beta * old_sp_b

    new_color_a = distance.batch_color_distance(src_pixels[sb:sb+1], tgt_pixels[a:a+1], metric=metric)[0]
    new_color_b = distance.batch_color_distance(src_pixels[sa:sa+1], tgt_pixels[b:b+1], metric=metric)[0]
    new_sp_a = distance.spatial_distance(src_pos_sb, tgt_pos_a)
    new_sp_b = distance.spatial_distance(src_pos_sa, tgt_pos_b)
    new = alpha * new_color_a + beta * new_sp_a + alpha * new_color_b + beta * new_sp_b

    return float(new - old)


if _HAS_NUMBA:

    @njit
    def _nb_color_dist_sq(a0, a1):
        d0 = a0[0] - a1[0]
        d1 = a0[1] - a1[1]
        d2 = a0[2] - a1[2]
        return d0 * d0 + d1 * d1 + d2 * d2

    @njit
    def nb_local_swap_delta(
        assignments,
        src_pixels_int32,
        tgt_pixels_int32,
        sidelen,
        a,
        b,
        alpha,
        beta
    ):
        if a == b:
            return 0.0
        N = sidelen * sidelen

        sa = assignments[a]
        sb = assignments[b]


        old_ca = _nb_color_dist_sq(src_pixels_int32[sa], tgt_pixels_int32[a])
        old_cb = _nb_color_dist_sq(src_pixels_int32[sb], tgt_pixels_int32[b])
        
        ax = a % sidelen
        ay = a // sidelen
        bx = b % sidelen
        by = b // sidelen
        src_ax = sa % sidelen
        src_ay = sa // sidelen
        src_bx = sb % sidelen
        src_by = sb // sidelen
        old_sp_a = ((src_ax - ax) ** 2 + (src_ay - ay) ** 2) ** 0.5
        old_sp_b = ((src_bx - bx) ** 2 + (src_by - by) ** 2) ** 0.5
        old = alpha * old_ca + beta * old_sp_a + alpha * old_cb + beta * old_sp_b

       
        new_ca = _nb_color_dist_sq(src_pixels_int32[sb], tgt_pixels_int32[a])
        new_cb = _nb_color_dist_sq(src_pixels_int32[sa], tgt_pixels_int32[b])
        new_sp_a = ((src_bx - ax) ** 2 + (src_by - ay) ** 2) ** 0.5
        new_sp_b = ((src_ax - bx) ** 2 + (src_ay - by) ** 2) ** 0.5
        new = alpha * new_ca + beta * new_sp_a + alpha * new_cb + beta * new_sp_b
        return new - old

else:
    nb_local_swap_delta = None  


def demo_init_and_score(source_img, target_img, sidelen=64, init_mode="random", seed=0, alpha=1.0, beta=0.02):
   
    def _normalize(img):
        arr = np.asarray(img)
        if arr.ndim == 3:
            h, w = arr.shape[0], arr.shape[1]
            if h != sidelen or w != sidelen:
                raise ValueError("Image must be resized to sidelen x sidelen before calling this helper.")
            return arr.reshape(-1, 3).astype(np.int32)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr.astype(np.int32)
        raise ValueError("Unsupported image shape")

    src = _normalize(source_img)
    tgt = _normalize(target_img)

    assigns = initialize_assignments(src, tgt, sidelen=sidelen, mode=init_mode, seed=seed)
    score = calc_total_heuristic(assigns, src, tgt, sidelen, alpha=alpha, beta=beta)
    return assigns, score

from __future__ import annotations
import numpy as np
import os, time
from typing import Optional, Tuple, Dict, Callable
from tqdm import trange

from . import greedy

FrameCallback = Optional[Callable[[np.ndarray, int], None]]

def _pick_b_within_radius(a: int, sidelen: int, max_dist: int, rng: np.random.Generator) -> int:
    if max_dist <= 0:
        return int(rng.integers(0, sidelen * sidelen))
    ax = a % sidelen
    ay = a // sidelen
    dx = int(rng.integers(-max_dist, max_dist + 1))
    dy = int(rng.integers(-max_dist, max_dist + 1))
    bx = max(0, min(sidelen - 1, ax + dx))
    by = max(0, min(sidelen - 1, ay + dy))
    return int(by * sidelen + bx)

def _save_checkpoint(path: str, assignments: np.ndarray, meta: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, assignments=assignments, **meta)

def run_swap_optimizer(
    src_pixels: np.ndarray,
    tgt_pixels: np.ndarray,
    sidelen: int,
    *,
    init_assignments: Optional[np.ndarray] = None,
    init_mode: str = "random",
    seed: Optional[int] = None,
    generations: int = 200,
    swaps_per_generation_per_pixel: float = 1.0,
    initial_max_dist: int = 8,
    min_max_dist: int = 1,
    shrink_every_gen: int = 10,
    shrink_factor: float = 0.75,
    alpha: float = 1.0,
    beta: float = 0.02,
    metric: str = "rgb",
    frame_callback: FrameCallback = None,
    frame_interval_generations: int = 1,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(seed)
    N = sidelen * sidelen
    src = np.asarray(src_pixels)
    tgt = np.asarray(tgt_pixels)
    if src.shape[0] != N or tgt.shape[0] != N:
        raise ValueError("src/tgt must be flattened arrays (N,3)")

    if init_assignments is None:
        assigns = greedy.initialize_assignments(src, tgt, sidelen=sidelen, mode=init_mode, seed=seed)
    else:
        assigns = np.asarray(init_assignments, dtype=np.int32).copy()

    nb_delta = getattr(greedy, "nb_local_swap_delta", None)
    use_numba = nb_delta is not None
    if use_numba:
        src_int = src.astype(np.int32)
        tgt_int = tgt.astype(np.int32)
    else:
        src_int = tgt_int = None

    stats = {"start_time": time.time(), "accepted_swaps": 0, "attempted_swaps": 0, "frames_emitted": 0}
    max_dist = int(initial_max_dist)
    frame_idx = 0
    gen_iter = trange(generations, desc="generations") if verbose else range(generations)

    for gen in gen_iter:
        swaps_this_gen = int(np.round(swaps_per_generation_per_pixel * N))
        accepted_in_gen = 0

        for _ in range(swaps_this_gen):
            stats["attempted_swaps"] += 1
            a = int(rng.integers(0, N))
            b = _pick_b_within_radius(a, sidelen, max_dist, rng)
            if a == b:
                continue

            if use_numba:
                delta = float(nb_delta(assigns, src_int, tgt_int, sidelen, a, b, float(alpha), float(beta)))
            else:
                delta = greedy.local_swap_delta(assigns, src, tgt, sidelen, a, b, alpha=alpha, beta=beta, metric=metric)

            if delta < 0.0:
                assigns[a], assigns[b] = assigns[b], assigns[a]
                accepted_in_gen += 1
                stats["accepted_swaps"] += 1

        if (gen + 1) % shrink_every_gen == 0 and gen > 0:
            new_max = max(min_max_dist, int(round(max_dist * shrink_factor)))
            if new_max < max_dist:
                max_dist = new_max

        if frame_callback is not None and ((gen % frame_interval_generations) == 0 or gen == generations - 1):
            frame_callback(assigns.copy(), frame_idx)
            stats["frames_emitted"] += 1
            frame_idx += 1

        if checkpoint_path is not None and ((gen + 1) % max(1, generations // 10) == 0 or gen == generations - 1):
            meta = {"gen": gen, "accepted_swaps": stats["accepted_swaps"], "attempted_swaps": stats["attempted_swaps"], "max_dist": max_dist}
            _save_checkpoint(checkpoint_path, assigns, meta)

        if verbose:
            gen_iter.set_postfix({"accepted": accepted_in_gen, "max_dist": max_dist})

    stats["end_time"] = time.time()
    stats["duration_s"] = stats["end_time"] - stats["start_time"]
    return assigns, stats

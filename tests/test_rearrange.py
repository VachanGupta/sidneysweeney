import numpy as np
from src.core import rearrange, utils

def test_rearrange_heuristic_small():
    sidelen = 8
    N = sidelen * sidelen
    src = np.zeros((sidelen, sidelen, 3), dtype=np.uint8)
    tgt = np.zeros_like(src)
    for y in range(sidelen):
        for x in range(sidelen):
            v = int((x + y) / (2 * sidelen) * 255)
            src[y, x] = [v, v, v]
            tgt[y, x] = [255 - v, 255 - v, 255 - v]

    assignments = rearrange.rearrange(src, tgt, sidelen=sidelen, mode="heuristic", alpha=1.0, beta=0.01)
    assert assignments.shape[0] == N
    assert set(assignments.tolist()) <= set(range(N))

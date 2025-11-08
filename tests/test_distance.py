import numpy as np
from src.core import distance

def test_rgb_distance_basic():
    a = np.array([255, 0, 0])
    b = np.array([0, 255, 0])
    d = distance.euclidean_rgb(a, b)
    assert d > 0
    assert abs(d - 130050) < 1e-3 

def test_spatial_distance():
    assert distance.spatial_distance((0, 0), (3, 4)) == 5.0

def test_combined_distance():
    a = np.array([100, 100, 100])
    b = np.array([200, 200, 200])
    val = distance.combined_distance(a, b, (0, 0), (1, 1))
    assert val > 0

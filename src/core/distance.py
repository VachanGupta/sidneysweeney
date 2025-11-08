import numpy as np
from math import sqrt

try:
    from skimage.color import rgb2lab
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# color space uitl
def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image [0-255] to CIE Lab [0-100].
    Requires scikit-image; otherwise returns normalized RGB.
    """
    if _HAS_SKIMAGE:
        img = image.astype(np.float32) / 255.0
        return rgb2lab(img)
    else:
        return image.astype(np.float32) / 255.0


# color dist metrics
def euclidean_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """
    Squared Euclidean distance in RGB space.
    """
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sum(diff * diff))


def euclidean_lab(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance in Lab space (perceptual difference).
    """
    if not _HAS_SKIMAGE:
        return euclidean_rgb(a, b)
    a_lab = rgb_to_lab(a[np.newaxis, np.newaxis, :])[0, 0]
    b_lab = rgb_to_lab(b[np.newaxis, np.newaxis, :])[0, 0]
    diff = a_lab - b_lab
    return float(np.sqrt(np.sum(diff * diff)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    1 - cosine similarity (range [0, 2])
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return 1.0 - num / denom


# spatial dist
def spatial_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Euclidean distance between two pixel coordinates (x, y).
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return sqrt(dx * dx + dy * dy)


# combined
def combined_distance(
    color_a: np.ndarray,
    color_b: np.ndarray,
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    alpha: float = 1.0,
    beta: float = 0.02,
    metric: str = "rgb"
) -> float:
    """
    Weighted sum of color distance and positional distance.

    D = alpha * color_distance + beta * spatial_distance
    """
    if metric == "lab":
        cdist = euclidean_lab(color_a, color_b)
    elif metric == "cosine":
        cdist = cosine_distance(color_a, color_b)
    else:
        cdist = euclidean_rgb(color_a, color_b)

    sdist = spatial_distance(pos_a, pos_b)
    return alpha * cdist + beta * sdist


# vector versoin
def batch_color_distance(src: np.ndarray, tgt: np.ndarray, metric: str = "rgb") -> np.ndarray:
    """
    Compute per-pixel color distances between two arrays of same shape (N,3).
    Returns vector of length N.
    """
    if metric == "lab" and _HAS_SKIMAGE:
        src_lab = rgb_to_lab(src.reshape(1, -1, 3))[0]
        tgt_lab = rgb_to_lab(tgt.reshape(1, -1, 3))[0]
        diff = src_lab - tgt_lab
    else:
        diff = src.astype(np.float32) - tgt.astype(np.float32)
    return np.sum(diff * diff, axis=1)

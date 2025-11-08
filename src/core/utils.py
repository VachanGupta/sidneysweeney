from PIL import Image
import numpy as np

def load_image(path: str, size: int):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)

def flatten_image(img: np.ndarray):
    return img.reshape(-1, 3).astype(np.int32)

def save_image(array, sidelen: int, path: str):
    img = Image.fromarray(array.reshape(sidelen, sidelen, 3).astype(np.uint8))
    img.save(path)


from __future__ import annotations
import os
from typing import List, Optional, Callable
from PIL import Image
import numpy as np
 
try:
    import imageio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False


def render_assignments_to_image(src_pixels: np.ndarray, assignments: np.ndarray, sidelen: int) -> Image.Image: 
    N = sidelen * sidelen
    if src_pixels.shape[0] != N:
        raise ValueError("src_pixels must be flattened (N,3) with N = sidelen^2")
    if assignments.shape[0] != N:
        raise ValueError("assignments must have length N")

  
    src = src_pixels.astype(np.uint8)
    assigns = assignments.astype(np.int32)

    canvas = src[assigns]   
    canvas = canvas.reshape((sidelen, sidelen, 3))
    return Image.fromarray(canvas, "RGB")


class FrameCollector: 

    def __init__(self, out_dir: Optional[str] = None, keep_frames: bool = True, png_prefix: str = "frame"):
        self.out_dir = out_dir
        self.keep_frames = keep_frames
        self.png_prefix = png_prefix
        self.frames: List[np.ndarray] = []
        self._counter = 0
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)

    def callback(self, assignments: np.ndarray, frame_idx: int, src_pixels: Optional[np.ndarray] = None, sidelen: Optional[int] = None):
         
        if src_pixels is None or sidelen is None:
             
            raise ValueError("FrameCollector.callback requires src_pixels and sidelen to be provided via a lambda wrapper.")

        img = render_assignments_to_image(src_pixels, assignments, sidelen)
        arr = np.asarray(img)

        if self.keep_frames:
            self.frames.append(arr)

        if self.out_dir is not None:
            path = os.path.join(self.out_dir, f"{self.png_prefix}_{frame_idx:04d}.png")
            img.save(path)

        self._counter += 1

    def save_gif(self, out_path: str, fps: int = 6, loop: int = 0):
        
        if len(self.frames) == 0:
            raise RuntimeError("No frames to save. Did you collect frames?")

        if _HAS_IMAGEIO:
             
            duration = 1.0 / float(fps)
            imageio.mimsave(out_path, self.frames, format="GIF", duration=duration, loop=loop)
        else:
           
            pil_frames = [Image.fromarray(f) for f in self.frames]
            first, *rest = pil_frames
            first.save(out_path, save_all=True, append_images=rest, duration=int(1000 / fps), loop=loop, optimize=True)


def save_frames_as_gif(frames: List[np.ndarray], out_path: str, fps: int = 6, loop: int = 0):
    
    if _HAS_IMAGEIO:
        duration = 1.0 / float(fps)
        imageio.mimsave(out_path, frames, format="GIF", duration=duration, loop=loop)
    else:
        pil_frames = [Image.fromarray(f) for f in frames]
        first, *rest = pil_frames
        first.save(out_path, save_all=True, append_images=rest, duration=int(1000 / fps), loop=loop, optimize=True)


if __name__ == "__main__":
    import argparse
    from src.core import anneal, utils as core_utils
    from src.core.greedy import initialize_assignments

    parser = argparse.ArgumentParser(description="Demo: anneal + frame collection")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--sidelen", type=int, default=64)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--out", default="out_demo")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    src_img = core_utils.load_image(args.source, args.sidelen)
    tgt_img = core_utils.load_image(args.target, args.sidelen)
    src_flat = core_utils.flatten_image(src_img)
    tgt_flat = core_utils.flatten_image(tgt_img)

    collector = FrameCollector(out_dir=args.out, keep_frames=True)

    frame_cb = lambda assigns, idx: collector.callback(assigns, idx, src_pixels=src_flat, sidelen=args.sidelen)

    assigns, stats = anneal.run_swap_optimizer(
        src_flat,
        tgt_flat,
        args.sidelen,
        init_mode="random",
        seed=42,
        generations=args.generations,
        swaps_per_generation_per_pixel=1.0,
        initial_max_dist=max(1, args.sidelen // 8),
        frame_callback=frame_cb,
        frame_interval_generations=max(1, args.generations // 8),
        checkpoint_path=os.path.join(args.out, "checkpoint.npz"),
        verbose=True,
    )

    gif_path = os.path.join(args.out, "result.gif")
    collector.save_gif(gif_path, fps=args.fps)
    print("Saved GIF:", gif_path)

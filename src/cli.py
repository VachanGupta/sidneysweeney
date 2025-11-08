import argparse
import os
from src.core import utils, anneal
from src.visualization.progress_gif import FrameCollector

def main():
    parser = argparse.ArgumentParser(description="Sydnify — rearrange pixels into Sydney Sweeney.")
    parser.add_argument("--source", required=True, help="Path to source image")
    parser.add_argument("--target", required=True, help="Path to target image (Sydney Sweeney image)")
    parser.add_argument("--sidelen", type=int, default=64, help="Resize both images to sidelen x sidelen")
    parser.add_argument("--generations", type=int, default=120, help="Number of annealing generations")
    parser.add_argument("--swaps-per-pixel", type=float, default=1.0, help="Swap attempts per pixel per generation")
    parser.add_argument("--out", default="out_sydnify", help="Output directory")
    parser.add_argument("--fps", type=int, default=6, help="GIF frame rate")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading images...")
    src_img = utils.load_image(args.source, args.sidelen)
    tgt_img = utils.load_image(args.target, args.sidelen)
    src_flat = utils.flatten_image(src_img)
    tgt_flat = utils.flatten_image(tgt_img)

    print("Preparing frame collector...")
    collector = FrameCollector(out_dir=args.out, keep_frames=True)
    frame_cb = lambda assigns, idx: collector.callback(assigns, idx, src_pixels=src_flat, sidelen=args.sidelen)

    print("Running annealer...")
    assigns, stats = anneal.run_swap_optimizer(
        src_flat,
        tgt_flat,
        args.sidelen,
        init_mode="random",
        seed=42,
        generations=args.generations,
        swaps_per_generation_per_pixel=args.swaps_per_pixel,
        initial_max_dist=max(1, args.sidelen // 8),
        frame_callback=frame_cb,
        frame_interval_generations=max(1, args.generations // 8),
        checkpoint_path=os.path.join(args.out, "checkpoint.npz"),
        verbose=True,
    )

    gif_path = os.path.join(args.out, "sydneyfy.gif")
    collector.save_gif(gif_path, fps=args.fps)
    print(f" Done! Saved GIF to {gif_path}")

if __name__ == "__main__":
    main()

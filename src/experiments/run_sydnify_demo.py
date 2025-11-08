from src.visualization.progress_gif import FrameCollector
from src.core import anneal, utils

src_flat = utils.flatten_image(utils.load_image("source.jpg", 64))
tgt_flat = utils.flatten_image(utils.load_image("sydney.jpg", 64))

collector = FrameCollector(out_dir="out_run", keep_frames=True)
frame_cb = lambda assigns, idx: collector.callback(assigns, idx, src_pixels=src_flat, sidelen=64)

assigns, stats = anneal.run_swap_optimizer(
    src_flat, tgt_flat, 64,
    init_mode="random",
    seed=123,
    generations=120,
    swaps_per_generation_per_pixel=1.0,
    frame_callback=frame_cb,
    frame_interval_generations=5,
)

collector.save_gif("out_run/sydnify.gif", fps=6)
print("✅ Done! Saved GIF to out_run/sydnify.gif")

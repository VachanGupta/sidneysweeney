from __future__ import annotations
import os, tempfile
import streamlit as st
import numpy as np
from src.core import utils, anneal
from src.visualization.progress_gif import FrameCollector

st.set_page_config(page_title="Sydnify", page_icon="🧠", layout="wide")

st.title("Sydnify — Turn any image into Sydney Sweeney")

with st.sidebar:
    st.header("⚙️ Settings")
    sidelen = st.slider("Resize (pixels per side)", 32, 192, 96, step=16)
    generations = st.slider("Generations", 20, 300, 120, step=10)
    swaps_per_pixel = st.slider("Swaps per pixel", 0.2, 3.0, 1.0, step=0.1)
    fps = st.slider("GIF FPS", 2, 15, 6)
    seed = st.number_input("Random seed", value=42, step=1)
    run_button = st.button("🚀 Run Sydnify")

st.markdown("Upload two images below — your **source** and the **Sydney target**:")

col1, col2 = st.columns(2)
with col1:
    src_file = st.file_uploader("Source image", type=["jpg", "jpeg", "png"], key="src")
with col2:
    tgt_file = st.file_uploader("Target image (Sydney Sweeney)", type=["jpg", "jpeg", "png"], key="tgt")

if run_button:
    if not src_file or not tgt_file:
        st.error("Please upload both images.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, "source.jpg")
        tgt_path = os.path.join(tmp, "target.jpg")
        with open(src_path, "wb") as f:
            f.write(src_file.getbuffer())
        with open(tgt_path, "wb") as f:
            f.write(tgt_file.getbuffer())

  
        st.write("Loading and resizing images...")
        src_img = utils.load_image(src_path, sidelen)
        tgt_img = utils.load_image(tgt_path, sidelen)
        src_flat = utils.flatten_image(src_img)
        tgt_flat = utils.flatten_image(tgt_img)

        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        collector = FrameCollector(out_dir=out_dir, keep_frames=True)
        frame_cb = lambda assigns, idx: collector.callback(assigns, idx, src_pixels=src_flat, sidelen=sidelen)

        st.write("Running annealer... this may take a minute ⏳")
        progress = st.progress(0)
        log_placeholder = st.empty()

        def progress_frame(assigns, idx):
            frame_cb(assigns, idx)
            progress.progress(min(1.0, (idx + 1) / (generations // 5 + 1)))
            log_placeholder.text(f"Generated frame {idx + 1}")

        assigns, stats = anneal.run_swap_optimizer(
            src_flat,
            tgt_flat,
            sidelen,
            init_mode="random",
            seed=seed,
            generations=generations,
            swaps_per_generation_per_pixel=swaps_per_pixel,
            initial_max_dist=max(1, sidelen // 8),
            frame_callback=progress_frame,
            frame_interval_generations=max(1, generations // 5),
            checkpoint_path=os.path.join(out_dir, "checkpoint.npz"),
            verbose=False,
        )

        gif_path = os.path.join(out_dir, "sydneyfy.gif")
        collector.save_gif(gif_path, fps=fps)

        st.success("✅ Done! Sydnify completed successfully.")
        st.image(gif_path, caption="Sydneyfied result", use_column_width=True)

        st.download_button("⬇️ Download GIF", data=open(gif_path, "rb").read(), file_name="sydneyfy.gif")

        st.markdown(f"**Accepted swaps:** {stats['accepted_swaps']} • **Time:** {stats['duration_s']:.2f}s")

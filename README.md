# Sydnify 

> *Rearranging pixels into art — one Sydney Sweeney at a time.*

Sydnify is a Python reimagination of the original *Obamify* algorithm, built from scratch with modular architecture.
It rearranges pixels from any source image into a target image (by default, Sydney Sweeney) using both **optimal transport** and **genetic swap-based heuristics**.

---

## Features (Planned)
- Pixel rearrangement engine (`core/`)
  - Optimal assignment via Hungarian algorithm
  - Genetic swap optimization
  - Simulated annealing variant
- Visualization tools (`visualization/`)
  - Frame previewer
  - GIF progress generator
- Web interface (`webapp/`)
  - Streamlit/Gradio live demo
- Benchmarks + tests (`experiments/`, `tests/`)

---

##  Setup
```bash
git clone <your_repo_url>
cd sydnify
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

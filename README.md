# OCT Denoising + Segmentation (Noise2Void + U‑Net)

**At a glance:** Self‑supervised denoising (Noise2Void‑style) and U‑Net segmentation on OCT B‑scans. This repo reflects work I contributed to at BU’s **Tian Lab**—from dataset creation and denoising to model training and evaluation. It is structured for **fast review**: static results below, 1‑click notebooks, and a simple local quickstart.

[![N2V Demo (Colab)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/oct-denoise-unet/blob/main/notebooks/01_n2v_demo.ipynb)
[![U‑Net Training (Colab)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/oct-denoise-unet/blob/main/notebooks/02_unet_training.ipynb)

---

## Why this matters for labs & employers
- **End‑to‑end ownership:** Created ground‑truth datasets (ImageJ/AnnotatorJ), tuned denoising (Noise2Void), trained U‑Net baselines, and organized results for review.
- **Rigor + practicality:** Reproducible scripts and notebooks, clean repo layout, and quick visual checks that reduce onboarding time for collaborators.
- **Transferable impact:** Skills translate to medical imaging, quality assurance, data pipelines, and production‑grade ML prototyping.

---

## Quick view: results (static previews)
<p align="center">
  <img src="examples/results_panel.png" width="98%"><br/>
  <em>Left: noisy OCT‑like B‑scan · Middle: denoised (N2V‑style preview) · Right: example segmentation mask</em>
</p>

Additional previews:
- Denoise before/after: `examples/n2v_before.png` → `examples/n2v_after.png`  
- Mask + overlay: `examples/unet_pred.png`, `examples/unet_overlay.png`

> Full pipeline notebooks are provided for verification, but a reviewer can evaluate the approach from these images alone.

---

## What’s in this repo
- **notebooks/** – two minimal, Colab‑ready notebooks (N2V demo, U‑Net training).  
- **scripts/** – CLI for synthetic data, training, and quick metrics.  
- **examples/** – static PNGs used above so reviewers can see results immediately.  
- **Makefile** – convenience targets (`setup`, `synth`, `unet`, `eval`).

### Local quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_synthetic_oct.py --out data/synth --n 20
python scripts/train_unet.py --data data/synth --out runs/unet_demo --epochs 2
python scripts/eval_metrics.py --pred runs/unet_demo/preds --gt data/synth/masks
```

---

## Project background (BU Tian Lab)
- Built **ground‑truth datasets** by annotating OCT lung images with **ImageJ/AnnotatorJ**.
- Improved input quality with **Noise2Void** parameter sweeps to reduce speckle‑like noise.
- Trained a **U‑Net** (with Dense‑block variants) and iterated hyperparameters for reliability.
- Organized results, experiment logs, and simple demos to streamline collaboration.

**Contact:** Euijin Jung · ajung23@bu.edu · (872) 381‑3969 · Chicago, IL

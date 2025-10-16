# OCT Denoising & Segmentation (Noise2Void + U‑Net)

Self‑supervised denoising (Noise2Void) and U‑Net segmentation for OCT B‑scans.  
This repository reflects the workflow I contributed to at BU’s **Tian Lab**: dataset creation with ImageJ/AnnotatorJ → denoising → segmentation → light, reproducible evaluation.

[![N2V Demo (Colab)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/oct-denoise-unet/blob/main/notebooks/01_n2v_demo.ipynb)
[![U‑Net Training (Colab)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/oct-denoise-unet/blob/main/notebooks/02_unet_training.ipynb)

---

## How this work adds value
- **End‑to‑end ownership:** built ground‑truth datasets (ImageJ/AnnotatorJ), tuned Noise2Void denoising, trained U‑Net baselines, and packaged results for fast review.
- **Rigor + speed:** clean scripts/notebooks and a small synthetic set so collaborators can verify quickly, then swap in real data.
- **Transferable impact:** methods and habits (data quality, self‑supervised denoising, segmentation, metrics) carry straight into clinical imaging and applied‑AI prototyping.

---

## Quick view (no need to run anything)
<p align="center">
  <img src="examples/results_panel.png" width="98%"><br/>
  <em>Left: noisy OCT‑like B‑scan · Middle: denoised preview (N2V‑style) · Right: example mask</em>
</p>

More previews:
- Denoise before/after: `examples/n2v_before.png` → `examples/n2v_after.png`  
- Mask + overlay: `examples/unet_pred.png`, `examples/unet_overlay.png`

---

## What’s here
- **notebooks/** – two minimal demos (Colab‑ready)  
- **scripts/** – CLI for synthetic data, training, and quick metrics  
- **examples/** – static PNGs so reviewers can see results immediately  
- **Makefile** – `make setup | synth | unet | eval`

### Local quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/make_synthetic_oct.py --out data/synth --n 20
python scripts/train_unet.py --data data/synth --out runs/unet_demo --epochs 2
python scripts/eval_metrics.py --pred runs/unet_demo/preds --gt data/synth/masks

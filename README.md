# OCT Denoising + Segmentation (Noise2Void + U‑Net)

> Self‑supervised denoising (Noise2Void) + U‑Net segmentation on OCT‑like B‑scans with a **5‑minute quickstart**. Synthetic examples included (no PHI).

[![Open N2V Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/oct-denoise-unet/blob/main/notebooks/01_n2v_demo.ipynb)
[![Open U‑Net Training in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/oct-denoise-unet/blob/main/notebooks/02_unet_training.ipynb)

## Quickstart (Local)
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Synthetic data -> denoise -> segment -> evaluate
python scripts/make_synthetic_oct.py --out data/synth --n 20
# (Optional: N2V; heavy on CPU/CI) python scripts/denoise_n2v.py --data data/synth/raw --out runs/n2v_demo --epochs 1
python scripts/train_unet.py --data data/synth --out runs/unet_demo --epochs 2
python scripts/eval_metrics.py --pred runs/unet_demo/preds --gt data/synth/masks
```

**Results (fill in after running):**
| Stage                     | PSNR (↑) | SSIM (↑) | Dice (↑) |
|--------------------------|----------|----------|----------|
| Raw noisy                |          |          |    —     |
| After N2V (denoised)     |          |          |    —     |
| U‑Net segmentation       |    —     |    —     |          |

## Repo map
```
notebooks/  -> interactive demos (Colab-friendly)
scripts/    -> CLI pipeline (synth, train, eval)
examples/   -> small images used in README previews
.github/    -> CI smoke test
.devcontainer/ -> Codespaces dev environment
```

## Codespaces (1‑click dev container)
Open in Codespaces and run the Makefile:
```bash
make setup
make synth
make unet
make eval
```

## Notes
- Real datasets should live under `data/` (not committed). Masks follow ImageJ/AnnotatorJ naming.
- Synthetic data lets reviewers/labs reproduce the pipeline instantly.
- This repo intentionally avoids PHI.

**Contact:** Euijin Jung · ajung23@bu.edu · (872) 381‑3969 · Chicago, IL

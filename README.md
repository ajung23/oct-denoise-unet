# OCT Denoising + Segmentation (Noise2Void + U‑Net)

> Self‑supervised denoising (Noise2Void) + U‑Net segmentation for OCT‑like B‑scans with **fast setup**.  
> This patch adds **Google Drive dataset integration** and a **real‑data quick check** notebook.

[![Open N2V Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/oct-denoise-unet/blob/main/notebooks/01_n2v_demo.ipynb)
[![Open U‑Net Training in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/oct-denoise-unet/blob/main/notebooks/02_unet_training.ipynb)
[![Open Real Data Check in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/oct-denoise-unet/blob/main/notebooks/03_real_data_quickcheck.ipynb)

---

## Dataset (Google Drive)
**Folder (provided by author):** https://drive.google.com/drive/folders/12nhKcPg3eGIwHcSWv2anTuD_23AwpZX4

> Make sure sharing is set to **Anyone with the link – Viewer** (no login required) if you want others to reproduce.  
> **PHI warning:** Do not upload PHI or identifiable information. Use de‑identified or synthetic data.

### Download via gdown
```bash
# 1) Install deps
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt  # includes gdown

# 2) Download the Drive folder into data/drive_raw
python scripts/download_gdrive.py --folder-id 12nhKcPg3eGIwHcSWv2anTuD_23AwpZX4 --out data/drive_raw

# 3) Organize into standard layout (data/real/raw, data/real/masks)
python scripts/organize_dataset.py --src data/drive_raw --dst data/real

# 4) Validate
python scripts/validate_dataset.py --root data/real
```

**Standard layout expected:**
```
data/
└── real/
    ├── raw/           # OCT images (.png/.tif)
    └── masks/         # binary masks from ImageJ/AnnotatorJ (same basenames)
```

If your masks are in a different format or naming, use the flags in `organize_dataset.py` to map patterns (see usage).

---

## Quickstart (Synthetic → works anywhere)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/make_synthetic_oct.py --out data/synth --n 20
# (optional) python scripts/denoise_n2v.py --data data/synth/raw --out runs/n2v_demo --epochs 1
python scripts/train_unet.py --data data/synth --out runs/unet_demo --epochs 2
python scripts/eval_metrics.py --pred runs/unet_demo/preds --gt data/synth/masks
```

### Results (fill these in after running)
| Stage                     | PSNR (↑) | SSIM (↑) | Dice (↑) |
|--------------------------|----------|----------|----------|
| Raw noisy                |          |          |    —     |
| After N2V (denoised)     |          |          |    —     |
| U‑Net segmentation       |    —     |    —     |          |

---

## Repo map (with this patch)
```
scripts/
  download_gdrive.py     # download Google Drive folder with gdown
  organize_dataset.py    # normalize into data/real/raw & data/real/masks
  validate_dataset.py    # sanity checks (counts, matched filenames)
notebooks/
  03_real_data_quickcheck.ipynb  # visualize pairs from data/real
DATA.md                  # dataset structure, patterns, PHI safety
Makefile                 # data-drive, organize, validate targets
requirements.txt         # adds gdown
.gitignore               # ensures data/ is ignored
```

# DATA

This project expects OCT data in a **standard layout** for training and evaluation:

```
data/
└── real/
    ├── raw/           # OCT images (.png/.tif)
    └── masks/         # binary masks (.png) from ImageJ/AnnotatorJ
```
- File basenames must match between `raw/` and `masks/` (e.g., `oct_001.png` in both).  
- Masks should be binary (0/255). Multi-class masks can be converted to binary per class.

## Using the provided Google Drive folder
**Link:** https://drive.google.com/drive/folders/12nhKcPg3eGIwHcSWv2anTuD_23AwpZX4

1. Ensure sharing is **Anyone with the link – Viewer** if others need to reproduce.  
2. Download with `scripts/download_gdrive.py`.  
3. Run `scripts/organize_dataset.py` to map to `data/real/raw` and `data/real/masks`.

### If your files have different names/patterns
Use pattern flags in `organize_dataset.py`, e.g.:
```bash
python scripts/organize_dataset.py --src data/drive_raw --dst data/real   --raw-pattern "*Bscan*.png" --mask-pattern "*mask*.png"
```

## PHI / IRB
- Do **not** store PHI or identifiable information in this repo.  
- Keep raw data out of git (see `.gitignore`).  
- When in doubt, use the provided **synthetic** examples or obtain PI/IRB approval.

## Quick validation
```bash
python scripts/validate_dataset.py --root data/real
```
This checks counts and prints a few sample pairs. For visual sanity checks, open `notebooks/03_real_data_quickcheck.ipynb`.

import argparse
from pathlib import Path
import random, imageio.v2 as imageio

def main():
    ap = argparse.ArgumentParser(description="Sanity-check data/real/[raw|masks] pairs.")
    ap.add_argument("--root", default="data/real", help="root containing raw/ and masks/")
    ap.add_argument("--samples", type=int, default=5, help="print a few sample pairs")
    args = ap.parse_args()

    raw_dir = Path(args.root, "raw")
    mask_dir = Path(args.root, "masks")
    raws = sorted([p for p in raw_dir.glob("*") if p.suffix.lower() in [".png", ".tif", ".tiff"]])
    masks = sorted([p for p in mask_dir.glob("*.png")])

    raw_set = {p.stem for p in raws}
    mask_set = {p.stem for p in masks}
    both = sorted(raw_set & mask_set)
    raw_only = sorted(raw_set - mask_set)
    mask_only = sorted(mask_set - raw_set)

    print(f"Total raw: {len(raws)} | masks: {len(masks)} | paired: {len(both)}")
    if raw_only:
        print(f"Raw without mask: {len(raw_only)} (e.g., {raw_only[:5]})")
    if mask_only:
        print(f"Masks without raw: {len(mask_only)} (e.g., {mask_only[:5]})")

    # Read a few to ensure they load
    for name in random.sample(both, min(args.samples, len(both))):
        r = next(p for p in raws if p.stem == name)
        m = next(p for p in masks if p.stem == name)
        img = imageio.imread(r)
        msk = imageio.imread(m)
        print(f"OK: {name} | raw shape {getattr(img,'shape',None)} | mask shape {getattr(msk,'shape',None)}")

if __name__ == "__main__":
    main()

import argparse, shutil
from pathlib import Path
import fnmatch

def collect_files(root, pattern):
    root = Path(root)
    return [p for p in root.rglob("*") if p.is_file() and fnmatch.fnmatch(p.name, pattern)]

def main():
    ap = argparse.ArgumentParser(description="Organize Google Drive dump into data/real/raw and data/real/masks")
    ap.add_argument("--src", required=True, help="source directory (downloaded folder)")
    ap.add_argument("--dst", required=True, help="destination root (e.g., data/real)")
    ap.add_argument("--raw-pattern", default="*.png", help="glob for raw OCT images (*.png/*.tif)")
    ap.add_argument("--mask-pattern", default="*mask*.png", help="glob for masks (*mask*.png)")
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    raw_dir = dst / "raw"
    mask_dir = dst / "masks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    raw_files = collect_files(src, args.raw_pattern)
    mask_files = collect_files(src, args.mask_pattern)

    # index masks by basename (without extension) to find matches
    mask_index = {m.stem: m for m in mask_files}

    copied = 0
    for rf in raw_files:
        basename = rf.stem
        # try exact match; otherwise try some common variants
        candidates = [
            basename,
            basename.replace("_raw","").replace("-raw",""),
            basename + "_mask",
            basename + "-mask",
            basename + "_label"
        ]
        mf = None
        for c in candidates:
            if c in mask_index:
                mf = mask_index[c]
                break
        if mf is None:
            # no mask; still copy raw for visibility
            target_raw = raw_dir / rf.name
            (shutil.move if args.move else shutil.copy2)(rf, target_raw)
            continue

        target_raw = raw_dir / (basename + rf.suffix)
        target_mask = mask_dir / (basename + ".png")
        (shutil.move if args.move else shutil.copy2)(rf, target_raw)
        (shutil.move if args.move else shutil.copy2)(mf, target_mask)
        copied += 1

    print(f"[ok] Organized dataset at {dst}. Paired samples: {copied}. Raw-only files may exist if masks were missing or patterns mismatched.")
    print("Tip: Adjust --raw-pattern/--mask-pattern or rename files if pairs were missed.")

if __name__ == "__main__":
    main()

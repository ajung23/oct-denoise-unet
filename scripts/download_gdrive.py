import argparse, os, sys
from pathlib import Path

def main():
    try:
        import gdown
    except ImportError:
        print("Please install gdown: pip install gdown", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Download a public Google Drive folder into a local directory.")
    ap.add_argument("--folder-id", required=True, help="Google Drive folder ID")
    ap.add_argument("--out", default="data/drive_raw", help="Output directory")
    args = ap.parse_args()

    url = f"https://drive.google.com/drive/folders/{args.folder_id}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[gdown] Downloading folder {url} -> {out}")
    gdown.download_folder(url=url, output=str(out), quiet=False, use_cookies=False)
    print("[ok] Download complete.")

if __name__ == "__main__":
    main()

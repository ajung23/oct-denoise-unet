import argparse, os, numpy as np, imageio.v2 as imageio
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def main(args):
    preds = sorted(Path(args.pred).glob("*.png"))
    gts   = sorted(Path(args.gt).glob("*.png"))
    if len(preds) != len(gts):
        print("Warning: length mismatch")
    n = min(len(preds), len(gts))
    psnrs, ssimv = [], []
    for i in range(n):
        p = imageio.imread(preds[i])
        g = imageio.imread(gts[i])
        psnrs.append(psnr(g, p, data_range=255))
        ssimv.append(ssim(g, p, data_range=255))
    print(f"PSNR: {np.mean(psnrs):.2f} dB, SSIM: {np.mean(ssimv):.3f} over {n} images")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    main(args)

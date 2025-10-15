import argparse, os, numpy as np, imageio.v2 as imageio
from pathlib import Path

def synth_bscan(h=256, w=512, layers=4, noise=0.35, seed=None):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)+0.2
    # Draw sinusoidal layers
    ys = np.linspace(0, 1, w, dtype=np.float32)
    for i in range(layers):
        y0 = (i+1)*h/(layers+1) + 6*np.sin(ys*2*np.pi*(i+1)*0.2 + rng.uniform(0, 2*np.pi))
        thickness = rng.integers(3, 8)
        for t in range(-thickness, thickness):
            yy = np.clip((y0+t).astype(int), 0, h-1)
            img[yy, np.arange(w)] = 0.6 + 0.1*rng.standard_normal(w)
    # Smooth-ish background texture
    img += 0.1*rng.standard_normal((h,w)).astype(np.float32)
    # Normalize to [0,1]
    img = (img - img.min())/(img.max()-img.min()+1e-6)
    # Speckle-ish multiplicative noise
    speckle = rng.gamma(shape=1.5, scale=1.0, size=(h,w)).astype(np.float32)
    speckle = speckle/np.max(speckle)
    noisy = np.clip(img * (1.0 - noise + noise*speckle), 0, 1)
    # Create a simple mask for one band
    mask = np.zeros_like(noisy)
    band = (img > 0.62).astype(np.uint8) * 255
    return (noisy*255).astype(np.uint8), band

def main(args):
    out = Path(args.out)
    (out/"raw").mkdir(parents=True, exist_ok=True)
    (out/"masks").mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        noisy, mask = synth_bscan(seed=i)
        imageio.imwrite(out/"raw"/f"oct_{i:03d}.png", noisy)
        imageio.imwrite(out/"masks"/f"oct_{i:03d}.png", mask)
    print(f"[OK] Wrote {args.n} synthetic pairs to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    main(args)

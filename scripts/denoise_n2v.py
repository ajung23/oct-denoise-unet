import argparse, os, numpy as np, imageio.v2 as imageio
from pathlib import Path
from skimage.util import img_as_float32
from n2v.models import N2VConfig, N2V
from csbdeep.utils import normalize
from tifffile import imread, imwrite

def load_stack(folder):
    imgs = []
    for p in sorted(Path(folder).glob("*.png")):
        imgs.append(imageio.imread(p))
    if not imgs:
        for p in sorted(Path(folder).glob("*.tif*")):
            arr = imread(str(p))
            if arr.ndim == 3:
                for i in range(arr.shape[0]):
                    imgs.append(arr[i])
            else:
                imgs.append(arr)
    x = np.stack(imgs, 0).astype(np.float32)
    x = normalize(x, 1, 99.8, axis=(1,2))
    return x

def main(args):
    os.makedirs(args.out, exist_ok=True)
    x = load_stack(args.data)
    # Simple 2D training (tiny epochs for demo)
    conf = N2VConfig(x, unet_kern_size=3, train_steps_per_epoch=50, train_epochs=args.epochs,
                     train_loss='mse', batch_norm=True, train_batch_size=8, n2v_perc_pix=0.198)
    model = N2V(conf, 'n2v_oct_demo', basedir=args.out)
    model.train(x, x, validation_split=0.1)
    # Predict denoised stack
    den = model.predict(x, axes='YXC' if x.ndim==4 else 'YX')
    # Save
    den_dir = os.path.join(args.out, "denoised")
    os.makedirs(den_dir, exist_ok=True)
    for i, img in enumerate(den):
        outp = os.path.join(den_dir, f"den_{i:03d}.png")
        imageio.imwrite(outp, (np.clip(img,0,1)*255).astype(np.uint8))
    print(f"[OK] Saved {len(den)} denoised frames to {den_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder with raw noisy OCT images (.png/.tif)")
    ap.add_argument("--out", default="runs/n2v_demo", help="Output folder")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    main(args)

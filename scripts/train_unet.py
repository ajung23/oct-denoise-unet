import argparse, os, numpy as np, imageio.v2 as imageio
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor

class UNetMini(nn.Module):
    # Small UNet for demo purposes
    def __init__(self, in_ch=1, out_ch=1, base=16):
        super().__init__()
        def block(cin, cout): 
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(True),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(True))
        self.enc1 = block(in_ch, base)
        self.enc2 = block(base, base*2); self.pool1 = nn.MaxPool2d(2)
        self.enc3 = block(base*2, base*4); self.pool2 = nn.MaxPool2d(2)
        self.bott = block(base*4, base*8)
        self.up2 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec2 = block(base*8, base*4)
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec1 = block(base*4, base*2)
        self.outc = nn.Conv2d(base*2, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bott(self.pool2(e3))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e3], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        y  = self.outc(d1)
        return y

class OCTSet(Dataset):
    def __init__(self, root):
        self.imgs = sorted(list(Path(root, "raw").glob("*.png")))
        self.masks = [Path(root, "masks", p.name) for p in self.imgs]
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        x = imageio.imread(self.imgs[i])
        m = imageio.imread(self.masks[i])
        x = to_tensor(x)[0:1]  # grayscale
        m = (to_tensor(m)[0:1] > 0.5).float()
        return x, m

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred*target).sum()
    denom = pred.sum()+target.sum()+eps
    return 1. - (2.*inter/denom)

def main(args):
    os.makedirs(args.out, exist_ok=True)
    ds = OCTSet(args.data)
    if len(ds) == 0:
        raise SystemExit("No data found. Expected data/raw/*.png and data/masks/*.png")
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetMini().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for x, m in dl:
            x, m = x.to(device), m.to(device)
            y = model(x)
            loss = dice_loss(y, m)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*x.size(0)
        print(f"Epoch {epoch+1}: loss={tot/len(ds):.4f}")
    # Save preds
    pred_dir = os.path.join(args.out, "preds"); os.makedirs(pred_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (x, m) in enumerate(DataLoader(ds, batch_size=1)):
            x = x.to(device)
            y = torch.sigmoid(model(x)).cpu().numpy()[0,0]
            imageio.imwrite(os.path.join(pred_dir, f"pred_{i:03d}.png"), (y*255).astype(np.uint8))
    torch.save(model.state_dict(), os.path.join(args.out, "unet_mini.pt"))
    print("[OK] Saved preds and model.")
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Root folder with raw/ and masks/ subfolders")
    ap.add_argument("--out", default="runs/unet_demo", help="Output folder")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    main(args)

# Assumes:
#   /content/images   (folder with test images)
#   /content/best.pt  (checkpoint)
#
# 1) Defines TinyCurveCCMNet
# 2) Loads checkpoint robustly
# 3) Predicts params on a small resized image (fast)


import os, glob
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

CKPT = "/content/global_best_phys_Curve.pt"
IN_DIR = "/content/images"
OUT_DIR = "/content/out_fullres"
os.makedirs(OUT_DIR, exist_ok=True)

# Params are predicted on this size (speed). Output stays full-res.
INFER_SIZE = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# Model definition (Curve + CCM)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, stride=s, padding=p, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return F.silu(x)

class TinyCurveCCMNet(nn.Module):
    def __init__(self,
                 gain_min=0.6, gain_max=2.0,
                 gamma_min=0.6, gamma_max=2.2,
                 ccm_strength=0.30,
                 bias_strength=0.06):
        super().__init__()
        self.gain_min, self.gain_max = gain_min, gain_max
        self.gamma_min, self.gamma_max = gamma_min, gamma_max
        self.ccm_strength = ccm_strength
        self.bias_strength = bias_strength

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )
        self.b1 = DepthwiseSeparableConv(16, 24, s=2)
        self.b2 = DepthwiseSeparableConv(24, 32, s=2)
        self.b3 = DepthwiseSeparableConv(32, 48, s=2)
        self.b4 = DepthwiseSeparableConv(48, 64, s=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 48),
            nn.SiLU(),
            nn.Linear(48, 18)  # 3 gain + 3 gamma + 9 ccm + 3 bias
        )

    def _map_range(self, x, lo, hi):
        x = torch.sigmoid(x)
        return lo + (hi - lo) * x

    def forward(self, x):
        feat = self.stem(x)
        feat = self.b1(feat)
        feat = self.b2(feat)
        feat = self.b3(feat)
        feat = self.b4(feat)

        p = self.head(self.pool(feat))  # (B,18)

        gain  = self._map_range(p[:, 0:3], self.gain_min, self.gain_max).view(-1,3,1,1)
        gamma = self._map_range(p[:, 3:6], self.gamma_min, self.gamma_max).view(-1,3,1,1)

        R = torch.tanh(p[:, 6:15]).view(-1,3,3) * self.ccm_strength
        I = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).expand(R.size(0), -1, -1)
        M = I + R

        b = torch.tanh(p[:, 15:18]).view(-1,3,1,1) * self.bias_strength

        # (We won't use y here for final full-res; we apply params to full-res instead)
        return gain, gamma, M, b

def load_state_dict_robust(model, ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu")
    state = None
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "net", "ema", "weights"]:
            if k in obj and isinstance(obj[k], (dict, torch.nn.Module)):
                state = obj[k].state_dict() if isinstance(obj[k], torch.nn.Module) else obj[k]
                break
        if state is None and all(isinstance(v, torch.Tensor) for v in obj.values()):
            state = obj
    elif isinstance(obj, torch.nn.Module):
        state = obj.state_dict()

    if state is None:
        raise RuntimeError("Could not find state_dict in checkpoint.")

    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Loaded ckpt:", ckpt_path)
    if missing:
        print("  Missing keys:", missing[:10], ("..." if len(missing) > 10 else ""))
    if unexpected:
        print("  Unexpected keys:", unexpected[:10], ("..." if len(unexpected) > 10 else ""))

model = TinyCurveCCMNet().to(device).eval()
load_state_dict_robust(model, CKPT)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def pil_to_tensor_rgb01(img_pil):
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # (1,3,H,W)
    return t

def tensor_to_pil_rgb01(t):
    if t.ndim == 4:
        t = t[0]
    t = torch.clamp(t, 0, 1).detach().cpu()
    arr = (t.permute(1,2,0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)

# Apply enhancement params to ANY resolution image tensor x_full (1,3,H,W)
def apply_params_fullres(x_full, gain, gamma, M, b):
    # curve
    z = torch.clamp(gain * x_full, 0.0, 1.0)
    z = torch.pow(z + 1e-6, gamma)

    # CCM: (B,3,3) x (B,3,HW)
    B, C, H, W = z.shape
    z_flat = z.view(B, 3, -1)
    y_flat = torch.bmm(M, z_flat)
    y = y_flat.view(B, 3, H, W) + b
    y = torch.clamp(y, 0.0, 1.0)
    return y


# Run inference on folder

img_paths = []
for ext in IMG_EXTS:
    img_paths += glob.glob(os.path.join(IN_DIR, f"*{ext}"))
img_paths = sorted(img_paths)

if not img_paths:
    raise RuntimeError(f"No images found in {IN_DIR}.")

print(f"Found {len(img_paths)} images.")

saved = []
with torch.no_grad():
    for pth in img_paths:
        img = Image.open(pth).convert("RGB")

        # full-res tensor
        x_full = pil_to_tensor_rgb01(img).to(device)

        # small tensor for param prediction
        img_small = img.resize((INFER_SIZE, INFER_SIZE), Image.BILINEAR)
        x_small = pil_to_tensor_rgb01(img_small).to(device)

        # predict params on small
        gain, gamma, M, b = model(x_small)

        # apply params to full-res
        y_full = apply_params_fullres(x_full, gain, gamma, M, b)

        out = tensor_to_pil_rgb01(y_full)
        out_path = os.path.join(OUT_DIR, Path(pth).stem + "_enh_fullres.png")
        out.save(out_path)  # PNG keeps quality (lossless)
        saved.append(out_path)

print("Saved full-res outputs to:", OUT_DIR)
print("Example:", saved[0])

# Preview grid

n_show = min(9, len(img_paths))
plt.figure(figsize=(4, 4*n_show))
for i in range(n_show):
    inp = Image.open(img_paths[i]).convert("RGB")
    out = Image.open(saved[i]).convert("RGB")

    ax1 = plt.subplot(n_show, 2, 2*i+2)
    ax1.imshow(inp); ax1.set_title(f"Input: {Path(img_paths[i]).name}"); ax1.axis("off")

    ax2 = plt.subplot(n_show, 2, 2*i+2)
    ax2.imshow(out); ax2.set_title("Output"); ax2.axis("off")

plt.tight_layout()
plt.show()

print("Done")

#!/usr/bin/env bash
#SBATCH --job-name=tinycurve_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=120:00:00
#SBATCH --output=/home/shahid-ahamed.hasib/Downloads/LiteEnhanceNet-main/slurm_tinycurve_ablation_%j.out
#SBATCH --error=/home/shahid-ahamed.hasib/Downloads/LiteEnhanceNet-main/slurm_tinycurve_ablation_%j.err

set -euo pipefail

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "============================================================"

PROJECT_ROOT="/home/shahid-ahamed.hasib/Downloads/LiteEnhanceNet-main"
SRC_EXP="$PROJECT_ROOT/experiments/phys_curve_prog5"

ABLATION_NAME="tinycurvenet_ccm_ablation"
ABLATION_DIR="$PROJECT_ROOT/experiments/$ABLATION_NAME"
PYTHON_BIN="/home/shahid-ahamed.hasib/venvs/uie_gpu/bin/python3"

mkdir -p "$ABLATION_DIR"
mkdir -p "$ABLATION_DIR/src"
mkdir -p "$ABLATION_DIR/checkpoints"
mkdir -p "$ABLATION_DIR/logs"
mkdir -p "$ABLATION_DIR/outputs"

echo
echo "========== Python / CUDA check =========="
"$PYTHON_BIN" - <<'PY'
import sys
import torch
print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
PY

echo
echo "========== Copy helper files =========="
cp "$SRC_EXP/src/dataset.py" "$ABLATION_DIR/src/dataset.py"
cp "$SRC_EXP/src/vis.py" "$ABLATION_DIR/src/vis.py" || true

echo
echo "========== Write ablation training script =========="
cat > "$ABLATION_DIR/src/run_tinycurve_ablation.py" <<'PY'
import os
import csv
import math
import time
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pyiqa

from dataset import PairedImageFolder


# ============================================================
# User settings
# ============================================================

SEED = 42

TRAIN_LOW_DIR = "/home/shahid-ahamed.hasib/Downloads/Data/UIEB_splits/train/low"
TRAIN_HIGH_DIR = "/home/shahid-ahamed.hasib/Downloads/Data/UIEB_splits/train/high"
VAL_LOW_DIR = "/home/shahid-ahamed.hasib/Downloads/Data/UIEB_splits/val/low"
VAL_HIGH_DIR = "/home/shahid-ahamed.hasib/Downloads/Data/UIEB_splits/val/high"

CROP = 384
BATCH_SIZE = 8
NUM_WORKERS = 4

EPOCHS = 80
LR = 2e-4
WEIGHT_DECAY = 0.0
AMP = True
GRAD_CLIP = 0.5
LOG_EVERY = 25

OUT_ROOT = Path("/home/shahid-ahamed.hasib/Downloads/LiteEnhanceNet-main/experiments/tinycurvenet_ccm_ablation")
CKPT_ROOT = OUT_ROOT / "checkpoints"
LOG_ROOT = OUT_ROOT / "logs"
TABLE_CSV = OUT_ROOT / "ablation_results.csv"
TABLE_MD = OUT_ROOT / "ablation_results.md"
TABLE_TEX = OUT_ROOT / "ablation_results_latex.tex"


# ============================================================
# Variants
# ============================================================

VARIANTS = [
    {
        "name": "only_gain_gamma",
        "display": "Only gain and gamma",
        "gain": True,
        "gamma": True,
        "ccm": False,
        "bias": False,
        "loss_type": "l1",
        "interpretation": "Limited correction capacity",
    },
    {
        "name": "without_gain",
        "display": "Without gain",
        "gain": False,
        "gamma": True,
        "ccm": True,
        "bias": True,
        "loss_type": "l1",
        "interpretation": "Weaker channel intensity correction",
    },
    {
        "name": "without_gamma",
        "display": "Without gamma",
        "gain": True,
        "gamma": False,
        "ccm": True,
        "bias": True,
        "loss_type": "l1",
        "interpretation": "Reduced nonlinear contrast control",
    },
    {
        "name": "without_ccm",
        "display": "Without CCM",
        "gain": True,
        "gamma": True,
        "ccm": False,
        "bias": True,
        "loss_type": "l1",
        "interpretation": "Tests contribution of global color mixing",
    },
    {
        "name": "without_bias",
        "display": "Without bias",
        "gain": True,
        "gamma": True,
        "ccm": True,
        "bias": False,
        "loss_type": "l1",
        "interpretation": "Tests contribution of additive color offset",
    },
    {
        "name": "full_l1",
        "display": "Full TinyCurveNet-CCM L1",
        "gain": True,
        "gamma": True,
        "ccm": True,
        "bias": True,
        "loss_type": "l1",
        "interpretation": "Full model trained with L1 reconstruction loss",
    },
    {
        "name": "full_mse",
        "display": "Full TinyCurveNet-CCM L2/MSE",
        "gain": True,
        "gamma": True,
        "ccm": True,
        "bias": True,
        "loss_type": "mse",
        "interpretation": "Full model trained with MSE for PSNR-oriented fidelity",
    },
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Model
# ============================================================

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


class TinyCurveCCMNetAblation(nn.Module):
    def __init__(
        self,
        use_gain=True,
        use_gamma=True,
        use_ccm=True,
        use_bias=True,
        gain_min=0.6,
        gain_max=2.0,
        gamma_min=0.6,
        gamma_max=2.2,
        ccm_strength=0.30,
        bias_strength=0.06,
    ):
        super().__init__()

        self.use_gain = bool(use_gain)
        self.use_gamma = bool(use_gamma)
        self.use_ccm = bool(use_ccm)
        self.use_bias = bool(use_bias)

        self.gain_min = gain_min
        self.gain_max = gain_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
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
            nn.Linear(48, 18),
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

        p = self.head(self.pool(feat))

        B, _, H, W = x.shape

        if self.use_gain:
            gain = self._map_range(
                p[:, 0:3],
                self.gain_min,
                self.gain_max,
            ).view(-1, 3, 1, 1)
        else:
            gain = torch.ones(B, 3, 1, 1, device=x.device, dtype=x.dtype)

        if self.use_gamma:
            gamma = self._map_range(
                p[:, 3:6],
                self.gamma_min,
                self.gamma_max,
            ).view(-1, 3, 1, 1)
        else:
            gamma = torch.ones(B, 3, 1, 1, device=x.device, dtype=x.dtype)

        eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)

        if self.use_ccm:
            R = torch.tanh(p[:, 6:15]).view(-1, 3, 3) * self.ccm_strength
            M = eye + R
        else:
            M = eye

        if self.use_bias:
            b = torch.tanh(p[:, 15:18]).view(-1, 3, 1, 1) * self.bias_strength
        else:
            b = torch.zeros(B, 3, 1, 1, device=x.device, dtype=x.dtype)

        z = torch.clamp(gain * x, 0.0, 1.0)
        z = torch.pow(z + 1e-6, gamma)

        zf = z.view(B, 3, -1)
        yf = torch.bmm(M, zf)
        y = yf.view(B, 3, H, W) + b
        y = torch.clamp(y, 0.0, 1.0)

        return y, gain, gamma, M, b


# ============================================================
# Metrics
# ============================================================

class Metrics:
    def __init__(self, device):
        self.device = device
        self.psnr = pyiqa.create_metric("psnr", device=device)
        self.ssim = pyiqa.create_metric("ssim", device=device)
        self.ms_ssim = pyiqa.create_metric("ms_ssim", device=device)
        self.lpips = pyiqa.create_metric("lpips", device=device)

    @torch.no_grad()
    def batch(self, pred, target):
        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        return {
            "PSNR": float(self.psnr(pred, target).mean().item()),
            "SSIM": float(self.ssim(pred, target).mean().item()),
            "MS-SSIM": float(self.ms_ssim(pred, target).mean().item()),
            "LPIPS": float(self.lpips(pred, target).mean().item()),
        }


# ============================================================
# Data
# ============================================================

def make_loaders():
    train_ds = PairedImageFolder(
        TRAIN_LOW_DIR,
        TRAIN_HIGH_DIR,
        img_size=CROP,
        train=True,
    )

    val_ds = PairedImageFolder(
        VAL_LOW_DIR,
        VAL_HIGH_DIR,
        img_size=CROP,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


# ============================================================
# Train / Validate
# ============================================================

def save_ckpt(path, model, opt, variant, epoch, best_score, best_metrics):
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "variant": variant,
            "epoch": epoch,
            "best_score": best_score,
            "best_metrics": best_metrics,
        },
        path,
    )


@torch.no_grad()
def validate(model, loader, device, metrics):
    model.eval()

    l1_fn = nn.L1Loss(reduction="mean")
    mse_fn = nn.MSELoss(reduction="mean")

    sums = defaultdict(float)
    total_images = 0

    for low, high, _ in loader:
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)

        pred, gain, gamma, M, b = model(low)

        bs = low.size(0)

        loss_l1 = float(l1_fn(pred, high).item())
        loss_mse = float(mse_fn(pred, high).item())
        mets = metrics.batch(pred, high)

        sums["L1"] += loss_l1 * bs
        sums["MSE"] += loss_mse * bs
        for k, v in mets.items():
            sums[k] += float(v) * bs

        total_images += bs

    for k in list(sums.keys()):
        sums[k] /= max(total_images, 1)

    return dict(sums)


def train_one_variant(variant, train_loader, val_loader, device, metrics):
    print("\n" + "=" * 80)
    print("VARIANT:", variant["display"])
    print("=" * 80)

    set_seed(SEED)

    model = TinyCurveCCMNetAblation(
        use_gain=variant["gain"],
        use_gamma=variant["gamma"],
        use_ccm=variant["ccm"],
        use_bias=variant["bias"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=AMP and device.type == "cuda")

    l1_fn = nn.L1Loss()
    mse_fn = nn.MSELoss()

    variant_dir = CKPT_ROOT / variant["name"]
    variant_dir.mkdir(parents=True, exist_ok=True)

    log_csv = LOG_ROOT / f"{variant['name']}_epochs.csv"

    best_score = float("inf")
    best_metrics = None
    best_epoch = -1

    if variant["loss_type"] == "mse":
        monitor_key = "MSE"
    else:
        monitor_key = "L1"

    for epoch in range(1, EPOCHS + 1):
        model.train()

        t0 = time.time()
        run_loss = 0.0
        steps = 0

        for step, (low, high, _) in enumerate(train_loader, start=1):
            low = low.to(device, non_blocking=True)
            high = high.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred, gain, gamma, M, b = model(low)

                if variant["loss_type"] == "mse":
                    loss = mse_fn(pred, high)
                else:
                    loss = l1_fn(pred, high)

            scaler.scale(loss).backward()

            if GRAD_CLIP > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            scaler.step(opt)
            scaler.update()

            run_loss += float(loss.detach().cpu())
            steps += 1

            if step % LOG_EVERY == 0:
                print(
                    f"[{variant['name']} | epoch {epoch:03d} | step {step:04d}] "
                    f"train_loss={run_loss / max(steps, 1):.6f}"
                )

        val = validate(model, val_loader, device, metrics)
        dt = time.time() - t0

        row = {
            "variant": variant["name"],
            "epoch": epoch,
            "train_loss": run_loss / max(steps, 1),
            "val_L1": val["L1"],
            "val_MSE": val["MSE"],
            "val_PSNR": val["PSNR"],
            "val_SSIM": val["SSIM"],
            "val_MS_SSIM": val["MS-SSIM"],
            "val_LPIPS": val["LPIPS"],
            "time_sec": dt,
        }

        write_header = not log_csv.exists()
        with open(log_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

        print(
            f"{variant['display']} | Epoch {epoch:03d}/{EPOCHS} | "
            f"L1={val['L1']:.6f} | MSE={val['MSE']:.6f} | "
            f"PSNR={val['PSNR']:.4f} | SSIM={val['SSIM']:.4f} | "
            f"MS-SSIM={val['MS-SSIM']:.4f} | LPIPS={val['LPIPS']:.4f} | "
            f"time={dt:.1f}s"
        )

        score = val[monitor_key]

        if score < best_score:
            best_score = score
            best_metrics = val
            best_epoch = epoch

            save_ckpt(
                variant_dir / "best.pt",
                model,
                opt,
                variant,
                epoch,
                best_score,
                best_metrics,
            )

            print(
                f" saved best.pt for {variant['name']} | "
                f"monitor={monitor_key}={best_score:.6f} | epoch={epoch}"
            )

        save_ckpt(
            variant_dir / "latest.pt",
            model,
            opt,
            variant,
            epoch,
            best_score,
            best_metrics,
        )

    result = {
        "Variant": variant["display"],
        "Gain": "Yes" if variant["gain"] else "No",
        "Gamma": "Yes" if variant["gamma"] else "No",
        "CCM": "Yes" if variant["ccm"] else "No",
        "Bias": "Yes" if variant["bias"] else "No",
        "Loss": variant["loss_type"].upper(),
        "Best Epoch": best_epoch,
        "L1": best_metrics["L1"],
        "MSE": best_metrics["MSE"],
        "PSNR": best_metrics["PSNR"],
        "SSIM": best_metrics["SSIM"],
        "MS-SSIM": best_metrics["MS-SSIM"],
        "LPIPS": best_metrics["LPIPS"],
        "Interpretation": variant["interpretation"],
        "Checkpoint": str(variant_dir / "best.pt"),
    }

    return result


# ============================================================
# Tables
# ============================================================

def yesno_tex(v):
    return r"\checkmark" if v == "Yes" else r"--"


def write_summary_tables(results):
    # CSV
    fieldnames = [
        "Variant",
        "Gain",
        "Gamma",
        "CCM",
        "Bias",
        "Loss",
        "Best Epoch",
        "L1",
        "MSE",
        "PSNR",
        "SSIM",
        "MS-SSIM",
        "LPIPS",
        "Interpretation",
        "Checkpoint",
    ]

    with open(TABLE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    # Markdown
    with open(TABLE_MD, "w") as f:
        f.write("| Variant | Gain | Gamma | CCM | Bias | Loss | Best Epoch | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | LPIPS ↓ | Interpretation |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in results:
            f.write(
                f"| {r['Variant']} | {r['Gain']} | {r['Gamma']} | {r['CCM']} | {r['Bias']} | "
                f"{r['Loss']} | {r['Best Epoch']} | {r['PSNR']:.4f} | {r['SSIM']:.4f} | "
                f"{r['MS-SSIM']:.4f} | {r['LPIPS']:.4f} | {r['Interpretation']} |\n"
            )

    # LaTeX
    with open(TABLE_TEX, "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Ablation study of TinyCurveNet-CCM components on the validation set.}
\label{tab:tinycurvenet_ccm_ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c c c c c c c c c l}
\toprule
\textbf{Variant} & \textbf{Gain} & \textbf{Gamma} & \textbf{CCM} & \textbf{Bias} & \textbf{Loss} & \textbf{Epoch} & \textbf{PSNR} $\uparrow$ & \textbf{SSIM} $\uparrow$ & \textbf{LPIPS} $\downarrow$ & \textbf{Interpretation} \\
\midrule
""")

        for r in results:
            f.write(
                f"{r['Variant']} & {yesno_tex(r['Gain'])} & {yesno_tex(r['Gamma'])} & "
                f"{yesno_tex(r['CCM'])} & {yesno_tex(r['Bias'])} & {r['Loss']} & "
                f"{r['Best Epoch']} & {r['PSNR']:.4f} & {r['SSIM']:.4f} & "
                f"{r['LPIPS']:.4f} & {r['Interpretation']} \\\\\n"
            )

        f.write(r"""\bottomrule
\end{tabular}%
}
\end{table}
""")

    print("\nSaved summary files:")
    print("CSV:   ", TABLE_CSV)
    print("MD:    ", TABLE_MD)
    print("LaTeX: ", TABLE_TEX)


# ============================================================
# Main
# ============================================================

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Creating dataloaders...")
    train_loader, val_loader = make_loaders()

    print("Creating metrics...")
    metrics = Metrics(device=device)

    results = []

    for variant in VARIANTS:
        result = train_one_variant(
            variant,
            train_loader,
            val_loader,
            device,
            metrics,
        )

        results.append(result)
        write_summary_tables(results)

    print("\n" + "=" * 80)
    print("ABLATION STUDY FINISHED")
    print("=" * 80)

    for r in results:
        print(
            f"{r['Variant']} | "
            f"Gain={r['Gain']} Gamma={r['Gamma']} CCM={r['CCM']} Bias={r['Bias']} | "
            f"Loss={r['Loss']} | Epoch={r['Best Epoch']} | "
            f"PSNR={r['PSNR']:.4f} SSIM={r['SSIM']:.4f} "
            f"MS-SSIM={r['MS-SSIM']:.4f} LPIPS={r['LPIPS']:.4f}"
        )

    print("\nResults table:")
    print(TABLE_CSV)
    print(TABLE_MD)
    print(TABLE_TEX)


if __name__ == "__main__":
    main()
PY

echo
echo "========== Check script syntax =========="
"$PYTHON_BIN" -m py_compile "$ABLATION_DIR/src/run_tinycurve_ablation.py"
"$PYTHON_BIN" -m py_compile "$ABLATION_DIR/src/dataset.py"

echo
echo "========== Run TinyCurveNet-CCM ablation study =========="
cd "$ABLATION_DIR"

echo "PWD: $(pwd)"
echo "Script: $ABLATION_DIR/src/run_tinycurve_ablation.py"

"$PYTHON_BIN" "$ABLATION_DIR/src/run_tinycurve_ablation.py"

echo
echo "============================================================"
echo "Ablation job finished."
echo "End time: $(date)"
echo "============================================================"

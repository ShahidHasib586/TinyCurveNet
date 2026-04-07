#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/shahid-ahamed.hasib/Downloads/LiteEnhanceNet-main"
SRC_EXP="$PROJECT_ROOT/experiments/phys_curve_prog5"
SRC_CKPT="$SRC_EXP/checkpoints/global_best.pt"

NEW_RUN_NAME="phys_curve_prog5_stage3_continue"
NEW_RUN_DIR="$PROJECT_ROOT/experiments/$NEW_RUN_NAME"
NEW_CKPT_DIR="$NEW_RUN_DIR/checkpoints"
NEW_CFG="$NEW_RUN_DIR/config_stage3_continue.yaml"

PYTHON_BIN="/home/shahid-ahamed.hasib/venvs/uie_gpu/bin/python3"
TRAIN_PY="$PROJECT_ROOT/experiments/phys_curve_prog5/src/train_progressive.py"

mkdir -p "$NEW_RUN_DIR"
mkdir -p "$NEW_CKPT_DIR"

echo "========== STEP 1: Inspect source checkpoint =========="
"$PYTHON_BIN" - <<'PY'
import torch
p="/home/shahid-ahamed.hasib/Downloads/LiteEnhanceNet-main/experiments/phys_curve_prog5/checkpoints/global_best.pt"
ckpt=torch.load(p, map_location="cpu")
print("Checkpoint:", p)
print("Keys:", list(ckpt.keys()))
print("stage_idx:", ckpt.get("stage_idx"))
print("epoch_in_stage:", ckpt.get("epoch_in_stage"))
print("best_val:", ckpt.get("best_val"))
print("num model tensors:", len(ckpt["model"]) if "model" in ckpt else "N/A")
PY

echo
echo "========== STEP 2: Write Stage-3-only config =========="
cat > "$NEW_CFG" <<'YAML'
seed: 42

data:
  train_low_dir: "/home/shahid-ahamed.hasib/Downloads/Data/UIEB_splits/train/low"
  train_high_dir: "/home/shahid-ahamed.hasib/Downloads/Data/UIEB_splits/train/high"
  val_low_dir: "/home/shahid-ahamed.hasib/Downloads/Data/UIEB_splits/val/low"
  val_high_dir: "/home/shahid-ahamed.hasib/Downloads/Data/UIEB_splits/val/high"
  num_workers: 4

train:
  amp: true
  weight_decay: 0.0
  log_every: 25
  save_every: 1

progressive:
  crops: [384]
  epochs_per_stage: 800
  batch_sizes: [8]
  lrs: [0.0002]
  full_eval_every: 20

loss:
  w_l1: 1.0
  w_ssim: 0.20
  w_lpips: 0.05

model:
  gain_min: 0.6
  gain_max: 2.0
  gamma_min: 0.6
  gamma_max: 2.2
  ccm_strength: 0.30
  bias_strength: 0.06

runtime:
  device: "cuda"
YAML

echo "Wrote config to: $NEW_CFG"

echo
echo "========== STEP 3: Create patched latest.pt for new run =========="
"$PYTHON_BIN" - <<'PY'
import math
import torch
from pathlib import Path

src = Path("/home/shahid-ahamed.hasib/Downloads/LiteEnhanceNet-main/experiments/phys_curve_prog5/checkpoints/global_best.pt")
dst_dir = Path("/home/shahid-ahamed.hasib/Downloads/LiteEnhanceNet-main/experiments/phys_curve_prog5_stage3_continue/checkpoints")
dst_dir.mkdir(parents=True, exist_ok=True)

ckpt = torch.load(src, map_location="cpu")

new_ckpt = {}
new_ckpt["model"] = ckpt["model"]
new_ckpt["opt"] = ckpt.get("opt", {})
new_ckpt["stage_idx"] = 0
new_ckpt["epoch_in_stage"] = -1
new_ckpt["best_val"] = float("inf")

torch.save(new_ckpt, dst_dir / "latest.pt")

# optional convenience copies
torch.save(new_ckpt, dst_dir / "stage3_seed_latest_reset.pt")

print("Saved:", dst_dir / "latest.pt")
print("Patched values:")
print("  stage_idx =", new_ckpt["stage_idx"])
print("  epoch_in_stage =", new_ckpt["epoch_in_stage"])
print("  best_val =", new_ckpt["best_val"])
PY

echo
echo "========== STEP 4: Run training =========="
cd "$NEW_RUN_DIR"

echo "PWD: $(pwd)"
echo "Using train script: $TRAIN_PY"
echo "Using config: $NEW_CFG"

"$PYTHON_BIN" "$TRAIN_PY" --config "$NEW_CFG"

echo
echo "Training command exited normally."
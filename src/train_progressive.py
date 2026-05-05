import os
import csv
import time
import yaml
import random
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PairedImageFolder
from model import TinyCurveNet
from metrics import IQAMetrics
from vis import save_labeled_triptych

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_loaders(cfg, crop, batch_size):
    train_ds = PairedImageFolder(
        cfg["data"]["train_low_dir"],
        cfg["data"]["train_high_dir"],
        img_size=crop,
        train=True,
    )
    val_ds = PairedImageFolder(
        cfg["data"]["val_low_dir"],
        cfg["data"]["val_high_dir"],
        img_size=crop,
        train=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader

def save_ckpt(path, model, opt, stage_idx, epoch_in_stage, best_val):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "stage_idx": stage_idx,
        "epoch_in_stage": epoch_in_stage,
        "best_val": best_val,
    }, path)

def append_csv(path, row, header=None):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if (not exists) and header is not None:
            w.writerow(header)
        w.writerow(row)

@torch.no_grad()
def validate_core(model, loader, device, iqa, sample_path=None):
    model.eval()
    l1 = nn.L1Loss()

    sums = defaultdict(float)
    n = 0
    sample_saved = False

    for low, high, names in loader:
        low = low.to(device)
        high = high.to(device)

        pred, gain, gamma, M, b = model(low)
        loss_l1 = l1(pred, high).item()
        fr = iqa.fr_batch(pred, high)

        sums["L1"] += loss_l1
        sums["PSNR"] += fr["PSNR"]
        sums["SSIM"] += fr["SSIM"]
        sums["MS-SSIM"] += fr["MS-SSIM"]
        sums["LPIPS"] += fr["LPIPS"]
        n += 1

        if (sample_path is not None) and (not sample_saved):
            save_labeled_triptych(low[0], pred[0], high[0], sample_path)
            sample_saved = True

    for k in list(sums.keys()):
        sums[k] /= max(n, 1)
    return dict(sums)

@torch.no_grad()
def validate_full(model, loader, device, iqa):
    model.eval()
    sums = defaultdict(float)
    n = 0

    for low, high, names in loader:
        low = low.to(device)
        high = high.to(device)

        pred, gain, gamma, M, b = model(low)
        mets = iqa.full_batch(pred, high)

        for k, v in mets.items():
            sums[k] += v
        n += 1

    for k in list(sums.keys()):
        sums[k] /= max(n, 1)
    return dict(sums)

def print_full_metrics(tag, mets):
    print(
        f"{tag} | "
        f"PSNR↑={mets['PSNR']:.4f}  "
        f"SSIM↑={mets['SSIM']:.4f}  "
        f"MS-SSIM↑={mets['MS-SSIM']:.4f}  "
        f"LPIPS↓={mets['LPIPS']:.4f}  "
        f"NIQE↓={mets['NIQE']:.4f}  "
        f"UIQM↑={mets['UIQM']:.4f}  "
        f"UCIQE↑={mets['UCIQE']:.4f}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/progressive.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt_dir = Path("checkpoints")
    out_dir = Path("outputs")
    log_dir = Path("logs")
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    out_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    model = TinyCurveNet(
        gain_min=cfg["model"]["gain_min"],
        gain_max=cfg["model"]["gain_max"],
        gamma_min=cfg["model"]["gamma_min"],
        gamma_max=cfg["model"]["gamma_max"],
        ccm_strength=cfg["model"]["ccm_strength"],
        bias_strength=cfg["model"]["bias_strength"],
    ).to(device)

    iqa = IQAMetrics(device=device)
    l1 = nn.L1Loss()

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    latest_path = ckpt_dir / "latest.pt"
    start_stage = 0
    start_epoch = 1
    best_val = 1e9

    crops = cfg["progressive"]["crops"]
    batch_sizes = cfg["progressive"]["batch_sizes"]
    lrs = cfg["progressive"]["lrs"]
    epochs_per_stage = cfg["progressive"]["epochs_per_stage"]
    full_eval_every = cfg["progressive"]["full_eval_every"]

    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        start_stage = ckpt.get("stage_idx", 0)
        start_epoch = ckpt.get("epoch_in_stage", 0) + 1
        best_val = ckpt.get("best_val", 1e9)
        print(f"Resuming from stage {start_stage+1}, epoch {start_epoch}")

    epoch_csv = log_dir / "epoch_metrics.csv"
    stage_csv = log_dir / "stage_metrics.csv"

    for s_idx in range(start_stage, len(crops)):
        crop = crops[s_idx]
        batch_size = batch_sizes[s_idx]
        lr = lrs[s_idx]

        print(f"\n========== STAGE {s_idx+1}/{len(crops)} | crop={crop} | batch={batch_size} | lr={lr} ==========\n")

        train_loader, val_loader = make_loaders(cfg, crop, batch_size)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["train"]["weight_decay"])

        stage_best = 1e9
        stage_dir = out_dir / f"stage_{s_idx+1:02d}_crop_{crop}"
        stage_dir.mkdir(exist_ok=True, parents=True)

        ep0 = start_epoch if s_idx == start_stage else 1

        for epoch in range(ep0, epochs_per_stage + 1):
            model.train()
            t0 = time.time()

            run_loss = 0.0
            batches = 0

            for step, (low, high, _) in enumerate(train_loader, start=1):
                low = low.to(device, non_blocking=True)
                high = high.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    pred, gain, gamma, M, b = model(low)

                    loss_l1 = l1(pred, high)

                    fr = iqa.fr_batch(pred.detach(), high.detach())
                    lpips_val = torch.tensor(fr["LPIPS"], device=device)
                    ssim_val = torch.tensor(fr["SSIM"], device=device)

                    loss = (
                        cfg["loss"]["w_l1"] * loss_l1
                        + cfg["loss"]["w_ssim"] * (1.0 - ssim_val)
                        + cfg["loss"]["w_lpips"] * lpips_val
                    )

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                run_loss += float(loss.item())
                batches += 1

                if step % cfg["train"]["log_every"] == 0:
                    print(f"[stage {s_idx+1} | epoch {epoch} | step {step}] loss={run_loss/max(batches,1):.6f}")

            sample_path = stage_dir / f"epoch_{epoch:03d}_triptych.png"
            val_core = validate_core(model, val_loader, device, iqa, sample_path=str(sample_path))
            dt = time.time() - t0

            append_csv(
                str(epoch_csv),
                [
                    s_idx+1, crop, epoch, lr,
                    val_core["L1"], val_core["PSNR"], val_core["SSIM"],
                    val_core["MS-SSIM"], val_core["LPIPS"], dt
                ],
                header=["stage","crop","epoch","lr","val_L1","val_PSNR","val_SSIM","val_MS_SSIM","val_LPIPS","time_sec"]
            )

            print(
                f"Stage {s_idx+1} Epoch {epoch} | "
                f"L1={val_core['L1']:.6f}  "
                f"PSNR↑={val_core['PSNR']:.4f}  "
                f"SSIM↑={val_core['SSIM']:.4f}  "
                f"MS-SSIM↑={val_core['MS-SSIM']:.4f}  "
                f"LPIPS↓={val_core['LPIPS']:.4f}  "
                f"time={dt:.1f}s"
            )

            if val_core["L1"] < stage_best:
                stage_best = val_core["L1"]
                save_ckpt(ckpt_dir / f"stage_{s_idx+1:02d}_best.pt", model, opt, s_idx, epoch, best_val)
                print(f" saved stage best: stage_{s_idx+1:02d}_best.pt")

            if val_core["L1"] < best_val:
                best_val = val_core["L1"]
                save_ckpt(ckpt_dir / "global_best.pt", model, opt, s_idx, epoch, best_val)
                print(" saved global_best.pt")

            save_ckpt(latest_path, model, opt, s_idx, epoch, best_val)

            if (epoch % full_eval_every == 0) or (epoch == epochs_per_stage):
                full_m = validate_full(model, val_loader, device, iqa)
                print_full_metrics(f"[FULL EVAL] stage={s_idx+1} epoch={epoch}", full_m)
                append_csv(
                    str(stage_csv),
                    [
                        s_idx+1, crop, epoch,
                        full_m["PSNR"], full_m["SSIM"], full_m["MS-SSIM"],
                        full_m["LPIPS"], full_m["NIQE"], full_m["UIQM"], full_m["UCIQE"]
                    ],
                    header=["stage","crop","epoch","PSNR","SSIM","MS_SSIM","LPIPS","NIQE","UIQM","UCIQE"]
                )

        # load stage best before moving to next stage
        stage_best_ckpt = torch.load(ckpt_dir / f"stage_{s_idx+1:02d}_best.pt", map_location=device)
        model.load_state_dict(stage_best_ckpt["model"], strict=True)
        start_epoch = 1

    # final report from global best
    final_ckpt = torch.load(ckpt_dir / "global_best.pt", map_location=device)
    model.load_state_dict(final_ckpt["model"], strict=True)

    final_crop = crops[-1]
    final_bs = batch_sizes[-1]
    _, val_loader = make_loaders(cfg, final_crop, final_bs)
    final_m = validate_full(model, val_loader, device, iqa)
    print_full_metrics("[FINAL GLOBAL BEST]", final_m)

    print("\nDone.")
    print("Best model:", ckpt_dir / "global_best.pt")

if __name__ == "__main__":
    main()

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
from losses import TinyCurveLoss


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
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "stage_idx": stage_idx,
            "epoch_in_stage": epoch_in_stage,
            "best_val": best_val,
        },
        path,
    )


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

    l1 = nn.L1Loss(reduction="sum")

    sums = defaultdict(float)
    total_images = 0
    sample_saved = False

    for low, high, names in loader:
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)

        pred, gain, gamma, M, b = model(low)

        batch_size = low.size(0)

        loss_l1 = l1(pred, high).item() / batch_size
        fr = iqa.fr_batch(pred, high)

        sums["L1"] += loss_l1 * batch_size
        sums["PSNR"] += fr["PSNR"] * batch_size
        sums["SSIM"] += fr["SSIM"] * batch_size
        sums["MS-SSIM"] += fr["MS-SSIM"] * batch_size
        sums["LPIPS"] += fr["LPIPS"] * batch_size

        total_images += batch_size

        if (sample_path is not None) and (not sample_saved):
            save_labeled_triptych(low[0], pred[0], high[0], sample_path)
            sample_saved = True

    for k in list(sums.keys()):
        sums[k] /= max(total_images, 1)

    return dict(sums)


@torch.no_grad()
def validate_full(model, loader, device, iqa):
    model.eval()

    sums = defaultdict(float)
    total_images = 0

    for low, high, names in loader:
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)

        pred, gain, gamma, M, b = model(low)

        batch_size = low.size(0)
        mets = iqa.full_batch(pred, high)

        for k, v in mets.items():
            sums[k] += v * batch_size

        total_images += batch_size

    for k in list(sums.keys()):
        sums[k] /= max(total_images, 1)

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

    device = torch.device(
        cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )

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

    criterion = TinyCurveLoss(
        w_l1=cfg["loss"].get("w_l1", 1.0),
        w_ssim=cfg["loss"].get("w_ssim", 0.20),
        w_msssim=cfg["loss"].get("w_msssim", 0.0),
        w_lpips=cfg["loss"].get("w_lpips", 0.05),
        lpips_net=cfg["loss"].get("lpips_net", "vgg"),
        use_lpips=cfg["loss"].get("use_lpips", True),
    ).to(device)

    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(cfg["train"]["amp"]) and device.type == "cuda"
    )

    latest_path = ckpt_dir / "latest.pt"

    start_stage = 0
    start_epoch = 1
    best_val = 1e9

    crops = cfg["progressive"]["crops"]
    batch_sizes = cfg["progressive"]["batch_sizes"]
    lrs = cfg["progressive"]["lrs"]
    epochs_per_stage = cfg["progressive"]["epochs_per_stage"]
    full_eval_every = cfg["progressive"]["full_eval_every"]

    resume_opt_state = None

    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device)

        model.load_state_dict(ckpt["model"], strict=True)
        resume_opt_state = ckpt.get("opt", None)

        start_stage = ckpt.get("stage_idx", 0)
        start_epoch = ckpt.get("epoch_in_stage", 0) + 1
        best_val = ckpt.get("best_val", 1e9)

        print(f"Resuming from stage {start_stage + 1}, epoch {start_epoch}")

    epoch_csv = log_dir / "epoch_metrics.csv"
    stage_csv = log_dir / "stage_metrics.csv"

    for s_idx in range(start_stage, len(crops)):
        crop = crops[s_idx]
        batch_size = batch_sizes[s_idx]
        lr = lrs[s_idx]

        print(
            f"\n========== STAGE {s_idx + 1}/{len(crops)} | "
            f"crop={crop} | batch={batch_size} | lr={lr} ==========\n"
        )

        train_loader, val_loader = make_loaders(cfg, crop, batch_size)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=cfg["train"]["weight_decay"],
        )

        if resume_opt_state is not None and s_idx == start_stage:
            try:
                opt.load_state_dict(resume_opt_state)
                print("Loaded optimizer state from latest checkpoint.")
            except Exception as e:
                print(f"Could not load optimizer state. Starting optimizer fresh. Reason: {e}")

        resume_opt_state = None

        stage_best = 1e9

        stage_dir = out_dir / f"stage_{s_idx + 1:02d}_crop_{crop}"
        stage_dir.mkdir(exist_ok=True, parents=True)

        ep0 = start_epoch if s_idx == start_stage else 1

        for epoch in range(ep0, epochs_per_stage + 1):
            model.train()

            t0 = time.time()

            run_loss = 0.0
            run_l1 = 0.0
            run_ssim = 0.0
            run_msssim = 0.0
            run_lpips = 0.0
            batches = 0

            for step, (low, high, _) in enumerate(train_loader, start=1):
                low = low.to(device, non_blocking=True)
                high = high.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    pred, gain, gamma, M, b = model(low)

                    # Correct differentiable loss.
                    # No pred.detach() here.
                    # No metric-to-tensor conversion here.
                    loss, loss_items = criterion(pred, high)

                scaler.scale(loss).backward()

                if cfg["train"].get("grad_clip", 0.0) > 0.0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(cfg["train"]["grad_clip"]),
                    )

                scaler.step(opt)
                scaler.update()

                run_loss += float(loss.detach().cpu())
                run_l1 += loss_items["l1"]
                run_ssim += loss_items["ssim_loss"]
                run_msssim += loss_items["msssim_loss"]
                run_lpips += loss_items["lpips_loss"]
                batches += 1

                if step % cfg["train"]["log_every"] == 0:
                    denom = max(batches, 1)

                    print(
                        f"[stage {s_idx + 1} | epoch {epoch} | step {step}] "
                        f"loss={run_loss / denom:.6f}  "
                        f"L1={run_l1 / denom:.6f}  "
                        f"SSIM_loss={run_ssim / denom:.6f}  "
                        f"MS_SSIM_loss={run_msssim / denom:.6f}  "
                        f"LPIPS_loss={run_lpips / denom:.6f}"
                    )

            sample_path = stage_dir / f"epoch_{epoch:03d}_triptych.png"

            val_core = validate_core(
                model,
                val_loader,
                device,
                iqa,
                sample_path=str(sample_path),
            )

            dt = time.time() - t0

            append_csv(
                str(epoch_csv),
                [
                    s_idx + 1,
                    crop,
                    epoch,
                    lr,
                    val_core["L1"],
                    val_core["PSNR"],
                    val_core["SSIM"],
                    val_core["MS-SSIM"],
                    val_core["LPIPS"],
                    dt,
                ],
                header=[
                    "stage",
                    "crop",
                    "epoch",
                    "lr",
                    "val_L1",
                    "val_PSNR",
                    "val_SSIM",
                    "val_MS_SSIM",
                    "val_LPIPS",
                    "time_sec",
                ],
            )

            print(
                f"Stage {s_idx + 1} Epoch {epoch} | "
                f"L1={val_core['L1']:.6f}  "
                f"PSNR↑={val_core['PSNR']:.4f}  "
                f"SSIM↑={val_core['SSIM']:.4f}  "
                f"MS-SSIM↑={val_core['MS-SSIM']:.4f}  "
                f"LPIPS↓={val_core['LPIPS']:.4f}  "
                f"time={dt:.1f}s"
            )

            # You are currently selecting by validation L1.
            # This is okay, but if you want perceptual best model, use val_core["LPIPS"]
            # or a combined validation score.
            if val_core["L1"] < stage_best:
                stage_best = val_core["L1"]

                save_ckpt(
                    ckpt_dir / f"stage_{s_idx + 1:02d}_best.pt",
                    model,
                    opt,
                    s_idx,
                    epoch,
                    best_val,
                )

                print(f" saved stage best: stage_{s_idx + 1:02d}_best.pt")

            if val_core["L1"] < best_val:
                best_val = val_core["L1"]

                save_ckpt(
                    ckpt_dir / "global_best.pt",
                    model,
                    opt,
                    s_idx,
                    epoch,
                    best_val,
                )

                print(" saved global_best.pt")

            save_ckpt(
                latest_path,
                model,
                opt,
                s_idx,
                epoch,
                best_val,
            )

            if (epoch % full_eval_every == 0) or (epoch == epochs_per_stage):
                full_m = validate_full(model, val_loader, device, iqa)

                print_full_metrics(
                    f"[FULL EVAL] stage={s_idx + 1} epoch={epoch}",
                    full_m,
                )

                append_csv(
                    str(stage_csv),
                    [
                        s_idx + 1,
                        crop,
                        epoch,
                        full_m["PSNR"],
                        full_m["SSIM"],
                        full_m["MS-SSIM"],
                        full_m["LPIPS"],
                        full_m["NIQE"],
                        full_m["UIQM"],
                        full_m["UCIQE"],
                    ],
                    header=[
                        "stage",
                        "crop",
                        "epoch",
                        "PSNR",
                        "SSIM",
                        "MS_SSIM",
                        "LPIPS",
                        "NIQE",
                        "UIQM",
                        "UCIQE",
                    ],
                )

        stage_best_path = ckpt_dir / f"stage_{s_idx + 1:02d}_best.pt"

        if stage_best_path.exists():
            stage_best_ckpt = torch.load(stage_best_path, map_location=device)
            model.load_state_dict(stage_best_ckpt["model"], strict=True)

        start_epoch = 1

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

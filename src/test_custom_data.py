# Assumes:
#   /content/images   (folder with test images)
#   /content/best.pt  (checkpoint)
#
# 1) Defines TinyCurveCCMNet
# 2) Loads checkpoint robustly
# 3) Predicts params on a small resized image (fast)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# Model


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
    def __init__(
        self,
        gain_min=0.6,
        gain_max=2.0,
        gamma_min=0.6,
        gamma_max=2.2,
        ccm_strength=0.30,
        bias_strength=0.06,
    ):
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
            nn.Linear(48, 18),  # 3 gain + 3 gamma + 9 ccm + 3 bias
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

        gain = self._map_range(p[:, 0:3], self.gain_min, self.gain_max).view(-1, 3, 1, 1)
        gamma = self._map_range(p[:, 3:6], self.gamma_min, self.gamma_max).view(-1, 3, 1, 1)

        r = torch.tanh(p[:, 6:15]).view(-1, 3, 3) * self.ccm_strength
        eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).expand(r.size(0), -1, -1)
        m = eye + r

        b = torch.tanh(p[:, 15:18]).view(-1, 3, 1, 1) * self.bias_strength
        return gain, gamma, m, b


# Helpers


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def pil_to_tensor_rgb01(img_pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_np_uint8(t: torch.Tensor) -> np.ndarray:
    if t.ndim == 4:
        t = t[0]
    t = torch.clamp(t, 0, 1).detach().cpu()
    return (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def make_compare_strip(inp_u8, raw_u8, post_u8):
    h = min(inp_u8.shape[0], raw_u8.shape[0], post_u8.shape[0])
    inp_u8 = inp_u8[:h, :, :]
    raw_u8 = raw_u8[:h, :, :]
    post_u8 = post_u8[:h, :, :]
    sep = np.zeros((h, 12, 3), dtype=np.uint8)
    return np.concatenate([inp_u8, sep, raw_u8, sep, post_u8], axis=1)


def collect_images_recursive(root_dir: Path):
    files = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return sorted(files)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# Apply full-resolution enhancement
# ============================================================

def apply_params_fullres(x_full, gain, gamma, m, b):
    z = torch.clamp(gain * x_full, 0.0, 1.0)
    z = torch.pow(z + 1e-6, gamma)

    batch, _, h, w = z.shape
    z_flat = z.view(batch, 3, -1)
    y_flat = torch.bmm(m, z_flat)
    y = y_flat.view(batch, 3, h, w) + b
    return torch.clamp(y, 0.0, 1.0)


# Post-processing


def dehaze_light(img_u8: np.ndarray, strength=0.14, radius=41) -> np.ndarray:
    rgb = img_u8.astype(np.float32) / 255.0
    dc = np.min(rgb, axis=2)
    dc = cv2.erode(dc, np.ones((7, 7), np.uint8))
    t = 1.0 - float(strength) * cv2.GaussianBlur(dc, (0, 0), radius / 6)
    t = np.clip(t, 0.80, 1.0)
    a = np.percentile(rgb.reshape(-1, 3), 99.5, axis=0)
    out = (rgb - a) / t[..., None] + a
    out = np.clip(out, 0, 1)
    return (out * 255.0 + 0.5).astype(np.uint8)


def wb_lab_chroma_safe(img_u8: np.ndarray, strength=0.55, dark_thresh=10) -> np.ndarray:
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    mask = (l > dark_thresh).astype(np.uint8)
    if mask.sum() < 200:
        return img_u8
    a_mean = cv2.mean(a, mask=mask)[0]
    b_mean = cv2.mean(b, mask=mask)[0]
    a_shift = float(np.clip((128.0 - a_mean) * float(strength), -10, 10))
    b_shift = float(np.clip((128.0 - b_mean) * float(strength), -10, 10))
    a2 = np.clip(a + a_shift, 0, 255)
    b2 = np.clip(b + b_shift, 0, 255)
    lab2 = cv2.merge([l, a2, b2]).astype(np.uint8)
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)


def clahe_on_luminance_blend(img_u8: np.ndarray, clip=1.1, tilegrid=8, alpha=0.20) -> np.ndarray:
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip), tileGridSize=(int(tilegrid), int(tilegrid))
    )
    l2 = clahe.apply(l)
    lb = cv2.addWeighted(l, 1.0 - float(alpha), l2, float(alpha), 0.0)
    lab2 = cv2.merge([lb, a, b])
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)


def edge_mask(img_u8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mag = np.abs(lap)
    p95 = np.percentile(mag, 95) + 1e-6
    m = np.clip(mag / p95, 0, 1)
    return cv2.GaussianBlur(m, (0, 0), 1.0)


def unsharp_edge_masked(img_u8: np.ndarray, amount=0.08, radius=1.2, edge_bias=0.88) -> np.ndarray:
    img = img_u8.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), float(radius))
    sharp = img + float(amount) * (img - blurred)
    m = edge_mask(img_u8)
    m2 = np.clip((m - edge_bias) / max(1e-6, (1.0 - edge_bias)), 0, 1)
    out = img * (1.0 - m2[..., None]) + sharp * m2[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def postprocess(
    img_u8: np.ndarray,
    do_dehaze=True,
    dehaze_strength=0.14,
    dehaze_radius=41,
    do_white_balance=True,
    wb_strength=0.55,
    do_clahe=True,
    clahe_clip=1.1,
    clahe_tilegrid=8,
    clahe_blend=0.20,
    do_sharpen=True,
    sharp_amount=0.08,
    sharp_radius=1.2,
    sharp_edge_bias=0.88,
) -> np.ndarray:
    out = img_u8
    if do_dehaze:
        out = dehaze_light(out, strength=dehaze_strength, radius=dehaze_radius)
    if do_white_balance:
        out = wb_lab_chroma_safe(out, strength=wb_strength)
    if do_clahe:
        out = clahe_on_luminance_blend(
            out, clip=clahe_clip, tilegrid=clahe_tilegrid, alpha=clahe_blend
        )
    if do_sharpen:
        out = unsharp_edge_masked(
            out,
            amount=sharp_amount,
            radius=sharp_radius,
            edge_bias=sharp_edge_bias,
        )
    return out


# Robust checkpoint loading


def torch_load_safe(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_state_dict_robust(model, ckpt_path):
    obj = torch_load_safe(ckpt_path, map_location="cpu")
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
        print("  Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("  Unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")


def load_model(args, device):
    model = TinyCurveCCMNet(
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        ccm_strength=args.ccm_strength,
        bias_strength=args.bias_strength,
    ).to(device).eval()

    load_state_dict_robust(model, args.model)
    return model



# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Recursive TinyCurveCCMNet inference with mirrored output folders"
    )

    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input root folder")
    parser.add_argument("--output", type=str, required=True, help="Output root folder")

    parser.add_argument("--infer-size", type=int, default=256, help="Resize used for parameter prediction")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")

    parser.add_argument("--gain-min", type=float, default=0.6)
    parser.add_argument("--gain-max", type=float, default=2.0)
    parser.add_argument("--gamma-min", type=float, default=0.6)
    parser.add_argument("--gamma-max", type=float, default=2.2)
    parser.add_argument("--ccm-strength", type=float, default=0.30)
    parser.add_argument("--bias-strength", type=float, default=0.06)

    parser.add_argument("--no-dehaze", action="store_true")
    parser.add_argument("--dehaze-strength", type=float, default=0.14)
    parser.add_argument("--dehaze-radius", type=int, default=41)

    parser.add_argument("--no-white-balance", action="store_true")
    parser.add_argument("--wb-strength", type=float, default=0.55)

    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--clahe-clip", type=float, default=1.1)
    parser.add_argument("--clahe-tilegrid", type=int, default=8)
    parser.add_argument("--clahe-blend", type=float, default=0.20)

    parser.add_argument("--no-sharpen", action="store_true")
    parser.add_argument("--sharp-amount", type=float, default=0.08)
    parser.add_argument("--sharp-radius", type=float, default=1.2)
    parser.add_argument("--sharp-edge-bias", type=float, default=0.88)

    parser.add_argument("--save-raw-only", action="store_true")

    return parser.parse_args()



# Main


def main():
    args = parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = load_model(args, device)

    paths = collect_images_recursive(input_root)
    print(f"Found images: {len(paths)}")
    print(f"Input root : {input_root}")
    print(f"Output root: {output_root}")

    if not paths:
        print("No valid images found.")
        return

    image_times = []
    model_times = []
    post_times = []
    save_times = []

    total_start = time.perf_counter()

    with torch.no_grad():
        for p in paths:
            try:
                img_pil = Image.open(p).convert("RGB")
                inp_u8 = np.asarray(img_pil).astype(np.uint8)

                x_full = pil_to_tensor_rgb01(img_pil).to(device)

                img_small = img_pil.resize((args.infer_size, args.infer_size), Image.BILINEAR)
                x_small = pil_to_tensor_rgb01(img_small).to(device)

                rel_parent = p.parent.relative_to(input_root)
                save_dir = output_root / rel_parent
                save_dir.mkdir(parents=True, exist_ok=True)
                base = p.stem

                img_start = time.perf_counter()

                sync_cuda()
                t0 = time.perf_counter()
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(args.amp and x_small.is_cuda),
                ):
                    gain, gamma, m, b = model(x_small)
                    y_full = apply_params_fullres(x_full, gain, gamma, m, b)
                sync_cuda()
                t1 = time.perf_counter()

                raw_u8 = tensor_to_np_uint8(y_full)

                t2 = time.perf_counter()
                if args.save_raw_only:
                    post_u8 = None
                    comp_u8 = None
                else:
                    post_u8 = postprocess(
                        raw_u8.copy(),
                        do_dehaze=not args.no_dehaze,
                        dehaze_strength=args.dehaze_strength,
                        dehaze_radius=args.dehaze_radius,
                        do_white_balance=not args.no_white_balance,
                        wb_strength=args.wb_strength,
                        do_clahe=not args.no_clahe,
                        clahe_clip=args.clahe_clip,
                        clahe_tilegrid=args.clahe_tilegrid,
                        clahe_blend=args.clahe_blend,
                        do_sharpen=not args.no_sharpen,
                        sharp_amount=args.sharp_amount,
                        sharp_radius=args.sharp_radius,
                        sharp_edge_bias=args.sharp_edge_bias,
                    )
                    comp_u8 = make_compare_strip(inp_u8, raw_u8, post_u8)
                t3 = time.perf_counter()

                Image.fromarray(raw_u8).save(save_dir / f"{base}_raw_model.png")
                if not args.save_raw_only:
                    Image.fromarray(post_u8).save(save_dir / f"{base}_post.png")
                    Image.fromarray(comp_u8).save(save_dir / f"{base}_compare.png")
                t4 = time.perf_counter()

                img_elapsed = t4 - img_start
                model_elapsed = t1 - t0
                post_elapsed = t3 - t2
                save_elapsed = t4 - t3

                image_times.append(img_elapsed)
                model_times.append(model_elapsed)
                post_times.append(post_elapsed)
                save_times.append(save_elapsed)

                print(
                    f"[OK] {p.relative_to(input_root)} | "
                    f"total={img_elapsed*1000:.2f} ms | "
                    f"model={model_elapsed*1000:.2f} ms | "
                    f"post={post_elapsed*1000:.2f} ms | "
                    f"save={save_elapsed*1000:.2f} ms | "
                    f"FPS={1.0/max(img_elapsed,1e-9):.2f}"
                )

            except Exception as e:
                print(f"[ERROR] {p} -> {e}")

    total_end = time.perf_counter()
    grand_total = total_end - total_start

    avg_total = np.mean(image_times) if image_times else 0.0
    avg_model = np.mean(model_times) if model_times else 0.0
    avg_post = np.mean(post_times) if post_times else 0.0
    avg_save = np.mean(save_times) if save_times else 0.0

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Processed images         : {len(image_times)} / {len(paths)}")
    print(f"Grand total time         : {grand_total:.3f} s")
    print(f"Average total/image      : {avg_total*1000:.2f} ms")
    print(f"Average model/image      : {avg_model*1000:.2f} ms")
    print(f"Average post/image       : {avg_post*1000:.2f} ms")
    print(f"Average save/image       : {avg_save*1000:.2f} ms")
    print(f"Average FPS              : {1.0/max(avg_total,1e-9):.2f}")
    print(f"Overall throughput FPS   : {len(image_times)/max(grand_total,1e-9):.2f}")
    print(f"Outputs saved in         : {output_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()

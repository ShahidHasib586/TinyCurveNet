#  © shahid ahamed Hasib 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
from pathlib import Path

import numpy as np
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


def collect_images_recursive(root_dir: Path):
    return sorted([
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ])


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def resize_keep_aspect_max(img: Image.Image, max_size: int) -> Image.Image:
    if max_size <= 0:
        return img
    w, h = img.size
    scale = max_size / max(w, h)
    if scale >= 1.0:
        return img
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BILINEAR)


def resize_square(img: Image.Image, size: int) -> Image.Image:
    if size <= 0:
        return img
    return img.resize((size, size), Image.BILINEAR)


# Enhancement

def apply_params_fullres(x_full, gain, gamma, m, b):
    z = torch.clamp(gain * x_full, 0.0, 1.0)
    z = torch.pow(z + 1e-6, gamma)

    batch, _, h, w = z.shape
    z_flat = z.reshape(batch, 3, -1)
    y_flat = torch.bmm(m, z_flat)
    y = y_flat.reshape(batch, 3, h, w) + b
    return torch.clamp(y, 0.0, 1.0)

# Checkpoint loading

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
    parser = argparse.ArgumentParser(description="Fast TinyCurveNet inference")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--infer-size", type=int, default=128)

    # Output speed controls
    parser.add_argument("--output-size", type=int, default=512,
                        help="Max output side. 512 recommended. Use 0 to keep original size.")
    parser.add_argument("--square-output", action="store_true",
                        help="Force output to output-size x output-size.")
    parser.add_argument("--save-format", type=str, default="jpg", choices=["jpg", "png"])
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--png-compress", type=int, default=1,
                        help="0 fastest/largest, 9 slowest/smallest")

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)

    parser.add_argument("--gain-min", type=float, default=0.6)
    parser.add_argument("--gain-max", type=float, default=2.0)
    parser.add_argument("--gamma-min", type=float, default=0.6)
    parser.add_argument("--gamma-max", type=float, default=2.2)
    parser.add_argument("--ccm-strength", type=float, default=0.30)
    parser.add_argument("--bias-strength", type=float, default=0.06)

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

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_root.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
        torch.backends.cudnn.benchmark = True

    model = load_model(args, device)

    paths = collect_images_recursive(input_root)
    if args.max_images > 0:
        paths = paths[:args.max_images]

    print(f"Found images: {len(paths)}")
    print(f"Input root : {input_root}")
    print(f"Output root: {output_root}")
    print(f"Infer size : {args.infer_size}")
    print(f"Output size: {args.output_size}")
    print(f"Save format: {args.save_format}")

    if not paths:
        print("No valid images found.")
        return

    image_times = []
    model_times = []
    save_times = []
    load_times = []

    total_start = time.perf_counter()

    with torch.inference_mode():
        for idx, p in enumerate(paths, 1):
            try:
                img_start = time.perf_counter()

                t_load0 = time.perf_counter()
                img_pil = Image.open(p).convert("RGB")

                # Make final model application only on the desired output resolution.
                if args.square_output and args.output_size > 0:
                    img_work = resize_square(img_pil, args.output_size)
                else:
                    img_work = resize_keep_aspect_max(img_pil, args.output_size)

                x_full = pil_to_tensor_rgb01(img_work).to(device, non_blocking=True)

                img_small = img_pil.resize((args.infer_size, args.infer_size), Image.BILINEAR)
                x_small = pil_to_tensor_rgb01(img_small).to(device, non_blocking=True)
                t_load1 = time.perf_counter()

                rel_parent = p.parent.relative_to(input_root)
                save_dir = output_root / rel_parent
                save_dir.mkdir(parents=True, exist_ok=True)
                base = p.stem

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
                if args.save_format == "jpg":
                    out_path = save_dir / f"{base}_raw_model.jpg"
                    Image.fromarray(raw_u8).save(
                        out_path,
                        quality=args.jpg_quality,
                        subsampling=0,
                        optimize=False,
                    )
                else:
                    out_path = save_dir / f"{base}_raw_model.png"
                    Image.fromarray(raw_u8).save(
                        out_path,
                        compress_level=args.png_compress,
                    )
                t3 = time.perf_counter()

                img_elapsed = t3 - img_start
                load_elapsed = t_load1 - t_load0
                model_elapsed = t1 - t0
                save_elapsed = t3 - t2

                image_times.append(img_elapsed)
                load_times.append(load_elapsed)
                model_times.append(model_elapsed)
                save_times.append(save_elapsed)

                print(
                    f"[OK {idx}/{len(paths)}] {p.relative_to(input_root)} | "
                    f"total={img_elapsed*1000:.2f} ms | "
                    f"load/pre={load_elapsed*1000:.2f} ms | "
                    f"model={model_elapsed*1000:.2f} ms | "
                    f"save={save_elapsed*1000:.2f} ms | "
                    f"FPS={1.0/max(img_elapsed,1e-9):.2f}"
                )

            except Exception as e:
                print(f"[ERROR] {p} -> {repr(e)}")

    grand_total = time.perf_counter() - total_start

    avg_total = float(np.mean(image_times)) if image_times else 0.0
    avg_load = float(np.mean(load_times)) if load_times else 0.0
    avg_model = float(np.mean(model_times)) if model_times else 0.0
    avg_save = float(np.mean(save_times)) if save_times else 0.0

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Processed images         : {len(image_times)} / {len(paths)}")
    print(f"Grand total time         : {grand_total:.3f} s")
    print(f"Average total/image      : {avg_total*1000:.2f} ms")
    print(f"Average load/pre/image   : {avg_load*1000:.2f} ms")
    print(f"Average model/image      : {avg_model*1000:.2f} ms")
    print(f"Average save/image       : {avg_save*1000:.2f} ms")
    print(f"Average FPS              : {1.0/max(avg_total,1e-9):.2f}")
    print(f"Overall throughput FPS   : {len(image_times)/max(grand_total,1e-9):.2f}")
    print(f"Outputs saved in         : {output_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()

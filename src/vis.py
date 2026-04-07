import os
import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF

@torch.no_grad()
def save_labeled_triptych(inp, out, gt, path, labels=("INPUT", "OUTPUT", "GT")):
    """
    inp,out,gt: tensors (3,H,W) in [0,1]
    """
    inp = TF.to_pil_image(inp.cpu().clamp(0,1))
    out = TF.to_pil_image(out.cpu().clamp(0,1))
    gt  = TF.to_pil_image(gt.cpu().clamp(0,1))

    w, h = inp.size
    bar_h = 28
    canvas = Image.new("RGB", (w*3, h+bar_h), color=(0,0,0))
    canvas.paste(inp, (0, bar_h))
    canvas.paste(out, (w, bar_h))
    canvas.paste(gt, (2*w, bar_h))

    draw = ImageDraw.Draw(canvas)
    for i, txt in enumerate(labels):
        x = i*w + 10
        y = 5
        draw.text((x, y), txt, fill=(255,255,255))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)

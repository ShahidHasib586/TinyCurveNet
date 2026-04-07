import torch
import pyiqa
import numpy as np
import cv2

class IQAMetrics:
    def __init__(self, device):
        self.device = device
        self.psnr = pyiqa.create_metric('psnr', device=device)
        self.ssim = pyiqa.create_metric('ssim', device=device)
        self.ms_ssim = pyiqa.create_metric('ms_ssim', device=device)
        self.lpips = pyiqa.create_metric('lpips', device=device)
        self.niqe = pyiqa.create_metric('niqe', device=device)

    @torch.no_grad()
    def fr_batch(self, pred, gt):
        pred = torch.clamp(pred.float(), 1e-6, 1.0-1e-6)
        gt = torch.clamp(gt, 1e-6, 1.0-1e-6)

        return {
            "PSNR": float(self.psnr(pred, gt).mean().item()),
            "SSIM": float(self.ssim(pred, gt).mean().item()),
            "MS-SSIM": float(self.ms_ssim(pred, gt).mean().item()),
            "LPIPS": float(self.lpips(pred, gt).mean().item()),
        }

    @torch.no_grad()
    def nr_batch(self, pred):
        pred = torch.clamp(pred.float(), 1e-6, 1.0-1e-6)

        niqe_val = float(self.niqe(pred).mean().item())

        pred_np = pred.detach().cpu().permute(0,2,3,1).numpy()

        uiqm_vals = []
        uciqe_vals = []

        for img in pred_np:
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)

            uiqm_vals.append(compute_uiqm(img))
            uciqe_vals.append(compute_uciqe(img))

        return {
            "NIQE": niqe_val,
            "UIQM": float(np.mean(uiqm_vals)),
            "UCIQE": float(np.mean(uciqe_vals)),
        }

    @torch.no_grad()
    def full_batch(self, pred, gt):
        out = {}
        out.update(self.fr_batch(pred, gt))
        out.update(self.nr_batch(pred))
        return out


def compute_uiqm(img):
    img = img.astype(np.float32) / 255.0

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    rg = R - G
    yb = 0.5*(R + G) - B

    uicm = np.sqrt(np.std(rg)**2 + np.std(yb)**2)

    gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    uism = cv2.Laplacian(gray, cv2.CV_64F).var()

    uiconm = np.std(gray)

    return 0.0282*uicm + 0.2953*uism + 3.5753*uiconm


def compute_uciqe(img):
    img = img.astype(np.float32) / 255.0

    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2LAB)

    L = lab[:,:,0] / 255.0
    a = lab[:,:,1]
    b = lab[:,:,2]

    chroma = np.sqrt(a**2 + b**2)

    sigma_c = np.std(chroma)
    con_l = np.max(L) - np.min(L)
    mu_s = np.mean(chroma)

    return 0.4680*sigma_c + 0.2745*con_l + 0.2576*mu_s

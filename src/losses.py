import torch
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim
import lpips


class TinyCurveLoss(nn.Module):
    """
    Differentiable loss for TinyCurveNet / TinyCurveCCMNet.

    Total loss:
        L = w_l1 * L1
          + w_ssim * (1 - SSIM)
          + w_msssim * (1 - MS-SSIM)
          + w_lpips * LPIPS

    pred and target must be in [0, 1].
    """

    def __init__(
        self,
        w_l1=1.0,
        w_ssim=0.20,
        w_msssim=0.0,
        w_lpips=0.05,
        lpips_net="vgg",
        use_lpips=True,
    ):
        super().__init__()

        self.w_l1 = float(w_l1)
        self.w_ssim = float(w_ssim)
        self.w_msssim = float(w_msssim)
        self.w_lpips = float(w_lpips)
        self.use_lpips = bool(use_lpips) and self.w_lpips > 0.0

        self.l1 = nn.L1Loss()

        if self.use_lpips:
            self.lpips_fn = lpips.LPIPS(net=lpips_net)
            self.lpips_fn.eval()

            # Freeze LPIPS network weights.
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
        else:
            self.lpips_fn = None

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        loss_l1 = self.l1(pred, target)

        loss_ssim = pred.new_tensor(0.0)
        if self.w_ssim > 0.0:
            ssim_val = ssim(
                pred,
                target,
                data_range=1.0,
                size_average=True,
            )
            loss_ssim = 1.0 - ssim_val

        loss_msssim = pred.new_tensor(0.0)
        if self.w_msssim > 0.0:
            msssim_val = ms_ssim(
                pred,
                target,
                data_range=1.0,
                size_average=True,
            )
            loss_msssim = 1.0 - msssim_val

        loss_lpips = pred.new_tensor(0.0)
        if self.use_lpips:
            # LPIPS expects input in [-1, 1], while your images are in [0, 1].
            pred_lpips = pred * 2.0 - 1.0
            target_lpips = target * 2.0 - 1.0
            loss_lpips = self.lpips_fn(pred_lpips, target_lpips).mean()

        total_loss = (
            self.w_l1 * loss_l1
            + self.w_ssim * loss_ssim
            + self.w_msssim * loss_msssim
            + self.w_lpips * loss_lpips
        )

        loss_items = {
            "total": float(total_loss.detach().cpu()),
            "l1": float(loss_l1.detach().cpu()),
            "ssim_loss": float(loss_ssim.detach().cpu()),
            "msssim_loss": float(loss_msssim.detach().cpu()),
            "lpips_loss": float(loss_lpips.detach().cpu()),
        }

        return total_loss, loss_items

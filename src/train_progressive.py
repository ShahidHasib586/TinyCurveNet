"""Compatibility entry point for the corrected differentiable trainer.

Select loss.objective=l1 to reproduce the L1-only objective of the old script.
The former implementation detached SSIM/LPIPS, so they did not affect gradients.
"""
from train import main

if __name__ == "__main__":
    main()

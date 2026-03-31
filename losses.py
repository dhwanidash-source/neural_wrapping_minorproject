"""
losses.py - Perceptual Loss Functions
=======================================
The paper explicitly optimizes for perceptual quality scores: SSIM, MS-SSIM, VMAF.
We use a combination of MSE and SSIM loss — simple but effective.

Why SSIM loss?
  MSE alone maximizes PSNR but can produce blurry outputs.
  SSIM measures structural similarity (edges, textures) — closer to human perception.
  The paper's core argument: optimize for perception, not just distortion.
  
Loss = α * MSE_Loss + (1-α) * SSIM_Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    SSIM ranges from -1 to 1 (higher = more similar).
    Loss = 1 - SSIM (so minimizing loss = maximizing SSIM).

    Implementation uses a Gaussian window to match the original SSIM paper.
    This is much cheaper than full-image SSIM and works on mini-batches.
    """
    def __init__(self, window_size=11, channels=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channels = channels
        self.window = self._create_gaussian_window(window_size, channels)

    def _gaussian_kernel(self, window_size, sigma=1.5):
        """1D Gaussian kernel."""
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_gaussian_window(self, window_size, channels):
        """2D Gaussian window for SSIM computation."""
        _1d = self._gaussian_kernel(window_size)
        _2d = _1d.unsqueeze(1) @ _1d.unsqueeze(0)  # outer product
        window = _2d.unsqueeze(0).unsqueeze(0)       # (1, 1, W, W)
        window = window.expand(channels, 1, window_size, window_size)
        return window.contiguous()

    def _ssim(self, x, y):
        """Compute SSIM between two batches of images."""
        C1 = 0.01 ** 2  # stability constants from original SSIM paper
        C2 = 0.03 ** 2

        window = self.window.to(x.device)
        padding = self.window_size // 2

        # Local means
        mu_x = F.conv2d(x, window, padding=padding, groups=self.channels)
        mu_y = F.conv2d(y, window, padding=padding, groups=self.channels)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy   = mu_x * mu_y

        # Local variances and covariance
        sigma_x_sq = F.conv2d(x*x, window, padding=padding, groups=self.channels) - mu_x_sq
        sigma_y_sq = F.conv2d(y*y, window, padding=padding, groups=self.channels) - mu_y_sq
        sigma_xy   = F.conv2d(x*y, window, padding=padding, groups=self.channels) - mu_xy

        # SSIM formula
        numerator   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim_map = numerator / denominator

        return ssim_map.mean()

    def forward(self, prediction, target):
        """Returns SSIM loss (1 - SSIM) so that minimizing = maximizing SSIM."""
        return 1.0 - self._ssim(prediction, target)


class PerceptualLoss(nn.Module):
    """
    Combined MSE + SSIM loss.

    The paper uses combinations of MSE (distortion) and perceptual scores.
    We keep it simple:
      Total Loss = alpha * MSE + (1 - alpha) * SSIM_Loss

    alpha=0.5 is a good starting point.
    alpha=0.0 → pure SSIM (very perceptual, sometimes blurry)
    alpha=1.0 → pure MSE (high PSNR, sometimes unpleasant artifacts)
    """
    def __init__(self, alpha=0.5, channels=3):
        """
        Args:
            alpha   : weight for MSE loss (1-alpha goes to SSIM)
            channels: number of image channels (3 for RGB)
        """
        super(PerceptualLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss  = nn.MSELoss()
        self.ssim_loss = SSIMLoss(channels=channels)

    def forward(self, prediction, target):
        mse  = self.mse_loss(prediction, target)
        ssim = self.ssim_loss(prediction, target)
        total = self.alpha * mse + (1 - self.alpha) * ssim
        return total, mse.item(), ssim.item()


# ── PSNR metric ────────────────────────────────────────────────────────────────

def psnr(prediction, target, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio in dB.
    Higher is better. Typical compressed images: 25-40 dB.
    The paper reports BD-rate which relates changes across PSNR/quality curves.
    """
    mse = F.mss_loss(prediction, target) if False else ((prediction - target) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(torch.tensor(max_val ** 2) / mse)


def compute_ssim(prediction, target):
    """Compute SSIM score (not loss) — range [0,1], higher is better."""
    loss_fn = SSIMLoss(channels=prediction.shape[1])
    with torch.no_grad():
        return 1.0 - loss_fn(prediction, target)


if __name__ == "__main__":
    print("=== Loss Function Tests ===\n")

    # Create fake batch
    pred   = torch.rand(4, 3, 64, 64)
    target = torch.rand(4, 3, 64, 64)

    loss_fn = PerceptualLoss(alpha=0.5)
    total, mse_val, ssim_val = loss_fn(pred, target)

    print(f"Total loss : {total.item():.4f}")
    print(f"MSE loss   : {mse_val:.4f}")
    print(f"SSIM loss  : {ssim_val:.4f}")

    # PSNR test
    p = psnr(pred, target)
    print(f"PSNR       : {p.item():.2f} dB")

    # Test identical images (should give 0 loss, inf PSNR)
    same = torch.rand(4, 3, 64, 64)
    total_same, _, _ = loss_fn(same, same)
    print(f"\nLoss on identical images: {total_same.item():.6f} (should be ~0)")

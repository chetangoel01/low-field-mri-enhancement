"""Loss functions for MRI Super Resolution.

Matches the competition metric (metric.py) exactly:
- SSIM: global means/variances, C1=0.01^2, C2=0.03^2, per-slice normalized to [0,1]
- PSNR: optimized via Charbonnier L1 loss
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Charbonnier L1 loss: sqrt((pred - target)^2 + eps^2).

    Smoother than L1 near zero, better gradient behavior.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps_sq))


class CompetitionSSIMLoss(nn.Module):
    """SSIM loss matching the competition metric exactly.

    Uses global means and variances (no windowing), matching metric.py:
    - C1 = 0.01^2 = 1e-4
    - C2 = 0.03^2 = 9e-4
    - Per-slice computation (each sample in batch is one slice)

    Loss = 1 - SSIM
    """

    def __init__(self):
        super().__init__()
        self.C1 = 0.01 ** 2  # 1e-4
        self.C2 = 0.03 ** 2  # 9e-4

    def forward(self, pred, target):
        # pred, target: (B, 1, H, W), already in [0, 1]
        # Compute per-sample SSIM (matching metric.py global stats approach)
        b = pred.shape[0]

        # Flatten spatial dims for each sample: (B, H*W)
        pred_flat = pred.reshape(b, -1)
        target_flat = target.reshape(b, -1)

        # Per-sample means
        mu_pred = pred_flat.mean(dim=1)      # (B,)
        mu_target = target_flat.mean(dim=1)  # (B,)

        # Per-sample variances and covariance
        sigma_pred_sq = ((pred_flat - mu_pred.unsqueeze(1)) ** 2).mean(dim=1)
        sigma_target_sq = ((target_flat - mu_target.unsqueeze(1)) ** 2).mean(dim=1)
        sigma_cross = ((pred_flat - mu_pred.unsqueeze(1)) * (target_flat - mu_target.unsqueeze(1))).mean(dim=1)

        # SSIM formula
        numerator = (2 * mu_pred * mu_target + self.C1) * (2 * sigma_cross + self.C2)
        denominator = (mu_pred ** 2 + mu_target ** 2 + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2)

        ssim = numerator / denominator  # (B,)

        # Loss = 1 - mean SSIM
        return 1.0 - ssim.mean()

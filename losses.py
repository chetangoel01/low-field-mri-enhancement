"""Loss functions for MRI Super Resolution.

Phase 2 metric: MS-SSIM (Multi-Scale Structural Similarity)
- 5 scales, 11x11 Gaussian windows (sigma=1.5), Wang et al. (2003) weights
- Phase 1 CompetitionSSIMLoss kept for backwards compatibility
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


class MSSSIMLoss(nn.Module):
    """MS-SSIM loss matching the Phase 2 competition metric exactly.

    Uses 5 scales, 11x11 Gaussian windows (sigma=1.5), and standard weights
    from Wang, Simoncelli & Bovik (2003). Matches phase_2/metric.py parameters:
    - C1 = (0.01 * data_range)^2, C2 = (0.03 * data_range)^2
    - Weights: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    - Intermediate scales: contrast-structure only
    - Final scale: luminance * contrast-structure

    Loss = 1 - MS_SSIM
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred, target: (B, 1, H, W), already in [0, 1]
        from pytorch_msssim import ms_ssim

        val = ms_ssim(
            pred, target,
            data_range=1.0,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            K=(0.01, 0.03),
        )
        # Guard NaN (near-uniform slices) and clamp to valid range
        val = torch.nan_to_num(val, nan=0.0)
        return 1.0 - val.clamp(0.0, 1.0)

"""Conditional denoising U-Net for DDPM-based MRI super-resolution.

SR3-style architecture: concatenate the upsampled LR image (5 adjacent axial
slices as context) with the noisy HR slice, then denoise conditioned on both.

Input:  (B, 6, 179, 221) = concat([noisy_HF (1ch), LF_context (5ch)])
Time:   (B,) integer timestep in [0, T-1]
Output: (B, 1, 179, 221) = predicted noise epsilon

No residual connection to input — this network predicts noise, not enhancement.

Uses GroupNorm instead of BatchNorm: each training batch mixes samples at
wildly different noise levels (t=3 is near-clean, t=997 is near-Gaussian),
so BatchNorm's batch statistics are meaningless. GroupNorm normalizes within
each sample independently, which is stable across all timesteps.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels):
    """GroupNorm with up to 32 groups (fewer when channels < 32)."""
    groups = 32
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep.

    Maps integer t ∈ [0, T-1] to a (B, dim) vector using alternating
    sin/cos of log-spaced frequencies, identical to Transformer positional
    encoding. Projected through a 2-layer MLP to produce a conditioning vector.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # MLP: sinusoidal(dim) → Linear(dim, dim*4) + SiLU → Linear(dim*4, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        # t: (B,) integer timesteps
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )  # (half,)
        args = t[:, None].float() * freqs[None, :]  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.mlp(emb)  # (B, dim)


class TimeCondBlock(nn.Module):
    """Two Conv2d(3x3) + GroupNorm + SiLU with additive time-step conditioning.

    GroupNorm is per-sample, so statistics are stable regardless of what
    timestep each sample in the batch is at. SiLU (Swish) is standard in
    DDPM architectures (smoother gradients than ReLU).

    The time embedding is projected to (out_ch,) and broadcast-added to
    the feature map after the first conv, letting the network condition on
    how much noise is present before applying the second conv.
    """

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        # First conv + GroupNorm + SiLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            _gn(out_ch),
            nn.SiLU(),
        )
        # Time projection: (B, time_emb_dim) → (B, out_ch)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        # Second conv + GroupNorm + SiLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            _gn(out_ch),
            nn.SiLU(),
        )

    def forward(self, x, t_emb):
        # x: (B, in_ch, H, W), t_emb: (B, time_emb_dim)
        x = self.conv1(x)
        # Add time shift as a broadcast spatial bias
        shift = self.time_proj(t_emb)[:, :, None, None]  # (B, out_ch, 1, 1)
        x = x + shift
        x = self.conv2(x)
        return x


class DiffusionUNet(nn.Module):
    """Conditional denoising U-Net for SR3-style DDPM.

    Architecture mirrors UNet2D (model.py) but:
    - in_channels=6: 1 noisy HF + 5 LF context slices concatenated
    - Every ConvBlock replaced by TimeCondBlock (time-conditioned)
    - No residual connection to input (predicts noise epsilon, not delta)
    - No output clamping (noise prediction is unbounded)

    Encoder: 4 downsampling blocks (base -> 2x -> 4x -> 8x)
    Bottleneck: 16x base features
    Decoder: 4 upsampling blocks with skip connections
    Final: 1x1 conv to 1 channel (predicted epsilon)
    """

    def __init__(self, in_channels=6, out_channels=1, base_features=64,
                 time_emb_dim=256):
        super().__init__()
        f = base_features  # 64
        d = time_emb_dim   # 256

        # Time embedding: sinusoidal(d) → MLP → (B, d)
        self.time_emb = SinusoidalTimeEmbedding(d)

        self.pool = nn.MaxPool2d(2)

        # Encoder (time-conditioned)
        self.enc1 = TimeCondBlock(in_channels, f,      d)   # (B, 64,  H,    W)
        self.enc2 = TimeCondBlock(f,           f * 2,  d)   # (B, 128, H/2,  W/2)
        self.enc3 = TimeCondBlock(f * 2,       f * 4,  d)   # (B, 256, H/4,  W/4)
        self.enc4 = TimeCondBlock(f * 4,       f * 8,  d)   # (B, 512, H/8,  W/8)

        # Bottleneck
        self.bottleneck = TimeCondBlock(f * 8, f * 16, d)   # (B, 1024, H/16, W/16)

        # Decoder (skip concat then time-conditioned)
        self.dec4 = TimeCondBlock(f * 16 + f * 8, f * 8,  d)  # 512
        self.dec3 = TimeCondBlock(f * 8  + f * 4, f * 4,  d)  # 256
        self.dec2 = TimeCondBlock(f * 4  + f * 2, f * 2,  d)  # 128
        self.dec1 = TimeCondBlock(f * 2  + f,     f,      d)  # 64

        # Final 1x1 conv → predicted noise (unbounded)
        self.final = nn.Conv2d(f, out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _up_and_concat(self, x, skip):
        """Bilinear upsample x to match skip spatial size, then concat."""
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x, t):
        """
        Args:
            x: (B, 6, H, W) = concat([noisy_HF (1ch), LF_context (5ch)])
            t: (B,) integer timesteps in [0, T-1]

        Returns:
            (B, 1, H, W) predicted noise epsilon
        """
        t_emb = self.time_emb(t)   # (B, time_emb_dim)

        # Encoder
        e1 = self.enc1(x,              t_emb)   # (B, 64,  H,    W)
        e2 = self.enc2(self.pool(e1),  t_emb)   # (B, 128, H/2,  W/2)
        e3 = self.enc3(self.pool(e2),  t_emb)   # (B, 256, H/4,  W/4)
        e4 = self.enc4(self.pool(e3),  t_emb)   # (B, 512, H/8,  W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4), t_emb)  # (B, 1024, H/16, W/16)

        # Decoder with skip connections
        d4 = self.dec4(self._up_and_concat(b,  e4), t_emb)  # (B, 512, H/8,  W/8)
        d3 = self.dec3(self._up_and_concat(d4, e3), t_emb)  # (B, 256, H/4,  W/4)
        d2 = self.dec2(self._up_and_concat(d3, e2), t_emb)  # (B, 128, H/2,  W/2)
        d1 = self.dec1(self._up_and_concat(d2, e1), t_emb)  # (B, 64,  H,    W)

        return self.final(d1)  # (B, 1, H, W) — raw noise prediction, no clamping

"""2.5D U-Net for MRI Super Resolution.

Processes 5 adjacent axial slices from trilinear-upsampled low-field MRI
and predicts the center high-field slice. Residual learning adds the center
input slice to the network output.

Input:  (B, 5, 179, 221)
Output: (B, 1, 179, 221)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two Conv2d(3x3) + BN + ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    """2.5D U-Net with residual learning from center input slice.

    Encoder: 4 downsampling blocks (base -> 2x -> 4x -> 8x)
    Bottleneck: 16x base features
    Decoder: 4 upsampling blocks with skip connections
    Final: 1x1 conv to 1 channel
    Residual: output += center input slice
    """

    def __init__(self, in_channels=5, out_channels=1, base_features=64):
        super().__init__()
        f = base_features  # 64

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)        # 64
        self.enc2 = ConvBlock(f, f * 2)              # 128
        self.enc3 = ConvBlock(f * 2, f * 4)          # 256
        self.enc4 = ConvBlock(f * 4, f * 8)          # 512

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)   # 1024

        # Decoder
        self.dec4 = ConvBlock(f * 16 + f * 8, f * 8)  # 512
        self.dec3 = ConvBlock(f * 8 + f * 4, f * 4)   # 256
        self.dec2 = ConvBlock(f * 4 + f * 2, f * 2)   # 128
        self.dec1 = ConvBlock(f * 2 + f, f)            # 64

        # Final 1x1 conv
        self.final = nn.Conv2d(f, out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 5, H, W)
        center = x[:, 2:3, :, :]  # center input slice for residual

        # Encoder
        e1 = self.enc1(x)                    # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))        # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))        # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))        # (B, 512, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))   # (B, 1024, H/16, W/16)

        # Decoder with skip connections
        d4 = self._up_and_concat(b, e4)
        d4 = self.dec4(d4)                   # (B, 512, H/8, W/8)

        d3 = self._up_and_concat(d4, e3)
        d3 = self.dec3(d3)                   # (B, 256, H/4, W/4)

        d2 = self._up_and_concat(d3, e2)
        d2 = self.dec2(d2)                   # (B, 128, H/2, W/2)

        d1 = self._up_and_concat(d2, e1)
        d1 = self.dec1(d1)                   # (B, 64, H, W)

        out = self.final(d1)                 # (B, 1, H, W)

        # Residual: add center input slice
        out = out + center

        # Clamp to [0, 1] since inputs are normalized to [0, 1]
        out = torch.clamp(out, 0.0, 1.0)

        return out

    def _up_and_concat(self, x, skip):
        """Bilinear upsample x to match skip spatial size, then concat."""
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

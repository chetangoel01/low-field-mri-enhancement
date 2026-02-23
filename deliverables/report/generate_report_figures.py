"""Generate report figures for MRI super-resolution project.

Creates:
  - figs/fig1.png: low-field upsampled vs high-field slices (train pair)
  - figs/fig2.png: 2.5D U-Net architecture diagram (schematic)
  - figs/fig3.png: bicubic baseline vs model submission output (test sample)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import logging

import matplotlib.pyplot as plt
from matplotlib import patches
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from extract_slices import base64_to_slice

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


ROOT = Path(__file__).resolve().parent
FIGS_DIR = ROOT / "figs"


def normalize_for_display(slice_2d: np.ndarray) -> np.ndarray:
    p1 = np.percentile(slice_2d, 1)
    p99 = np.percentile(slice_2d, 99)
    if p99 <= p1:
        return np.zeros_like(slice_2d, dtype=np.float32)
    return np.clip((slice_2d - p1) / (p99 - p1), 0.0, 1.0).astype(np.float32)


def load_nifti(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.float32)


def upsample_volume_trilinear(volume_xyz: np.ndarray, target_shape_xyz=(179, 221, 200)) -> np.ndarray:
    """Upsample an (X,Y,Z) volume using trilinear interpolation."""
    vol_zyx = np.transpose(volume_xyz, (2, 1, 0))  # (Z,Y,X)
    t = torch.from_numpy(vol_zyx)[None, None, ...]  # (1,1,Z,Y,X)
    target_zyx = (target_shape_xyz[2], target_shape_xyz[1], target_shape_xyz[0])
    out = F.interpolate(t, size=target_zyx, mode="trilinear", align_corners=False)
    out_zyx = out.squeeze(0).squeeze(0).cpu().numpy()
    return np.transpose(out_zyx, (2, 1, 0))  # back to (X,Y,Z)


def save_fig1(low_path: Path, high_path: Path, out_path: Path, z_slices: Iterable[int] = (40, 100, 160)) -> None:
    low = load_nifti(low_path)
    high = load_nifti(high_path)
    low_up = upsample_volume_trilinear(low, target_shape_xyz=high.shape)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    z_list = list(z_slices)
    for col, z in enumerate(z_list):
        low_sl = normalize_for_display(low_up[:, :, z])
        high_sl = normalize_for_display(high[:, :, z])

        axes[0, col].imshow(low_sl.T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"Upsampled LF (z={z})")
        axes[0, col].axis("off")

        axes[1, col].imshow(high_sl.T, cmap="gray", origin="lower")
        axes[1, col].set_title(f"High-Field GT (z={z})")
        axes[1, col].axis("off")

    fig.suptitle("Figure 1: Low-field upsampled vs high-field reference", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig2(out_path: Path) -> None:
    """Draw a clean publication-style 2.5D residual U-Net schematic."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 13)
    ax.axis("off")

    def block(x, y, w, h, label, fc, ec="#1f2937", fs=10):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.4,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color="#111827")
        return rect

    def arrow(x1, y1, x2, y2, dashed=False, color="#374151", lw=1.6):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                lw=lw,
                linestyle="--" if dashed else "-",
                color=color,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    def polyline(points, dashed=False, color="#374151", lw=1.6):
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            is_last = i == len(points) - 2
            if is_last:
                arrow(x1, y1, x2, y2, dashed=dashed, color=color, lw=lw)
            else:
                ax.plot([x1, x2], [y1, y2], linestyle="--" if dashed else "-", color=color, linewidth=lw)

    # Color palette.
    c_input = "#eef2ff"
    c_enc = "#dbeafe"
    c_bott = "#fde68a"
    c_dec = "#dcfce7"
    c_head = "#fce7f3"

    # Layout anchors.
    x_in = 0.8
    x_enc = 4.0
    x_bott = 9.2
    x_dec = 14.4
    x_head = 18.2

    y_levels = [9.8, 8.0, 6.2, 4.4]  # top -> bottom
    bw, bh = 2.4, 1.35

    # Blocks.
    block(x_in, 6.2, 2.9, 1.8, "Input\n5 x 179 x 221", c_input, fs=10)
    block(x_enc, y_levels[0], bw, bh, "Enc1\n64", c_enc)
    block(x_enc, y_levels[1], bw, bh, "Enc2\n128", c_enc)
    block(x_enc, y_levels[2], bw, bh, "Enc3\n256", c_enc)
    block(x_enc, y_levels[3], bw, bh, "Enc4\n512", c_enc)

    block(x_bott, 5.9, 2.7, 1.9, "Bottleneck\n1024", c_bott)

    block(x_dec, y_levels[3], bw, bh, "Dec4\n512", c_dec)
    block(x_dec, y_levels[2], bw, bh, "Dec3\n256", c_dec)
    block(x_dec, y_levels[1], bw, bh, "Dec2\n128", c_dec)
    block(x_dec, y_levels[0], bw, bh, "Dec1\n64", c_dec)

    block(x_head, 9.5, 3.0, 1.35, "1x1 Conv\n1 channel", c_head)
    block(x_head, 7.6, 3.0, 1.35, "Residual Add\n(+ center slice)", c_head)
    block(x_head, 5.7, 3.0, 1.35, "Output\n1 x 179 x 221", c_head)

    # Main encoder flow.
    arrow(x_in + 2.9, 7.1, x_enc, y_levels[0] + bh / 2)
    arrow(x_enc + bw / 2, y_levels[0], x_enc + bw / 2, y_levels[1] + bh)
    arrow(x_enc + bw / 2, y_levels[1], x_enc + bw / 2, y_levels[2] + bh)
    arrow(x_enc + bw / 2, y_levels[2], x_enc + bw / 2, y_levels[3] + bh)
    arrow(x_enc + bw, y_levels[3] + bh / 2, x_bott, 6.85)

    # Bottleneck to decoder.
    arrow(x_bott + 2.7, 6.85, x_dec, y_levels[3] + bh / 2)
    arrow(x_dec + bw / 2, y_levels[3] + bh, x_dec + bw / 2, y_levels[2])
    arrow(x_dec + bw / 2, y_levels[2] + bh, x_dec + bw / 2, y_levels[1])
    arrow(x_dec + bw / 2, y_levels[1] + bh, x_dec + bw / 2, y_levels[0])

    # Decoder to output head.
    arrow(x_dec + bw, y_levels[0] + bh / 2, x_head, 10.15)
    arrow(x_head + 1.5, 9.5, x_head + 1.5, 8.95)
    arrow(x_head + 1.5, 7.6, x_head + 1.5, 7.05)

    # Skip connections.
    skip_color = "#6b7280"
    y_bus = 11.6
    for y_enc, y_dec in zip(y_levels, reversed(y_levels)):
        x_start = x_enc + bw
        x_end = x_dec
        y_start = y_enc + bh / 2
        y_end = y_dec + bh / 2
        polyline(
            [(x_start, y_start), (x_start + 0.45, y_bus), (x_end - 0.45, y_bus), (x_end, y_end)],
            dashed=True,
            color=skip_color,
            lw=1.2,
        )
        y_bus -= 0.28
    ax.text(10.4, 12.15, "skip connections", fontsize=9, color=skip_color)

    # Residual path from center slice.
    residual_color = "#7c3aed"
    polyline(
        [(x_in + 2.9, 6.6), (6.0, 3.3), (16.9, 3.3), (x_head, 8.25)],
        dashed=True,
        color=residual_color,
        lw=1.4,
    )
    ax.text(9.0, 2.9, "center-slice residual path", fontsize=9, color=residual_color)

    ax.text(11.0, 1.0, "Figure 2: 2.5D Residual U-Net architecture", ha="center", fontsize=13, color="#111827")
    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def decode_submission_volume(csv_path: Path, sample_id: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    rows = df[df["row_id"].str.startswith(f"{sample_id}_slice_")].copy()
    if rows.empty:
        raise ValueError(f"No rows found for {sample_id} in {csv_path}")
    rows["slice_idx"] = rows["row_id"].str.extract(r"_slice_(\d+)$").astype(int)
    rows = rows.sort_values("slice_idx")
    slices = [base64_to_slice(pred) for pred in rows["prediction"].tolist()]
    return np.stack(slices, axis=2).astype(np.float32)


def find_submission_csv() -> Path:
    candidates = [
        ROOT / "submission_dedup.csv",
        ROOT / "submission.csv",
        ROOT / "submission_diffusion_v1_rerun_v2.csv",
        ROOT / "submission_diffusion_v1_rerun.csv",
        ROOT / "experiments_downloads" / "submission_dedup.csv",
        ROOT / "experiments_downloads" / "submission.csv",
        ROOT / "experiments_downloads" / "submission_diffusion.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No submission CSV found. Expected submission*.csv in project root or experiments_downloads.")


def save_fig3(test_low_path: Path, out_path: Path, sample_id="sample_019", z=100) -> None:
    low = load_nifti(test_low_path)
    low_up = upsample_volume_trilinear(low, target_shape_xyz=(179, 221, 200))
    sub_csv = find_submission_csv()
    pred_vol = decode_submission_volume(sub_csv, sample_id=sample_id)

    bicubic = normalize_for_display(low_up[:, :, z])
    pred = normalize_for_display(pred_vol[:, :, z])
    diff = np.abs(pred - bicubic)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(bicubic.T, cmap="gray", origin="lower")
    axes[0].set_title("Bicubic Upsampled Input")
    axes[0].axis("off")

    axes[1].imshow(pred.T, cmap="gray", origin="lower")
    axes[1].set_title("Model Prediction (Submission)")
    axes[1].axis("off")

    im = axes[2].imshow(diff.T, cmap="magma", origin="lower")
    axes[2].set_title("|Prediction - Bicubic|")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.045, pad=0.03)

    fig.suptitle(f"Figure 3: Qualitative output on {sample_id}, slice z={z}", fontsize=13)
    fig.text(
        0.5,
        0.01,
        f"Submission source: {sub_csv.name} (test sample; high-field GT not available)",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGS_DIR.mkdir(exist_ok=True)

    low_train = ROOT / "train" / "low_field" / "sample_017_lowfield.nii"
    high_train = ROOT / "train" / "high_field" / "sample_017_highfield.nii"
    low_test = ROOT / "test" / "low_field" / "sample_019_lowfield.nii"

    save_fig1(low_train, high_train, FIGS_DIR / "fig1.png")
    save_fig2(FIGS_DIR / "fig2.png")
    save_fig3(low_test, FIGS_DIR / "fig3.png")

    print("Generated:")
    print(f"  - {FIGS_DIR / 'fig1.png'}")
    print(f"  - {FIGS_DIR / 'fig2.png'}")
    print(f"  - {FIGS_DIR / 'fig3.png'}")


if __name__ == "__main__":
    main()

"""Inference for 2.5D U-Net MRI Super Resolution.

Features:
- Per-slice normalization matching competition metric
- 4-way TTA (horizontal flip, vertical flip, both)
- Optional histogram matching to training HF distribution
- Outputs submission CSV with base64-encoded slices

Usage:
    python inference.py --checkpoint experiments/unet_v1/checkpoints/best_model.pth --output submission.csv
    python inference.py --checkpoint best.pth --no_tta --output submission.csv
"""

import argparse
import os
import time

import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import yaml

from model import UNet2D
from dataset import normalize_slice
from extract_slices import slice_to_base64


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt['config']
    model = UNet2D(
        in_channels=config['data']['slice_context'],
        out_channels=1,
        base_features=config['model']['base_features'],
    ).to(device)

    # Load weights (handle both 'model' key formats)
    model.load_state_dict(ckpt['model'])
    model.eval()

    return model, config


def upsample_volume(vol, target_shape):
    """Trilinear upsample a 3D volume."""
    t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=target_shape, mode='trilinear', align_corners=False)
    return t.squeeze().numpy()


def compute_reference_histogram(config):
    """Compute reference histogram from training HF volumes for histogram matching.

    Returns a flattened array of normalized HF intensity values as reference.
    """
    from skimage.exposure import match_histograms
    hf_dir = config['data']['train_hf_dir']

    if not os.path.exists(hf_dir):
        print("Warning: training HF directory not found, skipping histogram matching")
        return None

    hf_files = sorted([f for f in os.listdir(hf_dir) if f.endswith(('.nii', '.nii.gz'))])
    val_samples = set(config['data']['val_samples'])

    # Collect normalized slices from training HF volumes
    all_values = []
    for hf_file in hf_files:
        sample_id = hf_file.split('_')[1]
        if sample_id in val_samples:
            continue

        hf_path = os.path.join(hf_dir, hf_file)
        hf_vol = nib.load(hf_path).get_fdata().astype(np.float32)
        target_shape = tuple(config['data']['target_shape'])
        if hf_vol.shape != target_shape:
            hf_vol = upsample_volume(hf_vol, target_shape)

        # Sample some slices (every 10th to keep memory reasonable)
        for z in range(0, hf_vol.shape[2], 10):
            s = hf_vol[:, :, z]
            s = normalize_slice(s)
            all_values.append(s.flatten())

    if not all_values:
        return None

    # Create a "reference" slice from the average distribution
    ref = np.concatenate(all_values)
    return ref


def predict_volume(model, lf_vol, config, device, use_tta=True, ref_histogram=None):
    """Predict a full high-field volume from a low-field volume.

    Args:
        model: trained UNet2D
        lf_vol: low-field volume, already upsampled to target shape
        config: configuration dict
        device: torch device
        use_tta: whether to use test-time augmentation
        ref_histogram: reference histogram for histogram matching (or None)

    Returns:
        predicted volume (179, 221, 200) as numpy float32
    """
    num_slices = config['data']['num_slices']
    half_ctx = config['data']['slice_context'] // 2

    predicted_slices = []

    for z_idx in range(num_slices):
        # Extract 5 adjacent slices with mirror padding
        input_slices = []
        for dz in range(-half_ctx, half_ctx + 1):
            z = z_idx + dz
            if z < 0:
                z = -z
            elif z >= num_slices:
                z = 2 * (num_slices - 1) - z
            z = max(0, min(z, num_slices - 1))
            s = lf_vol[:, :, z].copy()
            s = normalize_slice(s)
            input_slices.append(s)

        # Stack: (5, H, W)
        inp = np.stack(input_slices, axis=0).astype(np.float32)

        if use_tta:
            pred = _predict_with_tta(model, inp, device)
        else:
            pred = _predict_single(model, inp, device)

        # pred is (H, W) in [0, 1]

        # Histogram matching
        if ref_histogram is not None:
            try:
                from skimage.exposure import match_histograms
                pred = match_histograms(pred, ref_histogram.reshape(pred.shape) if ref_histogram.shape == pred.shape else ref_histogram[:pred.size].reshape(pred.shape))
            except Exception:
                # Fallback: simple quantile matching
                pred = _simple_hist_match(pred, ref_histogram)

        predicted_slices.append(pred)

    return np.stack(predicted_slices, axis=-1)  # (179, 221, 200)


def _predict_single(model, inp, device):
    """Single forward pass."""
    x = torch.from_numpy(inp).unsqueeze(0).to(device)  # (1, 5, H, W)
    with torch.no_grad(), autocast(enabled=True):
        pred = model(x)
    return pred[0, 0].cpu().numpy()  # (H, W)


def _predict_with_tta(model, inp, device):
    """4-way TTA: original, h-flip, v-flip, both flips."""
    preds = []

    # Original
    preds.append(_predict_single(model, inp, device))

    # Horizontal flip
    inp_hf = inp[:, :, ::-1].copy()
    p = _predict_single(model, inp_hf, device)
    preds.append(p[:, ::-1])  # flip back

    # Vertical flip
    inp_vf = inp[:, ::-1, :].copy()
    p = _predict_single(model, inp_vf, device)
    preds.append(p[::-1, :])  # flip back

    # Both flips
    inp_bf = inp[:, ::-1, ::-1].copy()
    p = _predict_single(model, inp_bf, device)
    preds.append(p[::-1, ::-1])  # flip back

    return np.mean(preds, axis=0)


def _simple_hist_match(source, reference):
    """Simple histogram matching via sorted value replacement."""
    s_shape = source.shape
    s_flat = source.flatten()
    r_flat = reference.flatten()

    # Sort both
    s_sorted_idx = np.argsort(s_flat)
    r_sorted = np.sort(r_flat)

    # Map source values to reference quantiles
    # Resample reference to match source length
    r_quantiles = np.interp(
        np.linspace(0, 1, len(s_flat)),
        np.linspace(0, 1, len(r_flat)),
        r_sorted
    )

    result = np.empty_like(s_flat)
    result[s_sorted_idx] = r_quantiles

    return result.reshape(s_shape).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='MRI Super Resolution Inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--no_tta', action='store_true')
    parser.add_argument('--no_hist_match', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save_volumes', action='store_true')
    parser.add_argument('--output_dir', type=str, default='predictions')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model, ckpt_config = load_model(args.checkpoint, device)

    # Load config for inference settings (use checkpoint config as base)
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = ckpt_config

    # Test directory
    test_dir = args.test_dir or config['data']['test_lf_dir']

    # Reference histogram for histogram matching
    ref_histogram = None
    if not args.no_hist_match and config.get('inference', {}).get('histogram_matching', True):
        print("Computing reference histogram from training data...")
        ref_histogram = compute_reference_histogram(config)
        if ref_histogram is not None:
            print(f"Reference histogram computed ({len(ref_histogram)} values)")

    # Find test volumes and keep one file per sample_id.
    candidates = sorted([f for f in os.listdir(test_dir) if f.endswith(('.nii', '.nii.gz'))])
    by_sample = {}
    for f in candidates:
        sample_id = '_'.join(f.split('_')[:2])
        prev = by_sample.get(sample_id)
        if prev is None or (prev.endswith('.nii') and f.endswith('.nii.gz')):
            by_sample[sample_id] = f
    test_files = [by_sample[k] for k in sorted(by_sample.keys())]
    print(f"Found {len(candidates)} candidate files, using {len(test_files)} unique test volumes")

    target_shape = tuple(config['data']['target_shape'])
    all_rows = []

    for tf in test_files:
        sample_id = '_'.join(tf.split('_')[:2])  # "sample_019"
        print(f"\nProcessing {sample_id}...")

        # Load and upsample
        lf_vol = nib.load(os.path.join(test_dir, tf)).get_fdata().astype(np.float32)
        print(f"  Original shape: {lf_vol.shape}")
        lf_vol = upsample_volume(lf_vol, target_shape)
        print(f"  Upsampled shape: {lf_vol.shape}")

        # Predict
        t0 = time.time()
        pred_vol = predict_volume(
            model, lf_vol, config, device,
            use_tta=not args.no_tta,
            ref_histogram=ref_histogram,
        )
        elapsed = time.time() - t0
        print(f"  Predicted in {elapsed:.1f}s, shape: {pred_vol.shape}")
        print(f"  Value range: [{pred_vol.min():.4f}, {pred_vol.max():.4f}]")

        # Save volume if requested
        if args.save_volumes:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, f"{sample_id}_predicted.nii.gz")
            nib.save(nib.Nifti1Image(pred_vol, np.eye(4)), out_path)
            print(f"  Saved to {out_path}")

        # Encode slices
        for z in range(config['data']['num_slices']):
            row_id = f"{sample_id}_slice_{z:03d}"
            slice_2d = pred_vol[:, :, z]
            b64 = slice_to_base64(slice_2d)
            all_rows.append({'row_id': row_id, 'prediction': b64})

    # Write submission CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nSubmission saved: {args.output} ({len(df)} rows)")


if __name__ == '__main__':
    main()

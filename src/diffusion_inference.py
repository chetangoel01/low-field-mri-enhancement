"""DDIM inference for conditional DDPM MRI super-resolution.

Loads a trained DiffusionUNet checkpoint and generates enhanced MRI volumes
using deterministic DDIM sampling (eta=0) conditioned on low-field MRI slices.

TTA: run DDIM on 4 flip variants of the LF conditioning; flip outputs back and
average. This exploits the stochastic diversity of different flip orientations.

Usage:
    python src/diffusion_inference.py --checkpoint experiments/diffusion_v1/checkpoints/best_model.pth --output submission.csv
    python src/diffusion_inference.py --checkpoint best.pth --ddim_steps 50 --no_tta --output sub.csv
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

from diffusion_model import DiffusionUNet
from diffusion_train import make_schedule, ddim_sample
from dataset import normalize_slice
from extract_slices import slice_to_base64


def load_diffusion_model(checkpoint_path, device):
    """Load DiffusionUNet from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    dcfg = config['diffusion']

    n_in = config['data']['slice_context'] + 1  # 5 LF context + 1 noisy HF
    model = DiffusionUNet(
        in_channels=n_in,
        out_channels=1,
        base_features=dcfg['base_features'],
        time_emb_dim=dcfg['time_emb_dim'],
    ).to(device)

    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, config


def upsample_volume(vol, target_shape):
    """Trilinear upsample 3D volume numpy array."""
    t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=target_shape, mode='trilinear', align_corners=False)
    return t.squeeze().numpy()


def predict_volume_ddim(model, lf_vol, config, schedule, n_steps, device,
                        use_tta=True, use_amp=True, batch_size=20):
    """Predict one enhanced HF volume from a LF volume using DDIM.

    Args:
        model:      DiffusionUNet in eval mode
        lf_vol:     (H, W, D_lf) raw low-field numpy volume
        config:     training config dict
        schedule:   DDPM schedule dict from make_schedule()
        n_steps:    DDIM denoising steps
        device:     torch device
        use_tta:    4-way flip TTA
        use_amp:    use autocast
        batch_size: slices to process per DDIM call

    Returns:
        predicted: (H, W, D_hf) numpy float32 volume in [0, 1]
    """
    target_shape = tuple(config['data']['target_shape'])  # (179, 221, 200)
    slice_context = config['data']['slice_context']        # 5
    half_ctx = slice_context // 2                          # 2
    num_slices = target_shape[2]                           # 200

    # Upsample LF to HF spatial size
    lf_up = upsample_volume(lf_vol, target_shape)
    # Per-slice normalization — must match dataset.py training normalization exactly.
    # DO NOT use volume-level percentile normalization here; the model was trained on
    # per-slice min/max [0,1] normalized inputs (normalize_slice in dataset.py).

    H, W, D = lf_up.shape
    predicted = np.zeros((H, W, num_slices), dtype=np.float32)

    # Build all slice indices first, then batch
    all_inputs = []  # list of (5, H, W) numpy arrays
    for z in range(num_slices):
        slices = []
        for dz in range(-half_ctx, half_ctx + 1):
            sz = z + dz
            # Mirror padding at boundaries
            if sz < 0:
                sz = -sz
            elif sz >= D:
                sz = 2 * (D - 1) - sz
            sz = max(0, min(sz, D - 1))
            slices.append(normalize_slice(lf_up[:, :, sz]))
        inp = np.stack(slices, axis=0).astype(np.float32)  # (5, H, W)
        all_inputs.append(inp)

    # Process in batches
    def _run_batch(inp_batch_np):
        """Run DDIM on a batch of LF conditioning slices."""
        inp_tensor = torch.from_numpy(np.stack(inp_batch_np)).to(device)  # (B, 5, H, W)
        pred = ddim_sample(model, inp_tensor, schedule, n_steps, device, use_amp)
        return pred.squeeze(1).cpu().numpy()  # (B, H, W)

    if not use_tta:
        for start in range(0, num_slices, batch_size):
            end = min(start + batch_size, num_slices)
            batch_inp = all_inputs[start:end]
            preds = _run_batch(batch_inp)
            for i, z in enumerate(range(start, end)):
                predicted[:, :, z] = preds[i]
    else:
        # 4 flip variants: original, h-flip, v-flip, both
        for flip_h, flip_v in [(False, False), (True, False), (False, True), (True, True)]:
            flip_preds = np.zeros_like(predicted)
            for start in range(0, num_slices, batch_size):
                end = min(start + batch_size, num_slices)
                batch_inp = []
                for inp in all_inputs[start:end]:
                    inp_f = inp.copy()
                    if flip_h:
                        inp_f = inp_f[:, :, ::-1].copy()
                    if flip_v:
                        inp_f = inp_f[:, ::-1, :].copy()
                    batch_inp.append(inp_f)
                preds = _run_batch(batch_inp)
                for i, z in enumerate(range(start, end)):
                    pred_slice = preds[i]
                    if flip_h:
                        pred_slice = pred_slice[:, ::-1]
                    if flip_v:
                        pred_slice = pred_slice[::-1, :]
                    flip_preds[:, :, z] = pred_slice
            predicted += flip_preds

        predicted /= 4.0  # average over 4 TTA variants

    return np.clip(predicted, 0.0, 1.0)


def generate_submission(model, config, schedule, test_dir, output_path,
                        n_steps, device, use_tta=True, use_amp=True,
                        batch_size=20, save_volumes=False, output_dir='predictions'):
    """Run inference on all test volumes and write submission CSV."""
    target_shape = tuple(config['data']['target_shape'])
    test_samples = ['sample_019', 'sample_020', 'sample_021', 'sample_022', 'sample_023']

    if save_volumes:
        os.makedirs(output_dir, exist_ok=True)

    all_rows = []

    for sample_id in test_samples:
        lf_path = os.path.join(test_dir, f'{sample_id}_lowfield.nii.gz')
        if not os.path.exists(lf_path):
            print(f"Warning: {lf_path} not found, skipping")
            continue

        print(f"Processing {sample_id}...", end=' ', flush=True)
        t0 = time.time()

        lf_vol = nib.load(lf_path).get_fdata().astype(np.float32)
        enhanced = predict_volume_ddim(
            model, lf_vol, config, schedule, n_steps, device,
            use_tta=use_tta, use_amp=use_amp, batch_size=batch_size,
        )
        # enhanced: (H, W, 200) in [0, 1]

        elapsed = time.time() - t0
        print(f"done in {elapsed:.1f}s  shape={enhanced.shape}")

        if save_volumes:
            out_nii = nib.Nifti1Image(enhanced, np.eye(4))
            nib.save(out_nii, os.path.join(output_dir, f'{sample_id}_predicted.nii.gz'))

        # Encode each of the 200 axial slices
        num_slices = target_shape[2]
        for z in range(num_slices):
            slice_2d = enhanced[:, :, z]
            row_id = f'{sample_id}_slice_{z:03d}'
            b64 = slice_to_base64(slice_2d)
            all_rows.append({'row_id': row_id, 'prediction': b64})

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}  ({len(df)} rows)")
    return df


def main():
    parser = argparse.ArgumentParser(description='DDIM inference for diffusion MRI SR')
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--config',      type=str, default=None,
                        help='Override config (default: use config from checkpoint)')
    parser.add_argument('--test_dir',    type=str, default=None)
    parser.add_argument('--output',      type=str, default='submission.csv')
    parser.add_argument('--ddim_steps',  type=int, default=None,
                        help='DDIM steps (default: from config diffusion.ddim_steps)')
    parser.add_argument('--batch_size',  type=int, default=20,
                        help='Slices per DDIM batch (default: 20)')
    parser.add_argument('--no_tta',      action='store_true')
    parser.add_argument('--device',      type=str, default=None)
    parser.add_argument('--save_volumes', action='store_true')
    parser.add_argument('--output_dir',  type=str, default='predictions')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, config = load_diffusion_model(args.checkpoint, device)

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    dcfg = config['diffusion']
    n_steps = args.ddim_steps or dcfg['ddim_steps']
    test_dir = args.test_dir or config['data']['test_lf_dir']
    use_amp = config['training']['use_amp']

    schedule = make_schedule(dcfg['T'], dcfg['beta_start'], dcfg['beta_end'], device)
    print(f"DDIM steps: {n_steps}, TTA: {not args.no_tta}")

    generate_submission(
        model=model,
        config=config,
        schedule=schedule,
        test_dir=test_dir,
        output_path=args.output,
        n_steps=n_steps,
        device=device,
        use_tta=not args.no_tta,
        use_amp=use_amp,
        batch_size=args.batch_size,
        save_volumes=args.save_volumes,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()

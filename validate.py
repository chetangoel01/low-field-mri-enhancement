"""Local validation using competition metric.

Runs inference on held-out validation volumes (017, 018) and scores
against ground truth using the exact competition metric (metric.py).

Usage:
    python validate.py --checkpoint experiments/unet_v1/checkpoints/best_model.pth
"""

import argparse
import os

import numpy as np
import nibabel as nib
import pandas as pd
import torch
import yaml

from inference import load_model, upsample_volume, predict_volume, compute_reference_histogram
from extract_slices import slice_to_base64, volume_to_submission_rows
from metric import score as competition_score


def main():
    parser = argparse.ArgumentParser(description='Local validation with competition metric')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--no_tta', action='store_true')
    parser.add_argument('--no_hist_match', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model, ckpt_config = load_model(args.checkpoint, device)

    # Config
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = ckpt_config

    val_samples = config['data']['val_samples']  # ["017", "018"]
    target_shape = tuple(config['data']['target_shape'])

    # Reference histogram
    ref_histogram = None
    if not args.no_hist_match and config.get('inference', {}).get('histogram_matching', True):
        print("Computing reference histogram...")
        ref_histogram = compute_reference_histogram(config)

    # Load ground truth from train.csv
    train_csv_path = config['data']['train_csv']
    if not os.path.exists(train_csv_path):
        print(f"Error: {train_csv_path} not found. Cannot validate.")
        return

    train_df = pd.read_csv(train_csv_path)
    print(f"Loaded train.csv: {len(train_df)} rows")

    # Run inference on validation volumes
    all_pred_rows = []

    lf_dir = config['data']['train_lf_dir']
    for sample_num in val_samples:
        sample_id = f"sample_{sample_num}"
        print(f"\nProcessing {sample_id}...")

        # Load low-field volume
        lf_path = os.path.join(lf_dir, f'{sample_id}_lowfield.nii')
        if not os.path.exists(lf_path):
            lf_path = lf_path + '.gz'
        if not os.path.exists(lf_path):
            print(f"  Warning: {lf_path} not found, skipping")
            continue

        lf_vol = nib.load(lf_path).get_fdata().astype(np.float32)
        lf_vol = upsample_volume(lf_vol, target_shape)

        # Predict
        pred_vol = predict_volume(
            model, lf_vol, config, device,
            use_tta=not args.no_tta,
            ref_histogram=ref_histogram,
        )
        print(f"  Predicted shape: {pred_vol.shape}, range: [{pred_vol.min():.4f}, {pred_vol.max():.4f}]")

        # Encode as submission rows
        rows = volume_to_submission_rows(pred_vol, sample_id)
        all_pred_rows.extend(rows)

    # Build submission DataFrame
    submission_df = pd.DataFrame(all_pred_rows)
    print(f"\nPrediction rows: {len(submission_df)}")

    # Filter ground truth to validation samples only
    # train.csv has columns: row_id, ground_truth
    val_row_ids = set(submission_df['row_id'])
    solution_df = train_df[train_df['row_id'].isin(val_row_ids)].copy()

    print(f"Ground truth rows: {len(solution_df)}")

    if len(solution_df) == 0:
        print("Error: No matching ground truth found in train.csv")
        return

    # Compute competition score
    try:
        final_score = competition_score(solution_df, submission_df, 'row_id')
        print(f"\n{'='*50}")
        print(f"Competition Score: {final_score:.4f}")
        print(f"{'='*50}")

        # Also compute per-sample scores
        for sample_num in val_samples:
            sample_id = f"sample_{sample_num}"
            sample_pred = submission_df[submission_df['row_id'].str.startswith(sample_id)]
            sample_gt = solution_df[solution_df['row_id'].str.startswith(sample_id)]
            if len(sample_pred) > 0 and len(sample_gt) > 0:
                s = competition_score(sample_gt, sample_pred, 'row_id')
                print(f"  {sample_id}: {s:.4f}")

    except Exception as e:
        print(f"Error computing score: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

# Final Code Submission - Best Model (2.5D U-Net, Phase 2)

This folder contains the training/inference code for the best-performing model described in the report: a 2.5D residual U-Net trained with `0.16 * Charbonnier + 0.84 * MS-SSIM`.

## Included Files

- `train.py` - training loop (EMA, AMP, cosine LR, validation, checkpointing)
- `inference.py` - test-time inference and CSV submission generation
- `model.py` - 2.5D residual U-Net architecture
- `dataset.py` - slice-based dataset and preprocessing utilities
- `losses.py` - Charbonnier and MS-SSIM losses
- `extract_slices.py` - submission encoding utilities
- `metric.py` - competition metric helper
- `config_best_unet_phase2_v4.yaml` - best model training configuration
- `requirements.txt` - Python dependencies
- `train.csv` - sample metadata file used by the project

## Expected Data Layout

From this folder's parent (or adjust paths in config):

- `train/low_field/*.nii`
- `train/high_field/*.nii`
- `test/low_field/*.nii`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train Best Model

```bash
python train.py --config config_best_unet_phase2_v4.yaml --experiment unet_phase2_v4
```

Resume training:

```bash
python train.py --config config_best_unet_phase2_v4.yaml --experiment unet_phase2_v4 --resume experiments/unet_phase2_v4/checkpoints/checkpoint_latest.pth
```

## Run Inference

```bash
python inference.py --config config_best_unet_phase2_v4.yaml --checkpoint experiments/unet_phase2_v4/checkpoints/best_model.pth --output submission.csv
```

## Notes

- Validation split in config: samples `017`, `018`
- Best reported internal validation MS-SSIM for this run family: `0.6943` (EMA)
- Best leaderboard entry in report uses the same 2.5D U-Net family

# Low-Field MRI Enhancement

Enhancing low-field MRI scans (112x138x40, 1.6x1.6x5.0mm voxels) to high-field quality (179x221x200, 1mm isotropic) using deep learning. Built for the CSGY 9223 Neuroinformatics MRI Super Resolution Challenge.

## Competition Metric

```
Score = 0.5 * SSIM + 0.5 * (PSNR / 50)
```

Computed per-slice on [0,1] normalized axial slices. SSIM uses global statistics (no windowing). Maximum possible score is 1.0.

| Method | Score |
|--------|-------|
| Bicubic interpolation (baseline) | 0.456 |
| 2.5D U-Net (best single model) | 0.4525 |

## Model Architecture

### 2.5D U-Net

The core model is a 2.5D U-Net that processes **5 adjacent axial slices** from trilinear-upsampled low-field MRI and predicts the center high-field slice. The "2.5D" approach gives the network z-axis context without the parameter explosion of full 3D convolutions.

```
Input:  (B, 5, 179, 221)  -- 5 adjacent axial slices
Output: (B, 1, 179, 221)  -- predicted center slice

Encoder:    5ch -> 64 -> 128 -> 256 -> 512  (MaxPool2d between stages)
Bottleneck: 1024 channels
Decoder:    512 -> 256 -> 128 -> 64         (bilinear upsample + skip concat)
Final:      64 -> 1                          (1x1 conv)
Residual:   output += center input slice
```

- **Parameters**: ~31.4M
- **Key design**: Residual learning from center input slice -- network only learns the enhancement delta
- **Weight init**: Kaiming normal (fan_out, ReLU)
- **Decoder upsampling**: Bilinear interpolation (not transposed convolution) handles non-power-of-2 dimensions (179x221) gracefully

### Why 2.5D?

With only 18 paired training volumes, the 2.5D slice-based approach yields **18 x 200 = 3,600 training examples** instead of ~800 volumetric patches. This data efficiency proved critical for stable training.

## Training

Single-phase optimization (no GAN) with Charbonnier L1 + competition-matched SSIM loss:

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (betas=0.9/0.99, weight_decay=1e-4) |
| LR schedule | Cosine annealing (1e-3 to 1e-6) with 500-step linear warmup |
| Batch size | 16 |
| Epochs | 400 |
| Loss | Charbonnier L1 (weight=1.0) + Competition SSIM (weight=1.0) |
| AMP | Enabled |
| EMA | Decay=0.999 |
| Gradient clipping | max_norm=1.0 |

**Augmentation** (train only): random 160x160 crop, 90-degree rotations, horizontal/vertical flip, intensity scaling [0.9-1.1] (p=0.5), Gaussian noise (std=0.01, p=0.3). Cropping to square patches enables 90-degree rotations despite non-square native slices (179x221).

**Validation**: All 18 paired volumes used for training (`val_samples: []`). Evaluation performed via Kaggle leaderboard on 5 held-out test volumes.

### Loss Functions

- **Charbonnier L1**: `sqrt((pred - target)^2 + eps^2)` -- smoother gradient near zero than standard L1
- **Competition SSIM**: Matches the exact competition metric (global mean/variance, C1=0.01^2, C2=0.03^2, per-slice). Loss = 1 - SSIM

## Data Pipeline

**Normalization**: Per-volume percentile clipping (1st/99th percentile), then scaled to [0,1].

**Input preparation**: Low-field volumes are trilinear-upsampled from native resolution (112x138x40) to high-field grid (179x221x200). For each slice index z, 5 adjacent axial slices (z-2 to z+2) are extracted with mirror-padding at volume boundaries.

**Data layout**:
```
train/
  low_field/     # 18 low-field NIfTI volumes (samples 001-018)
  high_field/    # 18 paired high-field NIfTI volumes
test/
  low_field/     # 5 test volumes (samples 019-023)
```

## Inference

Slice-by-slice prediction with 4-way test-time augmentation (original + horizontal flip + vertical flip + both flips):

1. Load test volume, trilinear upsample to (179, 221, 200)
2. For each z in 0-199: extract 5 adjacent slices, predict, average TTA predictions
3. Encode 200 slices as base64, write submission CSV

## Project Structure

```
.
├── model.py              # 2.5D U-Net architecture
├── losses.py             # Charbonnier L1 + Competition SSIM losses
├── simple_dataset.py     # SliceMRIDataset with augmentation
├── train.py              # Training loop (EMA, AMP, cosine LR, TensorBoard)
├── inference.py          # Inference with TTA, outputs submission CSV
├── config.yaml           # All hyperparameters and paths
├── requirements.txt      # Python dependencies
├── extract_slices.py     # [Competition-provided] Slice encoding/decoding
├── metric.py             # [Competition-provided] SSIM + PSNR scoring
├── sample_submission.csv # [Competition-provided] Bicubic baseline submission
├── hpc/                  # NYU HPC (Torch cluster) SLURM scripts
│   ├── train_job.sbatch
│   └── inference_job.sbatch
├── audit/                # Experiment run logs and snapshots
└── LITERATURE.md         # Domain background and reference approaches
```

## Usage

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training

```bash
# Start training
python train.py --config config.yaml --experiment unet_v1

# Resume from checkpoint
python train.py --config config.yaml --experiment unet_v1 --resume experiments/unet_v1/checkpoints/checkpoint_latest.pth
```

### Inference

```bash
# Generate submission CSV
python inference.py --checkpoint experiments/unet_v1/checkpoints/best_model.pth --output submission.csv

# Without TTA (faster)
python inference.py --checkpoint best_model.pth --no_tta --output submission.csv

# Save predicted NIfTI volumes
python inference.py --checkpoint best_model.pth --save_volumes --output_dir predictions
```

### NYU HPC

```bash
# Training
EXPERIMENT_NAME="unet_v1" sbatch hpc/train_job.sbatch

# Inference
EXPERIMENT_NAME="unet_v1" sbatch hpc/inference_job.sbatch

# Monitor
squeue -u $USER
tail -f experiments/unet_v1/logs/training.log
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- nibabel, numpy, pandas, scikit-image, tensorboard

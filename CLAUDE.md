# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Style Preferences

- Always output shell commands as a single line. Never split commands with `\` line continuations.

## Project Overview

MRI Super Resolution Challenge: enhances low-field MRI (112x138x40, 1.6x1.6x5.0mm voxels) to high-field quality (179x221x200, 1mm isotropic). Competition metric: `0.5 * SSIM + 0.5 * (PSNR / 50)`. Submission format: 200 base64-encoded axial slices per volume, row_id `sample_XXX_slice_YYY`.

**Current active model**: 2.5D U-Net (`model_type: "unet_2d"` in config.yaml). Previous model was 3D ESRGAN (still available via `model_type: "esrgan"`).

**Baseline score**: Bicubic interpolation (sample_submission.csv) = **~0.456**. Best achieved so far: **0.4525** (single U-Net, below baseline — see Leaderboard outcomes).

**Reference**: `LITERATURE.md` contains the full project brief — domain background, professor's suggested approaches, expected score ranges, data expansion strategies, and the exact competition metric implementation. Consult it for strategic planning.

## Commands

```bash
# Setup (ALWAYS use a venv for local work, never --break-system-packages)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Data pipeline (5 stages: analyze degradation → download IXI → synthesize low-field → organize → test loaders)
python run_pipeline.py --all --max-volumes 100
python run_pipeline.py --stage 3                      # run single stage

# Training — U-Net (single phase, config.yaml already set to unet_2d)
python train.py --config config.yaml --experiment unet_v1
python train.py --config config.yaml --experiment unet_v1 --resume experiments/unet_v1/checkpoints/checkpoint_latest.pth

# Training — ESRGAN (change config.yaml model_type to "esrgan" first)
python train.py --config config.yaml --experiment my_run --phases pretrain gan finetune

# Inference (auto-detects model type from checkpoint)
python inference.py --config config.yaml --checkpoint experiments/unet_v1/checkpoints/best_model.pth --output submission.csv
python inference.py --checkpoint path1.pt,path2.pt    # ensemble (ESRGAN only)
python inference.py --checkpoint best.pt --no_tta --save_volumes  # save NIfTI outputs

# Modal cloud (A100)
modal run modal_app.py --action upload --local-data ./ --detach
modal run modal_app.py --action train --experiment NAME --detach
modal run modal_app.py --action full --experiment NAME --detach

# NYU HPC (Torch) — see "NYU HPC Reference" section below for full details
EXPERIMENT_NAME="unet_v1" PHASES="pretrain" sbatch hpc/train_job.sbatch
EXPERIMENT_NAME="unet_v1" sbatch hpc/inference_job.sbatch
```

### Exact CLI Arguments (verified)

**train.py**: `--config`, `--experiment`, `--resume`, `--phases {pretrain,gan,finetune}`, `--device`, `--batch-size`. For U-Net, phases defaults to `['pretrain']` (single phase). Batch size from config.yaml (16 for U-Net, 2 for ESRGAN).

**inference.py**: `--config`, `--checkpoint` (required), `--test_dir`, `--output`, `--patch_size`, `--overlap`, `--no_tta`, `--device`, `--save_volumes`, `--output_dir`. Auto-detects model type from checkpoint.

**run_pipeline.py**: `--config`, `--all`, `--stage`, `--stages`, `--max-volumes`, `--skip-download`, `--check-deps`.

## Architecture

### Model Selection

Controlled by `config.yaml` → `esrgan.model.model_type`:
- `"unet_2d"` — **Active**. 2.5D U-Net, processes axial slices. Fast training, no GAN instability.
- `"esrgan"` — 3D RRDB network, processes volumetric patches. Multi-phase training with GAN.

---

### 2.5D U-Net Generator (`models/unet.py`) — ACTIVE

Processes 5 adjacent axial slices from trilinear-upsampled low-field MRI and predicts the center high-field slice. Residual learning adds the center input slice to the output.

- **Input**: (B, 5, 179, 221) — 5 adjacent axial slices
- **Output**: (B, 1, 179, 221) — predicted center slice
- **Encoder**: 4 downsampling blocks (64→128→256→512), each = 2x Conv2d(3x3) + BN + ReLU + MaxPool2d
- **Bottleneck**: 1024 channels
- **Decoder**: 4 upsampling blocks (bilinear interpolate + concat skip + 2x Conv2d)
- **Final**: Conv2d(1x1) → 1 channel
- **Residual connection**: output += center input slice
- **Weight init**: Kaiming normal (fan_out, relu)
- **Parameters**: ~31.4M (base_features=64)

Key advantage: 16 volumes × 200 slices = **3,200 training examples** (vs ~800 3D patches for ESRGAN).

### U-Net Training (`train.py` with `model_type: "unet_2d"`)

Single-phase training (no GAN, no discriminator):

| Setting | Value |
|---------|-------|
| Phase | pretrain only (single phase) |
| Epochs | 150 |
| LR | 1e-3, cosine annealed to ~1e-7 |
| Losses | Charbonnier (L1=1.0) + CompetitionSSIM (SSIM=0.5) |
| Optimizer | Adam, betas=(0.9, 0.99) |
| Batch size | 16 |
| EMA | decay=0.999 |
| Gradient clipping | max_norm=1.0 |
| AMP | enabled |
| LR warmup | 500 steps (linear) |

### U-Net Data (`simple_dataset.py` → `SliceMRIDataset`)

- Loads from `train/low_field` and `train/high_field` directly
- **Normalization**: Per-volume percentile clipping (1st/99th), then [0,1] range
- **Upsampling**: Low-field trilinear interpolated to (179, 221, 200)
- **Slicing**: For each z (0-199), extracts 5 adjacent axial slices as input, center high-field slice as target
- **Mirror-padding** at volume boundaries (z < 2 or z > 197)
- **Augmentation** (train only): horizontal flip, vertical flip, intensity scaling [0.9-1.1]
- **No 90-degree rotations** — slices are non-square (179×221), rotation changes dimensions and breaks batching
- **Validation holdout**: samples 017, 018
- **DataLoader**: batch_size=16, num_workers=4, pin_memory=True
- Volumes cached in memory after first load

### U-Net Inference (`inference.py` → `SliceBasedInference`)

Slice-by-slice (no patches, no blending needed):
1. Load test volume, trilinear upsample to (179, 221, 200), normalize
2. For each z (0-199): extract 5 adjacent slices (mirror-pad at boundaries), predict center slice
3. Stack all 200 predictions into (179, 221, 200) volume
4. TTA: horizontal flip only (average original + flipped prediction)
5. Clip to [0,1], encode with `extract_slices.py`

---

### 3D ESRGAN Generator (`models/generator.py`) — LEGACY

3D RRDB network, 1-channel in → 1-channel out. Uses **lite** variant when active.

| Parameter | Standard | Lite |
|-----------|----------|------|
| Base features | 64 | 48 |
| RRDB blocks | 16 | 8 |
| Growth rate | 32 | 24 |

- **Residual scaling (beta)**: 0.2 at both DenseBlock and RRDB levels
- **Each RRDB block**: 3 DenseBlock3D modules, each with 5 Conv3d layers (growth-based concatenation)
- **All convolutions**: 3x3x3 kernel, padding=1, no batch norm
- **Activation**: LeakyReLU(0.2) throughout
- **Channel Attention**: SE blocks (reduction=16)
- **Z-axis refinement**: Two 1x1x3 convolutions with LeakyReLU
- **Weight init**: Kaiming normal (a=0.2, fan_in, leaky_relu)

### ESRGAN Discriminator (`models/discriminator.py`)

3D Relativistic PatchGAN with spectral normalization on all Conv3d layers. Only used during ESRGAN GAN phase.

### ESRGAN Training (3-phase curriculum)

| Phase | Epochs | LR | Losses | Disc |
|-------|--------|----|--------|------|
| pretrain | 1-50 | 2e-4 | L1 + SSIM + FFT | No |
| gan | 51-100 | 1e-4 (G+D) | L1 + SSIM + perceptual + adversarial + FFT | Yes |
| finetune | 101-120 | 1e-5 | L1 + SSIM + perceptual (adversarial~0) | No |

Batch size: 2, Patch size: 64x64x64.

### ESRGAN Data (`simple_dataset.py` → `SimpleMRIDataset`)

- Extracts random 3D patches (64x64x64) from volumes
- Patches per volume: 50 (train), 25 (val)
- Augmentation: random flips (all axes), 90-degree rotations, intensity scaling

### ESRGAN Inference (`inference.py` → `PatchBasedInference`)

Patch-based with configurable overlap (default 75%) and Gaussian-weighted blending. TTA via 4 x/y flip combinations. Supports multi-checkpoint ensemble.

---

### Losses (`models/losses.py`)

| Loss | Used by | Details |
|------|---------|---------|
| Charbonnier | U-Net, ESRGAN | sqrt((pred-target)^2 + 1e-6^2) |
| CompetitionSSIMLoss | U-Net, ESRGAN | Global stats (no windowing), matches competition metric. C1=1e-4, C2=9e-4. |
| PerceptualLoss3D | ESRGAN only | VGG16 pretrained, layers relu1_2/relu2_2/relu3_3. Lazily initialized (only when weight > 0). |
| FFTLoss3D | ESRGAN only | L1 on magnitudes of 3D FFT |
| RelativisticAverageLoss | ESRGAN only | BCEWithLogitsLoss on relativistic scores. Lazily initialized. |

**U-Net loss weights**: L1=1.0, SSIM=0.5 (no perceptual, adversarial, or FFT — avoids VGG16 download)
**ESRGAN loss weights**: L1=1.0, SSIM=0.5, perceptual=0.1, adversarial=0.005, FFT=0.1

### Evaluation
- `metric.py`: Competition scoring function (`score(solution, submission, row_id_column_name)`)
- `extract_slices.py`: `volume_to_submission_rows()` and `create_submission_df()` for submission format
- `utils/metrics.py`: PSNR, SSIM, NRMSE, MAE with optional brain masking

## Configuration

`config.yaml` controls everything. Key field: `esrgan.model.model_type` selects between `"unet_2d"` and `"esrgan"`.

**U-Net-specific config keys**: `slice_context` (default 5), `base_features` (default 64)
**ESRGAN-specific config keys**: `generator_type`, `num_features`, `num_blocks`, `growth_rate`, `disc_features`, `disc_blocks`, `use_channel_attention`, `use_z_refinement`
**Shared keys**: `batch_size`, `use_amp`, `gradient_clip_norm`, `lr_warmup_steps`, `ema_decay`, `adam_betas`, `pretrain_epochs`, `pretrain_lr`, loss weights

Checkpoints include `model_type` field so inference auto-detects which model to load.

## Data Layout

- `train/low_field/`, `train/high_field/`: 18 real paired NIfTI volumes (samples 001-018)
- `test/low_field/`: 5 test volumes (samples 019-023)
- `external_data/ixi/`: Downloaded IXI high-field volumes
- `expanded_dataset/`: Organized real + synthetic training data
- `experiments/`: Checkpoints, logs, TensorBoard, predictions
- Validation holdout: samples 017, 018

**Local vs HPC**: The local directory contains only competition-provided files (`extract_slices.py`, `metric.py`, `sample_submission.csv`, `train.csv`), training/test data, and the `audit/` folder. The full codebase (`train.py`, `inference.py`, `config.yaml`, `models/`, `simple_dataset.py`, `utils/`, `hpc/`, `run_pipeline.py`, `modal_app.py`) lives on HPC at `/scratch/cg4652/mri-super-resolution/`. Use the rsync commands in the HPC section to sync between local and HPC. Competition-provided files (`extract_slices.py`, `metric.py`) are upstream — do not modify.

## Design Decisions

### U-Net (current)
- 2.5D approach: 2D U-Net with z-context via adjacent slices, avoids 3D conv parameter explosion
- 5 adjacent axial slices as input channels — gives z-context without full 3D processing
- Residual learning from center input slice — network only learns the enhancement delta
- BatchNorm is fine here — large batch size (16) provides stable statistics
- No 90-degree rotation augmentation — non-square slices (179×221) would change dimensions
- No GAN — single-phase L1+SSIM training avoids instability with small dataset
- Bilinear upsampling in decoder (not transposed conv) — handles non-power-of-2 dimensions (179×221) gracefully
- TTA limited to horizontal flip — fast and meaningful for brain MRI

### ESRGAN (legacy)
- All convolutions are 3D (Conv3d) — full volumetric processing
- Generator uses no batch norm to prevent MRI artifacts; discriminator uses spectral norm
- Z-axis refinement uses 1x1x3 kernels because input z-spacing (5mm) is 5x coarser than xy (1mm)
- Competition SSIM loss uses global stats (not windowed) to match exact competition metric
- VGG16 perceptual loss operates on 2D axial slices (pretrained features, not random weights)
- TTA skips z-axis flips because z-direction is anisotropic

### Shared
- Rician noise model for degradation synthesis (MRI-appropriate, not Gaussian)
- Gradient-weighted patch sampling via Sobel filters prefers structurally detailed regions (pipeline only)
- Per-volume percentile (1st/99th) normalization to [0,1]

## Deployment

Two equivalent deployment paths documented in `MODAL_SETUP.md` and `HPC_SETUP.md`:
- **Modal**: Cloud A100 GPU, `modal run modal_app.py` commands, pay-per-use
- **NYU HPC (Torch)**: Free with allocation, conda environment, `sbatch hpc/*.sbatch`, uses `$SCRATCH` storage

---

## NYU HPC Reference (Torch Cluster)

### Connection
```bash
ssh cg4652@login.torch.hpc.nyu.edu
# SSH keys NOT supported — uses browser MFA (microsoft.com/devicelogin + Duo)
# Requires NYU VPN if off-campus
```

### Cluster Specs
- **GPUs**: 232x NVIDIA H200 + 272x NVIDIA L40S (504 total)
- **Max per user**: 24 GPUs for jobs under 48 hours
- **Do NOT specify partitions manually** — scheduler auto-dispatches
- **Exception**: Use `--partition=h200_tandon` if jobs get stuck on `l40s_publ` with `QOSGrpGRES`

### User-Specific Configuration (cg4652)
- **NetID**: cg4652
- **Email**: cg4652@nyu.edu
- **SLURM accounts**: `torch_pr_62_general`, `torch_pr_62_tandon_priority`
- **Active account in sbatch scripts**: `torch_pr_62_tandon_priority`
- **Project dir on HPC**: `/scratch/cg4652/mri-super-resolution`
- **Conda env**: `mri-sr` (at `/scratch/cg4652/miniconda3/envs/mri-sr`)
- **Python environment**: PyTorch 2.5.1+cu121, Python via conda (NOT Singularity)

### Storage
| Path | Var | Quota | Backup | Purge |
|------|-----|-------|--------|-------|
| `/home/cg4652` | `$HOME` | 50GB / 30K files | Yes | No |
| `/scratch/cg4652` | `$SCRATCH` | 5TB / 5M files | No | 60 days inactive |
| `/archive/cg4652` | `$ARCHIVE` | 2TB / 20K files | Yes | No |

Check usage: `myquota`

### HPC Directory Layout
```
/scratch/cg4652/
├── mri-super-resolution/     # This project
│   ├── hpc/                  # SLURM sbatch scripts
│   ├── experiments/          # Training outputs
│   ├── train/                # Training data (low_field/, high_field/)
│   ├── test/                 # Test data (low_field/)
│   ├── logs/                 # SLURM job logs (train_JOBID.out/.err)
│   └── ...                   # All other project files
├── miniconda3/               # Conda installation
├── conda_envs/               # Additional conda envs
└── conda_pkgs/               # Conda package cache
```

### sbatch Scripts (all use conda, NOT Singularity)
All scripts hardcode `--account=torch_pr_62_tandon_priority` and activate the `mri-sr` conda env.

| Script | GPU | Time | Mem | Purpose |
|--------|-----|------|-----|---------|
| `hpc/train_job.sbatch` | 1 | 24h | 64GB | Training (any phase combo) |
| `hpc/inference_job.sbatch` | 1 | 2h | 32GB | Inference + submission CSV |
| `hpc/data_prep_job.sbatch` | 0 | 4h | 64GB | 5-stage data pipeline |
| `hpc/full_pipeline_job.sbatch` | 1 | 24h | 64GB | Data prep + train + infer |

### Common HPC Workflows

```bash
# Connect
ssh cg4652@login.torch.hpc.nyu.edu
cd /scratch/cg4652/mri-super-resolution

# --- U-Net training (current) ---
EXPERIMENT_NAME="unet_v1" PHASES="pretrain" sbatch hpc/train_job.sbatch

# Resume U-Net training
EXPERIMENT_NAME="unet_v1" PHASES="pretrain" RESUME_FROM="experiments/unet_v1/checkpoints/checkpoint_latest.pth" sbatch hpc/train_job.sbatch

# U-Net inference
EXPERIMENT_NAME="unet_v1" sbatch hpc/inference_job.sbatch

# --- ESRGAN training (legacy, change config.yaml model_type to "esrgan" first) ---
EXPERIMENT_NAME="esrgan_v2" PHASES="pretrain" sbatch hpc/train_job.sbatch
EXPERIMENT_NAME="esrgan_v2" PHASES="gan finetune" RESUME_FROM="experiments/esrgan_v2/checkpoints/checkpoint_latest.pth" sbatch hpc/train_job.sbatch

# Monitor
squeue -u $USER                                    # job status
tail -f experiments/unet_v1/logs/training.log       # training metrics (best)
tail -f logs/train_JOBID.out                        # stdout
tail -f logs/train_JOBID.err                        # errors
sacct -u $USER --starttime=today --format=JobID,JobName,State,Elapsed,NodeList

# Cancel
scancel JOBID

# Interactive GPU session
srun --account=torch_pr_62_tandon_priority --gres=gpu:1 --cpus-per-task=4 --mem=32GB --time=02:00:00 --pty /bin/bash
source /scratch/cg4652/miniconda3/etc/profile.d/conda.sh
conda activate mri-sr

# Sync code from local machine to HPC
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'experiments' --exclude 'expanded_dataset' --exclude '.git' "/Users/chetangoel/Desktop/CSGY 9223 - Neuroinformatics/MRI Super Resolution Challenge/" cg4652@dtn.torch.hpc.nyu.edu:/scratch/cg4652/mri-super-resolution/

# Sync results from HPC to local
rsync -avz cg4652@dtn.torch.hpc.nyu.edu:/scratch/cg4652/mri-super-resolution/experiments/ "/Users/chetangoel/Desktop/CSGY 9223 - Neuroinformatics/MRI Super Resolution Challenge/experiments/"
```

### Troubleshooting
- **`QOSGrpGRES` pending**: GPU quota hit. Wait for previous jobs to clear, or force partition: `sbatch --partition=h200_tandon hpc/train_job.sbatch`
- **`ModuleNotFoundError`**: Install in conda env: `conda activate mri-sr && pip install <package>`
- **Job finishes instantly**: Check `logs/train_JOBID.err` for the actual error
- **`unrecognized arguments`**: The HPC code may be out of sync with local. Rsync the latest code first.
- **Low GPU utilization**: Jobs with low GPU utilization get auto-canceled on Torch. Batch size 16 (U-Net) or 2 (ESRGAN 3D) should be fine.
- **`REMOTE HOST IDENTIFICATION HAS CHANGED`**: HPC changed host keys. Fix: `ssh-keygen -R dtn.torch.hpc.nyu.edu` then retry.
- **Tensor size mismatch in DataLoader** (`stack expects each tensor to be equal size`): Usually caused by augmentation changing spatial dims. Non-square slices (179×221) cannot use 90-degree rotations.

---

## Training Results Log

### U-Net Baseline Run
- **Status**: Completed.
- **Setup**: U-Net slice-based training with Charbonnier plus competition-style SSIM.
- **Outcome**: Strong validation behavior and stable optimization through pretrain.
- **Notes**: One failed attempt from augmentation behavior was rerun successfully.

### ESRGAN Lite Run
- **Status**: Completed.
- **Setup**: ESRGAN lite with pretrain, GAN, and finetune phases.
- **Outcome**: GAN phase showed instability; finetune partially recovered quality.
- **Notes**: Training was functional end-to-end but less stable than U-Net on this dataset scale.

### Key Learnings
- U-Net training was more stable and data-efficient in this project context.
- GAN-heavy training was more sensitive to limited real paired data.
- Slice-based pipelines avoided several memory and instability issues from volumetric GAN training.
- Non-square slice geometry remains incompatible with rotation augmentations that swap spatial axes.
- Competition-style SSIM (global statistics) remained useful for training-time monitoring.

---

## Recent Experiment Attempts (Feb 2026, Torch HPC)

Detailed job logs and failure signatures are in `audit/RUN_LOG_2026-02-11.md`.

### Features added
- Expanded-slice dataset, curriculum sampler, multi-seed orchestration, ensemble inference, dynamic loss schedule, early stopping.
- Key files changed on HPC: `simple_dataset.py`, `train.py`, `inference.py`, `config.yaml`, `hpc/submit_patha_jobs.sh`, `hpc/inference_ensemble_job.sbatch`.

### Known pitfalls (from this round)
- **z-index OOB**: Clamp z-indices when LF/HF volumes have different z-depths.
- **Shape mismatch in loss**: Resample HF target to match model output shape before loss computation.
- **SLURM `--wrap` corruption**: Never paste markdown-formatted commands into sbatch `--wrap` strings.

### Leaderboard outcomes
- Best single-model: `unet_realonly_patch_seed11` = **0.4525**.
- Ensembles did not improve over best single model (0.4438–0.4478).

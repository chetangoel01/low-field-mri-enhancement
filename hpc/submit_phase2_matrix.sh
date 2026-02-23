#!/bin/bash
# Submit 7-run phase-2 matrix on NYU Torch HPC.
# Usage:
#   bash hpc/submit_phase2_matrix.sh

set -euo pipefail

echo "Submitting Phase-2 matrix jobs..."

# -----------------------------
# Diffusion track (4 runs)
# -----------------------------
EXPERIMENT_NAME="diff_r1_v1" CONFIG_PATH="diffusion_config_recovery_v1.yaml" sbatch hpc/diffusion_train_job.sbatch
EXPERIMENT_NAME="diff_r1_v2" CONFIG_PATH="diffusion_config_recovery_v2.yaml" sbatch hpc/diffusion_train_job.sbatch
EXPERIMENT_NAME="diff_r1_v3" CONFIG_PATH="diffusion_config_recovery_v3.yaml" sbatch hpc/diffusion_train_job.sbatch
EXPERIMENT_NAME="diff_base_cfg" CONFIG_PATH="diffusion_config.yaml" sbatch hpc/diffusion_train_job.sbatch

# -----------------------------
# U-Net challenger track (3 runs)
# -----------------------------
EXPERIMENT_NAME="unet_base_r1" CONFIG_PATH="config.yaml" sbatch hpc/train_job.sbatch
EXPERIMENT_NAME="unet_se_r1" CONFIG_PATH="config_unet_se.yaml" sbatch hpc/train_job.sbatch
EXPERIMENT_NAME="unet_se_lowlr_r1" CONFIG_PATH="config_unet_se_lowlr.yaml" sbatch hpc/train_job.sbatch

echo "All 7 training jobs submitted."
echo "Monitor: squeue -u \$USER"

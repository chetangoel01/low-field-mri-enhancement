#!/bin/bash
#SBATCH --job-name=mri-sr-train
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Usage:
#   EXPERIMENT_NAME="unet_fresh_v1" sbatch hpc/train_job.sbatch
#   EXPERIMENT_NAME="unet_se_v1" CONFIG_PATH="configs/config_unet_se.yaml" sbatch hpc/train_job.sbatch
#   EXPERIMENT_NAME="unet_se_v1" CONFIG_PATH="configs/config_unet_se.yaml" RESUME_FROM="experiments/unet_se_v1/checkpoints/checkpoint_latest.pth" sbatch hpc/train_job.sbatch

set -e

# Environment
source /scratch/cg4652/miniconda3/etc/profile.d/conda.sh
conda activate mri-sr

cd "$SLURM_SUBMIT_DIR"

# Create log directory
mkdir -p logs

# Defaults
EXPERIMENT_NAME="${EXPERIMENT_NAME:-unet_fresh_v1}"
CONFIG_PATH="${CONFIG_PATH:-configs/config.yaml}"

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Config: $CONFIG_PATH"
echo "Resume: ${RESUME_FROM:-none}"
echo "============================================"

# Build command
CMD="python src/train.py --config $CONFIG_PATH --experiment $EXPERIMENT_NAME"

if [ -n "$RESUME_FROM" ]; then
    CMD="$CMD --resume $RESUME_FROM"
fi

echo "Running: $CMD"
$CMD

echo "Training complete."

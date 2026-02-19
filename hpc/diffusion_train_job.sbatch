#!/bin/bash
#SBATCH --job-name=mri-sr-diffusion-train
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/diffusion_train_%j.out
#SBATCH --error=logs/diffusion_train_%j.err

# Usage:
#   EXPERIMENT_NAME="diffusion_v1" sbatch hpc/diffusion_train_job.sbatch
#   EXPERIMENT_NAME="diffusion_v1" RESUME_FROM="experiments/diffusion_v1/checkpoints/checkpoint_latest.pth" sbatch hpc/diffusion_train_job.sbatch

set -e

source /scratch/cg4652/miniconda3/etc/profile.d/conda.sh
conda activate mri-sr

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

EXPERIMENT_NAME="${EXPERIMENT_NAME:-diffusion_v1}"

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Resume: ${RESUME_FROM:-none}"
echo "============================================"

CMD="python diffusion_train.py --config diffusion_config.yaml --experiment $EXPERIMENT_NAME"

if [ -n "$RESUME_FROM" ]; then
    CMD="$CMD --resume $RESUME_FROM"
fi

echo "Running: $CMD"
$CMD

echo "Training complete."

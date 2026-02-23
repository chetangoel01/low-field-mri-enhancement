#!/bin/bash
#SBATCH --job-name=mri-sr-infer
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err

# Usage:
#   EXPERIMENT_NAME="unet_fresh_v1" sbatch hpc/inference_job.sbatch
#   EXPERIMENT_NAME="unet_se_v1" CONFIG_PATH="configs/config_unet_se.yaml" sbatch hpc/inference_job.sbatch
#   EXPERIMENT_NAME="unet_se_v1" CHECKPOINT_PATH="experiments/unet_se_v1/checkpoints/checkpoint_latest.pth" OUTPUT_PATH="experiments/unet_se_v1/submission_latest.csv" sbatch hpc/inference_job.sbatch

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
CHECKPOINT_PATH="${CHECKPOINT_PATH:-experiments/${EXPERIMENT_NAME}/checkpoints/best_model.pth}"
OUTPUT_PATH="${OUTPUT_PATH:-experiments/${EXPERIMENT_NAME}/submission.csv}"

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUTPUT_PATH"
echo "============================================"

# Build inference command
CMD="python src/inference.py --config $CONFIG_PATH --checkpoint $CHECKPOINT_PATH --output $OUTPUT_PATH"

if [ -n "$NO_TTA" ]; then
    CMD="$CMD --no_tta"
fi

if [ -n "$NO_HIST_MATCH" ]; then
    CMD="$CMD --no_hist_match"
fi

echo "Running inference: $CMD"
$CMD

echo ""
echo "Inference complete. Submission saved to: $OUTPUT_PATH"

# Also run validation with same flags
echo ""
echo "Running validation..."
VAL_CMD="python src/validate.py --config $CONFIG_PATH --checkpoint $CHECKPOINT_PATH"
if [ -n "$NO_TTA" ]; then VAL_CMD="$VAL_CMD --no_tta"; fi
if [ -n "$NO_HIST_MATCH" ]; then VAL_CMD="$VAL_CMD --no_hist_match"; fi
$VAL_CMD

echo ""
echo "All done."

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
#   EXPERIMENT_NAME="unet_fresh_v1" NO_TTA=1 sbatch hpc/inference_job.sbatch

set -e

# Environment
source /scratch/cg4652/miniconda3/etc/profile.d/conda.sh
conda activate mri-sr

cd /scratch/cg4652/mri-sr-v2

# Create log directory
mkdir -p logs

# Defaults
EXPERIMENT_NAME="${EXPERIMENT_NAME:-unet_fresh_v1}"
CHECKPOINT="experiments/${EXPERIMENT_NAME}/checkpoints/best_model.pth"
OUTPUT="experiments/${EXPERIMENT_NAME}/submission.csv"

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT"
echo "============================================"

# Build inference command
CMD="python inference.py --checkpoint $CHECKPOINT --output $OUTPUT"

if [ -n "$NO_TTA" ]; then
    CMD="$CMD --no_tta"
fi

if [ -n "$NO_HIST_MATCH" ]; then
    CMD="$CMD --no_hist_match"
fi

echo "Running inference: $CMD"
$CMD

echo ""
echo "Inference complete. Submission saved to: $OUTPUT"

# Also run validation with same flags
echo ""
echo "Running validation..."
VAL_CMD="python validate.py --checkpoint $CHECKPOINT"
if [ -n "$NO_TTA" ]; then VAL_CMD="$VAL_CMD --no_tta"; fi
if [ -n "$NO_HIST_MATCH" ]; then VAL_CMD="$VAL_CMD --no_hist_match"; fi
$VAL_CMD

echo ""
echo "All done."

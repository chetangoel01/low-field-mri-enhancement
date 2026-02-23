#!/bin/bash
#SBATCH --job-name=mri-sr-val
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/val_%j.out
#SBATCH --error=logs/val_%j.err

set -e

source /scratch/cg4652/miniconda3/etc/profile.d/conda.sh
conda activate mri-sr

cd "$SLURM_SUBMIT_DIR"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-unet_fresh_v1}"
CHECKPOINT="experiments/${EXPERIMENT_NAME}/checkpoints/best_model.pth"

echo "Checkpoint: $CHECKPOINT"

echo "=== No histogram matching, with TTA ==="
python src/validate.py --checkpoint $CHECKPOINT --no_hist_match

echo ""
echo "=== No histogram matching, no TTA ==="
python src/validate.py --checkpoint $CHECKPOINT --no_hist_match --no_tta

echo ""
echo "=== With histogram matching, with TTA ==="
python src/validate.py --checkpoint $CHECKPOINT

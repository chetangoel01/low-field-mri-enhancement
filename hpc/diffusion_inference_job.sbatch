#!/bin/bash
#SBATCH --job-name=mri-sr-diffusion-infer
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/diffusion_infer_%j.out
#SBATCH --error=logs/diffusion_infer_%j.err

# Usage:
#   EXPERIMENT_NAME="diffusion_recovery_v1" sbatch hpc/diffusion_inference_job.sbatch
#   EXPERIMENT_NAME="diffusion_recovery_v1" CONFIG_PATH="configs/diffusion_config_recovery_v1.yaml" DDIM_STEPS=50 sbatch hpc/diffusion_inference_job.sbatch
#   EXPERIMENT_NAME="diffusion_recovery_v1" CHECKPOINT_PATH="experiments/diffusion_recovery_v1/checkpoints/checkpoint_latest.pth" OUTPUT_PATH="experiments/diffusion_recovery_v1/submission_latest.csv" sbatch hpc/diffusion_inference_job.sbatch

set -e

source /scratch/cg4652/miniconda3/etc/profile.d/conda.sh
conda activate mri-sr

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

EXPERIMENT_NAME="${EXPERIMENT_NAME:-diffusion_recovery_v1}"
CONFIG_PATH="${CONFIG_PATH:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-experiments/${EXPERIMENT_NAME}/checkpoints/best_model.pth}"
OUTPUT_PATH="${OUTPUT_PATH:-experiments/${EXPERIMENT_NAME}/submission_diffusion.csv}"

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Config override: ${CONFIG_PATH:-none}"
echo "DDIM steps: ${DDIM_STEPS:-from config}"
echo "TTA: ${NO_TTA:+disabled}"${NO_TTA:-enabled}
echo "============================================"

CMD="python src/diffusion_inference.py --checkpoint $CHECKPOINT_PATH --output $OUTPUT_PATH"

[ -n "$NO_TTA" ]     && CMD="$CMD --no_tta"
[ -n "$DDIM_STEPS" ] && CMD="$CMD --ddim_steps $DDIM_STEPS"
[ -n "$CONFIG_PATH" ] && CMD="$CMD --config $CONFIG_PATH"
[ -n "$BATCH_SIZE" ] && CMD="$CMD --batch_size $BATCH_SIZE"

echo "Running: $CMD"
$CMD

echo "Inference complete. Submission: $OUTPUT_PATH"

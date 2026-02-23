# Phase-2 7-Run Matrix (NYU Torch HPC)

## Goal
- Recover diffusion performance beyond current `0.54`.
- Run one strong non-diffusion challenger in parallel.
- Keep comparisons fair by using the same data split and metric pipeline.

## Runs

1. `diff_r1_v1`
- Script: `hpc/diffusion_train_job.sbatch`
- Config: `configs/diffusion_config_recovery_v1.yaml`
- Hypothesis: stronger MS-SSIM auxiliary loss without SNR weighting helps structure.

2. `diff_r1_v2`
- Script: `hpc/diffusion_train_job.sbatch`
- Config: `configs/diffusion_config_recovery_v2.yaml`
- Hypothesis: conservative aux strength + SNR weighting stabilizes training.

3. `diff_r1_v3`
- Script: `hpc/diffusion_train_job.sbatch`
- Config: `configs/diffusion_config_recovery_v3.yaml`
- Hypothesis: aggressive aux loss and early activation improve metric alignment.

4. `diff_base_cfg`
- Script: `hpc/diffusion_train_job.sbatch`
- Config: `configs/diffusion_config.yaml`
- Purpose: control run against current default diffusion setup.

5. `unet_base_r1`
- Script: `hpc/train_job.sbatch`
- Config: `configs/config.yaml`
- Purpose: non-attention U-Net control.

6. `unet_se_r1`
- Script: `hpc/train_job.sbatch`
- Config: `configs/config_unet_se.yaml`
- Hypothesis: SE attention improves structural fidelity.

7. `unet_se_lowlr_r1`
- Script: `hpc/train_job.sbatch`
- Config: `configs/config_unet_se_lowlr.yaml`
- Hypothesis: lower LR reduces over-sharpening/noise artifacts.

## Submit All 7

`bash hpc/submit_phase2_matrix.sh`

## Inference Commands (top checkpoints)

- Diffusion:
  - `EXPERIMENT_NAME="diff_r1_v1" CONFIG_PATH="configs/diffusion_config_recovery_v1.yaml" sbatch hpc/diffusion_inference_job.sbatch`
  - Optional: `DDIM_STEPS=50 NO_TTA=1`

- U-Net:
  - `EXPERIMENT_NAME="unet_se_r1" CONFIG_PATH="configs/config_unet_se.yaml" sbatch hpc/inference_job.sbatch`
  - Optional: `NO_TTA=1`

## Stop/Promote Criteria

- Stop if validation MS-SSIM is below bicubic baseline trend after the first meaningful checkpoint window.
- Promote if model exceeds current `0.54` with stable validation over multiple eval points.
- Ensemble only if two diverse families (diffusion + U-Net challenger) both beat `0.54`.

## Monitoring

- `squeue -u $USER`
- `tail -f experiments/diff_r1_v1/logs/training.log`
- `tail -f logs/diffusion_train_JOBID.out`
- `tail -f logs/train_JOBID.out`
- `sacct -u $USER --starttime=today --format=JobID,JobName,State,Elapsed,NodeList`

## Prompt

You are helping build a PyTorch solution for low-field to high-field MRI enhancement.

### Task Context

- We have paired training data:
  - Input low-field MRI volume shape: `(112, 138, 40)` (64mT)
  - Target high-field MRI volume shape: `(179, 221, 200)` (3T)
- Test set contains only low-field volumes.
- Goal: generate enhanced outputs matching high-field quality and produce competition submission CSV.

### Input Specification

- Training directories:
  - `train/low_field/*.nii`
  - `train/high_field/*.nii`
- Test directory:
  - `test/low_field/*.nii`
- Slice metadata file:
  - `train.csv` with `row_id, ground_truth`
- Use axial slice processing with 2.5D context:
  - For each z-index, model input is 5 adjacent slices from upsampled low-field volume.
  - Model target is the center high-field slice at the same z-index.

### Preprocessing Requirements

1. Upsample low-field volume from `(112,138,40)` to `(179,221,200)` using trilinear interpolation.
2. Percentile normalization: clip voxel intensities to 1st and 99th percentiles, then rescale to `[0,1]`. Apply independently to low-field and high-field volumes.
3. Use mirror-padding at z-boundaries for 5-slice context windows.
4. Training augmentation:
   - Random 160×160 crops (to enable 90° rotations on the non-square 179×221 slices)
   - Random 90-degree rotations
   - Horizontal and vertical flips
   - Intensity scaling in `[0.9, 1.1]` with probability `0.5`
   - Gaussian noise `std=0.01` with probability `0.3`
   - At inference, use full-resolution slices without cropping.

### Training Architecture We Want (Best Model Family)

Implement a **2.5D residual U-Net**:

- Input: `5 x 179 x 221`
- Output: `1 x 179 x 221`
- Encoder channels: `64 -> 128 -> 256 -> 512`
- Bottleneck: `1024`
- Decoder mirrors encoder with skip connections
- Residual learning: add center input slice to network output
- Attention: none (baseline best family in this submission package)

### Training Configuration

- Epochs: `400`
- Batch size: `16`
- Optimizer: `AdamW`, betas `(0.9, 0.99)`, weight decay `1e-4`
- LR schedule: cosine from `1e-3` to `1e-6`, with warmup `500` steps
- Gradient clipping: `1.0`
- AMP: enabled
- EMA decay: `0.999`
- Validation holdout volumes: samples `017`, `018`
- Early stopping patience: `50`

### Loss Function

Use weighted sum:

- `0.16 * Charbonnier L1`
- `0.84 * MS-SSIM`

### Inference & Output Requirements

1. Load test low-field NIfTI volume.
2. Upsample to `(179,221,200)`.
3. Predict each of 200 axial slices with 5-slice context.
4. Apply TTA (flip-based) and average predictions.
5. Reconstruct full 3D prediction volume per sample.
6. Convert each axial slice to base64 using `extract_slices.py` utilities.
7. Produce `submission.csv` with:
   - Columns: `row_id`, `prediction`
   - Row format: `sample_XXX_slice_YYY`
   - Exactly `1000` rows (5 test samples x 200 slices each)

### Deliverables to Generate

- `train.py` for model training/checkpointing
- `inference.py` for test-time prediction and CSV generation
- `model.py` for the 2.5D residual U-Net
- `dataset.py` for preprocessing and data loading
- `losses.py` for Charbonnier and MS-SSIM losses
- `config_best_unet_phase2_v4.yaml` for reproducible hyperparameters

Ensure code is modular, documented, and runnable from the command line.
# Low-Field → High-Field MRI Enhancement: Complete Agent Brief

## Table of Contents

1. [Mission Statement](#1-mission-statement)
2. [Deadline & Logistics](#2-deadline--logistics)
3. [Problem Definition](#3-problem-definition)
4. [Domain Background (From Lecture Materials)](#4-domain-background)
5. [Dataset Specification](#5-dataset-specification)
6. [Evaluation Metric (Exact Implementation)](#6-evaluation-metric)
7. [Baseline Performance](#7-baseline-performance)
8. [Submission Pipeline](#8-submission-pipeline)
9. [Suggested Approaches (From Professor)](#9-suggested-approaches)
10. [Architecture & Strategy Insights (From Experimentation)](#10-architecture--strategy-insights)
11. [Data Expansion Strategy](#11-data-expansion-strategy)
12. [Critical Implementation Details](#12-critical-implementation-details)
13. [Key Utility Code Reference](#13-key-utility-code-reference)
14. [Glossary](#14-glossary)

---

## 1. Mission Statement

Build a model that takes a low-field (64mT) T1-weighted brain MRI volume and produces an enhanced 3D volume that matches the corresponding high-field (3T) MRI as closely as possible. This is a paired 3D image enhancement / super-resolution problem evaluated on a Kaggle competition leaderboard. The task involves both spatial upsampling (the low-field volume is smaller) and quality enhancement (denoising, sharpening, contrast improvement).

---

## 2. Deadline & Logistics

- **Deadline**: February 13th, 2025 at 11:59 PM (midnight before February 14th)
- **Late penalty**: 1% per hour late (rounded down)
- **Submissions required**:
  - Kaggle: `submission.csv` file (1000 rows)
  - Brightspace: Zipped code with markdown/comments explaining approach
  - If LLMs were used: include the full chat history
- **Allowed**: Pre-built libraries (PyTorch, TensorFlow, scikit-image, etc.), pretrained models, additional external datasets
- **Language**: Python only
- **Compute**: NYU HPC available (sign up at NYU IT research computing)
- **Team size**: 1–3 members (max 2 graduate students per team)
- **Grading**: 100% for winning team, 99% for second, etc. Base grade from write-up quality. Sufficient performance above baseline → B/B+. Curved to B+ median.

---

## 3. Problem Definition

### Input → Output

| Property | Input (Low-Field) | Output (High-Field Target) |
|----------|-------------------|---------------------------|
| Volume shape | (112, 138, 40) | (179, 221, 200) |
| Voxel size | 1.6 × 1.6 × 5.0 mm | 1.0 × 1.0 × 1.0 mm (isotropic) |
| Axial slices | 40 | 200 |
| Each slice shape | 112 × 138 | 179 × 221 |
| Scanner strength | 64mT | 3T |

The model must perform both **spatial super-resolution** (upsampling from a smaller, anisotropic volume to a larger, isotropic one) and **quality enhancement** (denoising, sharpening, contrast correction).

### Key Physics Context

- SNR scales approximately as B₀² (where B₀ is field strength), so 3T has dramatically higher SNR than 64mT
- Low-field images suffer from: lower signal-to-noise ratio, reduced sharpness, contrast differences, anisotropic voxels (thick 5mm slices in z vs 1mm in-plane)
- The z-axis is the most challenging dimension: 40 slices → 200 slices (5× upsampling) vs ~1.6× in x and y

### Registration

- The high-field volumes have been registered (rigidly transformed) to the low-field coordinate space
- The low-field data is the original untouched acquisition
- Registration quality is good — most differences come from the actual enhancement challenge (SNR, sharpness, contrast), not misalignment

---

## 4. Domain Background

### From Lecture 2: Neuroimaging Technologies

These points are directly relevant to understanding what the model needs to learn:

**How MRI works**: Hydrogen protons align under a magnetic field. Radiofrequency pulses flip them, and as they relax back, they emit signals captured by receivers. Different tissues (water, fat, gray matter, white matter) have different signal responses.

**SNR and field strength**: SNR ∝ B₀ⁿ where n ≈ 2 experimentally. This means going from 64mT to 3T is roughly a (3/0.064)² ≈ 2200× improvement in SNR. In practice the gap is less extreme due to engineering, but low-field images are fundamentally noisier.

**Why low-field matters**: Low-field systems are cheaper, more portable, require less infrastructure (no liquid helium, no electromagnetic shielding), and can be deployed at the bedside. The trade-off is image quality.

**T1-weighted imaging**: This is a specific MRI contrast that highlights differences between gray matter, white matter, and CSF. It's the standard anatomical scan. All data in this competition is T1-weighted.

**Image reconstruction**: MRI data is acquired in k-space (frequency domain) and reconstructed via inverse FFT. Known artifacts include motion artifacts and magnetic field inhomogeneity.

### From Lecture Slides: Approaches Mentioned

The professor's slides (Lecture 2, pages 63–65) specifically mention:

1. **Simulation of HR-LR paired data**: Simulating large sets of diverse, aligned brain MRI at different resolutions to augment DL training
2. **GANs and Diffusion Models**: Training on paired images (64mT and 3T) where the model learns to predict high-frequency details, effectively "upscaling" the low-field image
3. **Deep Learning Denoising**: Specialized CNNs to strip thermal noise without blurring anatomical edges, enabling faster scans

---

## 5. Dataset Specification

### Files & Structure

```
mri_superres_dataset/
├── train/
│   ├── low_field/
│   │   ├── sample_001_lowfield.nii.gz
│   │   ├── sample_002_lowfield.nii.gz
│   │   └── ... (18 files)
│   └── high_field/
│       ├── sample_001_highfield.nii.gz
│       ├── sample_002_highfield.nii.gz
│       └── ... (18 files)
├── test/
│   └── low_field/
│       ├── sample_019_lowfield.nii.gz
│       ├── sample_020_lowfield.nii.gz
│       └── ... (5 files, samples 019–023)
├── train.csv              # 3600 rows (18 × 200 slices), columns: row_id, ground_truth
├── sample_submission.csv  # 1000 rows (5 × 200 slices), columns: row_id, prediction
└── extract_slices.py      # Utility script (MUST use for encoding)
```

### Volume Details

- **Training**: 18 paired volumes (samples 001–018). Each pair has a low-field (112×138×40) and high-field (179×221×200) NIfTI file.
- **Test**: 5 low-field volumes (samples 019–023). High-field targets are hidden on Kaggle.
- **Format**: NIfTI (.nii.gz). Load with `nibabel`: `nib.load(path).get_fdata()` → numpy array.
- **All scans**: T1-weighted anatomical brain MRI from healthy adult subjects.

### Row ID Format

`sample_XXX_slice_YYY` where:
- XXX = sample number (001–018 train, 019–023 test)
- YYY = slice index (000–199), representing axial slices through the z-dimension

### train.csv

Contains 3600 rows. Each row has a `row_id` and a `ground_truth` column containing a base64-encoded 2D array (179 × 221 pixels) representing one axial slice of the high-field volume.

---

## 6. Evaluation Metric

### CRITICAL: The exact metric implementation matters for loss design.

The competition uses a combined SSIM + PSNR score. Here is exactly how it works (from `metric.py`):

### Step 1: Normalization

Both the prediction and ground truth are **independently normalized to [0, 1]** before computing metrics:

```
def normalize(x):
    x_min, x_max = x.min(), x.max()
    if x_max - x_min > 0:
        return (x - x_min) / (x_max - x_min)
    return np.zeros_like(x)
```

**This is critical**: Absolute intensity values do not matter. Only the structural pattern within each slice matters. Your loss function should match this normalization behavior.

### Step 2: SSIM (Global, Not Windowed)

The SSIM is computed **globally** over the entire image (not using a sliding window like scikit-image's default):

```
mu1 = img1_norm.mean()       # global mean
mu2 = img2_norm.mean()
sigma1_sq = ((img1_norm - mu1) ** 2).mean()   # global variance
sigma2_sq = ((img2_norm - mu2) ** 2).mean()
sigma12 = ((img1_norm - mu1) * (img2_norm - mu2)).mean()  # global covariance
C1 = 0.01 ** 2 = 0.0001
C2 = 0.03 ** 2 = 0.0009
SSIM = (2*mu1*mu2 + C1) * (2*sigma12 + C2) / ((mu1² + mu2² + C1) * (sigma1_sq + sigma2_sq + C2))
```

**This differs from standard SSIM implementations** (e.g., scikit-image uses 7×7 or 11×11 sliding windows by default). Your SSIM loss should replicate this global computation.

### Step 3: PSNR

Computed on the [0,1] normalized images:
```
MSE = mean((img1_norm - img2_norm)²)
PSNR = 10 * log10(1.0 / MSE)
```
Clamped to [0, 50] dB. Perfect match = 50 dB.

### Step 4: Combined Score

```
score_per_slice = 0.5 × SSIM + 0.5 × (PSNR / 50)
final_score = mean(score_per_slice across all evaluated slices)
```

PSNR is divided by 50 to normalize it to approximately [0, 1] for fair weighting with SSIM.

### Leaderboard Split

- Total: 1000 test slices (5 samples × 200 slices)
- Public: 500 randomly selected slices (50%)
- Private: 500 remaining slices (50%)
- Random split across all samples and z-positions

---

## 7. Baseline Performance

The `sample_submission.csv` uses bicubic interpolation (simple upsampling without enhancement):

| Split | SSIM | PSNR (dB) | Combined Score |
|-------|------|-----------|----------------|
| Public (500 slices) | 0.593 | 15.90 | 0.455 |
| Private (500 slices) | 0.595 | 15.94 | 0.457 |
| Overall (1000 slices) | 0.594 | 15.92 | 0.456 |

**Goal: Significantly exceed ~0.46.**

---

## 8. Submission Pipeline

### Required Output

For each of the 5 test samples, produce an enhanced 3D volume with shape **(179, 221, 200)**. Then:

1. Extract all 200 axial slices per volume
2. Encode each slice to base64 using the provided `slice_to_base64()` function
3. Submit a CSV with 1000 rows (5 × 200)

### Encoding Details

The `slice_to_base64()` function:
1. Records the min and max of the slice
2. Normalizes to uint8 [0–255]
3. Saves as compressed npz (with slice data, shape, min_val, max_val)
4. Base64-encodes the result

**You MUST use this function**. Other encoding methods will fail validation.

### Submission Code Template

```python
from extract_slices import create_submission_df, load_nifti

predictions = {}
for sid in ['sample_019', 'sample_020', 'sample_021', 'sample_022', 'sample_023']:
    lf = load_nifti(f'test/low_field/{sid}_lowfield.nii.gz')
    enhanced = your_model(lf)  # must output shape (179, 221, 200)
    predictions[sid] = enhanced

submission_df = create_submission_df(predictions)
submission_df.to_csv('submission.csv', index=False)
# Should have 1000 rows
```

### Important Requirements

- Output shape must be exactly (179, 221, 200)
- All 200 slices per sample must be present (1000 total rows)
- No NaN or Inf values in predictions
- Row order doesn't matter (matched by row_id)

---

## 9. Suggested Approaches (From Professor)

The project PDF (Section 5) explicitly suggests these approaches:

### Baseline Methods
- Bicubic interpolation (provided baseline, score ≈ 0.46)
- Trilinear interpolation

### Deep Learning Approaches
- **3D U-Net**: Encoder-decoder architecture with skip connections
- **SRCNN/ESPCN**: Super-resolution CNNs (can be extended to 3D)
- **ESRGAN**: Enhanced super-resolution GAN for perceptual quality
- **Attention mechanisms**: Self-attention or transformer-based approaches
- **Residual learning**: Learn the difference between upsampled input and target

### Training Considerations (From PDF)
- Limited training data (18 samples) — consider data augmentation
- 3D context can help but increases memory requirements
- Patch-based training may be necessary for large volumes
- Loss functions: L1, L2, perceptual loss, adversarial loss

### From Lecture Slides (Additional)
- Diffusion models as an alternative to GANs
- Simulating paired data from external high-field datasets for augmentation
- CNN-based denoising that preserves anatomical edges

---

## 10. Architecture & Strategy Insights (From Experimentation)

### What Has Been Tried

**3D ESRGAN** was implemented with:
- 3D Residual-in-Residual Dense Block (RRDB) generator, no batch normalization, 16–23 RRDB blocks, residual scaling (×0.2)
- 3D Relativistic Average PatchGAN discriminator with spectral normalization
- Multi-phase training: L1+SSIM pre-training → GAN fine-tuning → metric-focused final tuning
- Multi-loss: L1, SSIM, perceptual, adversarial, FFT

**Result**: The 3D ESRGAN yielded only ~0.01 improvement over starting point and the estimated competition score (~0.373) was actually *below* the 0.46 baseline. The model was too parameter-heavy for 18 training samples and the GAN phase destabilized performance.

### Key Lessons Learned

1. **3D ESRGAN is too heavy for 18 samples.** The parameter count overwhelms the tiny dataset. Simpler models generalize better here.

2. **GAN training hurts with limited data.** The adversarial phase actively degraded performance. Drop it entirely — use only reconstruction losses.

3. **The metric normalization is unusual.** The competition SSIM normalizes each image independently to [0,1] and computes global (not windowed) SSIM. Your loss must replicate this.

4. **2D or 2.5D approaches are better suited** for this dataset size. Processing individual slices (or stacks of ~5 adjacent slices) gives you 18×200 = 3600 training examples vs 18 sparse 3D volumes.

### Recommended Architecture: 2.5D U-Net

Based on all experimentation, the highest-probability approach is:

**Input**: 5 adjacent axial slices from trilinear-upsampled low-field volume (5 × 179 × 221)
**Output**: 1 center slice prediction (1 × 179 × 221)
**Architecture**: Standard 2D U-Net (encoder: 64→128→256→512, bottleneck: 1024, decoder with skip connections)
**Key design**: Residual learning — predict the *residual* between upsampled low-field and high-field, not the full image directly
**Losses**: L1 (weight 1.0) + Competition-matching SSIM loss (weight 0.5). No perceptual, no adversarial.
**Optimizer**: Adam, lr=1e-3 with cosine annealing to 1e-5
**Batch size**: 16–32
**Epochs**: 100–200
**Augmentation**: Random horizontal flip, random 90° rotation, intensity jitter
**Training data**: Use all 18 samples, validate on random 10% of slices (not held-out volumes)
**Inference**: Slice-by-slice with mirror-padding at volume boundaries. Optional TTA (horizontal flip, average).

### Additional Improvements

- **Post-processing with histogram matching**: Match intensity histogram of predicted slices to average high-field training histogram. Free PSNR boost at inference time.
- **Test-time augmentation**: Flip predictions and average. Adds +1–2% for free.
- **Ensemble**: Train 2–3 variants (different depths, different loss weights) and average outputs.

### Expected Score Ranges

| Approach | Expected Score |
|----------|----------------|
| Baseline (bicubic) | ~0.456 |
| 2D Residual U-Net (simple) | ~0.50–0.55 |
| 2.5D U-Net + residual learning | ~0.55–0.62 |
| + Histogram matching + TTA | ~0.58–0.65 |
| + Ensemble of 3 models | ~0.62–0.70 |

---

## 11. Data Expansion Strategy

The project explicitly allows external datasets and pretrained models. With only 18 training volumes, data expansion is one of the highest-impact strategies.

### External Datasets

| Dataset | Volumes | Resolution | Access |
|---------|---------|------------|--------|
| IXI | ~580 | 1mm isotropic | Open, easy download |
| OASIS-3 | ~1000+ | 1mm isotropic | Free registration |
| HCP | 1100+ | 0.7mm isotropic | Free registration |
| ABIDE | ~1000 | Varies | Open |

**IXI is the recommended starting point** — open access, good quality, ~580 T1-weighted volumes.

### Synthetic Pair Generation

1. **Analyze real degradation**: Examine the 18 real low-field/high-field pairs to measure noise levels, blur characteristics, contrast mapping, and resolution differences.

2. **Apply degradation to external high-field data**:
   - Downsample to match low-field geometry (112 × 138 × 40)
   - Add Rician noise (MRI-appropriate noise model) calibrated to match real low-field noise levels
   - Apply Gaussian blur to match real low-field sharpness
   - Adjust contrast to match real low-field contrast characteristics
   - Upsample back to target resolution

3. **Training curriculum**:
   - **Stage 1**: Pre-train on synthetic pairs (500+ volumes) — learns general MRI enhancement
   - **Stage 2**: Fine-tune on real 18 pairs — adapts to actual low-field characteristics
   - **Stage 3**: Mix (80% real, 20% synthetic) for final training

### Impact Estimate

| Data Strategy | Effective Training Size | Expected Score Boost |
|---------------|------------------------|---------------------|
| 18 volumes only | ~3,600 slices | Baseline |
| + Heavy augmentation | ~14,000 effective | +2–3% |
| + Synthetic pairs (500 vol) | ~100,000+ slices | +5–8% |
| + Curriculum training | — | +2–3% |

### Patch-Based Expansion (Simpler Alternative)

Even without external data, patch extraction multiplies your data:

| Patch Size | Patches per Volume | Total Patches (18 vol) |
|------------|-------------------|------------------------|
| 64 × 64 | ~12 per slice × 200 | ~43,200 |
| 128 × 128 | ~3 per slice × 200 | ~10,800 |
| Full slice (179 × 221) | 200 per volume | 3,600 |

Combined with augmentation (8–16 variations each), effective training set grows significantly.

---

## 12. Critical Implementation Details

### Normalization Matching

The competition metric normalizes each slice independently to [0,1]. Your training pipeline should:
- Normalize input slices to [0,1] independently
- Train the network to output [0,1] (use sigmoid output or clamp)
- Compute losses on [0,1] normalized data
- At submission time, the `slice_to_base64()` function handles its own normalization (records min/max), so you can output in any range — but your loss should match the metric's normalization

### SSIM Loss Must Match Competition SSIM

The competition uses **global** SSIM (mean/variance computed over the entire image), NOT windowed SSIM. If you use `pytorch_msssim` or `scikit-image`, those default to windowed computation. You need a custom SSIM loss that:
1. Normalizes prediction and target independently to [0,1]
2. Computes global means, variances, covariance
3. Uses C1 = 0.0001, C2 = 0.0009

### Volume Upsampling

The low-field volume (112, 138, 40) must become (179, 221, 200). Use trilinear interpolation as the first step before feeding to the neural network:

```python
import torch.nn.functional as F
# volume shape: (112, 138, 40)
volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
upsampled = F.interpolate(volume_tensor, size=(179, 221, 200), mode='trilinear', align_corners=False)
# upsampled shape: (1, 1, 179, 221, 200)
```

### Z-Axis Boundary Handling

For 2.5D approaches using adjacent slices, mirror-pad at volume boundaries:
- Slice 0: use slices [2, 1, 0, 1, 2] as input (reflected padding)
- Slice 199: use slices [197, 198, 199, 198, 197]

### NaN/Inf Prevention

Predictions cannot contain NaN or Inf. Add safety clamps:
```python
output = torch.clamp(output, 0, max_reasonable_value)
```

---

## 13. Key Utility Code Reference

### Loading NIfTI Files

```python
import nibabel as nib

def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata()  # returns 3D numpy array (x, y, z)
```

### Slice Encoding (MUST USE)

```python
from extract_slices import slice_to_base64, base64_to_slice

# Encode: 2D numpy array → base64 string
b64 = slice_to_base64(slice_2d)

# Decode: base64 string → 2D numpy array (float32)
slice_2d = base64_to_slice(b64_string)
```

The encoding process:
1. Records slice min/max values
2. Normalizes to uint8 [0–255]
3. Saves as compressed npz with metadata
4. Base64-encodes the binary data

### Creating Submission

```python
from extract_slices import create_submission_df, volume_to_submission_rows

# Single volume → list of 200 row dicts
rows = volume_to_submission_rows(volume, 'sample_019')  # volume must be (179, 221, 200)

# Multiple volumes → submission DataFrame
predictions = {
    'sample_019': vol_019,  # each (179, 221, 200)
    'sample_020': vol_020,
    'sample_021': vol_021,
    'sample_022': vol_022,
    'sample_023': vol_023,
}
df = create_submission_df(predictions)
df.to_csv('submission.csv', index=False)
# df has 1000 rows (5 × 200)
```

### Local Evaluation

```python
from metric import score
import pandas as pd

solution = pd.read_csv('train.csv')  # or a validation split
submission = pd.read_csv('my_submission.csv')
result = score(solution, submission, 'row_id')
print(f"Score: {result}")
```

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| MRI | Magnetic Resonance Imaging |
| T1w | T1-weighted — a specific MRI contrast highlighting tissue types |
| 64mT | 64 milliTesla — low magnetic field strength (portable scanners) |
| 3T | 3 Tesla — high magnetic field strength (clinical standard) |
| NIfTI | Neuroimaging Informatics Technology Initiative — standard neuroimaging file format (.nii.gz) |
| SSIM | Structural Similarity Index Measure — perceptual image quality metric [0,1] |
| PSNR | Peak Signal-to-Noise Ratio — reconstruction fidelity metric in dB |
| SNR | Signal-to-Noise Ratio — fundamental image quality measure |
| Axial | Horizontal plane (top-down view through the brain) |
| Isotropic | Equal resolution in all spatial directions |
| Anisotropic | Unequal resolution (here: 5mm z-slices vs 1.6mm in-plane) |
| Voxel | 3D pixel — the smallest unit in a 3D volume |
| ESRGAN | Enhanced Super-Resolution GAN |
| RRDB | Residual-in-Residual Dense Block — ESRGAN's core building block |
| SRCNN | Super-Resolution CNN |
| ESPCN | Efficient Sub-Pixel CNN |
| Rician noise | The noise distribution characteristic of MRI magnitude images |
| k-space | Frequency domain where MRI data is acquired before reconstruction |
| TTA | Test-Time Augmentation — averaging predictions over augmented versions of input |
| EMA | Exponential Moving Average of model weights |

---

*This document consolidates the complete project PDF, Kaggle competition page, dataset description, metric implementation, lecture materials (Lectures 1 & 2), and all strategic insights from experimentation. It is intended to be a self-contained reference for any coding agent working on this project.*
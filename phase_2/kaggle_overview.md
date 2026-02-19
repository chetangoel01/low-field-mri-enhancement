Overview
Low-Field → High-Field MRI Enhancement Challenge
High-field MRI (e.g., 3T) produces excellent image quality but can be expensive to purchase and operate, requires specialized infrastructure, and is not equally available everywhere. Low-field and ultra-low-field MRI systems (e.g., 64mT) can be more affordable and easier to deploy, making MRI possible in settings where conventional scanners are impractical. The tradeoff is typically lower signal-to-noise ratio (SNR), reduced sharpness, and contrast differences.

This competition focuses on a practical research question:

Can we algorithmically enhance low-field brain MRI to better match high-field image quality?

Goal
Given a low-field (64mT) T1-weighted MRI volume, generate an enhanced image that matches the corresponding high-field (3T) MRI as closely as possible.

Participants will train on paired examples (low-field input → high-field target) and submit predictions on a hidden test set.

What you'll do
Train a model on paired low-field ↔ high-field MRI volumes
For each test subject, predict an enhanced 3D volume
Extract all 200 axial slices and submit as CSV
Evaluation
Your predictions are evaluated on all 200 axial slices from each test volume, giving comprehensive coverage of the entire brain. The public/private split is randomized across all 1000 slices (5 samples × 200 slices).

Data at a Glance
Split	Subjects	Slices	Low-Field Volume	High-Field Volume
Train	18	3,600	112 × 138 × 40	179 × 221 × 200
Test	5	1,000	112 × 138 × 40	Hidden
Low-field: 64mT scanner, voxel size 1.6 × 1.6 × 5.0 mm
High-field: 3T scanner, voxel size 1.0 × 1.0 × 1.0 mm (isotropic)
Start

4 days ago
Close
2 days to go
Description
Submissions are evaluated using Multi-Scale Structural Similarity (MS-SSIM), which measures perceptual image quality by comparing structural information between your predicted slices and the hidden high-field ground truth across multiple resolution scales.

Metric: MS-SSIM (Multi-Scale Structural Similarity)
Unlike standard SSIM which only evaluates structure at the original image resolution, MS-SSIM assesses quality at multiple spatial scales. This better captures both fine details (sharp edges, tissue boundaries) and large-scale anatomical structure (brain shape, ventricle geometry).

How it works:

At each scale, the luminance, contrast, and structure between the predicted and ground truth images are compared using local sliding windows (11×11 Gaussian-weighted, sigma=1.5)
The images are then downsampled by 2× and the process repeats across 5 scales
The per-scale quality scores are combined using empirically optimized weights
MS-SSIM formula:

MS-SSIM(x, y) = [l_M(x, y)]^α_M  ×  ∏_{j=1}^{M} [c_j(x, y)]^β_j × [s_j(x, y)]^γ_j
Where:

l, c, s = luminance, contrast, and structure comparisons
M = number of scales (5)
α, β, γ = weights per scale (from Wang et al., 2003)
Scale weights: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

Range: 0 to 1 (higher is better, 1 = identical images)

Why MS-SSIM?

Evaluates structural quality at multiple resolutions, rewarding models that preserve both fine anatomical detail and global brain structure
More robust than pixel-level metrics (MSE, PSNR) which can penalize models that improve structure but shift intensity slightly
Widely used in medical image quality assessment literature
Scoring
Each slice is scored independently:

score_slice = MS-SSIM(predicted_slice, ground_truth_slice)
Final Score:

score = mean(score_slice) across all evaluated slices
Public/Private Split
Total test slices: 1000 (5 samples × 200 slices)
Public leaderboard: 500 randomly selected slices (50%)
Private leaderboard: 500 remaining slices (50%)
The random split ensures that slices from all samples and all z-positions contribute to both leaderboards.

Submission File
For each test slice, you must predict an enhanced MRI slice encoded as a base64 string. The file should contain a header and have the following format:

row_id,prediction
sample_019_slice_000,<base64_encoded_slice>
sample_019_slice_001,<base64_encoded_slice>
sample_019_slice_002,<base64_encoded_slice>
etc.
row_id: Format is sample_XXX_slice_YYY where XXX is the sample number (019-023) and YYY is the slice index (000-199)
prediction: Base64-encoded predicted slice (179 × 221 pixels), created using the provided slice_to_base64() function from extract_slices.py
Total rows: 1000 (5 samples × 200 slices)

Use the provided extract_slices.py to ensure correct encoding:

from extract_slices import create_submission_df, load_nifti

# Create predictions for all test samples
predictions = {
    'sample_019': your_model(load_nifti('test/low_field/sample_019_lowfield.nii.gz')),
    'sample_020': your_model(load_nifti('test/low_field/sample_020_lowfield.nii.gz')),
    'sample_021': your_model(load_nifti('test/low_field/sample_021_lowfield.nii.gz')),
    'sample_022': your_model(load_nifti('test/low_field/sample_022_lowfield.nii.gz')),
    'sample_023': your_model(load_nifti('test/low_field/sample_023_lowfield.nii.gz')),
}

# Create submission DataFrame (1000 rows)
submission_df = create_submission_df(predictions)
submission_df.to_csv('submission.csv', index=False)
Evaluation
Submissions are evaluated using Multi-Scale Structural Similarity (MS-SSIM), which measures perceptual image quality by comparing structural information between your predicted slices and the hidden high-field ground truth across multiple resolution scales.

Metric: MS-SSIM (Multi-Scale Structural Similarity)
Unlike standard SSIM which only evaluates structure at the original image resolution, MS-SSIM assesses quality at multiple spatial scales. This better captures both fine details (sharp edges, tissue boundaries) and large-scale anatomical structure (brain shape, ventricle geometry).

How it works:

At each scale, the luminance, contrast, and structure between the predicted and ground truth images are compared using local sliding windows (11×11 Gaussian-weighted, sigma=1.5)
The images are then downsampled by 2× and the process repeats across 5 scales
The per-scale quality scores are combined using empirically optimized weights
MS-SSIM formula:

MS-SSIM(x, y) = [l_M(x, y)]^α_M  ×  ∏_{j=1}^{M} [c_j(x, y)]^β_j × [s_j(x, y)]^γ_j
Where:

l, c, s = luminance, contrast, and structure comparisons
M = number of scales (5)
α, β, γ = weights per scale (from Wang et al., 2003)
Scale weights: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

Range: 0 to 1 (higher is better, 1 = identical images)

Why MS-SSIM?

Evaluates structural quality at multiple resolutions, rewarding models that preserve both fine anatomical detail and global brain structure
More robust than pixel-level metrics (MSE, PSNR) which can penalize models that improve structure but shift intensity slightly
Widely used in medical image quality assessment literature
Scoring
Each slice is scored independently:

score_slice = MS-SSIM(predicted_slice, ground_truth_slice)
Final Score:

score = mean(score_slice) across all evaluated slices
Public/Private Split
Total test slices: 1000 (5 samples × 200 slices)
Public leaderboard: 500 randomly selected slices (50%)
Private leaderboard: 500 remaining slices (50%)
The random split ensures that slices from all samples and all z-positions contribute to both leaderboards.

Submission File
For each test slice, you must predict an enhanced MRI slice encoded as a base64 string. The file should contain a header and have the following format:

row_id,prediction
sample_019_slice_000,<base64_encoded_slice>
sample_019_slice_001,<base64_encoded_slice>
sample_019_slice_002,<base64_encoded_slice>
etc.
row_id: Format is sample_XXX_slice_YYY where XXX is the sample number (019-023) and YYY is the slice index (000-199)
prediction: Base64-encoded predicted slice (179 × 221 pixels), created using the provided slice_to_base64() function from extract_slices.py
Total rows: 1000 (5 samples × 200 slices)

Use the provided extract_slices.py to ensure correct encoding:

from extract_slices import create_submission_df, load_nifti

# Create predictions for all test samples
predictions = {
    'sample_019': your_model(load_nifti('test/low_field/sample_019_lowfield.nii.gz')),
    'sample_020': your_model(load_nifti('test/low_field/sample_020_lowfield.nii.gz')),
    'sample_021': your_model(load_nifti('test/low_field/sample_021_lowfield.nii.gz')),
    'sample_022': your_model(load_nifti('test/low_field/sample_022_lowfield.nii.gz')),
    'sample_023': your_model(load_nifti('test/low_field/sample_023_lowfield.nii.gz')),
}

# Create submission DataFrame (1000 rows)
submission_df = create_submission_df(predictions)
submission_df.to_csv('submission.csv', index=False)
Reference Scores
The sample_submission.csv provides a simple baseline using bicubic interpolation. For reference, we also show scores from a state-of-the-art (SOTA) diffusion-based model.

Submission	Public Score	Private Score
Baseline (bicubic interpolation)	0.5130	0.5153
SOTA (diffusion model)	0.6399	0.6343
Your goal is to exceed the baseline and approach or surpass SOTA!


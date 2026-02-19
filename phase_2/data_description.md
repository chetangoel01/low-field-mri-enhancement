Dataset Description
This dataset contains paired low-field (64mT) and high-field (3T) T1-weighted brain MRI scans for training image enhancement models.

Files
train/ - Training data folder
train/low_field/ - Low-field MRI volumes (18 NIfTI files)
train/high_field/ - Corresponding high-field MRI volumes (18 NIfTI files)
test/ - Test data folder
test/low_field/ - Low-field MRI volumes for prediction (5 NIfTI files)
train.csv - Training ground truth slices (3600 rows: 18 samples × 200 slices)
sample_submission.csv - Example submission with baseline predictions (1000 rows)
extract_slices.py - Utility script for slice extraction and encoding
metric.py - Evaluation metric implementation (MS-SSIM)
NIfTI Volume Specifications
Field	Low-Field (64mT)	High-Field (3T)
Shape	112 × 138 × 40	179 × 221 × 200
Voxel Size	1.6 × 1.6 × 5.0 mm	1.0 × 1.0 × 1.0 mm
Number of Slices	40	200
Each Slice Shape	112 × 138	179 × 221
Row ID Format
Each row represents a single axial slice:

row_id = sample_XXX_slice_YYY
XXX: Sample number (001-018 for train, 019-023 for test)
YYY: Slice index (000-199)
Examples:

sample_001_slice_000 - First slice of first training sample
sample_019_slice_100 - Center slice of first test sample
sample_023_slice_199 - Last slice of last test sample
Column Descriptions
train.csv (3600 rows)
Column	Description
row_id	Unique slice identifier (e.g., sample_001_slice_050)
ground_truth	Base64-encoded high-field slice (179 × 221 pixels)
sample_submission.csv (1000 rows)
Column	Description
row_id	Unique slice identifier (e.g., sample_019_slice_050)
prediction	Base64-encoded predicted slice (179 × 221 pixels)
How to Use the Extraction Script
from extract_slices import (
    load_nifti,
    slice_to_base64,
    base64_to_slice,
    volume_to_submission_rows,
    create_submission_df
)

# Load a volume
volume = load_nifti("train/high_field/sample_001_highfield.nii.gz")
print(f"Volume shape: {volume.shape}")  # (179, 221, 200)

# Your model produces a predicted volume
predicted_volume = your_model(low_field_input)  # Must be (179, 221, 200)

# Option 1: Create rows for one sample
rows = volume_to_submission_rows(predicted_volume, 'sample_019')
# rows is a list of 200 dicts: [{'row_id': 'sample_019_slice_000', 'prediction': '...'}, ...]

# Option 2: Create full submission
predictions = {
    'sample_019': pred_019,
    'sample_020': pred_020,
    'sample_021': pred_021,
    'sample_022': pred_022,
    'sample_023': pred_023,
}
submission_df = create_submission_df(predictions)
submission_df.to_csv('submission.csv', index=False)
print(f"Submission rows: {len(submission_df)}")  # 1000
Important Notes
Registration: High-field volumes have been registered to low-field space. The low-field data is untouched.

Output Shape: Your predicted volume must be (179 × 221 × 200) to match the high-field target dimensions.

All Slices: You must submit predictions for all 200 slices per sample (1000 total rows).

Encoding: Use the provided slice_to_base64() function - other encoding methods will cause submission errors.

Row Order: The order of rows in your submission doesn't matter - rows are matched by row_id.

Acronyms
Term	Definition
MRI	Magnetic Resonance Imaging
T1w	T1-weighted (a specific MRI contrast)
64mT	64 milliTesla (low magnetic field strength)
3T	3 Tesla (high magnetic field strength)
NIfTI	Neuroimaging Informatics Technology Initiative (file format)
MS-SSIM	Multi-Scale Structural Similarity Index
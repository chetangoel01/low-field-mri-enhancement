"""
Simple Dataset for direct training from raw train folder.
Bypasses the expanded_dataset requirement for quick training.
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import torch.nn.functional as F


class SimpleMRIDataset(Dataset):
    """
    Simple MRI dataset that loads directly from train/low_field and train/high_field.
    Extracts random patches during training.
    """
    
    def __init__(
        self,
        low_field_dir: str,
        high_field_dir: str,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        patches_per_volume: int = 50,
        target_shape: Optional[Tuple[int, int, int]] = None,
        augment: bool = True,
        val_samples: Optional[List[str]] = None,
        is_val: bool = False
    ):
        """
        Args:
            low_field_dir: Path to low field images
            high_field_dir: Path to high field images
            patch_size: Size of patches to extract
            patches_per_volume: Number of patches per volume
            target_shape: Target shape for high-field (upsampling target)
            augment: Whether to apply augmentation
            val_samples: List of sample IDs to use for validation
            is_val: Whether this is validation set
        """
        self.low_field_dir = Path(low_field_dir)
        self.high_field_dir = Path(high_field_dir)
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.target_shape = target_shape
        self.augment = augment and not is_val
        self.is_val = is_val
        
        # Default validation samples
        if val_samples is None:
            val_samples = ['sample_017', 'sample_018']
        
        # Find all paired samples
        self.samples = []
        
        for lf_file in sorted(self.low_field_dir.glob('*.nii*')):
            sample_id = lf_file.stem.replace('_lowfield', '').replace('.nii', '')
            
            # Find matching high-field file
            hf_pattern = f"{sample_id}*"
            hf_files = list(self.high_field_dir.glob(hf_pattern))
            
            if hf_files:
                # Check if this is train or val
                is_val_sample = any(vs in sample_id for vs in val_samples)
                
                if (is_val and is_val_sample) or (not is_val and not is_val_sample):
                    self.samples.append({
                        'id': sample_id,
                        'low_field': str(lf_file),
                        'high_field': str(hf_files[0])
                    })
        
        print(f"{'Val' if is_val else 'Train'} dataset: {len(self.samples)} volumes, "
              f"{len(self.samples) * patches_per_volume} total patches")
        
        # Cache for loaded volumes
        self._cache = {}
    
    def __len__(self):
        return len(self.samples) * self.patches_per_volume
    
    def _load_volume(self, path: str) -> np.ndarray:
        """Load and cache a volume."""
        if path not in self._cache:
            nifti = nib.load(path)
            volume = nifti.get_fdata().astype(np.float32)
            
            # Normalize to [0, 1]
            p1, p99 = np.percentile(volume, [1, 99])
            if p99 - p1 > 0:
                volume = np.clip(volume, p1, p99)
                volume = (volume - p1) / (p99 - p1)
            
            self._cache[path] = volume
        
        return self._cache[path]
    
    def _upsample_volume(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Upsample volume to target shape."""
        tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=False)
        return upsampled.squeeze().numpy()
    
    def _extract_random_patch(
        self,
        low_field: np.ndarray,
        high_field: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract aligned random patches from both volumes."""
        # First upsample low-field to match high-field
        if self.target_shape:
            low_field_up = self._upsample_volume(low_field, self.target_shape)
        else:
            low_field_up = self._upsample_volume(low_field, high_field.shape)
        
        # Random position
        d, h, w = high_field.shape
        pd, ph, pw = self.patch_size
        
        d_start = np.random.randint(0, max(1, d - pd + 1))
        h_start = np.random.randint(0, max(1, h - ph + 1))
        w_start = np.random.randint(0, max(1, w - pw + 1))
        
        # Extract patches
        lf_patch = low_field_up[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        hf_patch = high_field[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        # Pad if necessary
        if lf_patch.shape != self.patch_size:
            lf_padded = np.zeros(self.patch_size, dtype=np.float32)
            hf_padded = np.zeros(self.patch_size, dtype=np.float32)
            lf_padded[:lf_patch.shape[0], :lf_patch.shape[1], :lf_patch.shape[2]] = lf_patch
            hf_padded[:hf_patch.shape[0], :hf_patch.shape[1], :hf_patch.shape[2]] = hf_patch
            lf_patch, hf_patch = lf_padded, hf_padded
        
        return lf_patch, hf_patch
    
    def _augment(self, lf_patch: np.ndarray, hf_patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations."""
        # Random flips
        if np.random.random() > 0.5:
            lf_patch = np.flip(lf_patch, axis=0).copy()
            hf_patch = np.flip(hf_patch, axis=0).copy()
        if np.random.random() > 0.5:
            lf_patch = np.flip(lf_patch, axis=1).copy()
            hf_patch = np.flip(hf_patch, axis=1).copy()
        if np.random.random() > 0.5:
            lf_patch = np.flip(lf_patch, axis=2).copy()
            hf_patch = np.flip(hf_patch, axis=2).copy()
        
        # Random 90-degree rotations
        k = np.random.randint(0, 4)
        if k > 0:
            lf_patch = np.rot90(lf_patch, k, axes=(1, 2)).copy()
            hf_patch = np.rot90(hf_patch, k, axes=(1, 2)).copy()
        
        # Random intensity scaling (only on low-field input)
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            lf_patch = np.clip(lf_patch * scale, 0, 1)
        
        return lf_patch, hf_patch
    
    def __getitem__(self, idx):
        volume_idx = idx // self.patches_per_volume
        sample = self.samples[volume_idx]
        
        # Load volumes
        low_field = self._load_volume(sample['low_field'])
        high_field = self._load_volume(sample['high_field'])
        
        # Extract patch
        lf_patch, hf_patch = self._extract_random_patch(low_field, high_field)
        
        # Augment
        if self.augment:
            lf_patch, hf_patch = self._augment(lf_patch, hf_patch)
        
        # Convert to tensors [1, D, H, W]
        input_tensor = torch.from_numpy(lf_patch).float().unsqueeze(0)
        target_tensor = torch.from_numpy(hf_patch).float().unsqueeze(0)
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'sample_id': sample['id']
        }


def get_simple_dataloaders(
    config_path: str,
    batch_size: int = 4,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Get simple dataloaders that work directly with raw train data.
    
    Args:
        config_path: Path to configs/config.yaml
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    paths = config.get('paths', {})
    low_field_dir = paths.get('train_low_field', 'train/low_field')
    high_field_dir = paths.get('train_high_field', 'train/high_field')
    
    training = config.get('training', {})
    patch_size = tuple(training.get('patch_size', [64, 64, 64]))
    patches_per_volume = training.get('patches_per_volume', 50)
    
    dims = config.get('dimensions', {})
    target_shape = tuple(dims.get('high_field', {}).get('shape', [179, 221, 200]))
    
    org = config.get('organization', {})
    val_samples = org.get('val_samples', ['sample_016', 'sample_017', 'sample_018'])
    
    # Create datasets
    train_dataset = SimpleMRIDataset(
        low_field_dir=low_field_dir,
        high_field_dir=high_field_dir,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
        target_shape=target_shape,
        augment=True,
        val_samples=val_samples,
        is_val=False
    )
    
    val_dataset = SimpleMRIDataset(
        low_field_dir=low_field_dir,
        high_field_dir=high_field_dir,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume // 2,
        target_shape=target_shape,
        augment=False,
        val_samples=val_samples,
        is_val=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


class SliceMRIDataset(Dataset):
    """
    2.5D MRI dataset that yields (N adjacent axial slices, center target slice) pairs.

    Loads paired low-field / high-field volumes, upsamples low-field to high-field
    shape via trilinear interpolation, then extracts axial slices.
    """

    def __init__(
        self,
        low_field_dir: str,
        high_field_dir: str,
        slice_context: int = 5,
        target_shape: Optional[Tuple[int, int, int]] = None,
        augment: bool = True,
        val_samples: Optional[List[str]] = None,
        is_val: bool = False,
        patch_size_2d: Optional[Tuple[int, int]] = None,
    ):
        self.low_field_dir = Path(low_field_dir)
        self.high_field_dir = Path(high_field_dir)
        self.slice_context = slice_context
        self.half_ctx = slice_context // 2
        self.target_shape = target_shape
        self.augment = augment and not is_val
        self.is_val = is_val
        self.patch_size_2d = patch_size_2d

        if val_samples is None:
            val_samples = ['sample_017', 'sample_018']

        # Find paired samples
        self.samples = []
        for lf_file in sorted(self.low_field_dir.glob('*.nii*')):
            sample_id = lf_file.stem.replace('_lowfield', '').replace('.nii', '')
            hf_files = list(self.high_field_dir.glob(f"{sample_id}*"))
            if hf_files:
                is_val_sample = any(vs in sample_id for vs in val_samples)
                if (is_val and is_val_sample) or (not is_val and not is_val_sample):
                    self.samples.append({
                        'id': sample_id,
                        'low_field': str(lf_file),
                        'high_field': str(hf_files[0]),
                    })

        # Cache for loaded and preprocessed volumes
        self._cache = {}

        # Determine number of axial slices from target shape
        if target_shape is not None:
            self.num_slices = target_shape[2]  # z-dimension
        else:
            self.num_slices = 200  # default high-field z

        total = len(self.samples) * self.num_slices
        print(f"{'Val' if is_val else 'Train'} slice dataset: "
              f"{len(self.samples)} volumes, {total} slices")

    def __len__(self):
        return len(self.samples) * self.num_slices

    def _load_and_preprocess(self, sample: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load, normalize, and upsample a volume pair. Returns (low_up, high)."""
        key = sample['id']
        if key not in self._cache:
            lf = nib.load(sample['low_field']).get_fdata().astype(np.float32)
            hf = nib.load(sample['high_field']).get_fdata().astype(np.float32)

            # Percentile normalization
            for vol_name, vol in [('lf', lf), ('hf', hf)]:
                p1, p99 = np.percentile(vol, [1, 99])
                if p99 - p1 > 0:
                    vol = np.clip(vol, p1, p99)
                    vol = (vol - p1) / (p99 - p1)
                if vol_name == 'lf':
                    lf = vol
                else:
                    hf = vol

            # Upsample low-field to high-field shape
            shape = self.target_shape if self.target_shape else hf.shape
            tensor = torch.from_numpy(lf).float().unsqueeze(0).unsqueeze(0)
            lf_up = F.interpolate(tensor, size=shape, mode='trilinear',
                                  align_corners=False).squeeze().numpy()

            self._cache[key] = (lf_up, hf)
        return self._cache[key]

    def __getitem__(self, idx):
        vol_idx = idx // self.num_slices
        z = idx % self.num_slices
        sample = self.samples[vol_idx]

        lf_up, hf = self._load_and_preprocess(sample)

        # Extract adjacent slices with mirror-padding at boundaries
        depth = lf_up.shape[2]
        input_slices = np.zeros((self.slice_context, lf_up.shape[0], lf_up.shape[1]),
                                dtype=np.float32)
        for i, dz in enumerate(range(-self.half_ctx, self.half_ctx + 1)):
            sz = z + dz
            # Mirror-pad
            if sz < 0:
                sz = -sz
            elif sz >= depth:
                sz = 2 * (depth - 1) - sz
            sz = max(0, min(sz, depth - 1))
            input_slices[i] = lf_up[:, :, sz]

        target_slice = hf[:, :, z]

        if self.patch_size_2d is not None:
            input_slices, target_slice = self._crop_slices(input_slices, target_slice, random_crop=self.augment)

        # Augmentation
        if self.augment:
            input_slices, target_slice = self._augment_slices(input_slices, target_slice)

        input_tensor = torch.from_numpy(input_slices.copy()).float()      # (C, H, W)
        target_tensor = torch.from_numpy(target_slice.copy()).float().unsqueeze(0)  # (1, H, W)

        return {
            'input': input_tensor,
            'target': target_tensor,
            'sample_id': sample['id'],
        }

    def _augment_slices(
        self,
        input_slices: np.ndarray,
        target_slice: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random augmentations applied consistently to input and target."""
        # Horizontal flip (axis 2 = W for input, axis 1 = W for target)
        if np.random.random() > 0.5:
            input_slices = np.flip(input_slices, axis=2).copy()
            target_slice = np.flip(target_slice, axis=1).copy()

        # Vertical flip (axis 1 = H for input, axis 0 = H for target)
        if np.random.random() > 0.5:
            input_slices = np.flip(input_slices, axis=1).copy()
            target_slice = np.flip(target_slice, axis=0).copy()

        # No 90-degree rotations: slices are non-square (179x221),
        # rotation would change dimensions and break batching

        # Intensity scaling (input only)
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            input_slices = np.clip(input_slices * scale, 0, 1)

        # Gaussian noise (input only)
        if np.random.random() < 0.3:
            sigma = np.random.uniform(0.01, 0.03)
            noise = np.random.normal(0, sigma, input_slices.shape).astype(np.float32)
            input_slices = np.clip(input_slices + noise, 0, 1)

        # Random gamma (input only)
        if np.random.random() < 0.3:
            gamma = np.random.uniform(0.8, 1.2)
            input_slices = np.clip(np.power(input_slices, gamma), 0, 1)

        # Random brightness shift (input only)
        if np.random.random() < 0.3:
            shift = np.random.uniform(-0.05, 0.05)
            input_slices = np.clip(input_slices + shift, 0, 1)

        return input_slices, target_slice

    def _crop_slices(
        self,
        input_slices: np.ndarray,
        target_slice: np.ndarray,
        random_crop: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply aligned crop to input and target slices."""
        ph, pw = self.patch_size_2d
        h, w = target_slice.shape
        ph = min(ph, h)
        pw = min(pw, w)

        if random_crop:
            y0 = np.random.randint(0, max(1, h - ph + 1))
            x0 = np.random.randint(0, max(1, w - pw + 1))
        else:
            y0 = max(0, (h - ph) // 2)
            x0 = max(0, (w - pw) // 2)

        input_crop = input_slices[:, y0:y0 + ph, x0:x0 + pw]
        target_crop = target_slice[y0:y0 + ph, x0:x0 + pw]
        return input_crop, target_crop


class ExpandedSliceMRIDataset(Dataset):
    """
    2.5D slice dataset backed by expanded_dataset/manifest.csv.

    Supports filtering by split ('train'/'val') and data type
    ('real'/'synthetic') for curriculum learning.
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        split: str = 'train',
        data_types: Optional[List[str]] = None,
        slice_context: int = 5,
        target_shape: Optional[Tuple[int, int, int]] = None,
        augment: bool = True,
        slices_per_volume: Optional[int] = None,
        patch_size_2d: Optional[Tuple[int, int]] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.split = split
        self.slice_context = slice_context
        self.half_ctx = slice_context // 2
        self.target_shape = target_shape
        self.augment = augment and split == 'train'
        self.patch_size_2d = patch_size_2d

        rows = []
        with open(self.manifest_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('split') != split:
                    continue
                if data_types and row.get('type') not in data_types:
                    continue
                rows.append({
                    'id': row['sample_id'],
                    'type': row['type'],
                    'low_field': str(self.data_root / row['low_field_path']),
                    'high_field': str(self.data_root / row['high_field_path']),
                })
        self.samples = rows

        # Cache for loaded and preprocessed volumes
        self._cache = {}

        if slices_per_volume is not None:
            self.num_slices = slices_per_volume
        elif target_shape is not None:
            self.num_slices = target_shape[2]
        else:
            self.num_slices = 200

        total = len(self.samples) * self.num_slices
        type_counts: Dict[str, int] = {}
        for s in self.samples:
            t = s['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"{split.capitalize()} expanded slice dataset: {len(self.samples)} volumes, {total} slices, types={type_counts}")

    def __len__(self):
        return len(self.samples) * self.num_slices

    def _load_and_preprocess(self, sample: dict) -> Tuple[np.ndarray, np.ndarray]:
        key = sample['id']
        if key not in self._cache:
            lf = nib.load(sample['low_field']).get_fdata().astype(np.float32)
            hf = nib.load(sample['high_field']).get_fdata().astype(np.float32)

            # Percentile normalization
            for vol_name, vol in [('lf', lf), ('hf', hf)]:
                p1, p99 = np.percentile(vol, [1, 99])
                if p99 - p1 > 0:
                    vol = np.clip(vol, p1, p99)
                    vol = (vol - p1) / (p99 - p1)
                if vol_name == 'lf':
                    lf = vol
                else:
                    hf = vol

            # Resample both low-field and high-field to a shared target shape.
            # Some synthetic volumes may have different HF dimensions (e.g., 256 width).
            shape = self.target_shape if self.target_shape else hf.shape

            lf_tensor = torch.from_numpy(lf).float().unsqueeze(0).unsqueeze(0)
            lf_up = F.interpolate(lf_tensor, size=shape, mode='trilinear', align_corners=False).squeeze().numpy()

            if tuple(hf.shape) != tuple(shape):
                hf_tensor = torch.from_numpy(hf).float().unsqueeze(0).unsqueeze(0)
                hf_resampled = F.interpolate(hf_tensor, size=shape, mode='trilinear', align_corners=False).squeeze().numpy()
            else:
                hf_resampled = hf

            self._cache[key] = (lf_up, hf_resampled)
        return self._cache[key]

    def _augment_slices(self, input_slices: np.ndarray, target_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Horizontal flip (axis 2 = W for input, axis 1 = W for target)
        if np.random.random() > 0.5:
            input_slices = np.flip(input_slices, axis=2).copy()
            target_slice = np.flip(target_slice, axis=1).copy()

        # Vertical flip (axis 1 = H for input, axis 0 = H for target)
        if np.random.random() > 0.5:
            input_slices = np.flip(input_slices, axis=1).copy()
            target_slice = np.flip(target_slice, axis=0).copy()

        # Intensity scaling (input only)
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            input_slices = np.clip(input_slices * scale, 0, 1)

        # Gaussian noise (input only)
        if np.random.random() < 0.3:
            sigma = np.random.uniform(0.01, 0.03)
            noise = np.random.normal(0, sigma, input_slices.shape).astype(np.float32)
            input_slices = np.clip(input_slices + noise, 0, 1)

        # Random gamma (input only)
        if np.random.random() < 0.3:
            gamma = np.random.uniform(0.8, 1.2)
            input_slices = np.clip(np.power(input_slices, gamma), 0, 1)

        return input_slices, target_slice

    def __getitem__(self, idx):
        vol_idx = idx // self.num_slices
        z = idx % self.num_slices
        sample = self.samples[vol_idx]

        lf_up, hf = self._load_and_preprocess(sample)
        # Use the shared valid depth between input and target volumes.
        depth = min(lf_up.shape[2], hf.shape[2])
        if depth <= 0:
            raise ValueError(f"Invalid depth for sample {sample['id']}: lf={lf_up.shape}, hf={hf.shape}")

        # Clamp z in case dataset-wide num_slices is larger than this sample depth.
        z = max(0, min(z, depth - 1))

        input_slices = np.zeros((self.slice_context, lf_up.shape[0], lf_up.shape[1]), dtype=np.float32)
        for i, dz in enumerate(range(-self.half_ctx, self.half_ctx + 1)):
            sz = z + dz
            if sz < 0:
                sz = -sz
            elif sz >= depth:
                sz = 2 * (depth - 1) - sz
            sz = max(0, min(sz, depth - 1))
            input_slices[i] = lf_up[:, :, sz]

        target_slice = hf[:, :, z]

        if self.patch_size_2d is not None:
            input_slices, target_slice = self._crop_slices(input_slices, target_slice, random_crop=self.augment)

        if self.augment:
            input_slices, target_slice = self._augment_slices(input_slices, target_slice)

        input_tensor = torch.from_numpy(input_slices.copy()).float()
        target_tensor = torch.from_numpy(target_slice.copy()).float().unsqueeze(0)

        return {
            'input': input_tensor,
            'target': target_tensor,
            'sample_id': sample['id'],
            'data_type': sample['type'],
        }

    def _crop_slices(
        self,
        input_slices: np.ndarray,
        target_slice: np.ndarray,
        random_crop: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply aligned crop to input and target slices."""
        if self.patch_size_2d is None:
            return input_slices, target_slice
        ph, pw = self.patch_size_2d
        h, w = target_slice.shape
        ph = min(ph, h)
        pw = min(pw, w)

        if random_crop:
            y0 = np.random.randint(0, max(1, h - ph + 1))
            x0 = np.random.randint(0, max(1, w - pw + 1))
        else:
            y0 = max(0, (h - ph) // 2)
            x0 = max(0, (w - pw) // 2)

        input_crop = input_slices[:, y0:y0 + ph, x0:x0 + pw]
        target_crop = target_slice[y0:y0 + ph, x0:x0 + pw]
        return input_crop, target_crop


class SliceCurriculumSampler(torch.utils.data.Sampler):
    """
    Curriculum sampler for ExpandedSliceMRIDataset.

    Uses phase data ratios (real/synthetic) as a function of epoch.
    """

    def __init__(self, dataset: ExpandedSliceMRIDataset, phase_config: List[dict], current_epoch: int = 1):
        self.dataset = dataset
        self.phase_config = phase_config
        self.current_epoch = current_epoch

        self.real_indices = []
        self.synthetic_indices = []
        for i in range(len(dataset)):
            vol_idx = i // dataset.num_slices
            data_type = dataset.samples[vol_idx]['type']
            if data_type == 'real':
                self.real_indices.append(i)
            else:
                self.synthetic_indices.append(i)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _get_current_ratio(self) -> Dict[str, float]:
        for phase in self.phase_config:
            epoch_range = phase.get('epochs', [1, 10**9])
            if epoch_range[0] <= self.current_epoch <= epoch_range[1]:
                return phase.get('data_ratio', {'real': 0.5, 'synthetic': 0.5})
        return {'real': 0.5, 'synthetic': 0.5}

    def __iter__(self):
        ratio = self._get_current_ratio()
        real_ratio = float(ratio.get('real', 0.5))
        synth_ratio = float(ratio.get('synthetic', 0.5))
        total_ratio = real_ratio + synth_ratio
        if total_ratio > 0:
            real_ratio /= total_ratio
            synth_ratio /= total_ratio

        total_samples = len(self.dataset)
        n_real = int(total_samples * real_ratio)
        n_synth = total_samples - n_real

        indices = []
        if n_real > 0 and self.real_indices:
            real_samples = np.random.choice(self.real_indices, size=n_real, replace=True)
            indices.extend(real_samples.tolist())
        if n_synth > 0 and self.synthetic_indices:
            synth_samples = np.random.choice(self.synthetic_indices, size=n_synth, replace=True)
            indices.extend(synth_samples.tolist())

        np.random.shuffle(indices)
        return iter(indices[:total_samples])

    def __len__(self):
        return len(self.dataset)


def get_slice_dataloaders(
    config_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    current_epoch: int = 1,
    use_expanded: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Get dataloaders for 2.5D slice-based training."""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    paths = config.get('paths', {})
    low_field_dir = paths.get('train_low_field', 'train/low_field')
    high_field_dir = paths.get('train_high_field', 'train/high_field')

    dims = config.get('dimensions', {})
    target_shape = tuple(dims.get('high_field', {}).get('shape', [179, 221, 200]))

    esrgan = config.get('esrgan', {})
    model_cfg = esrgan.get('model', {})
    slice_context = model_cfg.get('slice_context', 5)

    if use_expanded:
        output_dir = paths.get('output_dir', 'expanded_dataset')
        manifest_path = Path(output_dir) / 'manifest.csv'
        if manifest_path.exists():
            slice_cfg = config.get('slice_training', {})
            slices_per_volume_train = int(slice_cfg.get('slices_per_volume_train', target_shape[2]))
            slices_per_volume_val = int(slice_cfg.get('slices_per_volume_val', target_shape[2]))
            phase_config = slice_cfg.get('phases', config.get('training', {}).get('phases', []))
            train_patch_size = slice_cfg.get('train_patch_size_2d', None)
            patch_size_2d = tuple(train_patch_size) if train_patch_size is not None else None

            train_dataset = ExpandedSliceMRIDataset(
                manifest_path=str(manifest_path),
                data_root=str(output_dir),
                split='train',
                data_types=['real', 'synthetic'],
                slice_context=slice_context,
                target_shape=target_shape,
                augment=True,
                slices_per_volume=slices_per_volume_train,
                patch_size_2d=patch_size_2d,
            )

            val_dataset = ExpandedSliceMRIDataset(
                manifest_path=str(manifest_path),
                data_root=str(output_dir),
                split='val',
                data_types=['real'],
                slice_context=slice_context,
                target_shape=target_shape,
                augment=False,
                slices_per_volume=slices_per_volume_val,
                patch_size_2d=None,
            )

            if phase_config:
                train_sampler = SliceCurriculumSampler(
                    dataset=train_dataset,
                    phase_config=phase_config,
                    current_epoch=current_epoch,
                )
                shuffle = False
            else:
                train_sampler = None
                shuffle = True

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            return train_loader, val_loader
        print(f"Expanded manifest not found at {manifest_path}, falling back to raw train/ directories")

    org = config.get('organization', {})
    val_samples = org.get('val_samples', ['sample_017', 'sample_018'])
    slice_cfg = config.get('slice_training', {})
    train_patch_size = slice_cfg.get('train_patch_size_2d', None)
    patch_size_2d = tuple(train_patch_size) if train_patch_size is not None else None

    train_dataset = SliceMRIDataset(
        low_field_dir=low_field_dir,
        high_field_dir=high_field_dir,
        slice_context=slice_context,
        target_shape=target_shape,
        augment=True,
        val_samples=val_samples,
        is_val=False,
        patch_size_2d=patch_size_2d,
    )

    val_dataset = SliceMRIDataset(
        low_field_dir=low_field_dir,
        high_field_dir=high_field_dir,
        slice_context=slice_context,
        target_shape=target_shape,
        augment=False,
        val_samples=val_samples,
        is_val=True,
        patch_size_2d=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    train_loader, val_loader = get_simple_dataloaders('configs/config.yaml', batch_size=2)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Input: {batch['input'].shape}")
    print(f"  Target: {batch['target'].shape}")
    print(f"  Sample IDs: {batch['sample_id']}")

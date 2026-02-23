"""MRI Super Resolution Dataset with per-slice normalization.

Loads paired low-field / high-field NIfTI volumes, trilinear-upsamples the
low-field data to match high-field shape, and serves 2D axial slices with
5-slice context for the 2.5D U-Net.

Critical design choice: per-slice [0,1] normalization matches the competition
metric (metric.py lines 70-74) which normalizes each slice independently.
"""

import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def normalize_slice(s):
    """Normalize a 2D slice to [0, 1] using per-slice min/max.

    Matches metric.py normalize() exactly.
    """
    s_min = s.min()
    s_max = s.max()
    if s_max - s_min > 0:
        return (s - s_min) / (s_max - s_min)
    return np.zeros_like(s)


class SliceMRIDataset(Dataset):
    """Dataset that serves 2D axial slices with z-context.

    Each sample: 5 adjacent normalized LF slices (input) + 1 center HF slice (target).
    Per-slice normalization to [0,1] on each 2D slice independently.
    Training: random square crop + 90-degree rotation augmentation.
    """

    def __init__(self, config, is_train=True):
        super().__init__()
        self.config = config
        self.is_train = is_train
        self.context = config['data']['slice_context']  # 5
        self.half_ctx = self.context // 2  # 2
        self.num_slices = config['data']['num_slices']  # 200
        target_shape = config['data']['target_shape']  # [179, 221, 200]
        self.target_shape = tuple(target_shape)
        val_samples = set(config['data'].get('val_samples', []))

        lf_dir = config['data']['train_lf_dir']
        hf_dir = config['data']['train_hf_dir']

        # Discover volume files
        lf_files = sorted([f for f in os.listdir(lf_dir) if f.endswith(('.nii', '.nii.gz'))])

        self.volumes = []

        for lf_file in lf_files:
            sample_id = lf_file.split('_')[1]

            # Split train/val (if val_samples is empty, all go to train)
            if val_samples:
                if is_train and sample_id in val_samples:
                    continue
                if not is_train and sample_id not in val_samples:
                    continue

            hf_file = lf_file.replace('lowfield', 'highfield')
            lf_path = os.path.join(lf_dir, lf_file)
            hf_path = os.path.join(hf_dir, hf_file)

            if not os.path.exists(hf_path):
                continue

            print(f"Loading {'train' if is_train else 'val'} volume: {sample_id}")

            lf_vol = nib.load(lf_path).get_fdata().astype(np.float32)
            hf_vol = nib.load(hf_path).get_fdata().astype(np.float32)

            lf_vol = self._upsample_volume(lf_vol, self.target_shape)

            if hf_vol.shape != tuple(self.target_shape):
                hf_vol = self._upsample_volume(hf_vol, self.target_shape)

            self.volumes.append((lf_vol, hf_vol, sample_id))

        print(f"Loaded {len(self.volumes)} {'train' if is_train else 'val'} volumes, "
              f"{len(self.volumes) * self.num_slices} total slices")

        # Augmentation config
        self.aug = config.get('augmentation', {})
        self.crop_size = self.aug.get('crop_size', None) if is_train else None

    def _upsample_volume(self, vol, target_shape):
        """Trilinear upsample a 3D volume to target_shape."""
        t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=target_shape, mode='trilinear', align_corners=False)
        return t.squeeze().numpy()

    def __len__(self):
        return len(self.volumes) * self.num_slices

    def __getitem__(self, idx):
        vol_idx = idx // self.num_slices
        z_idx = idx % self.num_slices

        lf_vol, hf_vol, sample_id = self.volumes[vol_idx]

        # Extract 5 adjacent LF slices with mirror-padding at boundaries
        input_slices = []
        for dz in range(-self.half_ctx, self.half_ctx + 1):
            z = z_idx + dz
            if z < 0:
                z = -z
            elif z >= self.num_slices:
                z = 2 * (self.num_slices - 1) - z
            z = max(0, min(z, self.num_slices - 1))

            s = lf_vol[:, :, z].copy()
            s = normalize_slice(s)
            input_slices.append(s)

        # Target: center HF slice, normalized per-slice
        target = hf_vol[:, :, z_idx].copy()
        target = normalize_slice(target)

        # Stack input: (5, H, W)
        input_tensor = np.stack(input_slices, axis=0).astype(np.float32)
        target_tensor = target[np.newaxis].astype(np.float32)  # (1, H, W)

        # Augmentation (train only)
        if self.is_train:
            input_tensor, target_tensor = self._augment(input_tensor, target_tensor)

        return torch.from_numpy(input_tensor), torch.from_numpy(target_tensor)

    def _augment(self, inp, tgt):
        """Apply augmentations to input (C, H, W) and target (1, H, W).

        Order: crop -> rotation -> flip -> intensity
        """
        # Random square crop
        if self.crop_size is not None:
            cs = self.crop_size
            _, h, w = inp.shape
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
            inp = inp[:, top:top+cs, left:left+cs].copy()
            tgt = tgt[:, top:top+cs, left:left+cs].copy()

        # 90-degree rotation (now possible since crops are square)
        if self.aug.get('rotation_90', False) and self.crop_size is not None:
            k = random.randint(0, 3)  # 0, 90, 180, 270 degrees
            if k > 0:
                inp = np.rot90(inp, k, axes=(1, 2)).copy()
                tgt = np.rot90(tgt, k, axes=(1, 2)).copy()

        # Horizontal flip
        if self.aug.get('horizontal_flip', False) and random.random() < 0.5:
            inp = inp[:, :, ::-1].copy()
            tgt = tgt[:, :, ::-1].copy()

        # Vertical flip
        if self.aug.get('vertical_flip', False) and random.random() < 0.5:
            inp = inp[:, ::-1, :].copy()
            tgt = tgt[:, ::-1, :].copy()

        # Intensity scaling (input only — target is fixed ground truth, must not be scaled)
        scale_range = self.aug.get('intensity_scale', None)
        if scale_range and random.random() < self.aug.get('intensity_scale_prob', 0.5):
            scale = random.uniform(scale_range[0], scale_range[1])
            inp = np.clip(inp * scale, 0.0, 1.0)

        # Gaussian noise (input only)
        noise_std = self.aug.get('gaussian_noise_std', 0.0)
        if noise_std > 0 and random.random() < self.aug.get('gaussian_noise_prob', 0.3):
            noise = np.random.normal(0, noise_std, inp.shape).astype(np.float32)
            inp = np.clip(inp + noise, 0.0, 1.0)

        return inp, tgt

"""
Data augmentation pipelines using Albumentations for multi-branch training.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# RGB branch augmentations
rgb_transform = A.Compose(
    [
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
            p=0.4,
        ),
        A.GaussNoise(var_limit=(5, 20), p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]
)

# ELA branch transformations
# IMPORTANT NOTE:
# Do NOT apply JPEG compression augmentation here. Doing so would overwrite or destroy
# the genuine JPEG compression discrepancies (Error Level Analysis signal) that the
# ELA branch relies on to detect splicing and copy-paste boundaries.
ela_transform = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.25, 0.25, 0.25],
        ),
        ToTensorV2(),
    ]
)

# FFT / Noise branch transformations
# Albumentations handles 4-channel inputs natively (Normalize mean/std of length 4).
fft_noise_transform = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.Normalize(
            mean=[0.5, 0.5, 0.5, 0.5],
            std=[0.25, 0.25, 0.25, 0.25],
        ),
        ToTensorV2(),
    ]
)

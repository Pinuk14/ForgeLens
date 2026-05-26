"""
PyTorch dataset and data loaders for multi-branch forensic training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from backend.app.forensics.preprocessors.ela import generate_ela
from backend.app.forensics.preprocessors.fft import generate_fft_noise_stack
from backend.app.forensics.preprocessors.metadata import extract_metadata
from training.datasets.augmentation import (
    ela_transform,
    fft_noise_transform,
    rgb_transform,
)

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# Supported image file extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ForensicDataset(Dataset):
    """Forensic dataset yielding multi-branch inputs on the fly.

    Folder layout expected under root_dir/{split}/:
        - authentic/
        - ai_generated/
        - manipulated/

    Labels are mapped based on the curriculum training phase:
        - Phase 1: authentic (0) & ai_generated (1)
        - Phase 2: authentic (0) & manipulated (2)
        - Phase 3: authentic (0) & ai_generated (1) & manipulated (2)
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        phase: int = 3,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.phase = phase

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Map folders and classes depending on the curriculum learning phase
        self.samples: list[tuple[Path, int]] = []
        
        # Class folder names and their corresponding label IDs
        class_mapping = {
            "authentic": 0,
            "ai_generated": 1,
            "manipulated": 2,
        }

        # Select which classes are active in the current phase
        if phase == 1:
            active_folders = ["authentic", "ai_generated"]
        elif phase == 2:
            active_folders = ["authentic", "manipulated"]
        elif phase == 3:
            active_folders = ["authentic", "ai_generated", "manipulated"]
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 1, 2, or 3.")

        for folder in active_folders:
            folder_path = split_dir / folder
            label = class_mapping[folder]
            if not folder_path.is_dir():
                logger.warning(
                    "Directory %s not found for phase %d. Skipping.",
                    folder_path,
                    phase,
                )
                continue

            # Gather all image files
            for file_path in folder_path.iterdir():
                if file_path.suffix.lower() in _IMAGE_EXTENSIONS:
                    self.samples.append((file_path, label))

        logger.info(
            "Initialized ForensicDataset for split '%s' (Phase %d) with %d samples.",
            split,
            phase,
            len(self.samples),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Load and preprocess a single sample.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, int]
            - rgb_tensor: (3, 224, 224) normalized RGB image
            - ela_tensor: (3, 224, 224) normalized ELA map
            - fft_noise_tensor: (4, 224, 224) normalized FFT+Noise stacked tensor
            - meta_tensor: (16,) normalized metadata feature vector
            - label: int class ID
        """
        img_path, label = self.samples[index]

        # 1. Read raw image bytes for ELA and metadata extraction
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        # 2. Read BGR image for noise residual & FFT
        # (cv2.imread expects strings/os.PathLike)
        bgr_image = cv2.imread(str(img_path))
        if bgr_image is None:
            raise IOError(f"Could not load image using OpenCV: {img_path}")

        # 3. Preprocess on the fly
        ela_arr = generate_ela(image_bytes)
        fft_noise_arr = generate_fft_noise_stack(bgr_image)
        meta = extract_metadata(image_bytes)

        # Convert baseline BGR to RGB for the standard backbone branch
        rgb_arr = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # 4. Apply branch-specific augmentations
        # Albumentations transforms return a dict with the transformed "image"
        transformed_rgb = rgb_transform(image=rgb_arr)["image"]
        transformed_ela = ela_transform(image=ela_arr)["image"]
        transformed_fft_noise = fft_noise_transform(image=fft_noise_arr)["image"]

        # 5. Tensorize metadata feature vector
        meta_tensor = torch.tensor(meta["feature_vector"], dtype=torch.float32)

        return (
            transformed_rgb,
            transformed_ela,
            transformed_fft_noise,
            meta_tensor,
            label,
        )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def get_weighted_sampler(dataset: ForensicDataset) -> WeightedRandomSampler:
    """Compute per-class weights and return a WeightedRandomSampler.

    Balances class frequencies at the source image level to prevent training
    biases, particularly in highly imbalanced classes.
    """
    labels = [label for _, label in dataset.samples]
    if not labels:
        raise ValueError("Cannot compute sampler weights on an empty dataset.")

    # Calculate class counts
    class_counts: dict[int, int] = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    # Class weight = 1 / class count
    class_weights = {
        label: 1.0 / count for label, count in class_counts.items()
    }

    # Weight per sample is its class's weight
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

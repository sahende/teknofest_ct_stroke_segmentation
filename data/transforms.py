# data/transforms.py

import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(norm_file="data_split/norm.json"):
    """
    Build Albumentations-based preprocessing and augmentation pipelines 
    for training and validation datasets.

    The function loads dataset-level normalization statistics (mean and std) 
    from a JSON file and defines two transformation pipelines:

    1. train_transform: Includes spatial augmentations (resize, rotation, translation),
       followed by normalization and tensor conversion.
    2. val_transform: Applies only resizing, normalization, and tensor conversion
       (no stochastic augmentations to ensure evaluation consistency).

    Parameters
    ----------
    norm_file : str, optional (default="data_split/norm.json")
        Path to a JSON file containing the normalization parameters 
        computed from the training dataset.
        Expected format: {"mean": float or [float, float, float],
                          "std": float or [float, float, float]}

    Returns
    -------
    train_transform : albumentations.Compose
        Augmentation pipeline for training images and masks.
    val_transform : albumentations.Compose
        Preprocessing pipeline for validation/test images and masks.
    """

    # Load dataset normalization parameters from JSON
    with open(norm_file, "r") as f:
        norm = json.load(f)

    # Ensure mean and std are lists of length 3 (one per RGB channel).
    # If stored as single values (grayscale dataset), replicate across 3 channels.
    mean = norm["mean"] if isinstance(norm.get("mean"), (list, tuple)) else [norm["mean"]] * 3
    std = norm["std"] if isinstance(norm.get("std"), (list, tuple)) else [norm["std"]] * 3

    # Training pipeline:
    # - Resize images/masks to a fixed size (224x224)
    # - Apply mild rotation (±10°) with probability 0.5
    # - Apply affine translation (up to 5% shift) with probability 0.5
    # - Normalize pixel values using dataset mean/std
    # - Convert to PyTorch tensor
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=10, p=0.5),
        A.Affine(translate_percent={"x": 0.05, "y": 0.05}, rotate=0, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    # Validation pipeline:
    # - Resize only (same as training size)
    # - Normalize using the same statistics
    # - Convert to PyTorch tensor
    # Note: No random augmentations are applied to keep evaluation deterministic.
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    return train_transform, val_transform

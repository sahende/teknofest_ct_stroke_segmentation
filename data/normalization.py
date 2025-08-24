# data/normalization.py

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

def compute_normalization_params(metadata_file, dataset_dir, output_path):
    """
    Compute global normalization parameters (mean and standard deviation) 
    for a dataset of grayscale CT images.

    The function iterates over all images listed in a metadata JSON file,
    loads them from the dataset directory, converts them to normalized 
    floating-point arrays, and computes pixel-wise statistics.

    These parameters are later used to normalize the dataset so that 
    the model trains more stably and converges faster.

    Parameters
    ----------
    metadata_file : str
        Path to a JSON file containing image identifiers (keys are image IDs).
        Example format: {"0001": {...}, "0002": {...}, ...}
    dataset_dir : str
        Directory containing PNG images (filenames expected as <img_id>.png).
    output_path : str
        Path to save the computed mean and standard deviation values as JSON.

    Returns
    -------
    mean : float
        Mean pixel intensity (normalized to [0, 1]).
    std : float
        Standard deviation of pixel intensities (normalized to [0, 1]).
    """

    # Load metadata that defines which images belong to the dataset
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    img_ids = list(metadata.keys())

    # Variables for online computation of mean and variance
    n_pixels = 0          # Total number of pixels across all images
    sum_pixels = 0.0      # Sum of pixel values
    sum_squared = 0.0     # Sum of squared pixel values (needed for variance)

    # Iterate over all image IDs defined in the metadata
    for img_id in tqdm(img_ids):
        img_path = os.path.join(dataset_dir, f"{img_id}.png")

        # Skip if the file is missing (robustness for incomplete datasets)
        if not os.path.exists(img_path):
            print(f"⚠️ {img_path} not found, skipping.")
            continue

        # Load image in grayscale ("L" mode ensures single channel)
        # Convert to NumPy array with float32 precision
        # Normalize values to [0, 1] by dividing by 255
        image = Image.open(img_path).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0

        # Accumulate statistics for mean and variance calculation
        sum_pixels += np.sum(image)
        sum_squared += np.sum(image ** 2)
        n_pixels += image.size  # Count of pixels in the current image

    # Final mean = E[X]
    mean = float(sum_pixels / n_pixels)

    # Variance = E[X^2] - (E[X])^2
    # Standard deviation = sqrt(variance)
    std = float(np.sqrt(sum_squared / n_pixels - mean ** 2))

    # Save normalization parameters to JSON for reproducibility
    normalization_params = {"mean": mean, "std": std}
    with open(output_path, "w") as f:
        json.dump(normalization_params, f, indent=4)

    # Print summary to console for quick inspection
    print(f"✅ Mean: {mean:.6f}, Std: {std:.6f}")

    return mean, std

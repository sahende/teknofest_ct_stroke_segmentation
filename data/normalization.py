# data/normalization.py
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

def compute_normalization_params(metadata_file, dataset_dir, output_path):
    """
    Computes mean and standard deviation of pixel values from grayscale CT images.
    
    Args:
        metadata_file (str): Path to JSON file with image IDs.
        dataset_dir (str): Directory containing PNG images.
        output_path (str): Where to save the mean and std as a JSON.
        
    Returns:
        mean (float), std (float)
    """
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    img_ids = list(metadata.keys())

    n_pixels = 0
    sum_pixels = 0.0
    sum_squared = 0.0

    for img_id in tqdm(img_ids):
        img_path = os.path.join(dataset_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            print(f"⚠️ {img_path} not found, skipping.")
            continue

        image = Image.open(img_path).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0

        sum_pixels += np.sum(image)
        sum_squared += np.sum(image ** 2)
        n_pixels += image.size

    mean = float(sum_pixels / n_pixels)
    std = float(np.sqrt(sum_squared / n_pixels - mean ** 2))

    normalization_params = {"mean": mean, "std": std}
    with open(output_path, "w") as f:
        json.dump(normalization_params, f, indent=4)

    print(f"✅ Mean: {mean:.6f}, Std: {std:.6f}")
    return mean, std
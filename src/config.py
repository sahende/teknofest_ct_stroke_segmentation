# src/config.py

import os
import torch

# ===============================
# Dataset Paths
# ===============================
DATA_DIR = "data_split"

# Training data
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train/images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train/masks")

# Validation data
VAL_IMG_DIR = os.path.join(DATA_DIR, "val/images")
VAL_MASK_DIR = os.path.join(DATA_DIR, "val/masks")

# Test data
TEST_IMG_DIR = os.path.join(DATA_DIR, "test/images")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test/masks")

# Output directory for checkpoints, logs, and results
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Checkpoint path
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_swin_upernet_cbam_tiny.pth")


# ===============================
# Training Hyperparameters
# ===============================
BATCH_SIZE = 4                     # Number of samples per batch
LR = 5e-5                          # Learning rate for optimizer
NUM_EPOCHS = 100                   # Maximum training epochs
ACCUMULATION_STEPS = 2             # Gradient accumulation steps
SEED = 42                           # Seed for reproducibility
SCHEDULER_PATIENCE = 2             # Early stopping patience for LR scheduler
SCHEDULER_FACTOR = 0.5             # LR reduction factor when plateau is detected


# ===============================
# Model Parameters
# ===============================
NUM_CLASSES = 1                    # Output channels (1 for binary segmentation)
FP_DIM = 256                        # Feature Pyramid output channels
PPM_OUT = 512                        # Pyramid Pooling Module output channels
BACKBONE = "swin_tiny_patch4_window7_224"  # Pretrained backbone architecture
PRETRAINED = True                   # Use ImageNet pretrained weights


# ===============================
# Device & Mixed Precision
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically select GPU if available
AMP_ENABLED = True                                      # Enable Automatic Mixed Precision (for speed & memory)


# ===============================
# Normalization
# ===============================
NORM_FILE = os.path.join(DATA_DIR, "norm.json")  # Path to precomputed mean/std for dataset normalization

import os
import json
import torch

# Paths
DATA_DIR = "data_split"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train/images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train/masks")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val/images")
VAL_MASK_DIR = os.path.join(DATA_DIR, "val/masks")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test/images")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test/masks")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_swin_upernet_cbam_tiny.pth")

# Training params
BATCH_SIZE = 4
LR = 5e-5
NUM_EPOCHS = 100
ACCUMULATION_STEPS = 2
SEED = 42
SCHEDULER_PATIENCE=2
SCHEDULER_FACTOR=0.5

# Model params
NUM_CLASSES = 1
FP_DIM = 256
PPM_OUT = 512
BACKBONE = "swin_tiny_patch4_window7_224"
PRETRAINED = True

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED=True

# Normalization
NORM_FILE = os.path.join(DATA_DIR, "norm.json")






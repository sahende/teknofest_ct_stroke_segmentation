# notebook/run_eval.py
# ====================================================
# 🔹 Run evaluation with TTA & Dice and visualizition
# ====================================================


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import torch
from huggingface_hub import hf_hub_download
from src.evaluate import run_evaluation
from model.swin_upernet_cbam import SwinUPerNetCBAM
import src.config as cfg
from utils.visualization import visualize_batch
from data.dataset import SegmentationDataset
from torch.utils.data import DataLoader
from data.transforms import get_transforms

# -------------------------
# Device
device = torch.device(cfg.DEVICE)

# -------------------------
# Init model (same config as training)
model = SwinUPerNetCBAM(
    backbone_name=cfg.BACKBONE,  
    num_classes=cfg.NUM_CLASSES,
    fpn_dim=cfg.FP_DIM,          
    ppm_out=cfg.PPM_OUT,
    pretrained=cfg.PRETRAINED    
).to(cfg.DEVICE)



# -------------------------
# Load weights from Hugging Face
hf_model_repo = "Sahende/teknofest_ct_stroke_segmentation"
checkpoint_file = hf_hub_download(repo_id=hf_model_repo, filename="pytorch_model.bin")
state_dict = torch.load(checkpoint_file, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()
print("✅ Model loaded from Hugging Face successfully.")

# -------------------------
# Run evaluation
avg_dice, f1, iou = run_evaluation(
    model=model,
    checkpoint_path=None,   # HF'den yüklendiği için burası opsiyonel
    test_img_dir=cfg.TEST_IMG_DIR,
    test_mask_dir=cfg.TEST_MASK_DIR,
    batch_size=4,
    device=device,
    threshold=0.5
)

print(f"\n✅ Test Result -> Dice={avg_dice:.4f}, F1={f1:.4f}, IoU={iou:.4f}")

# Load transforms
_, val_transform = get_transforms()
# -------------------------
# Prepare test dataset + dataloader
test_dataset = SegmentationDataset(
    image_dir=cfg.TEST_IMG_DIR,
    mask_dir=cfg.TEST_MASK_DIR,
    transform=val_transform # augmentations are not needed for visualization
)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# -------------------------
# Visualize predictions
print("🔍 Visualizing predictions for one batch...")
visualize_batch(test_loader, model=model, device=device, batch_index=0, max_samples=5)


# =====================================================
#  (Optional) Run training instead of evaluation
# =====================================================
# from src.train import train
# train()

# notebook/run_eval.py
# ==================================
# 🔹 Run evaluation with TTA & Dice
# ==================================


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import torch
from huggingface_hub import hf_hub_download
from src.evaluate import run_evaluation
from model.swin_upernet_cbam import SwinUPerNetCBAM
import src.config as cfg

# -------------------------
# Device
device = torch.device(cfg.DEVICE)

# -------------------------
# Init model (same config as training)
model = SwinUPerNetCBAM(
    backbone_name=cfg.BACKBONE,  # cfg.BACKBONE_NAME yerine cfg.BACKBONE
    num_classes=cfg.NUM_CLASSES,
    fpn_dim=cfg.FP_DIM,          # cfg.FPN_DIM yerine cfg.FP_DIM
    ppm_out=cfg.PPM_OUT,
    pretrained=cfg.PRETRAINED    # cfg.PRETRAINED_BACKBONE yerine cfg.PRETRAINED
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


# =====================================================
#  (Optional) Run training instead of evaluation
# =====================================================
# from src.train import train
# train()

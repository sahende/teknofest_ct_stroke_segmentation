# src/evaluate.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SegmentationDataset
from data.transforms import get_transforms
from utils.metrics import compute_metrics, dice_coefficient
import src.config as cfg

# -------------------------
# TTA Inference
def tta_inference_probs(model, images, device):
    """
    Apply Test Time Augmentation (TTA) and return averaged probabilities.
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device)

        p0 = torch.sigmoid(model(images))  # original

        # horizontal flip
        x_hflip = torch.flip(images, dims=[3])
        p1 = torch.sigmoid(model(x_hflip))
        p1 = torch.flip(p1, dims=[3])

        # vertical flip
        x_vflip = torch.flip(images, dims=[2])
        p2 = torch.sigmoid(model(x_vflip))
        p2 = torch.flip(p2, dims=[2])

        # transpose (swap H <-> W)
        x_trans = images.permute(0, 1, 3, 2).contiguous()
        p3 = torch.sigmoid(model(x_trans))
        p3 = p3.permute(0, 1, 3, 2).contiguous()

        return (p0 + p1 + p2 + p3) / 4.0

# -------------------------
# Main evaluation function
def run_evaluation(model, checkpoint_path, test_img_dir, test_mask_dir, batch_size=4, device=cfg.DEVICE, threshold=0.5):
    device = torch.device(device)
    
    # transforms (same as validation)
    _, val_transform = get_transforms(cfg.NORM_FILE)

    # dataset & loader
    test_dataset = SegmentationDataset(test_img_dir, test_mask_dir, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        pass

    
    model.to(device)
    model.eval()

    # evaluation loop
    test_dice_sum = 0.0
    n_batches = 0
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing with TTA"):
            images, masks = images.to(device), masks.to(device)

            probs = tta_inference_probs(model, images, device)
            batch_dice = dice_coefficient(probs, masks, from_logits=False)
            test_dice_sum += batch_dice
            n_batches += 1

            probs_cpu = probs.cpu().numpy().astype(np.float16)
            masks_cpu = masks.cpu().numpy().astype(np.uint8)
            preds_bin = (probs_cpu >= threshold).astype(np.uint8)

            y_true_all.append(masks_cpu)
            y_pred_all.append(preds_bin)

    avg_test_dice = test_dice_sum / max(1, n_batches)
    y_true_all = np.concatenate(y_true_all, axis=0).ravel()
    y_pred_all = np.concatenate(y_pred_all, axis=0).ravel()
    test_f1, test_iou = compute_metrics(y_true_all, y_pred_all)


    return avg_test_dice, test_f1, test_iou
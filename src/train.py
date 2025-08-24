# src/train.py

"""
Training script for SwinUPerNet-CBAM segmentation model.

Features:
- Config-driven parameters (from src.config)
- Mixed precision training (AMP)
- Gradient accumulation
- Checkpointing (resume + best model saving)
- Dice, F1, IoU metrics
- Validation after each epoch
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch import amp

from data.dataset import SegmentationDataset
from utils.helpers import set_seed, load_checkpoint, save_checkpoint
from utils.metrics import dice_coefficient, compute_metrics
from model.swin_upernet_cbam import SwinUPerNetCBAM
import src.config as cfg
from data.transforms import get_transforms


def train():
    # -------------------------
    # Set seed for reproducibility and define device
    # -------------------------
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)

    # -------------------------
    # Define image transformations
    # TRAIN_TRANSFORM: augmentations for robust training
    # VAL_TRANSFORM: minimal preprocessing for validation
    # -------------------------
    TRAIN_TRANSFORM, VAL_TRANSFORM = get_transforms(cfg.NORM_FILE)

    # -------------------------
    # Prepare datasets and dataloaders
    # -------------------------
    train_dataset = SegmentationDataset(cfg.TRAIN_IMG_DIR, cfg.TRAIN_MASK_DIR, transform=TRAIN_TRANSFORM)
    val_dataset = SegmentationDataset(cfg.VAL_IMG_DIR, cfg.VAL_MASK_DIR, transform=VAL_TRANSFORM)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # -------------------------
    # Initialize the SwinUPerNet-CBAM model
    # Backbone: Swin Transformer with optional pretrained weights
    # PPM + FPN + CBAM modules integrated for segmentation
    # -------------------------
    model = SwinUPerNetCBAM(
        backbone_name=cfg.BACKBONE,  
        num_classes=cfg.NUM_CLASSES,
        fpn_dim=cfg.FP_DIM,          
        ppm_out=cfg.PPM_OUT,
        pretrained=cfg.PRETRAINED    
    ).to(cfg.DEVICE)

    # -------------------------
    # Enable gradient checkpointing to reduce GPU memory consumption
    # Only applied if the backbone supports it
    # -------------------------
    try:
        if hasattr(model.backbone, 'gradient_checkpointing_enable'):
            model.backbone.gradient_checkpointing_enable()
            print("✅ Enabled backbone gradient checkpointing")
    except Exception as e:
        print("⚠️ Could not enable gradient checkpointing:", e)

    # -------------------------
    # Define loss function, optimizer, and LR scheduler
    # BCEWithLogitsLoss is used for binary segmentation tasks
    # ReduceLROnPlateau scheduler monitors validation F1 score
    # -------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        patience=cfg.SCHEDULER_PATIENCE, factor=cfg.SCHEDULER_FACTOR, verbose=True
    )

    # Automatic Mixed Precision (AMP) for faster training and lower memory usage
    scaler = GradScaler()
    accum_steps = cfg.ACCUMULATION_STEPS  # Gradient accumulation for effective larger batch size

    # -------------------------
    # Load checkpoint if available
    # Supports resuming training with optimizer and scheduler states
    # -------------------------
    model, optimizer, scheduler, start_epoch, best_f1 = load_checkpoint(
        model, optimizer, scheduler, cfg.CHECKPOINT_PATH, device
    )

    # -------------------------
    # Main training loop
    # -------------------------
    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        model.train()  # Enable training mode
        running_loss = 0.0
        running_dice = 0.0
        y_true_all, y_pred_all = [], []
        optimizer.zero_grad()

        # -------------------------
        # Iterate over training batches
        # -------------------------
        for step, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
            images, masks = images.to(device), masks.to(device)

            # Mixed precision context
            with amp.autocast(device_type='cuda', enabled=cfg.AMP_ENABLED):
                outputs = model(images)
                loss = criterion(outputs, masks) / accum_steps  # Scale loss for accumulation

            # Backpropagate scaled loss
            scaler.scale(loss).backward()

            # Optimizer step after gradient accumulation
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Accumulate training metrics
            running_loss += (loss.item() * accum_steps)
            running_dice += dice_coefficient(outputs.detach(), masks.detach(), from_logits=True)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            y_true_all.append(masks.cpu().numpy())
            y_pred_all.append(preds)

        # -------------------------
        # Compute average training metrics for the epoch
        # -------------------------
        avg_loss = running_loss / len(train_loader)
        avg_dice = running_dice / len(train_loader)
        y_true_all = np.concatenate(y_true_all).ravel()
        y_pred_all = np.concatenate(y_pred_all).ravel()
        train_f1, train_iou = compute_metrics(y_true_all, y_pred_all)

        # -------------------------
        # Validation loop
        # -------------------------
        model.eval()  # Disable dropout/batchnorm updates
        val_loss = 0.0
        val_dice = 0.0
        y_true_all, y_pred_all = [], []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                images, masks = images.to(device), masks.to(device)
                with amp.autocast(device_type='cuda', enabled=cfg.AMP_ENABLED):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, masks, from_logits=True)

                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                y_true_all.append(masks.cpu().numpy())
                y_pred_all.append(preds)

        # -------------------------
        # Compute average validation metrics for the epoch
        # -------------------------
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        y_true_all = np.concatenate(y_true_all).ravel()
        y_pred_all = np.concatenate(y_pred_all).ravel()
        val_f1, val_iou = compute_metrics(y_true_all, y_pred_all)

        # -------------------------
        # Log training and validation metrics
        # -------------------------
        print(f"\nEpoch {epoch}: "
              f"Train Loss={avg_loss:.4f}, Train Dice={avg_dice:.4f}, "
              f"Train F1={train_f1:.4f}, Train IoU={train_iou:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}, "
              f"Val F1={val_f1:.4f}, Val IoU={val_iou:.4f}")

        # -------------------------
        # Update learning rate based on validation F1
        # -------------------------
        scheduler.step(val_f1)

        # -------------------------
        # Save checkpoint if validation F1 improves
        # Stores model, optimizer, scheduler states and best F1
        # -------------------------
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(
                cfg.CHECKPOINT_PATH,
                epoch, model.state_dict(),
                optimizer.state_dict(),
                scheduler.state_dict(),
                best_f1
            )

# utils/metrics.py
import torch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score

import torch

def dice_coefficient(preds, targets, epsilon=1e-6, from_logits=True):
    """
    Dice Coefficient Metric

    Args:
        preds (torch.Tensor): Model output (logits or probability map).
        Shape: (N, C, H, W)
        targets (torch.Tensor): Ground truth masks (N, H, W) or one-hot encoded masks (N, C, H, W).
        epsilon (float): Small value to avoid division by zero.
        from_logits (bool): If True, applies sigmoid/softmax to raw logits.

    Returns:
        dice (float): Dice coefficient score in the range [0, 1].
    """

    if from_logits:
        if preds.shape[1] == 1:  # Binary segmentation
            preds = torch.sigmoid(preds)
        else:  # Multi-class segmentation
            preds = torch.softmax(preds, dim=1)

    # If targets are not one-hot, turn to one-hot
    if preds.shape != targets.shape:
        if preds.shape[1] == 1:  # Binary
            targets = targets.unsqueeze(1)  # (N, 1, H, W)
        else:  # Multi-class
            targets = torch.nn.functional.one_hot(
                targets.long(), num_classes=preds.shape[1]
            ).permute(0, 3, 1, 2).float()

    preds = (preds > 0.5).float()  # Threshold
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()


def compute_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    f1 = f1_score(y_true, y_pred, zero_division=0)
    iou = jaccard_score(y_true, y_pred, zero_division=0)
    return f1, iou

# utils/metrics.py

import torch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score


def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6, from_logits: bool = True) -> float:
    """
    Compute Dice Coefficient between predictions and ground-truth.

    Dice = (2 * |X ∩ Y|) / (|X| + |Y|)

    Parameters
    ----------
    preds : torch.Tensor
        Model outputs. Shape: (N, C, H, W).
        Can be raw logits (from_logits=True) or probabilities (from_logits=False).
    targets : torch.Tensor
        Ground truth masks.
        Shape: (N, H, W) for binary/multi-class or (N, C, H, W) if already one-hot.
    epsilon : float, optional
        Small constant to avoid division by zero, by default 1e-6.
    from_logits : bool, optional
        If True, applies sigmoid (binary) or softmax (multi-class) to logits.

    Returns
    -------
    float
        Mean Dice coefficient over batch, ∈ [0, 1].
    """

    # Apply sigmoid/softmax if logits
    if from_logits:
        if preds.shape[1] == 1:  # Binary segmentation
            preds = torch.sigmoid(preds)
        else:  # Multi-class
            preds = torch.softmax(preds, dim=1)

    # Align target shape to predictions
    if preds.shape != targets.shape:
        if preds.shape[1] == 1:  # Binary → add channel dim
            targets = targets.unsqueeze(1)
        else:  # Multi-class → one-hot encode
            targets = torch.nn.functional.one_hot(
                targets.long(), num_classes=preds.shape[1]
            ).permute(0, 3, 1, 2).float()

    # Threshold predictions
    preds = (preds > 0.5).float()
    targets = targets.float()

    # Dice formula
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice.mean().item()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute classical classification metrics (F1-score and IoU).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth mask (flattened or will be flattened).
    y_pred : np.ndarray
        Predicted mask (flattened or will be flattened).

    Returns
    -------
    tuple (f1, iou) : (float, float)
        - f1 : F1-score ∈ [0, 1]
        - iou : Jaccard Index (IoU) ∈ [0, 1]
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    f1 = f1_score(y_true, y_pred, zero_division=0)
    iou = jaccard_score(y_true, y_pred, zero_division=0)

    return f1, iou

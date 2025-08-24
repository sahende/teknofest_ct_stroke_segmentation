# utils/helpers.py

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across:
    - Python `random`
    - NumPy
    - PyTorch (CPU & CUDA)

    Also enforces deterministic behavior in cuDNN by:
    - `torch.backends.cudnn.deterministic = True`
    - `torch.backends.cudnn.benchmark = False`

    ⚠️ Note: Enabling determinism can slightly degrade performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_path=None, device='cuda'):
    """
    Load model (and optionally optimizer + scheduler) state from checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance to load weights into.
    optimizer : torch.optim.Optimizer, optional
        Optimizer to resume training state.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler to resume state.
    checkpoint_path : str, optional
        Path to checkpoint file (.pth or .pt).
    device : str, optional
        Target device to map checkpoint ('cuda' or 'cpu').

    Returns
    -------
    model : torch.nn.Module
        Model with loaded weights.
    optimizer : torch.optim.Optimizer or None
        Loaded optimizer state (if available).
    scheduler : torch.optim.lr_scheduler._LRScheduler or None
        Loaded scheduler state (if available).
    start_epoch : int
        Epoch to resume from (default=1 if no checkpoint).
    best_f1 : float
        Best validation F1-score stored in checkpoint (default=0).
    """
    start_epoch = 1
    best_f1 = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Case 1: Full training checkpoint dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        # Case 2: Raw state_dict only
        else:
            pretrained_dict = checkpoint

        # Filter out unmatched layers (e.g., different head dimension)
        model_dict = model.state_dict()
        filtered = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }
        print(f"ℹ️ {len(filtered)}/{len(model_dict)} layers matched and loaded from checkpoint.")

        # Update weights
        model_dict.update(filtered)
        model.load_state_dict(model_dict)

        # Restore optimizer & scheduler states if available
        if optimizer and scheduler and isinstance(checkpoint, dict):
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 1) + 1
                best_f1 = checkpoint.get('best_f1', 0)
                print(f"✅ Loaded optimizer/scheduler. Resuming from epoch {start_epoch-1}, best_f1={best_f1:.4f}")
            except Exception as e:
                print(f"⚠️ Could not load optimizer/scheduler: {e}")
        else:
            print("ℹ️ Checkpoint had no optimizer/scheduler states; training with fresh optimizer.")
    else:
        print("ℹ️ No checkpoint found. Training from scratch.")

    return model, optimizer, scheduler, start_epoch, best_f1


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_f1):
    """
    Save model, optimizer, scheduler and training metadata to checkpoint.

    Parameters
    ----------
    path : str
        Target file path to save checkpoint.
    epoch : int
        Current epoch number.
    model : torch.nn.Module
        Model whose weights will be saved.
    optimizer : torch.optim.Optimizer
        Optimizer state to save.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Scheduler state to save.
    best_f1 : float
        Best F1 score achieved so far (for resuming / early stopping).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model,
        'optimizer_state_dict': optimizer,
        'scheduler_state_dict': scheduler,
        'best_f1': best_f1
    }, path)
    print(f"✅ Checkpoint saved at {path}")

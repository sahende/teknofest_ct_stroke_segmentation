# utils/helpers.py
import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_path=None, device='cuda'):
    start_epoch = 1
    best_f1 = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            pretrained_dict = checkpoint

        model_dict = model.state_dict()
        filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        print(f"ℹ️ {len(filtered)}/{len(model_dict)} layers matched and loaded from checkpoint.")
        model_dict.update(filtered)
        model.load_state_dict(model_dict)

        if optimizer and scheduler and isinstance(checkpoint, dict):
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 1) + 1
                best_f1 = checkpoint.get('best_f1', 0)
                print(f"✅ Loaded optimizer/scheduler and resume from epoch {start_epoch-1}, best_f1={best_f1:.4f}")
            except Exception as e:
                print(f"⚠️ Could not load optimizer/scheduler: {e}")
        else:
            print("ℹ️ Checkpoint had no optimizer/scheduler states; starting fresh optimizer.")
    else:
        print("ℹ️ No checkpoint found. Training from scratch.")

    return model, optimizer, scheduler, start_epoch, best_f1

def save_checkpoint(path, epoch, model, optimizer, scheduler, best_f1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model,
        'optimizer_state_dict': optimizer,
        'scheduler_state_dict': scheduler,
        'best_f1': best_f1
    }, path)
    print(f"✅ Checkpoint saved at {path}")


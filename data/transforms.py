# data/transforms.py
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(norm_file="data_split/norm.json"):
    # normalization
    with open(norm_file, "r") as f:
        norm = json.load(f)

    mean = norm["mean"] if isinstance(norm.get("mean"), (list, tuple)) else [norm["mean"]] * 3
    std = norm["std"] if isinstance(norm.get("std"), (list, tuple)) else [norm["std"]] * 3

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=10, p=0.5),
        A.Affine(translate_percent={"x":0.05, "y":0.05}, rotate=0, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    return train_transform, val_transform

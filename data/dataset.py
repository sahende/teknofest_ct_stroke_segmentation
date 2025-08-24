# data/dataset.py

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset class for semantic segmentation tasks.

    This dataset assumes that for every input image there exists a 
    corresponding segmentation mask with the same filename in a 
    separate mask directory. The dataset handles image loading, 
    mask binarization, and optional augmentations.

    Attributes
    ----------
    image_dir : str
        Path to the directory containing input images.
    mask_dir : str
        Path to the directory containing corresponding ground truth masks.
    filenames : list[str]
        Sorted list of filenames in the image directory. Assumes mask directory
        contains masks with identical filenames.
    transform : albumentations.Compose, optional
        Transformation/augmentation pipeline applied to both image and mask
        (e.g., rotation, flipping, normalization).
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Initialize the dataset with image and mask directories.

        Parameters
        ----------
        image_dir : str
            Directory containing the input images.
        mask_dir : str
            Directory containing the segmentation masks.
        transform : albumentations.Compose, optional
            Data augmentation pipeline (applied consistently to both 
            images and masks).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Collect all image filenames and ensure a consistent ordering.
        # This guarantees that images and masks are matched correctly
        # when retrieved by index.
        self.filenames = sorted(os.listdir(image_dir))
        
        # Store optional transformation function/pipeline
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of images in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Retrieve the image and corresponding mask at a given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        image : torch.Tensor
            Transformed image tensor of shape (C, H, W).
        mask : torch.Tensor
            Corresponding binary mask tensor of shape (1, H, W).
        """
        # Construct full file paths for image and mask
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.filenames[idx])

        # Load the image and convert it to RGB (3-channel).
        # np.array() is used to obtain a NumPy array for augmentation support.
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load the mask and convert it to grayscale (single channel).
        # Values are binarized: pixels > 127 -> 1 (foreground), else 0 (background).
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            # Apply the same augmentation pipeline to both image and mask.
            # Albumentations ensures consistent spatial transformations.
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # already converted to tensor via pipeline
            mask = augmented['mask'].unsqueeze(0).float()  # add channel dim (1, H, W)
        else:
            # If no transform is provided:
            # - Convert the image into a PyTorch tensor using Albumentations' ToTensorV2.
            # - Convert mask to tensor manually, preserving binary values.
            image = ToTensorV2()(image=image)['image']
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

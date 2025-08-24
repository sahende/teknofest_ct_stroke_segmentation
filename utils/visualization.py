# utils/visualization.py

import matplotlib.pyplot as plt
import torch


def visualize_batch(dataloader, model=None, device="cuda", batch_index=0, max_samples=10):
    """
    Visualize a batch of images, ground truth masks, and optionally model predictions.

    This function is particularly useful for:
    - Inspecting dataset samples
    - Validating preprocessing/augmentation
    - Quick sanity check of model predictions during training or validation

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader yielding batches of (image, mask) pairs.
    model : torch.nn.Module, optional
        Trained model to generate predictions. If None, only images and masks are shown.
    device : str, default="cuda"
        Device for running model inference.
    batch_index : int, default=0
        Index of batch in DataLoader to visualize.
    max_samples : int, default=10
        Maximum number of samples to plot from the batch.

    Returns
    -------
    None
        Displays a matplotlib figure with images, masks, and optionally predictions.
    """
    # Save model training state to restore later
    model_was_training = False
    if model:
        model_was_training = model.training
        model.eval()  # Ensure deterministic predictions (disable dropout & batchnorm updates)

    # Retrieve batch from dataloader
    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            if idx == batch_index:
                images = images.to(device)
                if model:
                    outputs = model(images)
                    # Sigmoid for binary segmentation
                    preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
                else:
                    preds = None
                break

    # Convert tensors to numpy for matplotlib visualization
    images = images.cpu().numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
    masks = masks.cpu().numpy()
    num_samples = min(len(images), max_samples)

    # Create figure
    plt.figure(figsize=(12, num_samples * 3))
    for i in range(num_samples):
        # 1️⃣ Input image
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(images[i])
        plt.title("Image")
        plt.axis("off")

        # 2️⃣ Ground truth mask
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(masks[i][0], cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        # 3️⃣ Prediction (if available)
        if preds is not None:
            plt.subplot(num_samples, 3, i * 3 + 3)
            plt.imshow(preds[i][0], cmap="gray")
            plt.title("Prediction")
            plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Restore model's original training state (important if called during training loop)
    if model and model_was_training:
        model.train()

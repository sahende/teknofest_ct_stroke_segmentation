# utils/visualization.py
import matplotlib.pyplot as plt
import torch

def visualize_batch(dataloader, model=None, device="cuda", batch_index=0, max_samples=10):
    """
    Visualize a specific batch of images, masks, and optionally predictions.
    
    If a model is provided, the function will generate predictions for the 
    selected batch and display them alongside the input images and ground truth masks.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing images and masks.
        model (torch.nn.Module, optional): Trained model for generating predictions. 
                                           If None, only images and masks are shown.
        device (str): Device to run the model on ('cuda' or 'cpu').
        batch_index (int): Index of the batch to visualize.
        max_samples (int): Maximum number of samples from the batch to display.

    Returns:
        None. Displays a matplotlib figure with images, masks, and predictions.
    """
    # Store model's current training state to restore later
    model_was_training = False
    if model:
        model_was_training = model.training
        model.eval()  # Switch to evaluation mode for inference (disable dropout, batchnorm updates)

    # Retrieve the desired batch
    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            if idx == batch_index:
                images = images.to(device)
                if model:
                    outputs = model(images)
                    preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
                else:
                    preds = None
                break

    # Convert tensors to numpy arrays for visualization
    images = images.cpu().numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
    masks = masks.cpu().numpy()
    num_samples = min(len(images), max_samples)

    # Plot images, masks, and predictions
    plt.figure(figsize=(12, num_samples * 3))
    for i in range(num_samples):
        # Input image
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(images[i])
        plt.title("Image")
        plt.axis("off")

        # Ground truth mask
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(masks[i][0], cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        # Prediction (if available)
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

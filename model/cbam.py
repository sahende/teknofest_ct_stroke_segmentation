# model/cbam.py

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.
    
    This module emphasizes informative feature channels by computing
    attention weights across channels. It uses both average-pooling and 
    max-pooling descriptors, followed by a shared MLP (implemented as 1x1 convolutions).
    
    Paper: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
    """

    def __init__(self, in_planes, ratio=16):
        """
        Parameters
        ----------
        in_planes : int
            Number of input feature channels.
        ratio : int, optional (default=16)
            Reduction ratio used in the bottleneck fully connected layers.
        """
        super().__init__()

        # Global average pooling and max pooling (channel descriptors)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # shape: (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP: implemented as two 1x1 convolutions
        # Reduces channel dimension -> non-linear mapping -> restore dimension
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        # Final sigmoid to scale outputs to [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply average pooling and max pooling separately,
        # then pass each through the shared MLP
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        # Combine descriptors and compute channel attention map
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.
    
    This module emphasizes important spatial locations within feature maps.
    It aggregates channel information using average-pooling and max-pooling,
    concatenates them, and applies a convolution to produce a spatial attention map.
    """

    def __init__(self, kernel_size=7):
        """
        Parameters
        ----------
        kernel_size : int, optional (default=7)
            Kernel size for the convolution operation.
            A larger kernel increases receptive field for spatial attention.
        """
        super().__init__()

        # Convolution on 2-channel input (avg + max pooled maps)
        self.conv = nn.Conv2d(2, 1, kernel_size, 
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise average pooling (1 feature map)
        avg_out = torch.mean(x, dim=1, keepdim=True)

        # Channel-wise max pooling (1 feature map)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate along channel dimension -> (B, 2, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)

        # Convolution + sigmoid produces spatial attention mask
        return self.sigmoid(self.conv(combined))


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Sequentially applies:
    1. Channel Attention
    2. Spatial Attention
    
    The output is the input feature map refined in both channel and spatial dimensions.
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        """
        Parameters
        ----------
        in_planes : int
            Number of input channels.
        ratio : int, optional (default=16)
            Reduction ratio for channel attention.
        kernel_size : int, optional (default=7)
            Convolution kernel size for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # First refine channels
        out = x * self.channel_attention(x)

        # Then refine spatial locations
        out = out * self.spatial_attention(out)

        return out

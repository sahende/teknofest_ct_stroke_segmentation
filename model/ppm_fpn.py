# model/ppm_fpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import CBAMBlock


class PPM(nn.Module):
    """
    Pyramid Pooling Module (PSPNet style) with CBAM enhancement.

    This module aggregates contextual information at multiple spatial scales 
    using adaptive average pooling, followed by 1x1 convolutions, normalization, 
    and non-linear activation. Outputs from different scales are upsampled 
    and concatenated, then compressed via a bottleneck conv. Finally, 
    a CBAM block refines features through channel + spatial attention.

    Reference: PSPNet - "Pyramid Scene Parsing Network" (Zhao et al., CVPR 2017)
    """

    def __init__(self, in_ch, pool_sizes=(1, 2, 3, 6), 
                 norm_layer=nn.BatchNorm2d, out_ch=512):
        """
        Parameters
        ----------
        in_ch : int
            Number of input feature channels.
        pool_sizes : tuple of int, optional
            Pooling grid sizes for pyramid pooling (default=(1,2,3,6)).
        norm_layer : nn.Module, optional
            Normalization layer (default=BatchNorm2d).
        out_ch : int, optional
            Number of output channels for each pooled feature map and bottleneck.
        """
        super().__init__()

        # Multi-scale pooling + 1x1 conv + norm + ReLU for each scale
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),          # context pooling
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                norm_layer(out_ch),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])

        # Bottleneck: fuse original + pooled features
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch + len(pool_sizes) * out_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True)
        )

        # CBAM attention refinement
        self.cbam = CBAMBlock(out_ch)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # Start with original feature map
        priors = [x]

        # Pool at different scales, project, upsample, collect
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            priors.append(y)

        # Concatenate along channel dimension
        out = torch.cat(priors, dim=1)

        # Fuse with bottleneck conv
        out = self.bottleneck(out)

        # Refine with CBAM attention
        out = self.cbam(out)

        return out


class FPNFuse(nn.Module):
    """
    Feature Pyramid Network (FPN)-style feature fusion with CBAM refinement.

    Combines multi-level features from backbone (c1, c2, c3, c4).
    Each level is first projected to a common dimension using 1x1 convs,
    then fused in a top-down manner (higher-level features upsampled and added).
    Each fused level is further refined with a 3x3 smoothing conv and CBAM.

    Reference: FPN - "Feature Pyramid Networks for Object Detection"
               (Lin et al., CVPR 2017)
    """

    def __init__(self, in_channels_list, fpn_dim=256, norm_layer=nn.BatchNorm2d):
        """
        Parameters
        ----------
        in_channels_list : list[int]
            Number of channels for input feature maps [c1, c2, c3, c4].
        fpn_dim : int, optional
            Number of channels for each FPN output (default=256).
        norm_layer : nn.Module, optional
            Normalization layer (default=BatchNorm2d).
        """
        super().__init__()

        # Lateral 1x1 convs to project each feature map to fpn_dim
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, fpn_dim, 1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])

        # 3x3 convs to smooth after top-down fusion
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True)
            ) for _ in in_channels_list
        ])

        # CBAM applied to each pyramid level
        self.cbam_blocks = nn.ModuleList([CBAMBlock(fpn_dim) for _ in in_channels_list])

    def forward(self, c1, c2, c3, c4):
        # Project features to common dimension
        p4 = self.lateral_convs[3](c4)
        p3 = self.lateral_convs[2](c3)
        p2 = self.lateral_convs[1](c2)
        p1 = self.lateral_convs[0](c1)

        # Top-down fusion: propagate semantic info from high -> low levels
        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=False)
        p2 = p2 + F.interpolate(p3, size=p2.shape[2:], mode='bilinear', align_corners=False)
        p1 = p1 + F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)

        # Smooth each pyramid level
        p4 = self.smooth_convs[3](p4)
        p3 = self.smooth_convs[2](p3)
        p2 = self.smooth_convs[1](p2)
        p1 = self.smooth_convs[0](p1)

        # Apply CBAM refinement per level
        p4 = self.cbam_blocks[3](p4)
        p3 = self.cbam_blocks[2](p3)
        p2 = self.cbam_blocks[1](p2)
        p1 = self.cbam_blocks[0](p1)

        return p1, p2, p3, p4

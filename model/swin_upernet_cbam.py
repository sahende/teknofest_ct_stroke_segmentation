# model/swin_upernet_cbam.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from .ppm_fpn import PPM, FPNFuse


class SwinUPerNetCBAM(nn.Module):
    """
    Swin Transformer + UPerNet + CBAM for semantic segmentation.

    Architecture:
    1. Backbone: Swin Transformer (timm), providing multi-scale feature maps.
    2. PPM: Pyramid Pooling Module on the deepest Swin feature (context aggregation).
    3. FPN: Feature Pyramid Network fusion of (c1, c2, c3, PPM(c4)),
       with CBAM refinement at each scale.
    4. Decoder: Concatenates all pyramid outputs, fuses with 3x3 conv,
       and projects to segmentation logits.
    5. Output: Segmentation mask (upsampled to input size).

    Reference:
      - UPerNet: "Unified Perceptual Parsing for Scene Understanding" (Xiao et al., ECCV 2018)
      - Swin Transformer: "Hierarchical Vision Transformer using Shifted Windows" (Liu et al., ICCV 2021)
      - CBAM: "Convolutional Block Attention Module" (Woo et al., ECCV 2018)
    """

    def __init__(self, 
                 backbone_name: str = 'swin_tiny_patch4_window7_224',
                 num_classes: int = 1,
                 fpn_dim: int = 256,
                 ppm_out: int = 512,
                 pretrained: bool = True):
        """
        Parameters
        ----------
        backbone_name : str, optional
            Name of Swin Transformer backbone (from timm).
        num_classes : int, optional
            Number of output segmentation classes (default=1).
        fpn_dim : int, optional
            Feature dimension for FPN outputs (default=256).
        ppm_out : int, optional
            Output channels from PPM (default=512).
        pretrained : bool, optional
            If True, load pretrained Swin weights.
        """
        super().__init__()

        # 1. Swin Transformer backbone
        # features_only=True → returns multi-scale features [c1, c2, c3, c4]
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True)
        swin_channels = self.backbone.feature_info.channels()

        # 2. Pyramid Pooling Module on deepest stage (c4)
        self.ppm = PPM(in_ch=swin_channels[-1], pool_sizes=(1, 2, 3, 6), out_ch=ppm_out)

        # 3. Feature Pyramid Fusion with CBAM
        # fuse [c1, c2, c3, ppm(c4)]
        self.fpn = FPNFuse(
            in_channels_list=(swin_channels[0], swin_channels[1], swin_channels[2], ppm_out),
            fpn_dim=fpn_dim
        )

        # 4. Decoder head: fuse all pyramid features
        self.final_conv = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, fpn_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # 5. Classifier → per-pixel logits
        self.classifier = nn.Conv2d(fpn_dim, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Segmentation logits of shape (B, num_classes, H, W).
        """
        # 1. Extract multi-scale features from Swin backbone
        # Each feature is [B, H, W, C], so permute to [B, C, H, W]
        features = [f.permute(0, 3, 1, 2).contiguous() for f in self.backbone(x)]
        c1, c2, c3, c4 = features

        # 2. Apply PPM on c4 for global context
        p4_ppm = self.ppm(c4)

        # 3. Fuse multi-scale features with FPN + CBAM
        p1, p2, p3, p4 = self.fpn(c1, c2, c3, p4_ppm)

        # 4. Upsample all pyramid levels to match p1 resolution
        p2_up = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False)

        # 5. Concatenate and fuse with conv
        concat = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)
        fused = self.final_conv(concat)

        # 6. Classifier → segmentation logits
        out = self.classifier(fused)

        # 7. Upsample to original input resolution
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

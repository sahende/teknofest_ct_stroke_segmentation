# model/swin_upernet_cbam.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from .ppm_fpn import PPM, FPNFuse

class SwinUPerNetCBAM(nn.Module):
    def __init__(self, backbone_name='swin_tiny_patch4_window7_224', num_classes=1, fpn_dim=256, ppm_out=512, pretrained=True):
        super().__init__()
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True)
        swin_channels = self.backbone.feature_info.channels()
        self.ppm = PPM(in_ch=swin_channels[-1], pool_sizes=(1,2,3,6), out_ch=ppm_out)
        self.fpn = FPNFuse(in_channels_list=(swin_channels[0], swin_channels[1], swin_channels[2], ppm_out), fpn_dim=fpn_dim)

        self.final_conv = nn.Sequential(
            nn.Conv2d(fpn_dim*4, fpn_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Conv2d(fpn_dim, num_classes, 1)

    def forward(self, x):
        features = [f.permute(0,3,1,2).contiguous() for f in self.backbone(x)]
        c1, c2, c3, c4 = features
        p4_ppm = self.ppm(c4)
        p1, p2, p3, p4 = self.fpn(c1, c2, c3, p4_ppm)

        p2_up = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False)

        concat = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)
        fused = self.final_conv(concat)
        out = self.classifier(fused)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

# model/ppm_fpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import CBAMBlock

class PPM(nn.Module):
    def __init__(self, in_ch, pool_sizes=(1,2,3,6), norm_layer=nn.BatchNorm2d, out_ch=512):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                norm_layer(out_ch),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch + len(pool_sizes)*out_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAMBlock(out_ch)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        priors = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(H,W), mode='bilinear', align_corners=False)
            priors.append(y)
        out = torch.cat(priors, dim=1)
        out = self.bottleneck(out)
        out = self.cbam(out)
        return out

class FPNFuse(nn.Module):
    def __init__(self, in_channels_list, fpn_dim=256, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, fpn_dim, 1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True)
            ) for _ in in_channels_list
        ])
        self.cbam_blocks = nn.ModuleList([CBAMBlock(fpn_dim) for _ in in_channels_list])

    def forward(self, c1, c2, c3, c4):
        p4 = self.lateral_convs[3](c4)
        p3 = self.lateral_convs[2](c3)
        p2 = self.lateral_convs[1](c2)
        p1 = self.lateral_convs[0](c1)

        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=False)
        p2 = p2 + F.interpolate(p3, size=p2.shape[2:], mode='bilinear', align_corners=False)
        p1 = p1 + F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)

        p4 = self.smooth_convs[3](p4)
        p3 = self.smooth_convs[2](p3)
        p2 = self.smooth_convs[1](p2)
        p1 = self.smooth_convs[0](p1)

        p4 = self.cbam_blocks[3](p4)
        p3 = self.cbam_blocks[2](p3)
        p2 = self.cbam_blocks[1](p2)
        p1 = self.cbam_blocks[0](p1)

        return p1, p2, p3, p4

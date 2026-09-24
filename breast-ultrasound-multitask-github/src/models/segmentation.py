"""Segmentation, fusion, and classification models."""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class ASPP(nn.Module):
    """Five parallel dilated-convolution branches summed together."""
    def __init__(self, in_ch=3, branch_ch=16):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, branch_ch, 1, bias=False), nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(in_ch, branch_ch, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(in_ch, branch_ch, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True))
        self.b4 = nn.Sequential(nn.Conv2d(in_ch, branch_ch, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True))
        self.b5 = nn.Sequential(nn.Conv2d(in_ch, branch_ch, 3, padding=24, dilation=24, bias=False), nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.b1(x) + self.b2(x) + self.b3(x) + self.b4(x) + self.b5(x)

class ANetSegmenter(nn.Module):
    """ASPP + U-Net-style encoder-decoder used for lesion segmentation."""
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.aspp = ASPP(in_ch=in_ch, branch_ch=16)
        self.stem = DoubleConv(16, base_ch)
        self.pool = nn.AvgPool2d(2)
        self.enc1 = DoubleConv(base_ch, base_ch * 2)
        self.enc2 = DoubleConv(base_ch * 2, base_ch * 4)
        self.enc3 = DoubleConv(base_ch * 4, base_ch * 8)
        self.enc4 = DoubleConv(base_ch * 8, base_ch * 16)
        self.bottleneck = DoubleConv(base_ch * 16, base_ch * 32)
        up = lambda: nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up4, self.dec4 = up(), DoubleConv(base_ch * 32 + base_ch * 16, base_ch * 16)
        self.up3, self.dec3 = up(), DoubleConv(base_ch * 16 + base_ch * 8, base_ch * 8)
        self.up2, self.dec2 = up(), DoubleConv(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.up1, self.dec1 = up(), DoubleConv(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up0, self.dec0 = up(), DoubleConv(base_ch * 2 + base_ch, base_ch)
        self.final_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        feat = self.aspp(x)
        s0 = self.stem(feat)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), s0], dim=1))
        return self.final_conv(d0)

class FusionAdapter(nn.Module):
    """Concatenate RGB image with a 3-channel copy of the predicted mask."""
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
    def forward(self, image, mask_prob):
        mask3 = mask_prob.repeat(1, 3, 1, 1)
        return self.block(torch.cat([image, mask3], dim=1))

class ResNet50Classifier(nn.Module):
    """ImageNet-initialized ResNet50 with a custom 3-class classification head."""
    def __init__(self, num_classes=3):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.gap(x))

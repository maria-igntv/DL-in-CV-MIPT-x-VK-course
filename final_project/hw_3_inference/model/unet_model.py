"""U-Net with MobileNetV2 encoder for image enhancement.

Lightweight encoder-decoder for comparison against 3D LUT.
~2.3M parameters with pretrained encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class UNetEnhancer(nn.Module):
    """U-Net with MobileNetV2 encoder for photo enhancement.

    Input:  (B, 3, H, W)  RGB in [0, 1]
    Output: (B, 3, H, W)  RGB in [0, 1]
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        # MobileNetV2 feature blocks with output channels and stride
        self.enc1 = mobilenet.features[:2]  # 16 ch, /2
        self.enc2 = mobilenet.features[2:4]  # 24 ch, /4
        self.enc3 = mobilenet.features[4:7]  # 32 ch, /8
        self.enc4 = mobilenet.features[7:14]  # 96 ch, /16
        self.enc5 = mobilenet.features[14:]  # 1280 ch, /32

        self.bottleneck = nn.Sequential(
            nn.Conv2d(1280, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up5 = nn.ConvTranspose2d(256, 96, 2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(192, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.ConvTranspose2d(96, 32, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(48, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(11, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        # ImageNet normalization buffers
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x):
        return (x - self.mean) / self.std

    @staticmethod
    def _pad_cat(up, skip):
        dh = skip.size(2) - up.size(2)
        dw = skip.size(3) - up.size(3)
        up = F.pad(
            up,
            [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2],
        )
        return torch.cat([up, skip], dim=1)

    def forward(self, x):
        # Encoder (input is [0,1], normalize for pretrained encoder)
        inp_norm = self._normalize(x)
        e1 = self.enc1(inp_norm)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Bottleneck
        b = self.bottleneck(e5)

        # Decoder with skip connections
        d5 = self._pad_cat(self.up5(b), e4)
        d5 = self.dec5(d5)

        d4 = self._pad_cat(self.up4(d5), e3)
        d4 = self.dec4(d4)

        d3 = self._pad_cat(self.up3(d4), e2)
        d3 = self.dec3(d3)

        d2 = self._pad_cat(self.up2(d3), e1)
        d2 = self.dec2(d2)

        d1 = self._pad_cat(self.up1(d2), x)  # skip from original input
        d1 = self.dec1(d1)

        return d1

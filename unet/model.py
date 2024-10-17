# model.py

import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.bottleneck = self.conv_block(256, 512)

        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = self.conv_block(128, 64)

        self.conv_final = nn.Conv3d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder path
        u3 = self.upconv3(b)
        d3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(d3)

        u2 = self.upconv2(d3)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(d2)

        u1 = self.upconv1(d2)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output layer
        out = self.conv_final(d1)
        # Reduce depth dimension
        out = torch.mean(out, dim=2)  # Shape: (batch_size, out_channels, height, width)
        return out

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.encoder1 = conv_block(in_channels, 64)   # Input to 64 channels
        self.encoder2 = conv_block(64, 128)            # 64 to 128 channels
        self.encoder3 = conv_block(128, 256)           # 128 to 256 channels
        self.encoder4 = conv_block(256, 512)           # 256 to 512 channels
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.decoder1 = conv_block(512, 256)           # 512 to 256 channels
        self.decoder2 = conv_block(256, 128)           # 256 to 128 channels
        self.decoder3 = conv_block(128, 64)            # 128 to 64 channels
        self.decoder4 = conv_block(64, out_channels)   # 64 to out_channels

        # Transpose Convolutions
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # Upsample 512 to 256
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # Upsample 256 to 128
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # Upsample 128 to 64

    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)                # shape: (batch_size, 64, H, W)
        e2 = self.encoder2(self.pool(e1))    # shape: (batch_size, 128, H/2, W/2)
        e3 = self.encoder3(self.pool(e2))    # shape: (batch_size, 256, H/4, W/4)
        e4 = self.encoder4(self.pool(e3))    # shape: (batch_size, 512, H/8, W/8)

        # Decoder path
        d1 = self.upconv1(e4)                # shape: (batch_size, 256, H/4, W/4)
        d1 = d1 + e3                          # Adding skip connection from encoder3 (256 channels)
        d1 = self.decoder1(d1)                # shape: (batch_size, 256, H/4, W/4)

        d2 = self.upconv2(d1)                 # shape: (batch_size, 128, H/2, W/2)
        d2 = d2 + e2                          # Adding skip connection from encoder2 (128 channels)
        d2 = self.decoder2(d2)                # shape: (batch_size, 128, H/2, W/2)

        d3 = self.upconv3(d2)                 # shape: (batch_size, 64, H, W)
        d3 = d3 + e1                          # Adding skip connection from encoder1 (64 channels)
        d3 = self.decoder3(d3)                # shape: (batch_size, 64, H, W)

        out = self.decoder4(d3)               # shape: (batch_size, out_channels, H, W)
        return out

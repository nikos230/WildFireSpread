import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU()
            )

        # encoder
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 128)
        self.encoder4 = conv_block(256, 512)
        
        # pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # decoder
        self.decoder1 = conv_block(512, 256)
        self.decoder2 = conv_block(256, 128)
        self.decoder3 = conv_block(128, 64)
        self.decoder4 = conv_block(64, out_channels)

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)



    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        d1 = self.upconv1(e4) + e3
        d1 = self.decoder1(d1)
        d2 = self.upconv2(d1) + e2
        d2 = self.decoder2(d2)
        d3 = self.upconv3(d2) + e1
        d3 = self.decoder3(d3)

        out = self.decoder4(d3)
        return out
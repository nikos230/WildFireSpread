import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, kernel_size, pool_size, 
                 use_batchnorm=True, final_activation=None):
        """
        Parameters:
        - in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        - out_channels: Number of output channels (e.g., segmentation classes)
        - num_filters: List of filters for each level in the encoder and decoder
        - kernel_size: Convolution kernel size (default is 3x3x3)
        - pool_size: Pooling size for downsampling (default is (1, 2, 2))
        - use_batchnorm: Whether to use batch normalization in the conv blocks (default is True)
        - final_activation: Activation function to apply at the final output (e.g., nn.Sigmoid() or nn.Softmax(dim=1))
        """
        super(UNet3D, self).__init__()
        
        # Store parameters for flexibility
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.use_batchnorm = use_batchnorm
        self.final_activation = final_activation

        # Encoder
        self.encoders = nn.ModuleList([self.conv_block(in_channels, num_filters[0])])
        self.pools = nn.ModuleList([nn.MaxPool3d(kernel_size=self.pool_size)])
        for i in range(1, len(num_filters)):
            self.encoders.append(self.conv_block(num_filters[i-1], num_filters[i]))
            self.pools.append(nn.MaxPool3d(kernel_size=self.pool_size))

        # Bottleneck
        self.bottleneck = self.conv_block(num_filters[-1], num_filters[-1] * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(1, len(num_filters))):
            self.upconvs.append(nn.ConvTranspose3d(num_filters[i]*2, num_filters[i], kernel_size=self.pool_size, stride=self.pool_size))
            self.decoders.append(self.conv_block(num_filters[i]*2, num_filters[i]))
        
        self.upconvs.append(nn.ConvTranspose3d(num_filters[0]*2, num_filters[0], kernel_size=self.pool_size, stride=self.pool_size))
        self.decoders.append(self.conv_block(num_filters[0]*2, num_filters[0]))

        # Final convolution
        self.conv_final = nn.Conv3d(num_filters[0], out_channels, kernel_size=1)



    def conv_block(self, in_channels, out_channels):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(inplace=True)
        ]
        if self.use_batchnorm:
            layers.insert(1, nn.BatchNorm3d(out_channels))
        
        layers += [
            nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(inplace=True)
        ]
        if self.use_batchnorm:
            layers.insert(3, nn.BatchNorm3d(out_channels))
        
        return nn.Sequential(*layers)



    def forward(self, x):
        # Encoder path
        enc_outputs = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            enc_outputs.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for upconv, decoder, enc_output in zip(self.upconvs, self.decoders, reversed(enc_outputs)):
            x = upconv(x)
            x = torch.cat([x, enc_output], dim=1)  # Skip connection
            x = decoder(x)

        # Final convolution
        x = self.conv_final(x)

        # Reduce depth dimension (optional, adjust if necessary)
        x = torch.mean(x, dim=2)  # Shape: (batch_size, out_channels, height, width)

        # Apply final activation (if any)
        if self.final_activation:
            x = self.final_activation(x)

        return x

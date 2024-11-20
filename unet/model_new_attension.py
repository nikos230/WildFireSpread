import torch
import torch.nn as nn

class AttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # Element-wise multiplication for attention

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, kernel_size, pool_size, 
                 use_batchnorm=True, final_activation=None, dropout_rate=0.3, num_layers=1):
        super(UNet3D, self).__init__()
        
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.use_batchnorm = use_batchnorm
        self.final_activation = final_activation
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        
        # Encoder
        self.encoders = nn.ModuleList([self.conv_block(in_channels, num_filters[0], dropout_rate=self.dropout_rate, num_layers=self.num_layers)])
        self.pools = nn.ModuleList([nn.MaxPool3d(kernel_size=self.pool_size)])
        for i in range(1, len(num_filters)):
            self.encoders.append(self.conv_block(num_filters[i-1], num_filters[i], dropout_rate=self.dropout_rate, num_layers=self.num_layers))
            self.pools.append(nn.MaxPool3d(kernel_size=self.pool_size))

        # Bottleneck
        self.bottleneck = self.conv_block(num_filters[-1], num_filters[-1] * 2, dropout_rate=self.dropout_rate, num_layers=self.num_layers)

        # Decoder with Attention
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attentions = nn.ModuleList()  # Attention blocks for each skip connection
        for i in reversed(range(1, len(num_filters))):
            self.upconvs.append(nn.ConvTranspose3d(num_filters[i]*2, num_filters[i], kernel_size=self.pool_size, stride=self.pool_size))
            self.attentions.append(AttentionBlock3D(F_g=num_filters[i], F_l=num_filters[i], F_int=num_filters[i] // 2))
            self.decoders.append(self.conv_block(num_filters[i]*2, num_filters[i], dropout_rate=self.dropout_rate, num_layers=self.num_layers))
        
        self.upconvs.append(nn.ConvTranspose3d(num_filters[0]*2, num_filters[0], kernel_size=self.pool_size, stride=self.pool_size))
        self.attentions.append(AttentionBlock3D(F_g=num_filters[0], F_l=num_filters[0], F_int=num_filters[0] // 2))
        self.decoders.append(self.conv_block(num_filters[0]*2, num_filters[0], dropout_rate=self.dropout_rate, num_layers=self.num_layers))

        # Final convolution
        self.conv_final = nn.Conv3d(num_filters[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_rate=0.3, num_layers=1):
        layers = []
        
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel_size, padding=(1, 1, 1)))
        if self.use_batchnorm:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, padding=(1, 1, 1)))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Dropout3d(p=dropout_rate))
        
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

        # Decoder path with attention
        for upconv, attention, decoder, enc_output in zip(self.upconvs, self.attentions, self.decoders, reversed(enc_outputs)):
            x = upconv(x)
            enc_output = attention(g=x, x=enc_output)  # Apply attention
            x = torch.cat([x, enc_output], dim=1)  # Skip connection
            x = decoder(x)

        # Final convolution
        x = self.conv_final(x)

        # Reduce depth dimension (optional, adjust if necessary)
        x = torch.mean(x, dim=2)

        # Apply final activation (if any)
        if self.final_activation:
            x = self.final_activation(x)

        return x

# Example model instantiation
model = UNet3D(in_channels=28, out_channels=1, num_filters=[64, 128, 256], 
               kernel_size=3, pool_size=(1, 2, 2), use_batchnorm=True, 
               final_activation=nn.Sigmoid(), dropout_rate=0.3)

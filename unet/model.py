import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, kernel_size, pool_size, 
                 use_batchnorm=True, final_activation=None, dropout_rate=0.3, num_layers=1):
        super(UNet2D, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.use_batchnorm = use_batchnorm
        self.final_activation = final_activation
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # Encoder
        self.encoders = nn.ModuleList([self.conv_block(in_channels, num_filters[0], dropout_rate=self.dropout_rate, num_layers=self.num_layers)])
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=self.pool_size)])
        for i in range(1, len(num_filters)):
            self.encoders.append(self.conv_block(num_filters[i-1], num_filters[i], dropout_rate=self.dropout_rate, num_layers=self.num_layers))
            self.pools.append(nn.MaxPool2d(kernel_size=self.pool_size))

        # Bottleneck
        self.bottleneck = self.conv_block(num_filters[-1], num_filters[-1] * 2, dropout_rate=self.dropout_rate)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in reversed(range(1, len(num_filters))):
            self.upconvs.append(nn.ConvTranspose2d(num_filters[i]*2, num_filters[i], kernel_size=self.pool_size, stride=self.pool_size))
            self.decoders.append(self.conv_block(num_filters[i]*2, num_filters[i], dropout_rate=self.dropout_rate, num_layers=self.num_layers))

        self.upconvs.append(nn.ConvTranspose2d(num_filters[0]*2, num_filters[0], kernel_size=self.pool_size, stride=self.pool_size))
        self.decoders.append(self.conv_block(num_filters[0]*2, num_filters[0], dropout_rate=self.dropout_rate, num_layers=self.num_layers))

        # Final convolution
        self.conv_final = nn.Conv2d(num_filters[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_rate=0.3, num_layers=2):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=1))
        if self.use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=1))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())

        layers.append(nn.Dropout2d(p=dropout_rate))
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

        # Decoder path without attention in skip connections
        for upconv, decoder, enc_output in zip(self.upconvs, self.decoders, reversed(enc_outputs)):
            x = upconv(x)
            x = torch.cat([x, enc_output], dim=1)  # Skip connection without attention
            x = decoder(x)

        # Final convolution
        x = self.conv_final(x)

        # Apply final activation (if any)
        if self.final_activation:
            x = self.final_activation(x)

        return x        




class UNet2D_GradCam(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_filters, kernel_size, pool_size, 
                 use_batchnorm=True, final_activation=None, dropout_rate=0.3, num_layers=1):
        super(UNet2D_GradCam, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.use_batchnorm = use_batchnorm
        self.final_activation = final_activation
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # Encoder
        self.encoders = nn.ModuleList([self.conv_block(in_channels, num_filters[0], dropout_rate=self.dropout_rate, num_layers=self.num_layers)])
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=self.pool_size)])
        for i in range(1, len(num_filters)):
            self.encoders.append(self.conv_block(num_filters[i-1], num_filters[i], dropout_rate=self.dropout_rate, num_layers=self.num_layers))
            self.pools.append(nn.MaxPool2d(kernel_size=self.pool_size))

        # Bottleneck
        self.bottleneck = self.conv_block(num_filters[-1], num_filters[-1] * 2, dropout_rate=self.dropout_rate)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in reversed(range(1, len(num_filters))):
            self.upconvs.append(nn.ConvTranspose2d(num_filters[i]*2, num_filters[i], kernel_size=self.pool_size, stride=self.pool_size))
            self.decoders.append(self.conv_block(num_filters[i]*2, num_filters[i], dropout_rate=self.dropout_rate, num_layers=self.num_layers))

        self.upconvs.append(nn.ConvTranspose2d(num_filters[0]*2, num_filters[0], kernel_size=self.pool_size, stride=self.pool_size))
        self.decoders.append(self.conv_block(num_filters[0]*2, num_filters[0], dropout_rate=self.dropout_rate, num_layers=self.num_layers))

        # Final convolution
        self.conv_final = nn.Conv2d(num_filters[0], out_channels, kernel_size=1)

        # Grad-CAM specific
        self.gradients = None
        self.activations = None
        self.target_layer_name = "bottleneck"  # Default target layer

    def conv_block(self, in_channels, out_channels, dropout_rate=0.3, num_layers=2):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=1))
        if self.use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=1))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Dropout2d(p=dropout_rate))
        return nn.Sequential(*layers)

    def save_gradients(self, grad):
        """Hook to save gradients during backward pass."""
        self.gradients = grad

    def forward(self, x):
        # Encoder path
        enc_outputs = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            enc_outputs.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        if self.target_layer_name == "bottleneck":
            x.requires_grad_(True)
            x.register_hook(self.save_gradients)
            self.activations = x

        # Decoder path
        for upconv, decoder, enc_output in zip(self.upconvs, self.decoders, reversed(enc_outputs)):
            x = upconv(x)
            x = torch.cat([x, enc_output], dim=1)  # Skip connection
            x = decoder(x)

        # Final convolution
        x = self.conv_final(x)

        # Apply final activation (if any)
        if self.final_activation:
            x = self.final_activation(x)

        return x

    def get_grad_cam(self, inputs):
        """Compute Grad-CAM heatmap for each input channel."""
        self.eval()  # Ensure model is in evaluation mode
        inputs.requires_grad = True  # Enable gradient computation on inputs
        
        outputs = self(inputs)  # Forward pass

        # Create one-hot tensor for the target class (assume target is class index 0 for simplicity)
        target_class_idx = torch.argmax(outputs, dim=1)
        one_hot_output = torch.zeros_like(outputs)
        one_hot_output[:, target_class_idx] = 1  # Focus on target class

        # Perform backward pass to get gradients
        self.zero_grad()  # Zero out previous gradients
        outputs.backward(gradient=one_hot_output, retain_graph=True)

        # Verify gradients and activations
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations are not available. Did you call backward()?")

        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight activations by the pooled gradients
        activations = self.activations.squeeze(0)
        for i in range(activations.shape[0]):  # Iterate over feature map channels
            activations[i, :, :] *= pooled_gradients[i]

        # Compute heatmap as mean across feature map channels
        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()

        # Apply ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1

        return heatmap




class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class ViTSegmentation2(nn.Module):
    def __init__(self, in_channels, num_classes, image_size=64, patch_size=1, embed_dim=256, num_heads=32, 
                 depth=2, mlp_dim=256, dropout_rate=0.3):
        super(ViTSegmentation2, self).__init__()

        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding layer
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            Permute((0, 2, 1))  # Shape: (B, num_patches, embed_dim)
        )

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Decoder for patch size 2x2
        self.decoder = nn.Sequential(
            # Reduce the number of channels
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=5, stride=1, padding=2),
            nn.GELU(),

            # Upsample from 32x32 to 64x64
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1),
            #nn.GELU(),

            # Final convolution to adjust output to num_classes
            nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1, stride=1, padding=0)
        )

        # decoder for patch size 8x8
        # self.decoder = nn.Sequential(
        #     # Adjust channels first
        #     nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, stride=1, padding=1),
        #     nn.GELU(),

        #     # Upsample from 8x8 to 16x16
        #     nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1),
        #     nn.GELU(),

        #     # Upsample from 16x16 to 32x32
        #     nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=4, stride=2, padding=1),
        #     nn.GELU(),

        #     # Upsample from 32x32 to 64x64
        #     nn.ConvTranspose2d(embed_dim // 8, num_classes, kernel_size=4, stride=2, padding=1)
        # )





        # decoder patch size 1x1
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=7, stride=1, padding=3),
        #     #nn.LeakyReLU(negative_slope=1, inplace=True),
        #     nn.GELU(),
        #     #nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2, padding=0),  # Upsample by 2x
        #     #nn.LeakyReLU(negative_slope=1, inplace=True),
        #     nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1, stride=1, padding=0)  # Final output matches target size
        # )





        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        batch_size = x.size(0)

        # Patch embedding
        patches = self.patch_embedding(x) + self.positional_encoding

        # Transformer encoder
        transformer_output = self.transformer_encoder(patches)  # Shape: (B, num_patches, embed_dim)

        # Reshape transformer output to feature map
        H = W = self.image_size // self.patch_size
        feature_map = transformer_output.permute(0, 2, 1).view(batch_size, self.embed_dim, H, W)
        #print("Feature map shape:", feature_map.shape)

        # Decoder: Upsample to input resolution
        seg_map = self.decoder(feature_map)
        #print("Segmentation map shape:", seg_map.shape)

        return seg_map




class UNet2D_struct_(nn.Module):
    def __init__(self, in_channels, out_channels, use_previous_output=True):
        super(UNet2D_struct_, self).__init__()

        if use_previous_output:
            in_channels += out_channels

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 4))

        #self.enc2 = self.conv_block(32, 64)
        #self.pool2 = nn.MaxPool2d(kernel_size=2)

        # self.enc3 = self.conv_block(128, 256)
        # self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = self.conv_block(32, 64)


        # Decoder
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(4, 4))
        self.dec3 = self.conv_block(64, 32)

        #self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        #self.dec2 = self.conv_block(64, 32)

        # self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.dec1 = self.conv_block(128, 64)

        # Final output layer
        self.conv_final = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.GELU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.GELU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.GELU(),
            nn.Dropout2d(p=0.4)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        #enc2 = self.enc2(pool1)
        #pool2 = self.pool2(enc2)

        # e3 = self.enc3(p2)
        # p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(pool1)

  

        # Decoder path
        upconv3 = self.upconv3(b)
        dec3 = torch.cat([upconv3, enc1], dim=1)
        dec3 = self.dec3(dec3)

        #upconv2 = self.upconv2(dec3)
        #dec2 = torch.cat([upconv2, enc1], dim=1)
        #dec2 = self.dec2(dec2)

        # u1 = self.upconv1(d2)
        # d1 = torch.cat([u1, e1], dim=1)
        # d1 = self.dec1(d1)

        # Output layer
        out = self.conv_final(dec3)
        return out



class UNet2D_struct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet2D_struct, self).__init__()


        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 4))

        #self.enc2 = self.conv_block(32, 64)
        #self.pool2 = nn.MaxPool2d(kernel_size=2)

        # self.enc3 = self.conv_block(128, 256)
        # self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = self.conv_block(32, 64)

        # LSTM Layer (assuming you still want to process it using an LSTM)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)


        # Decoder
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(4, 4))
        self.dec3 = self.conv_block(64, 32)

        #self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        #self.dec2 = self.conv_block(64, 32)

        # self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.dec1 = self.conv_block(128, 64)

        # Final output layer
        self.conv_final = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(p=0.4)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        #enc2 = self.enc2(pool1)
        #pool2 = self.pool2(enc2)

        # e3 = self.enc3(p2)
        # p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(pool1)

        # LSTM processing (reshape the input to fit LSTM format: [batch, seq, feature])
        batch_size, channels, height, width = b.size()
        b = b.view(batch_size, channels, -1).transpose(1, 2)  # Reshape to [batch_size, seq_len, feature_size] for LSTM

        # Pass through LSTM layer
        lstm_out, (hn, cn) = self.lstm(b)  # Output of LSTM is (batch_size, seq_len, hidden_size)

        # Correcting the output dimensions: ensure that the channels are 128 (as expected by the upconv3 layer)
        lstm_out = lstm_out[:, -1, :].unsqueeze(2).unsqueeze(3)  # Take the last time-step, add spatial dims
        lstm_out = lstm_out.expand(-1, -1, height, width)  # Expand to match spatial dimensions

        # Decoder path
        upconv3 = self.upconv3(lstm_out)
        dec3 = torch.cat([upconv3, enc1], dim=1)
        dec3 = self.dec3(dec3)

        #upconv2 = self.upconv2(dec3)
        #dec2 = torch.cat([upconv2, enc1], dim=1)
        #dec2 = self.dec2(dec2)

        # u1 = self.upconv1(d2)
        # d1 = torch.cat([u1, e1], dim=1)
        # d1 = self.dec1(d1)

        # Output layer
        out = self.conv_final(dec3)
        return out








 
       
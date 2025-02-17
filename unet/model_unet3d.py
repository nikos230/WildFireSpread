import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, kernel_size, pool_size, 
                 use_batchnorm=True, final_activation=None, dropout_rate=0.3, num_layers=1):
        """
        Parameters:
        - in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        - out_channels: Number of output channels (e.g., segmentation classes)
        - num_filters: List of filters for each level in the encoder and decoder
        - kernel_size: Convolution kernel size (default is 3x3x3)
        - pool_size: Pooling size for downsampling (default is (1, 2, 2))
        - use_batchnorm: Whether to use batch normalization in the conv blocks (default is True)
        - final_activation: Activation function to apply at the final output (e.g., nn.Sigmoid() or nn.Softmax(dim=1))
        - dropout_rate: Dropout rate applied after GELU activations to prevent overfitting
        """
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

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(1, len(num_filters))):
            self.upconvs.append(nn.ConvTranspose3d(num_filters[i]*2, num_filters[i], kernel_size=self.pool_size, stride=self.pool_size))
            self.decoders.append(self.conv_block(num_filters[i]*2, num_filters[i], dropout_rate=self.dropout_rate, num_layers=self.num_layers))
        
        self.upconvs.append(nn.ConvTranspose3d(num_filters[0]*2, num_filters[0], kernel_size=self.pool_size, stride=self.pool_size))
        self.decoders.append(self.conv_block(num_filters[0]*2, num_filters[0], dropout_rate=self.dropout_rate, num_layers=self.num_layers))

        # Final convolution
        self.conv_final = nn.Conv3d(num_filters[0], out_channels, kernel_size=1)

        # grad-cam
        self.gradients = None
        self.activations = None
        self.target_layer_name = 'bottleneck'

    def conv_block(self, in_channels, out_channels, dropout_rate=0.3, num_layers=1):
        """
        A helper function to create a convolutional block consisting of multiple Conv3D layers, 
        BatchNorm (if enabled), GELU activation, and optional Dropout.
        
        Parameters:
        - in_channels: Number of input channels for the first Conv3D layer.
        - out_channels: Number of output channels for each Conv3D layer.
        - dropout_rate: The dropout rate applied after each GELU.
        - num_layers: Number of Conv3D layers to include in the block.
        """
        layers = []
        
        # Add the first convolutional layer
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel_size, padding=(1, 1, 1)))
        if self.use_batchnorm:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.GELU())
        
        # Add subsequent convolutional layers (num_layers - 1 times)
        for _ in range(num_layers - 1):
            layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, padding=(1, 1, 1)))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.GELU())
        
        # Add Dropout after GELU of the last layer
        layers.append(nn.Dropout3d(p=dropout_rate))



        return nn.Sequential(*layers)


    def save_gradients(self, grad):
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

        # grad-cam
        # if self.target_layer_name == 'bottleneck':
        #     x.requires_grad_(True)
        #     x.register_hook(self.save_gradients)
        #     self.activations = x

        # Decoder path
        for upconv, decoder, enc_output in zip(self.upconvs, self.decoders, reversed(enc_outputs)):
            x = upconv(x)
            x = torch.cat([x, enc_output], dim=1)  # Skip connection
            x = decoder(x)

        # Final convolution
        x = self.conv_final(x)

        # Reduce depth dimension (optional, adjust if necessary)
        x = torch.mean(x, dim=2)  # Shape: (batch_size, out_channels, height, width)
        #x = torch.mean(x, dim=[1, 2, 3])

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











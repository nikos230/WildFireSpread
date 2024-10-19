import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=1):
        super(ConvBlock3D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)    



class EncoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, pool_kernel=(1, 2, 2)):
        super(EncoderBlock3D, self).__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels, kernel_size, padding)
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel)

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p 

def crop_tensor(input_tensor, target_tensor):
    """Crops the input_tensor to match the spatial dimensions of the target_tensor."""
    _, _, d_in, h_in, w_in = input_tensor.size()
    _, _, d_tgt, h_tgt, w_tgt = target_tensor.size()

    delta_d = (d_in - d_tgt) // 2
    delta_h = (h_in - h_tgt) // 2
    delta_w = (w_in - w_tgt) // 2

    return input_tensor[:, :, delta_d:d_in - delta_d, delta_h:h_in - delta_h, delta_w:w_in - delta_w]
    
class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, upconv_kernel=(1, 2, 2), stride=(1, 2, 2)):
        super(DecoderBlock3D, self).__init__()
        
        # Adjust output_padding based on mismatch in dimensions
        self.upconv = nn.ConvTranspose3d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=upconv_kernel,    # Use upconv_kernel here
            padding=padding, 
            stride=stride,
            output_padding=(0, 0, 1)  # Adjust based on size mismatch (try different values)
        )
        
        self.conv_block = ConvBlock3D(in_channels, out_channels, kernel_size, padding)

    def forward(self, x, skip_connection):
        # Apply transposed convolution to upsample
        x = self.upconv(x)
        
        # Check if the sizes match for concatenation
        if x.size() != skip_connection.size():
            # Optional step: Crop or resize tensors if required
            pass
        
        # Concatenate the upsampled tensor and the skip connection tensor along the channel dimension
        x = torch.cat([x, skip_connection], dim=1)
        
        # Apply the conv block after concatenation
        x = self.conv_block(x)
        return x
        
            
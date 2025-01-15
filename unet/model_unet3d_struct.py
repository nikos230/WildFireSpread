import torch
import torch.nn as nn

class UNet3D_struct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D_struct, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 4, 4))

        #self.enc2 = self.conv_block(32, 64)
        #self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.bottleneck = self.conv_block(32, 64)

        # LSTM Layer (assuming you still want to process it using an LSTM)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)

        # Decoder
        self.upconv3 = nn.ConvTranspose3d(64, 32, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.dec3 = self.conv_block(64, 32)

        #self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        #self.dec2 = self.conv_block(64, 32)

        # Final output layer
        self.conv_final = nn.Conv3d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
           # nn.Dropout3d(p=0.4),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Dropout3d(p=0.4)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        #enc2 = self.enc2(pool1)
        #pool2 = self.pool2(enc2)

        # Bottleneck
        b = self.bottleneck(pool1)

        # LSTM processing (reshape the input to fit LSTM format: [batch, seq, feature])
        batch_size, channels, depth, height, width = b.size()
        b = b.view(batch_size, channels, -1).transpose(1, 2)  # Reshape to [batch_size, seq_len, feature_size] for LSTM

        # Pass through LSTM layer
        lstm_out, (hn, cn) = self.lstm(b)  # Output of LSTM is (batch_size, seq_len, hidden_size)

        # Correcting the output dimensions: ensure that the channels are 128 (as expected by the upconv3 layer)
        lstm_out = lstm_out[:, -1, :].unsqueeze(2).unsqueeze(3).unsqueeze(4)  # Take the last time-step, add spatial dims
        lstm_out = lstm_out.expand(-1, -1, depth, height, width)  # Expand to match spatial dimensions

        # Decoder path
        upconv3 = self.upconv3(lstm_out)
        dec3 = torch.cat([upconv3, enc1], dim=1)
        dec3 = self.dec3(dec3)

        #upconv2 = self.upconv2(dec3)
        #dec2 = torch.cat([upconv2, enc1], dim=1)
        #dec2 = self.dec2(dec2)

        # Output layer
        out = self.conv_final(dec3)
        out = torch.mean(out, dim=2)
        return out




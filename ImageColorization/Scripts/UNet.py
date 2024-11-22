import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder (Contracting Path)
        self.encoder1 = self.conv_block(1, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.encoder3 = self.conv_block(64, 128)
        self.encoder4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder (Expanding Path)
        self.decoder4 = self.upconv_block(512, 256)
        self.decoder3 = self.upconv_block(256, 128)
        self.decoder2 = self.upconv_block(128, 64)
        self.decoder1 = self.upconv_block(64, 32)

        # Final Convolution
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)  # Output: 3 channels for RGB

    def conv_block(self, in_channels, out_channels):
        """
        Convolutional block: two convolutional layers with ReLU and batch normalization.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2, 2)  # Downsampling
        )

    def upconv_block(self, in_channels, out_channels):
        """
        Upconvolutional block: transpose convolution followed by two convolutional layers.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # Encoder Path
        enc1 = self.encoder1(x)  # Output: (Batch, 32, 56, 56)
        enc2 = self.encoder2(enc1)  # Output: (Batch, 64, 28, 28)
        enc3 = self.encoder3(enc2)  # Output: (Batch, 128, 14, 14)
        enc4 = self.encoder4(enc3)  # Output: (Batch, 256, 7, 7)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # Output: (Batch, 512, 3, 3)

        # Decoder Path
        dec4 = self.decoder4(bottleneck) + enc4  # Output: (Batch, 256, 7, 7)
        dec3 = self.decoder3(dec4) + enc3  # Output: (Batch, 128, 14, 14)
        dec2 = self.decoder2(dec3) + enc2  # Output: (Batch, 64, 28, 28)
        dec1 = self.decoder1(dec2) + enc1  # Output: (Batch, 32, 56, 56)

        # Final Convolution
        output = self.final_conv(dec1)  # Output: (Batch, 3, 112, 112)
        return torch.sigmoid(output)  # Constrain output to [0, 1]

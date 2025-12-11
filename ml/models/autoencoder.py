"""Lightweight 2D CNN autoencoder for image denoising."""

import torch
import torch.nn as nn
from typing import Tuple


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, BatchNorm, and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """
        Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
        """
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    """Encoder network that progressively downsamples the input."""
    
    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        """
        Initialize encoder.
        
        Args:
            in_channels: Number of input channels (1 for grayscale)
            base_channels: Base number of channels (will be scaled in deeper layers)
        """
        super().__init__()
        
        # Encoder: 4 downsampling stages
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 8)
        )
        self.pool4 = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Forward pass with skip connections.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Tuple of (bottleneck, skip_connections)
        """
        # Encoder with skip connections
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        skip_connections = [e1, e2, e3, e4]
        
        return p4, skip_connections


class Decoder(nn.Module):
    """Decoder network that progressively upsamples to reconstruct the image."""
    
    def __init__(self, out_channels: int = 1, base_channels: int = 32):
        """
        Initialize decoder.
        
        Args:
            out_channels: Number of output channels (1 for grayscale)
            base_channels: Base number of channels (should match encoder)
        """
        super().__init__()
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16),
            ConvBlock(base_channels * 16, base_channels * 16)
        )
        
        # Decoder: 4 upsampling stages with skip connections
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = nn.Sequential(
            ConvBlock(base_channels * 16, base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 8)
        )
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4)
        )
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2)
        )
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels),
            ConvBlock(base_channels, base_channels)
        )
        
        # Final output layer
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, skip_connections: list) -> torch.Tensor:
        """
        Forward pass with skip connections.
        
        Args:
            x: Bottleneck tensor
            skip_connections: List of skip connection tensors from encoder
            
        Returns:
            Reconstructed output tensor (B, C, H, W)
        """
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.up4(x)
        x = torch.cat([x, skip_connections[3]], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = torch.cat([x, skip_connections[2]], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, skip_connections[1]], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, skip_connections[0]], dim=1)
        x = self.dec1(x)
        
        # Output layer with sigmoid to ensure [0, 1] range
        x = torch.sigmoid(self.output(x))
        
        return x


class DenoisingAutoencoder(nn.Module):
    """
    Lightweight U-Net style autoencoder for single-channel medical image denoising.
    
    Architecture:
        - Encoder: 4 downsampling stages (max pooling)
        - Bottleneck: 2 conv blocks at lowest resolution
        - Decoder: 4 upsampling stages (transposed conv) with skip connections
        - Output: Sigmoid activation for [0, 1] range
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        """
        Initialize the denoising autoencoder.
        
        Args:
            in_channels: Number of input channels (1 for grayscale coronary images)
            out_channels: Number of output channels (1 for grayscale)
            base_channels: Base number of filters (default 32 for lightweight model)
        """
        super().__init__()
        
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = Decoder(out_channels, base_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input noisy image tensor (B, 1, H, W)
            
        Returns:
            Denoised image tensor (B, 1, H, W)
        """
        # Encode with skip connections
        encoded, skip_connections = self.encoder(x)
        
        # Decode with skip connections
        decoded = self.decoder(encoded, skip_connections)
        
        return decoded
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(in_channels: int = 1, out_channels: int = 1, base_channels: int = 32) -> DenoisingAutoencoder:
    """
    Factory function to create a denoising autoencoder model.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base number of channels
        
    Returns:
        Initialized DenoisingAutoencoder model
    """
    model = DenoisingAutoencoder(in_channels, out_channels, base_channels)
    
    # Initialize weights using He initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    return model

import torch
import torch.nn as nn
from src.model.components.pixel_norm import PixelNorm

class Conv2dBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels:int, use_pixel_norm:bool=False):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,  # This will maintain the spatial dimensions of the input
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)  # LeakyReLU is often used in GANs to allow a small gradient when the unit is not active, which can help with training stability.
        ]

        if use_pixel_norm:
            layers.append(PixelNorm())  # Add pixel normalization if specified
        
        layers.extend([
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,  # This will maintain the spatial dimensions of the input
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ])

        if use_pixel_norm:
            layers.append(PixelNorm())  # Add pixel normalization if specified
        
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

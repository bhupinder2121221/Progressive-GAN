import torch
import torch.nn as nn
from src.model.components.conv2d import Conv2dBlock
from src.model.components.mini_batch_std import MiniBatchStdDev
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, image_channels:int = 3, base_channels:int = 256, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_channels = base_channels
        self.image_channels = image_channels


        # Channels by resolution stage:
        # or steps you ca say as well. Like step 0 => 4x4
        # step 1 => 8x8 etc.

        # This decreasing channels with increasing resolution is a common design choice in GAN discriminators to manage computational complexity and memory usage, while still allowing the model to learn rich representations at lower resolutions and finer details at higher resolutions.
        # Unless you have a super computer with tons of memory :) .
        self.channels = [
            base_channels,          # 4x4     # channels 256
            base_channels,          # 8x8     # channels 256
            base_channels // 2,     # 16x16   # channels 128
            base_channels // 4,     # 32x32   # channels 64
            base_channels // 8,     # 64x64   # channels 32
            base_channels // 16,    # 128x128 # channels 16
        ]

        self.from_rgb = nn.ModuleList()
        self.progression_blocks = nn.ModuleList()

        # Adding rgb block and progression block for each resolution stage
        for ch in self.channels:
            self.from_rgb.append(
                nn.Conv2d(
                    in_channels=image_channels,
                    out_channels=ch,
                    stride=1,
                    kernel_size=1
                )
            )

        # adding progession blocks for each resolution stage
        for i in range(len(self.channels) - 1, 0, -1):
            self.progression_blocks.append(
                Conv2dBlock(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i-1],
                    use_pixel_norm=False
                )
            )

        # add mini  batch std layer for adding divergence feature to detect mode collapse
        self.minibatch_std = MiniBatchStdDev()


        self.final_block = nn.Sequential(
            
            nn.Conv2d(
                in_channels=self.channels[0] + 1,  # +1 for the minibatch std feature
                out_channels=self.channels[0],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # refinement
            nn.Conv2d(
                in_channels=self.channels[0],
                out_channels=self.channels[0],
                kernel_size=4,
                stride=1,
                padding=0
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # flatten and output a single scalar value for real/fake classification
            nn.Flatten(),
            nn.Linear(self.channels[0], 1)  # Output a single scalar value for real/fake classification
        )

    def fade_in(self, x: torch.Tensor, prev_x: torch.Tensor, alpha: float) -> torch.Tensor:
        # Blend the upsampled previous output with the current output using the alpha value
        return alpha * x + (1 - alpha) * prev_x
    
    def forward(self, x: torch.Tensor, alpha: float, steps: int) -> torch.Tensor:
        """
        steps:
          0 => 4x4
          1 => 8x8
          2 => 16x16
          3 => 32x32
          4 => 64x64
          5 => 128x128
        """
        if steps == 0:
            # image to feature map of shape (b, 256, 4, 4)
            x = self.from_rgb[0](x)  # 3 channels to 256 channels
            x = self.minibatch_std(x)  # Add the minibatch standard deviation feature to help detect mode collapse
            return self.final_block(x)  # Process through the final block to get the real/fake classification output
        # else

        current_index = steps
        block_index = len(self.progression_blocks) - steps  # Calculate the block index based on the current step

        # get the features from the current resolution stage
        out = self.from_rgb[current_index](x)  # Convert the input image to feature map for the current resolution stage
        out = self.progression_blocks[block_index](out)  # Process through the progression block for the current resolution stage
        out = F.avg_pool2d(out, kernel_size=2, stride=2)  # Downsample the features to match the previous resolution stage

        # get the features from the previous resolution stage
        previous_out = F.avg_pool2d(x, kernel_size=2, stride=2)  # Downsample the input image to match the previous resolution stage
        previous_out = self.from_rgb[current_index - 1](previous_out)  # Convert the downsampled image to feature map for the previous resolution stage

        # blend the downscalled and learned features.
        out = self.fade_in(out, previous_out, alpha)  # Blend the features from the current and previous resolution stages using the alpha value

        for i in range(block_index + 1, len(self.progression_blocks)):
            out = self.progression_blocks[i](out)
            out = F.avg_pool2d(out, kernel_size=2, stride=2)

        out = self.minibatch_std(out)
        return self.final_block(out)

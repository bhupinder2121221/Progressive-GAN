import torch
import torch.nn as nn
from src.model.components.pixel_norm import PixelNorm
from src.model.components.conv2d import Conv2dBlock
from config import LATENT_DIM, IMAGE_CHANNELS, BASE_CHANNELS
class Generator(nn.Module):

    def __init__(self, latent_dim: int = LATENT_DIM, image_channels:int = IMAGE_CHANNELS, base_channels: int = BASE_CHANNELS, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # variables
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.base_channels = base_channels

        # Channels by resolution stage:
        # index 0 => 4x4
        # index 1 => 8x8
        # index 2 => 16x16
        # index 3 => 32x32
        # index 4 => 64x64
        # index 5 => 128x128

        # So with increasing resolution we are decreasing the number of channels to manage the computational complexity and memory usage, while still allowing the model to learn rich representations at lower resolutions and finer details at higher resolutions.
        self.channels = [
            base_channels,          # 4x4
            base_channels,          # 8x8
            base_channels // 2,     # 16x16
            base_channels // 4,     # 32x32
            base_channels // 8,     # 64x64
            base_channels // 16,    # 128x128
        ]


        # b,512,1,1 => b, 512, 4, 4 => b, 512, 4, 4
        self.initial = nn.Sequential(
            ######################################################
            ############# Feature Upscaling #############
            # Add pixcel norm for stabilizing the training and improving convergence.
            # this will prenvent from exploding feature values by normalizing the pixel values across the channel dimension, which helps in maintaining a stable training process and allows the model to learn more effectively.
            PixelNorm(),
            # N-out = (N-in -1)*strides + kernel_size - 2*padding + output_padding
            # = (1-1)*1 + 4 - 0 + 0 
            # = 4
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=self.channels[0],
                kernel_size=4,
                stride=1,
                padding=0,  # This will produce a 4x4 feature map from the latent vector.
            ),
            # activation function
            nn.LeakyReLU(negative_slope=0.2, inplace=True),




            ######################################################
            ############# Feature Refinement #############
            # This block will consist of two convolutional layers with LeakyReLU activations and pixel normalization.
            PixelNorm(),
            # N out = (N-in + 2*padding - kernel_size) // stride + 1
            # = (4 +2*1 - 3)//1 +1)
            # = 4
            nn.Conv2d(
                in_channels=self.channels[0],
                out_channels=self.channels[0],
                kernel_size=3,
                stride=1,
                padding=1,  # This will maintain the spatial dimensions of the feature map.
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            PixelNorm()
            ################################################
        )


        # This will hold the convolutional blocks for each resolution stage. Each block will consist of two convolutional layers with LeakyReLU activations and pixel normalization, followed by an upsampling layer to increase the spatial dimensions of the feature maps.
        self.progression_block = nn.ModuleList() 
        self.to_rgb = nn.ModuleList()

        # rgb for 4x4
        self.to_rgb.append(
            nn.Conv2d(
                in_channels=self.channels[0],
                out_channels=image_channels,
                kernel_size=1,
                stride=1,
                padding=0,  # This will convert the feature map to the desired number of image channels (e.g., 3 for RGB) without changing the spatial dimensions.
            )
        )

        # Now add blocks for each resolution stage
        for i in range(1, len(self.channels)):
            in_channels = self.channels[i-1]
            out_channels = self.channels[i]

            # add conv block for feature refinement at each resolution stage.
            self.progression_block.append(
                Conv2dBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    use_pixel_norm=True,  # Add pixel normalization for stabilizing the training and improving convergence.
                )
            )

            # Features channels (c, h, w) => Image channels (3, h, w)
            self.to_rgb.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=image_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,  # This will convert the feature map to the desired number of image channels (e.g., 3 for RGB) without changing the spatial dimensions.
                )
            )
    
    def fade_in(self, alpha:float, upscalled_x: torch.Tensor, generated_x: torch.Tensor) -> torch.Tensor:
        # This will blend the upscaled features from the previous resolution stage with the newly generated features from the current resolution stage based on the alpha value. 
        # The alpha value will control the contribution of each set of features during the transition between resolution stages, allowing for a smooth progression in the quality of generated images as the model trains.
        
        # When alphs = low, means more refinement so adding upscalled features to generated features.
        # When alpha = high, means more contribution from generated features and less from upscalled features.
        return alpha * generated_x + (1 - alpha) * upscalled_x
    
    def forward(self, z: torch.Tensor, alpha: float, steps: int) -> torch.Tensor:
        """
        z = latent vector of shape (b, latent_dim)  i.e. noise vector which will be transformed into an image by the generator.
        alpha = fade-in parameter
        steps = current resolution stage (0 for 4x4, 1 for 8x8, etc.)
        steps:
          0 => 4x4
          1 => 8x8
          2 => 16x16
          3 => 32x32
          4 => 64x64
          5 => 128x128
        """

        # latent vector b,z,1,1 to feature map of shape (b, 512, 4, 4)
        x = self.initial(z)

        if steps == 0:
            # we already have the output of 4x4 resolution stage, so we can directly convert it to RGB image using the to_rgb layer for this stage.
            return torch.tanh(self.to_rgb[steps](x))  # Apply tanh activation to scale the output pixel values between -1 and 1, which is common for GANs.
        for step in range(steps):
            # upscale the x 
            upscalled_x = nn.functional.interpolate(input=x, scale_factor=2, mode='nearest')  # Upscale the feature map by a factor of 2 using nearest neighbor interpolation. This will double the spatial dimensions (height and width) of the feature map.
            # refine the features using the convolutional block for the current resolution stage.
            x = self.progression_block[step](upscalled_x)  # Pass the upscaled features through the convolutional block for the current resolution stage to refine them.

        
        # new generated image for the current resolution stage.
        new_image = self.to_rgb[steps](x)  # Convert the refined features to an RGB image using the to_rgb layer for the current resolution stage.

        # Upscalled image from the previous resolution stage.
        prev_feature = upscalled_x
        prev_image = self.to_rgb[steps-1](prev_feature)  # Convert the upscaled features from the previous resolution stage to an RGB image using the to_rgb layer for that stage.

        # so now we have both the upscalled image on previous resolution and the new generated image for the current resolution stage, we will blend them using the fade-in function based on the alpha value to create a smooth transition between resolution stages.
        return self.fade_in(alpha=alpha, upscalled_x=prev_image, generated_x=new_image)  # Blend the upscaled image from the previous resolution stage with the newly generated image for the current resolution stage based on the alpha value to create a smooth transition between resolution stages.

            


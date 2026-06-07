import torch
import torch.nn as nn


class MiniBatchStdDev(nn.Module):
    """
    Add a feature represent the standard deviation of the features across the batch. 
    This is used in the discriminator to help it detect mode collapse, where the generator produces limited variety in its outputs. 
    By adding this feature, the discriminator can better identify when the generator is producing similar outputs, 
    which encourages the generator to produce more diverse images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_std = torch.std(x, dim=0, unbiased=False)  # Calculate the standard deviation across the batch dimension (dim=0). Setting unbiased=False gives the population standard deviation, which is more appropriate for this use case.
        # So acrross batch we found the divengence.
        # Meaning if earlier the shape was (b,c, h, w) then on dim 0 i.e. on b we found the std accorss all channels, pixels. 
        # so we got the shape (c, h, w).
        # This tells us how much the features vary across the batch, which can help the discriminator identify when the generator is producing similar outputs (mode collapse).


        # Now we will get mean scalar value of the standard deviation across all features to get a single scalar value.
        mean_std = batch_std.mean()  # Calculate the mean of the standard deviations across all features to get a single scalar value.
        # this gives us a single value telling about the diveristy in mini batch.

        # Now we will create a new feature map of shape (b, 1, h, w) where each pixel value is the mean standard deviation calculated above.
        mapped_std = mean_std.expand(x.size(0), 1, x.size(2), x.size(3))  # Expand the mean standard deviation to match the batch size and spatial dimensions of the input tensor. This creates a new feature map where each pixel value is the same mean standard deviation.

        # Now add this feature to the featres we already have so we addedd diversity in the feature
        return torch.cat([x, mapped_std], dim=1)  # Concatenate the new feature map with the original input tensor along the channel dimension (dim=1). This adds the mean standard deviation as an additional feature to the input, allowing the discriminator to use this information when making its decisions. The output shape will be (b, c+1, h, w) where c is the original number of channels.
        # now the size (b,c,h,w) becomes (b, c+1, h, w)
        

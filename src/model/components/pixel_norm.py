import torch
import torch.nn as nn


class PixelNorm(nn.Module):
    """
    This will normalize the pixel of the featire accorss the channel dimension. 
    It is used in the generator to stabilize the training and improve convergence.
    """
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for normalization, 
        # x = x / sqrt(mean(x^2) + epsilon)
        return x / (torch.sqrt(torch.mean(x**2, dim=1, keepdim=True)) + self.epsilon)  # Here dim =1 because we want to normalize across the channel dimension. keepdim=True is used to maintain the original dimensions of the tensor after the mean operation, which allows for proper broadcasting during division.
        
    

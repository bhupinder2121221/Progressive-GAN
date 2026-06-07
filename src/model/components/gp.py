import torch
import torch.nn as nn


def gradient_penality(discriminator: nn.Module, real_image: torch.Tensor, fake_image: torch.Tensor, steps: int, alpha: float, device: torch.device):
    batch_size, channels, height, width = real_image.shape
    epsinol = torch.rand(batch_size, 1, 1, 1).to(device)  # Random weight for interpolation
    interpolated_images = real_image * epsinol + fake_image * (1 - epsinol)  # Interpolate between real and fake images
    interpolated_images.requires_grad_(True)  # Enable gradient computation for the interpolated images

    mixed_scores = discriminator(interpolated_images, alpha, steps)  # Get discriminator scores for the interpolated images

    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(mixed_scores),  # Gradient of the output with respect to the input
        create_graph=True,  # Create a computational graph for the gradients
        retain_graph=True,  # Retain the graph for further gradient computations
    )[0]

    gradients = gradients.view(gradients.size(0), -1)  # Flatten the gradients to calculate the norm
    gradient_norm = gradients.norm(2, dim=1)  # Calculate the L2 norm of the gradients for each sample in the batch
    penality = torch.mean((gradient_norm - 1) ** 2)  # Calculate the gradient penalty using the formula (||gradients|| - 1)^2
    return penality

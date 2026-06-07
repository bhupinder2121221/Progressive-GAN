from pathlib import Path
import torch
from torch.nn import functional as F


from torchvision.utils import save_image


def save_samples(
    generator: torch.nn.Module,
    fixed_noise: torch.Tensor,
    steps: int,
    alpha: float,
    step_count: int,
    image_size: int,
    output_dir: str = "outputs",
):
    output_path = Path(output_dir) / f"size_{image_size}"
    output_path.mkdir(parents=True, exist_ok=True)

    generator.eval()

    with torch.no_grad():
        fake = generator(fixed_noise, alpha, steps)  

        # Generator outputs [-1, 1] because of tanh.
        # Convert back to [0, 1] before saving image.
        fake = fake * 0.5 + 0.5

        save_image(
            fake,
            output_path / f"sample_{step_count}.png",
            nrow=4,  # Adjust this based on how many samples you want per row in the saved image grid.
        )
        fake_to_save = F.interpolate(fake, size=(128, 128), mode="nearest")
        output_path = Path(output_path) / f"size_interpolate_128"
        output_path.mkdir(parents=True, exist_ok=True)
        save_image(
            fake_to_save,
            output_path / f"sample_{step_count}_128.png",
            nrow=4,  # Adjust this based on how many samples you want per row in the saved image grid.
        )

    generator.train()

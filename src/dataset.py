from pathlib import Path
from typing import List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import VALID_EXTENSIONS



# This will transform the raw images into tensors and normalize them according to model requirements. 
# You can add more transformations here if needed.
def transformation_pipeline(image_size: int) -> transforms.Compose:
    transformation = transforms.Compose(
        [
            transforms.Resize(image_size),  #  W = image_size   and it maintains the aspect ratio by default
            transforms.CenterCrop(image_size),  # H and W      this will crop the center of the image to ensure it is square and matches the desired size
            transforms.ToTensor(),  # Convert PIL Image to Tensor and scale pixel values to [0, 1]
            transforms.Normalize(                         # Normalize the tensor values between -1 and 1, as we are using tanh activation in the generator. The mean and std values are set to 0.5 to achieve this scaling.
                mean=[0.5, 0.5, 0.5],  # RGB channels mean 
                std=[0.5, 0.5, 0.5],   # RGB channels std
            )
        ]
    )
    return transformation


class ImageDataset(Dataset):
    def __init__(self, repo_dir: str, image_size: int):
        super().__init__()
        self.repo_dir = Path(repo_dir)
        self.image_size = image_size
        self.image_paths = []
        for ext in VALID_EXTENSIONS:
            self.image_paths.extend(self.repo_dir.glob(f'**/*{ext}'))
        
        if not self.image_paths:
            raise ValueError(f"No valid images found in {repo_dir}. Please check the directory and ensure it contains images with extensions: {VALID_EXTENSIONS}")
        
        self.transform = transformation_pipeline(image_size)

    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load the image, apply transformations, and return the processed tensor.
        self.image_path = self.image_paths[idx]
        # Open the image and convert it to RGB (in case it's grayscale or has an alpha channel), then apply transformations.
        image = Image.open(self.image_path).convert('RGB')  # Ensure all images are in RGB format
        return self.transform(image)



import torch 
import os
from config import DEVICE


def save_checkpoint(
        generator: torch.nn.Module, 
        discriminator: torch.nn.Module, 
        opt_g: torch.optim.Optimizer, 
        opt_d: torch.optim.Optimizer,
        epoch: int, 
        image_size: int,
        final=False,
        path: str = "../checkpoints/"):
    if not final:
        path = f"{path}/ckpt_size_{image_size}.pth"
    else:
        path = f"{path}/final_ckpt_size_{image_size}.pth"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "epoch": epoch
    }, path)

def load_checkpoint(
        generator: torch.nn.Module, 
        discriminator: torch.nn.Module, 
        opt_g: torch.optim.Optimizer, 
        opt_d: torch.optim.Optimizer,   
        image_size: int,
        path: str = "../checkpoints/")-> int:
    path = f"{path}/ckpt_size_{image_size}.pth"
    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        opt_g.load_state_dict(checkpoint["opt_g"])
        opt_d.load_state_dict(checkpoint["opt_d"])
        return checkpoint["epoch"]

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0
    
def load_latest_image_size_checkpoint(
        path: str = "../checkpoints/") -> int:
    latest_image_size = 0
    for file in os.listdir(path):
        if file.startswith("ckpt_size_") and file.endswith(".pth"):
            try:
                image_size = int(file.split("_")[2].split(".")[0])
                if image_size > latest_image_size:
                    latest_image_size = image_size
            except ValueError:
                continue
    return latest_image_size

import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model.components.gp import gradient_penality
from tqdm import tqdm

from config import (
    DEVICE,
    START_IMG_SIZE,
    TARGET_IMG_SIZE,
    LATENT_DIM,
    BASE_CHANNELS,
    LAMBDA_GP,
    LR_G,
    LR_D,
    BETA1,
    BETA2,
    EPOCHS_PER_RESOLUTION,
    BATCH_SIZES,
    IMAGE_CHANNELS,
    SAMPLE_EVERY,
    CHECKPOINT_EVERY,
    VALID_EXTENSIONS
)

from src.dataset import ImageDataset
from src.model.generator import Generator
from src.model.discriminator import Discriminator
from src.helper_functions.checkpoints import save_checkpoint, load_checkpoint, load_latest_image_size_checkpoint
from src.helper_functions.sample import save_samples

def image_size_to_step(image_size:int)->int:
    return int(math.log2(image_size // START_IMG_SIZE))

def train_one_resolution(
        generator: torch.nn.Module,
        discriminator: Discriminator,
        opt_g: optim.Adam,
        opt_d: optim.Adam,
        image_size: int,
        data_path: str,
        fixed_noise
):
    # try loading model
    start_epoch = load_checkpoint(
        generator=generator,
        discriminator=discriminator,
        opt_g=opt_g,
        opt_d=opt_d,
        image_size=image_size
    )



    # identify the step based on the image size
    step = image_size_to_step(image_size)

    # get the batch size and number of epochs for the current step
    batch_size = BATCH_SIZES[image_size]
    epochs = EPOCHS_PER_RESOLUTION[image_size]

    dataset = ImageDataset(repo_dir=data_path, image_size=image_size)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    total_batches = len(loader)*epochs
    global_batch = 0

    print(f"Training for image size: {image_size}x{image_size}")
    print(f"Batch size: {batch_size}, Epochs: {epochs}, Step: {step}")

    for epoch in range(start_epoch, epochs):
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} Image Size: {image_size}x{image_size}")

        for real_image in progress:
            real_image = real_image.to(DEVICE)
            current_batch_size = real_image.size(0)


            # for fading in, we calculate alpha based on the global batch number and total batches for this resolution
            alpha = min(1.0, (epoch + 1) / (epochs * 0.5))

            #################
            # Train Discriminator
            #################

            noise = torch.randn(current_batch_size, LATENT_DIM, 1, 1).to(DEVICE)  # generate noise for the current batch size
            fake_image = generator(noise, alpha, step)  # generate fake images using the generator

            real_score = discriminator(real_image, alpha, step)  # get discriminator's score for real images
            fake_score = discriminator(fake_image.detach(), alpha, step)  # get discriminator's score for fake images (detach to avoid backprop through generator)

            gp = gradient_penality(
                discriminator=discriminator,
                real_image=real_image,
                fake_image=fake_image.detach(),
                alpha=alpha,
                steps=step,
                device=DEVICE
            )

            d_loss = (
                -(torch.mean(real_score) - torch.mean(fake_score))   # Wasserstein loss for discriminator
                + LAMBDA_GP * gp   # add the gradient penalty term to the discriminator loss
                + 0.001 * torch.mean(real_score**2) # R1 regularization term to stabilize training
            )

            opt_d.zero_grad()  # zero the gradients for the discriminator
            d_loss.backward()  # backpropagate the discriminator loss
            opt_d.step()  # update the discriminator's parameters


            #################
            # Train Generator
            #################
            for _ in range(2):  # train the generator more times than the discriminator to help it catch up
                noise = torch.randn(current_batch_size, LATENT_DIM, 1, 1).to(DEVICE)  # generate noise for the current batch size
                fake_image = generator(noise, alpha, step)  # generate fake images using the generator

                fake_score = discriminator(fake_image, alpha, step)  # get discriminator's score for the newly generated fake images

                g_loss = -torch.mean(fake_score)  # Wasserstein loss for generator (negate because we want to maximize the score for fake images)

                opt_g.zero_grad()  # zero the gradients for the generator
                g_loss.backward()  # backpropagate the generator loss
                opt_g.step()  # update the generator's parameters

            progress.set_postfix({
                "D Loss": d_loss.item(),
                "G Loss": g_loss.item(),
                "Alpha": alpha
            })


            
            global_batch += 1
        
        if epoch % SAMPLE_EVERY == 0:
            save_samples(
                generator=generator,
                fixed_noise=fixed_noise,
                alpha=alpha,
                steps=step,
                step_count=epoch,
                image_size=image_size
            )
        
        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                generator=generator,
                discriminator=discriminator,
                opt_g=opt_g,
                opt_d=opt_d,
                epoch=epoch,
                image_size=image_size
            )


def main():
    data_dir = "data"

    # create the generator model
    generator = Generator(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        image_channels=IMAGE_CHANNELS
    ).to(DEVICE)

    # create the discriminator model
    discriminator = Discriminator(
        base_channels=BASE_CHANNELS,
        image_channels=IMAGE_CHANNELS
    ).to(DEVICE)

    # create optimizers for both generator and discriminator
    optimizer_g = optim.Adam(generator.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    # create a fixed noise vector for sampling during training
    fixed_noise = torch.randn(16, LATENT_DIM, 1, 1).to(DEVICE)


    resolutions = [4, 8, 16, 32, 64]

    if TARGET_IMG_SIZE >= 128:
        resolutions.append(128)

    # check at the chpoints which is the latest resolution we have trained on and start from there
    image_size_checkpoint = load_latest_image_size_checkpoint()
    if image_size_checkpoint in resolutions:
        resolutions = [size for size in resolutions if size >= image_size_checkpoint]
        print(f"Resuming training from image size: {image_size_checkpoint}x{image_size_checkpoint}")

    for image_size in resolutions:
        train_one_resolution(
            generator=generator,
            discriminator=discriminator,
            opt_g=optimizer_g,
            opt_d=optimizer_d,
            image_size=image_size,
            data_path=data_dir,
            fixed_noise=fixed_noise
        )

    save_checkpoint(
        generator=generator,
        discriminator=discriminator,
        opt_g=optimizer_g,
        opt_d=optimizer_d,
        epoch=0,
        final=True,
        image_size=TARGET_IMG_SIZE
    )


if __name__ == "__main__":
    # This is a common practice to ensure that the code runs on the GPU if available. 
    # By creating a small tensor and moving it to the GPU, we can trigger the CUDA initialization process. 
    # This can help avoid some of the overhead associated with the first CUDA operation during training, which can lead to smoother performance.
    if torch.cuda.is_available():
        _ = torch.zeros(1).cuda()
    main()



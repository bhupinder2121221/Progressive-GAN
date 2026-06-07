import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Progerressively growing GAN configuration
START_IMG_SIZE = 4
TARGET_IMG_SIZE = 64  



LATENT_DIM = 512      # 512 is classic, but 256 is safer for 6GB
BASE_CHANNELS = 256   # reduce if OOM

LAMBDA_GP = 10
LR_G = 2e-4
LR_D = 1e-4

# fade-in parameters
BETA1 = 0.0
BETA2 = 0.99

# Number of epochs per resolution phase
# For each resolution, we train for a certain number of epochs before moving to the next resolution.
# This is a simple schedule, but you can adjust it based on your dataset and compute resources.
# this is fade-in epochs, after fade-in we can train for more epochs if desired
EPOCHS_PER_RESOLUTION = {
    4: 1200,
    8: 1100,
    16: 900,
    32: 1500,
    64: 1500,
    128: 500,
}

# Batch sizes for each resolution. 
# Higher resolutions typically require smaller batch sizes due to memory constraints.
BATCH_SIZES = {
    4: 128,
    8: 128,
    16: 64,
    32: 32,
    64: 16,
    128: 8,
}
# The number of channels in the generated images. For RGB images, this is 3. For grayscale, it would be 1.
IMAGE_CHANNELS = 3

# How often to sample and save generated images during training, and how often to save model checkpoints.
SAMPLE_EVERY = 10

# How often to save model checkpoints during training. Adjust this based on your training time and storage capacity.
CHECKPOINT_EVERY = 50


# Valid image extensions to consider when loading the dataset. This helps filter out non-image files.
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
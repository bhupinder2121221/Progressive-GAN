# PGAN — Progressively Growing GAN

A PyTorch implementation of a **Progressively Growing GAN (PGAN)** that trains a generator and discriminator starting from 4×4 images and doubles the resolution at each phase, up to 64×64 (extendable to 128×128). The network uses Wasserstein loss with Gradient Penalty (WGAN-GP) and fade-in transitions between resolutions for stable, high-quality image synthesis.

---

## Author

**Bhupinder**  
AI Engineer  
B.Tech CSE — Guru Nanak Dev University, Amritsar
[*My Website*](https://bhupinder-ai.vercel.app/)
---

## How It Works

Training happens in stages. At each stage the generator and discriminator grow by one layer:

```
4×4  →  8×8  →  16×16  →  32×32  →  64×64  →  (128×128)
```

Each resolution has its own fade-in period (controlled by `alpha` going from 0 → 1) so new layers blend in smoothly before the model moves to the next size.

---

## Project Structure

```
PGAN/
├── config.py                        # All hyperparameters and settings
├── main.py                          # Entry point — runs the full training loop
├── requirements.txt                 # Python dependencies
│
├── data/                            # ← PUT YOUR TRAINING IMAGES HERE
│
├── checkpoints/                     # Saved model checkpoints (auto-created)
│
├── outputs/                         # Generated sample images (auto-created)
│   ├── size_4/
│   ├── size_8/
│   ├── size_16/
│   ├── size_32/
│   ├── size_64/
│   └── size_interpolate_128/
│
└── src/
    ├── dataset.py                   # Dataset loader with auto-resize & crop
    ├── model/
    │   ├── generator.py
    │   ├── discriminator.py
    │   └── components/
    │       ├── conv2d.py            # Equalized learning-rate conv layer
    │       ├── gp.py                # Gradient penalty
    │       ├── mini_batch_std.py    # Mini-batch standard deviation
    │       └── pixel_norm.py        # Pixel normalization
    └── helper_functions/
        ├── checkpoints.py           # Save / load checkpoint helpers
        └── sample.py                # Save generated image grids
```

---

## Where to Add Your Pictures

Place all your training images inside the **`data/`** folder.  
Sub-folders are supported — the dataset loader scans recursively.

```
data/
├── image_001.jpg
├── image_002.png
└── subfolder/
    └── image_003.jpeg
```

Accepted formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

The loader automatically resizes and center-crops every image to the current training resolution, so mixed sizes are fine.

---

## Where the Output Will Be

| Output type | Location |
|---|---|
| Generated image samples | `outputs/size_<N>/` for each resolution |
| Interpolated samples (128px) | `outputs/size_interpolate_128/` |
| Model checkpoints | `checkpoints/` |

Samples are saved every `SAMPLE_EVERY` epochs (default: every 10 epochs).  
Checkpoints are saved every `CHECKPOINT_EVERY` epochs (default: every 50 epochs).  
Training automatically resumes from the latest checkpoint if one exists.

---

## Installation

```bash
# 1. Clone / open the project
cd PGAN

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 4. Install dependencies (includes PyTorch with CUDA 12.1 support)
pip install -r requirements.txt
```

---

## How to Run

```bash
python main.py
```

The script will:
1. Detect your GPU automatically (falls back to CPU if no CUDA device is found).
2. Start training from 4×4 resolution (or resume from the last saved checkpoint).
3. Progressively grow through each resolution defined in `EPOCHS_PER_RESOLUTION`.
4. Save sample grids to `outputs/` and checkpoints to `checkpoints/` as it trains.

---

## Configuration (`config.py`)

All settings live in `config.py`. You do **not** need to touch `main.py` for normal use.

| Parameter | Default | Meaning |
|---|---|---|
| `DEVICE` | `"cuda"` / `"cpu"` | Auto-selected based on GPU availability |
| `START_IMG_SIZE` | `4` | First training resolution (4×4) |
| `TARGET_IMG_SIZE` | `64` | Final training resolution |
| `LATENT_DIM` | `512` | Size of the random noise vector fed to the generator |
| `BASE_CHANNELS` | `256` | Base number of feature maps (reduce if you run out of VRAM) |
| `LAMBDA_GP` | `10` | Gradient penalty weight for WGAN-GP stability |
| `LR_G` | `2e-4` | Generator learning rate |
| `LR_D` | `1e-4` | Discriminator learning rate |
| `BETA1` | `0.0` | Adam β₁ (0 is standard for GANs) |
| `BETA2` | `0.99` | Adam β₂ |
| `EPOCHS_PER_RESOLUTION` | see below | How many epochs to train at each resolution |
| `BATCH_SIZES` | see below | Batch size per resolution (smaller for larger images) |
| `IMAGE_CHANNELS` | `3` | `3` for RGB, `1` for grayscale |
| `SAMPLE_EVERY` | `10` | Save a sample image grid every N epochs |
| `CHECKPOINT_EVERY` | `50` | Save a checkpoint every N epochs |
| `VALID_EXTENSIONS` | `.jpg .jpeg .png .bmp .tiff` | File types the dataset loader will pick up |

### Epochs per resolution (default schedule)

| Resolution | Epochs |
|---|---|
| 4×4 | 1200 |
| 8×8 | 1100 |
| 16×16 | 900 |
| 32×32 | 1500 |
| 64×64 | 1500 |
| 128×128 | 500 |

### Batch sizes per resolution (default)

| Resolution | Batch size |
|---|---|
| 4×4 | 128 |
| 8×8 | 128 |
| 16×16 | 64 |
| 32×32 | 32 |
| 64×64 | 16 |
| 128×128 | 8 |

> **Tip:** If you run out of GPU memory, lower `BASE_CHANNELS` (e.g. to `128`) or reduce the batch size for the larger resolutions in `BATCH_SIZES`.

---

## Requirements

- Python 3.9+
- PyTorch 2.5.1 (CUDA 12.1 build included in `requirements.txt`)
- torchvision 0.20.1
- pillow, tqdm, matplotlib, numpy

---

## References

- Karras et al. (2018) — [*Progressive Growing of GANs for Improved Quality, Stability, and Variation*](https://arxiv.org/abs/1710.10196)
- Gulrajani et al. (2017) — [*Improved Training of Wasserstein GANs*](https://arxiv.org/abs/1704.00028)

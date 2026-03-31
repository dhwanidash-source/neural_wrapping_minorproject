# neural_wrapping_minorproject
# Neural Post-Processor for JPEG Artifact Removal

> Inspired by **"Perceptual Video Compression with Neural Wrapping"** — Khan et al., CVPR 2025 (Sony Interactive Entertainment)

A lightweight Python/PyTorch implementation of the neural post-processor concept from the paper. Instead of wrapping neural networks around AV1/VVC video codecs, this project applies the same idea to JPEG images — making it trainable on a standard laptop CPU with no GPU required.

---

## What This Project Does

Standard compression (JPEG, AV1, H.264) introduces blocking artifacts, blurring, and ringing. This project trains a small convolutional neural network to remove those artifacts and recover perceptual quality — the same goal as the paper's post-processor component **O**.

| | Paper (Full System) | This Project |
|---|---|---|
| Codec | AV1 / VVC video | JPEG images |
| Networks | Pre-processor + Post-processor | Post-processor only |
| Dataset | Vimeo-90k video | BSD500 images (~65MB) |
| Loss | SSIM + MS-SSIM + VMAF | MSE + SSIM |
| Hardware | GPU cluster | CPU laptop |

---

## Results

Evaluated on 10 images × 5 JPEG quality levels (50 total pairs). Every entry shows positive gain.

| JPEG Quality | PSNR Compressed | PSNR Enhanced | PSNR Gain | SSIM Compressed | SSIM Enhanced | SSIM Gain |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| q = 10 | 25.79 dB | 26.35 dB | **+0.56 dB** | 0.7433 | 0.7727 | **+0.0294** |
| q = 20 | 27.96 dB | 28.64 dB | **+0.68 dB** | 0.8318 | 0.8613 | **+0.0295** |
| q = 30 | 29.39 dB | 30.06 dB | **+0.67 dB** | 0.8682 | 0.8903 | **+0.0221** |
| q = 40 | 30.50 dB | 31.11 dB | **+0.61 dB** | 0.8913 | 0.9095 | **+0.0182** |
| q = 50 | 31.27 dB | 31.73 dB | **+0.46 dB** | 0.9097 | 0.9236 | **+0.0139** |
| **Average** | **28.98 dB** | **29.58 dB** | **+0.60 dB** | **0.8689** | **0.8915** | **+0.0226** |

---

## Project Structure
```
├── model.py          # Lightweight residual CNN (the post-processor)
├── dataset.py        # Downloads BSD500, creates JPEG compression pairs on the fly
├── losses.py         # SSIM + MSE perceptual loss functions
├── train.py          # Training loop with checkpointing
├── evaluate.py       # PSNR / SSIM metrics on test images
├── demo.py           # Visual before/after comparison (saves demo_output.png)
├── plot_results.py   # Training curves + quality RD curves
└── requirements.txt  # Dependencies
```

---

## Model Architecture

A residual CNN that learns to predict and subtract the compression artifact:
```
Input (JPEG-compressed image)
  → Conv(3→64) + ReLU                   [entry]
  → 8 × ResidualBlock(Conv→BN→ReLU→Conv→BN + skip)   [body]
  → Conv(64→3)                           [exit]
  → Add to input (residual correction)
  → Clamp [0, 1]
Output (artifact-removed image)
```

| Config | Features | Blocks | Parameters | CPU Training Time |
|---|:---:|:---:|:---:|:---:|
| Lightweight | 16 | 2 | ~22K | ~5 min |
| Default | 32 | 4 | ~92K | ~30–60 min |
| Full | 64 | 8 | ~345K | ~90 min |

---

## Setup

**Requirements:** Python 3.8+, no GPU needed.
```bash

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies (CPU-only PyTorch)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install Pillow scikit-image matplotlib numpy
```

---

## How to Run

### 1. Verify everything works
```bash
python model.py      # Prints parameter count and output shape
python losses.py     # Prints loss values
python dataset.py    # Downloads BSD500 (~65MB) and prints batch shapes
```

### 2. Train
```bash
# Quick test run (~5 minutes)
python train.py --epochs 5 --num_features 16 --num_blocks 2

# Default training (~30-60 minutes on CPU)
python train.py --epochs 50 --num_features 32 --num_blocks 4 --batch_size 32
```

### 3. Evaluate
```bash
python evaluate.py --num_features 32 --num_blocks 4
```

### 4. Visual demo
```bash
python demo.py --qualities 10 20 30
# Saves demo_output.png — side-by-side: Clean | Compressed | Enhanced | Diff
```

### 5. Plot curves
```bash
python plot_results.py
# Saves training_curves.png and rd_curves.png
```

---

## Loss Function
```
Total Loss = α × MSE(enhanced, clean) + (1 − α) × SSIM_Loss(enhanced, clean)
```

Default `α = 0.5`. Set `--alpha 0.0` for pure perceptual (SSIM) or `--alpha 1.0` for pure PSNR optimization.

---

## Dataset

The project uses **BSD500** (Berkeley Segmentation Dataset) — 500 natural photographs, ~65MB, downloaded automatically on first run. Training pairs (compressed, clean) are generated on the fly by JPEG-compressing random 64×64 patches at a random quality level between 10 and 50. No separate download of compressed images is needed.

If the download fails, the code automatically falls back to a synthetic dataset so you can still test the pipeline.

---

## Reference
```
@inproceedings{khan2025neural,
  title     = {Perceptual Video Compression with Neural Wrapping},
  author    = {Khan, Muhammad Umar Karim and Chadha, Aaron and
               Anam, Mohammad Ashraful and Andreopoulos, Yiannis},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer
               Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  organization = {Sony Interactive Entertainment}
}
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: torch` | Run `pip install torch` |
| Dataset download fails | Falls back to synthetic data automatically |
| Loss goes NaN | Add `--lr 0.0001` to lower the learning rate |
| Very slow training | Use `--num_features 16 --num_blocks 2` |
| `best_model.pth` not found | Run `train.py` first |

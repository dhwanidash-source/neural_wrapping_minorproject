"""
dataset.py - Data Preparation
==============================
We need pairs of:
  - clean images  (ground truth)
  - JPEG-compressed versions (simulating codec compression, like AV1/VVC in the paper)

The paper uses gaming video sequences. We use the BSD500 dataset (Berkeley Segmentation),
which is free, small (~65MB), and commonly used in image restoration research.

Dataset flow:
  Clean image → JPEG compress (quality 10-50) → compressed image
  Train: predict clean from compressed
  This directly mimics the paper's postprocessor: IC (codec output) → enhanced output

JPEG quality levels map roughly to codec QP values:
  quality=10  → very compressed (like high QP in AV1)
  quality=30  → moderate compression
  quality=50  → light compression
"""

import os
import io
import random
import urllib.request
import tarfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ── Download helpers ───────────────────────────────────────────────────────────

BSDS_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
# Note: If BSDS is unavailable, we fall back to generating synthetic patches

def download_bsds500(data_dir="data"):
    """
    Download the BSDS500 dataset (~65MB).
    Contains 500 natural images: 200 train, 100 val, 200 test.
    """
    os.makedirs(data_dir, exist_ok=True)
    tgz_path = os.path.join(data_dir, "BSR_bsds500.tgz")

    if os.path.exists(os.path.join(data_dir, "BSR")):
        print("BSDS500 already downloaded.")
        return os.path.join(data_dir, "BSR", "BSDS500", "data", "images")

    print("Downloading BSDS500 dataset (~65MB)...")
    print("This will take a minute depending on your connection.")

    try:
        urllib.request.urlretrieve(BSDS_URL, tgz_path,
            reporthook=lambda b, bs, t: print(f"\r  {b*bs/1e6:.1f}/{t/1e6:.1f} MB", end=""))
        print("\nExtracting...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(data_dir)
        os.remove(tgz_path)
        print("Done!")
        return os.path.join(data_dir, "BSR", "BSDS500", "data", "images")
    except Exception as e:
        print(f"\nCould not download BSDS500: {e}")
        print("Falling back to synthetic dataset...")
        return None


def collect_images(image_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """Walk a directory tree and collect all image file paths."""
    paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if any(f.lower().endswith(ext) for ext in extensions):
                paths.append(os.path.join(root, f))
    return sorted(paths)


# ── JPEG compression helper ────────────────────────────────────────────────────

def jpeg_compress(pil_image, quality):
    """
    Compress a PIL Image with JPEG at the given quality level and return it.
    This simulates what a standard codec does (like AV1 in the paper).

    quality: 1-95 (lower = more compression = more artifacts)
    """
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()  # .copy() needed to detach from buffer


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class JpegArtifactDataset(Dataset):
    """
    Dataset that returns (compressed_patch, clean_patch) pairs.

    For each sample:
      1. Load a clean image
      2. Crop a random patch (e.g. 64x64)
      3. JPEG compress the patch at a random quality level
      4. Return (compressed_tensor, clean_tensor)

    Why random quality?
      The paper trains a SINGLE model for the entire rate-quality curve.
      We do the same: one model handles multiple JPEG quality levels.
    """
    def __init__(self, image_paths, patch_size=64,
                 quality_range=(10, 50), patches_per_image=8):
        """
        Args:
            image_paths     : list of paths to clean images
            patch_size      : size of square patches to crop
            quality_range   : (min_quality, max_quality) for JPEG compression
            patches_per_image: how many patches to extract per image per epoch
        """
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.quality_range = quality_range
        self.patches_per_image = patches_per_image

        # Simple transforms: just convert to tensor [0,1]
        self.to_tensor = T.ToTensor()

        print(f"Dataset: {len(image_paths)} images, "
              f"~{len(image_paths)*patches_per_image} patches/epoch, "
              f"JPEG quality {quality_range[0]}-{quality_range[1]}")

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx):
        # Pick which image
        img_idx = idx % len(self.image_paths)
        img = Image.open(self.image_paths[img_idx]).convert("RGB")

        # Random crop — ensure image is large enough
        w, h = img.size
        ps = self.patch_size
        if w < ps or h < ps:
            img = img.resize((max(w, ps), max(h, ps)), Image.BICUBIC)
            w, h = img.size

        left = random.randint(0, w - ps)
        top  = random.randint(0, h - ps)
        clean_patch = img.crop((left, top, left + ps, top + ps))

        # JPEG compress at random quality (simulates different codec bitrates)
        quality = random.randint(*self.quality_range)
        compressed_patch = jpeg_compress(clean_patch, quality)

        # Convert to tensors
        clean_tensor = self.to_tensor(clean_patch)          # [3, H, W] in [0,1]
        compressed_tensor = self.to_tensor(compressed_patch) # [3, H, W] in [0,1]

        return compressed_tensor, clean_tensor


class SyntheticDataset(Dataset):
    """
    Fallback dataset using random noise images if BSDS is unavailable.
    Not great for training but good for testing your code pipeline.
    """
    def __init__(self, n_samples=1000, patch_size=64, quality_range=(10, 50)):
        self.n_samples = n_samples
        self.patch_size = patch_size
        self.quality_range = quality_range
        self.to_tensor = T.ToTensor()
        print(f"Using synthetic dataset: {n_samples} random samples")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Random colorful synthetic image patch
        arr = np.random.randint(0, 256, (self.patch_size, self.patch_size, 3), dtype=np.uint8)
        clean = Image.fromarray(arr)
        quality = random.randint(*self.quality_range)
        compressed = jpeg_compress(clean, quality)
        return self.to_tensor(compressed), self.to_tensor(clean)


# ── Factory function ───────────────────────────────────────────────────────────

def get_dataloaders(data_dir="data", patch_size=64, batch_size=32,
                    quality_range=(10, 50), num_workers=2):
    """
    Main entry point: returns (train_loader, val_loader).

    Steps:
      1. Try to download BSDS500
      2. Split into train/val (80/20)
      3. Create DataLoaders
    """
    image_dir = download_bsds500(data_dir)

    if image_dir and os.path.exists(image_dir):
        all_paths = collect_images(image_dir)
        print(f"Found {len(all_paths)} images total.")

        random.shuffle(all_paths)
        split = int(0.8 * len(all_paths))
        train_paths = all_paths[:split]
        val_paths   = all_paths[split:]

        train_dataset = JpegArtifactDataset(train_paths, patch_size, quality_range, patches_per_image=16)
        val_dataset   = JpegArtifactDataset(val_paths,   patch_size, quality_range, patches_per_image=4)
    else:
        # Fallback to synthetic
        train_dataset = SyntheticDataset(2000, patch_size, quality_range)
        val_dataset   = SyntheticDataset(400,  patch_size, quality_range)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


if __name__ == "__main__":
    print("=== Dataset Test ===\n")
    train_loader, val_loader = get_dataloaders(batch_size=4)
    compressed, clean = next(iter(train_loader))
    print(f"Compressed batch shape: {compressed.shape}")
    print(f"Clean batch shape     : {clean.shape}")
    print(f"Compressed value range: [{compressed.min():.3f}, {compressed.max():.3f}]")
    print(f"Clean value range     : [{clean.min():.3f}, {clean.max():.3f}]")
    print("\nDataset looks good!")

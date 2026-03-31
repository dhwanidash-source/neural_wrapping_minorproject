"""
evaluate.py - Evaluation & Metrics
====================================
After training, this script evaluates the model on test images.

We report the same metrics the paper uses:
  - PSNR  (Peak Signal-to-Noise Ratio) — standard quality measure
  - SSIM  (Structural Similarity)       — perceptual quality
  - BD-Rate savings (simplified)        — how much the network saves

We compare THREE versions for each test image:
  1. Clean (ground truth)
  2. JPEG-compressed (baseline, no post-processing)
  3. Enhanced (our model's output)

The improvement from (2) → (3) is what we're measuring.
In the paper: improvement from AV1/VVC → AV1/VVC + Neural Wrapper.
"""

import os
import io
import json
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

from model import NeuralPostProcessor


# ── JPEG helper ────────────────────────────────────────────────────────────────

def jpeg_compress_pil(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def pil_to_np(img):
    """PIL Image → float32 numpy array [0, 1]"""
    return np.array(img).astype(np.float32) / 255.0


def np_to_tensor(arr):
    """numpy [H, W, 3] → tensor [1, 3, H, W]"""
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def tensor_to_np(t):
    """tensor [1, 3, H, W] → numpy [H, W, 3]"""
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy()


# ── Evaluation functions ───────────────────────────────────────────────────────

def evaluate_image(model, clean_img, quality, device):
    """
    Evaluate one image at one JPEG quality level.
    Returns dict with PSNR/SSIM for compressed and enhanced versions.
    """
    # Compress
    compressed_img = jpeg_compress_pil(clean_img, quality)

    clean_np      = pil_to_np(clean_img)
    compressed_np = pil_to_np(compressed_img)

    # Run model
    compressed_tensor = np_to_tensor(compressed_np).to(device)
    with torch.no_grad():
        enhanced_tensor = model(compressed_tensor)
    enhanced_np = np.clip(tensor_to_np(enhanced_tensor), 0, 1)

    # Compute metrics
    # PSNR
    psnr_compressed = psnr_fn(clean_np, compressed_np, data_range=1.0)
    psnr_enhanced   = psnr_fn(clean_np, enhanced_np,   data_range=1.0)

    # SSIM (channel_axis for multichannel)
    ssim_compressed = ssim_fn(clean_np, compressed_np, data_range=1.0, channel_axis=2)
    ssim_enhanced   = ssim_fn(clean_np, enhanced_np,   data_range=1.0, channel_axis=2)

    return {
        "quality": quality,
        "psnr_compressed": float(psnr_compressed),
        "psnr_enhanced":   float(psnr_enhanced),
        "psnr_gain":       float(psnr_enhanced - psnr_compressed),
        "ssim_compressed": float(ssim_compressed),
        "ssim_enhanced":   float(ssim_enhanced),
        "ssim_gain":       float(ssim_enhanced - ssim_compressed),
    }


def print_results_table(all_results):
    """Print a nicely formatted comparison table."""
    # Group by quality level
    from collections import defaultdict
    by_quality = defaultdict(list)
    for r in all_results:
        by_quality[r["quality"]].append(r)

    print("\n" + "=" * 80)
    print(f"{'Quality':>8} | "
          f"{'PSNR Compressed':>17} | "
          f"{'PSNR Enhanced':>15} | "
          f"{'PSNR Gain':>11} | "
          f"{'SSIM Compressed':>17} | "
          f"{'SSIM Enhanced':>15} | "
          f"{'SSIM Gain':>11}")
    print("-" * 80)

    for q in sorted(by_quality.keys()):
        results = by_quality[q]
        avg_psnr_comp = np.mean([r["psnr_compressed"] for r in results])
        avg_psnr_enh  = np.mean([r["psnr_enhanced"]   for r in results])
        avg_psnr_gain = np.mean([r["psnr_gain"]        for r in results])
        avg_ssim_comp = np.mean([r["ssim_compressed"]  for r in results])
        avg_ssim_enh  = np.mean([r["ssim_enhanced"]    for r in results])
        avg_ssim_gain = np.mean([r["ssim_gain"]        for r in results])

        print(f"  q={q:3d}   | "
              f"{avg_psnr_comp:17.2f} | "
              f"{avg_psnr_enh:15.2f} | "
              f"{avg_psnr_gain:+11.2f} | "
              f"{avg_ssim_comp:17.4f} | "
              f"{avg_ssim_enh:15.4f} | "
              f"{avg_ssim_gain:+11.4f}")

    print("=" * 80)

    # Overall averages
    print(f"\nOverall Average:")
    print(f"  PSNR gain : {np.mean([r['psnr_gain'] for r in all_results]):+.2f} dB")
    print(f"  SSIM gain : {np.mean([r['ssim_gain'] for r in all_results]):+.4f}")
    print()
    print("Positive gains = our model improves quality over raw JPEG compression.")
    print("This mirrors the BD-rate savings reported in the paper (Tab. 1 & 2).")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Neural Post-Processor")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--test_images", type=str, default="data",
                        help="Directory with test images (uses BSDS test split)")
    parser.add_argument("--qualities", type=int, nargs="+", default=[10, 20, 30, 40, 50],
                        help="JPEG quality levels to evaluate at")
    parser.add_argument("--num_features", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=50,
                        help="Maximum number of test images to use")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_results", type=str, default="results.json",
                        help="Save JSON results to this file")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Please run train.py first!")
        return

    model = NeuralPostProcessor(
        num_features=args.num_features,
        num_blocks=args.num_blocks
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("Model loaded successfully!")

    # Find test images
    from dataset import collect_images
    image_paths = collect_images(args.test_images)
    if not image_paths:
        print(f"No images found in {args.test_images}")
        return

    # Use a subset
    image_paths = image_paths[:args.max_images]
    print(f"\nEvaluating on {len(image_paths)} images at qualities {args.qualities}...")
    print()

    all_results = []
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            # Resize if very large (for speed)
            w, h = img.size
            if max(w, h) > 512:
                scale = 512 / max(w, h)
                img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)

            for quality in args.qualities:
                result = evaluate_image(model, img, quality, device)
                result["image"] = os.path.basename(path)
                all_results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(image_paths)} images...")

        except Exception as e:
            print(f"  Skipping {path}: {e}")

    # Print results table
    print_results_table(all_results)

    # Save results
    with open(args.save_results, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {args.save_results}")
    print("Use these to make your own RD-curve plots (like Fig. 2 in the paper)!")


if __name__ == "__main__":
    main()

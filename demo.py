"""
demo.py - Visual Demo
======================
Run this script to see side-by-side comparisons of:
  [Clean] vs [JPEG Compressed] vs [Our Model Enhanced]

This is equivalent to Fig. 4 in the paper which shows:
  "Source | Codec | Codec + proposed"

Usage:
    python demo.py --image path/to/your/image.jpg --quality 20
    python demo.py  (uses a built-in test image)
"""

import os
import io
import argparse
import urllib.request
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

from model import NeuralPostProcessor


# ── Helpers ────────────────────────────────────────────────────────────────────

def jpeg_compress(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def load_model(checkpoint_path, num_features=64, num_blocks=8, device="cpu"):
    model = NeuralPostProcessor(num_features=num_features, num_blocks=num_blocks)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def enhance(model, pil_img, device):
    """Run the neural post-processor on a PIL image."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    out_np = np.clip(out.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
    return (out_np * 255).astype(np.uint8)


def get_test_image():
    """
    Try to load a sample image. Uses a free-to-use test image.
    Falls back to generating a colorful synthetic image.
    """
    # Try to use a standard test image (Lena/Baboon/Cameraman alternatives)
    sample_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
    try:
        data = urllib.request.urlopen(sample_url, timeout=5).read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        print("Using sample image from Wikipedia.")
        return img
    except Exception:
        pass

    # Fallback: colorful synthetic image
    print("Using synthetic test image.")
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    # Colorful blocks to make JPEG artifacts visible
    arr[:128, :128] = [220, 50, 50]    # red
    arr[:128, 128:] = [50, 150, 220]   # blue
    arr[128:, :128] = [50, 200, 100]   # green
    arr[128:, 128:] = [220, 180, 50]   # yellow
    # Add some texture/gradients to make artifacts interesting
    for i in range(0, 256, 16):
        arr[i:i+8, :] = arr[i:i+8, :] * 85 // 100
    return Image.fromarray(arr)


# ── Main visualization ─────────────────────────────────────────────────────────

def run_demo(args):
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"\nERROR: No checkpoint found at '{args.checkpoint}'")
        print("Please train the model first: python train.py")
        print("\nShowing what the demo WOULD look like with a random (untrained) model...")
        model = NeuralPostProcessor(num_features=args.num_features, num_blocks=args.num_blocks)
        model.to(device).eval()
    else:
        print(f"Loading model from {args.checkpoint}...")
        model = load_model(args.checkpoint, args.num_features, args.num_blocks, device)

    # Load image
    if args.image and os.path.exists(args.image):
        clean_img = Image.open(args.image).convert("RGB")
        # Resize if large
        w, h = clean_img.size
        if max(w, h) > 512:
            scale = 512 / max(w, h)
            clean_img = clean_img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
        print(f"Using image: {args.image} ({clean_img.size})")
    else:
        clean_img = get_test_image()

    qualities = args.qualities
    n_qualities = len(qualities)

    # ── Create figure ──────────────────────────────────────────────────────────
    # Layout: rows = quality levels, cols = [clean, compressed, enhanced, diff]
    fig = plt.figure(figsize=(16, 4 * n_qualities))
    fig.suptitle(
        "Neural Post-Processor Results\n"
        "(Inspired by 'Perceptual Video Compression with Neural Wrapping', CVPR 2025)",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(n_qualities, 4, figure=fig,
                           hspace=0.4, wspace=0.08)

    clean_np = np.array(clean_img)

    for row_idx, quality in enumerate(qualities):
        # Process image
        compressed_img = jpeg_compress(clean_img, quality)
        enhanced_np    = enhance(model, compressed_img, device)
        compressed_np  = np.array(compressed_img)

        # Compute metrics
        psnr_comp = psnr_fn(clean_np, compressed_np, data_range=255)
        psnr_enh  = psnr_fn(clean_np, enhanced_np,   data_range=255)
        ssim_comp = ssim_fn(clean_np.astype(float)/255,
                            compressed_np.astype(float)/255,
                            data_range=1.0, channel_axis=2)
        ssim_enh  = ssim_fn(clean_np.astype(float)/255,
                            enhanced_np.astype(float)/255,
                            data_range=1.0, channel_axis=2)

        # Difference map (amplified for visibility)
        diff = np.abs(enhanced_np.astype(int) - compressed_np.astype(int))
        diff_amp = np.clip(diff * 3, 0, 255).astype(np.uint8)

        # Plot
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax1 = fig.add_subplot(gs[row_idx, 1])
        ax2 = fig.add_subplot(gs[row_idx, 2])
        ax3 = fig.add_subplot(gs[row_idx, 3])

        ax0.imshow(clean_np);      ax0.axis("off")
        ax1.imshow(compressed_np); ax1.axis("off")
        ax2.imshow(enhanced_np);   ax2.axis("off")
        ax3.imshow(diff_amp);      ax3.axis("off")

        # Titles (only on first row)
        if row_idx == 0:
            ax0.set_title("Clean (Ground Truth)", fontsize=10, pad=4)
            ax1.set_title("JPEG Compressed\n(Baseline)", fontsize=10, pad=4)
            ax2.set_title("Neural Enhanced\n(Our Output)", fontsize=10, pad=4)
            ax3.set_title("Diff × 3\n(What model changed)", fontsize=10, pad=4)

        # Quality metrics as y-labels
        ax0.set_ylabel(f"JPEG q={quality}", fontsize=9, rotation=90, labelpad=4)

        # Metric annotations under each image
        ax1.set_xlabel(f"PSNR: {psnr_comp:.1f} dB\nSSIM: {ssim_comp:.4f}",
                       fontsize=8, color="red")
        ax2.set_xlabel(f"PSNR: {psnr_enh:.1f} dB (+{psnr_enh-psnr_comp:.1f})\n"
                       f"SSIM: {ssim_enh:.4f} (+{ssim_enh-ssim_comp:.4f})",
                       fontsize=8, color="green")

        print(f"  Quality {quality:3d} | "
              f"PSNR: {psnr_comp:.1f} → {psnr_enh:.1f} dB ({psnr_enh-psnr_comp:+.1f}) | "
              f"SSIM: {ssim_comp:.4f} → {ssim_enh:.4f} ({ssim_enh-ssim_comp:+.4f})")

    # Save figure
    out_path = args.output
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved comparison figure to: {out_path}")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Demo")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a test image (uses sample if not given)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--qualities", type=int, nargs="+", default=[10, 20, 30],
                        help="JPEG quality levels to demonstrate")
    parser.add_argument("--num_features", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--output", type=str, default="demo_output.png")
    args = parser.parse_args()

    run_demo(args)

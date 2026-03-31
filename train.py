"""
train.py - Training the Neural Post-Processor
===============================================
This is the main training script. Here's what happens:

  For each epoch:
    For each batch (compressed_image, clean_image):
      1. Forward pass: enhanced = model(compressed)
      2. Compute loss: PerceptualLoss(enhanced, clean)
      3. Backward pass: compute gradients
      4. Update weights: optimizer.step()

  Every few epochs:
    - Validate on held-out images
    - Log PSNR and SSIM metrics
    - Save the best model checkpoint

This directly corresponds to the paper's postprocessor training:
  IC (codec output) → O(IC) → optimized with perceptual loss vs source

The key difference from the paper: we don't jointly train a pre-processor
or a differentiable codec proxy. We keep it simple: just train the post-processor.
"""

import os
import time
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import NeuralPostProcessor, count_parameters
from dataset import get_dataloaders
from losses import PerceptualLoss, psnr, compute_ssim


# ── Argument parsing ───────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Train Neural Post-Processor")

    # Model
    parser.add_argument("--num_features", type=int, default=64,
                        help="Number of feature channels (32=fast, 64=better)")
    parser.add_argument("--num_blocks", type=int, default=8,
                        help="Number of residual blocks (4=fast, 8=better)")

    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (reduce if OOM)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="MSE weight in loss (0=pure SSIM, 1=pure MSE)")

    # Data
    parser.add_argument("--patch_size", type=int, default=64,
                        help="Training patch size")
    parser.add_argument("--quality_min", type=int, default=10,
                        help="Min JPEG quality (lower = more compression)")
    parser.add_argument("--quality_max", type=int, default=50,
                        help="Max JPEG quality")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory to store/load datasets")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers (0 on Windows)")

    # Saving
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Where to save model checkpoints")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Print log every N batches")

    # Hardware
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps (Apple Silicon)")

    return parser.parse_args()


# ── Helper functions ───────────────────────────────────────────────────────────

def get_device(device_arg):
    """Auto-detect the best available device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS (Metal Performance Shaders)")
        else:
            device = torch.device("cpu")
            print("Using CPU (no GPU detected — training will be slower)")
    else:
        device = torch.device(device_arg)
    return device


def validate(model, val_loader, loss_fn, device):
    """
    Run validation: compute average loss, PSNR, and SSIM on validation set.
    These are the same metrics the paper reports.
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0

    with torch.no_grad():
        for compressed, clean in val_loader:
            compressed = compressed.to(device)
            clean = clean.to(device)

            enhanced = model(compressed)
            loss, _, _ = loss_fn(enhanced, clean)

            total_loss += loss.item()
            total_psnr += psnr(enhanced, clean).item()
            total_ssim += compute_ssim(enhanced, clean).item()
            n_batches += 1

    return (total_loss / n_batches,
            total_psnr / n_batches,
            total_ssim / n_batches)


# ── Main training loop ─────────────────────────────────────────────────────────

def train(args):
    print("=" * 60)
    print("  Neural Post-Processor Training")
    print("  Inspired by: Perceptual Video Compression with Neural Wrapping")
    print("=" * 60)
    print()

    # 1. Device setup
    device = get_device(args.device)

    # 2. Data loading
    print("\n[Step 1] Loading data...")
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        quality_range=(args.quality_min, args.quality_max),
        num_workers=args.num_workers,
    )

    # 3. Model setup
    print("\n[Step 2] Building model...")
    model = NeuralPostProcessor(
        in_channels=3,
        num_features=args.num_features,
        num_blocks=args.num_blocks,
    ).to(device)
    count_parameters(model)

    # 4. Loss function
    loss_fn = PerceptualLoss(alpha=args.alpha).to(device)

    # 5. Optimizer + scheduler
    # Adam is standard for image restoration.
    # CosineAnnealingLR decays LR smoothly — helps final convergence.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 6. Checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    best_ssim = 0.0
    best_checkpoint = os.path.join(args.save_dir, "best_model.pth")
    latest_checkpoint = os.path.join(args.save_dir, "latest_model.pth")

    # 7. Training history (for plotting later)
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}

    print(f"\n[Step 3] Starting training for {args.epochs} epochs...")
    print(f"  Loss = {args.alpha:.1f} * MSE + {1-args.alpha:.1f} * SSIM")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        n_batches = 0

        for batch_idx, (compressed, clean) in enumerate(train_loader):
            compressed = compressed.to(device)
            clean = clean.to(device)

            # Forward pass
            enhanced = model(compressed)

            # Compute perceptual loss
            loss, mse_val, ssim_val = loss_fn(enhanced, clean)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping prevents exploding gradients (good practice)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # Print progress
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f} "
                      f"| MSE: {mse_val:.4f} | SSIM_loss: {ssim_val:.4f}")

        # Step LR scheduler
        scheduler.step()
        avg_train_loss = epoch_loss / n_batches
        epoch_time = time.time() - epoch_start

        # Validation every epoch (or every 5 for speed)
        val_loss, val_psnr_score, val_ssim_score = validate(model, val_loader, loss_fn, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr_score)
        history["val_ssim"].append(val_ssim_score)

        print(f"\nEpoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PSNR: {val_psnr_score:.2f} dB | "
              f"Val SSIM: {val_ssim_score:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {epoch_time:.1f}s\n")

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_ssim": val_ssim_score,
            "val_psnr": val_psnr_score,
            "args": vars(args),
        }, latest_checkpoint)

        # Save best model (based on SSIM, matching paper's perceptual focus)
        if val_ssim_score > best_ssim:
            best_ssim = val_ssim_score
            torch.save(model.state_dict(), best_checkpoint)
            print(f"  ★ New best SSIM: {best_ssim:.4f} — saved to {best_checkpoint}\n")

    # Save training history for plotting
    import json
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Best Val SSIM : {best_ssim:.4f}")
    print(f"  Best model    : {best_checkpoint}")
    print(f"  History saved : {args.save_dir}/history.json")
    print("=" * 60)
    print("\nNext step: run  python evaluate.py  to see full metrics")
    print("           run  python demo.py      to see visual comparisons")


if __name__ == "__main__":
    args = get_args()
    train(args)

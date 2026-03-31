"""
plot_results.py - Plot Training Curves & RD Curves
====================================================
Generates two types of plots:
  1. Training curve (loss, PSNR, SSIM over epochs)
  2. Rate-Distortion style curve (quality vs SSIM)
     — this is what Fig. 2 in the paper shows for their method

Run after training:  python plot_results.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict


def plot_training_history(history_path="checkpoints/history.json",
                          output_path="training_curves.png"):
    """Plot loss, PSNR, and SSIM over training epochs."""
    if not os.path.exists(history_path):
        print(f"No history file found at {history_path}. Run training first.")
        return

    with open(history_path) as f:
        h = json.load(f)

    epochs = list(range(1, len(h["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training History — Neural Post-Processor\n"
                 "(Inspired by Perceptual Video Compression with Neural Wrapping)",
                 fontsize=11, fontweight="bold")

    # Loss
    axes[0].plot(epochs, h["train_loss"], label="Train Loss", color="blue")
    axes[0].plot(epochs, h["val_loss"],   label="Val Loss",   color="orange")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss\n(Lower is better)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # PSNR
    axes[1].plot(epochs, h["val_psnr"], color="green")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Validation PSNR\n(Higher is better, ~30+ dB is good)")
    axes[1].grid(alpha=0.3)

    # SSIM
    axes[2].plot(epochs, h["val_ssim"], color="red")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("SSIM")
    axes[2].set_title("Validation SSIM\n(Higher is better, paper targets this)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Training curves saved to: {output_path}")
    plt.show()


def plot_rd_curves(results_path="results.json",
                   output_path="rd_curves.png"):
    """
    Plot Quality vs SSIM curves — like Fig. 2 in the paper.
    Compares: JPEG only vs JPEG + Neural Post-Processor
    """
    if not os.path.exists(results_path):
        print(f"No results file found at {results_path}. Run evaluate.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Group by quality
    by_q = defaultdict(lambda: {"ssim_comp": [], "ssim_enh": [],
                                 "psnr_comp": [], "psnr_enh": []})
    for r in results:
        q = r["quality"]
        by_q[q]["ssim_comp"].append(r["ssim_compressed"])
        by_q[q]["ssim_enh"].append(r["ssim_enhanced"])
        by_q[q]["psnr_comp"].append(r["psnr_compressed"])
        by_q[q]["psnr_enh"].append(r["psnr_enhanced"])

    qualities = sorted(by_q.keys())
    ssim_comp = [np.mean(by_q[q]["ssim_comp"]) for q in qualities]
    ssim_enh  = [np.mean(by_q[q]["ssim_enh"])  for q in qualities]
    psnr_comp = [np.mean(by_q[q]["psnr_comp"]) for q in qualities]
    psnr_enh  = [np.mean(by_q[q]["psnr_enh"])  for q in qualities]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Quality Curves: JPEG vs JPEG + Neural Post-Processor\n"
                 "(Analogous to Fig. 2 in the paper — higher curves = better)",
                 fontsize=11, fontweight="bold")

    # SSIM curve
    axes[0].plot(qualities, ssim_comp, "o-", color="orange", label="JPEG only (baseline)")
    axes[0].plot(qualities, ssim_enh,  "s-", color="green",  label="JPEG + Neural PP (ours)")
    axes[0].fill_between(qualities, ssim_comp, ssim_enh, alpha=0.15, color="green",
                         label="SSIM improvement")
    axes[0].set_xlabel("JPEG Quality →"); axes[0].set_ylabel("SSIM (higher=better)")
    axes[0].set_title("SSIM Quality Curve\n(The paper optimizes for this)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # PSNR curve
    axes[1].plot(qualities, psnr_comp, "o-", color="orange", label="JPEG only (baseline)")
    axes[1].plot(qualities, psnr_enh,  "s-", color="blue",   label="JPEG + Neural PP (ours)")
    axes[1].fill_between(qualities, psnr_comp, psnr_enh, alpha=0.15, color="blue",
                         label="PSNR improvement")
    axes[1].set_xlabel("JPEG Quality →"); axes[1].set_ylabel("PSNR dB (higher=better)")
    axes[1].set_title("PSNR Quality Curve")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"RD curves saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_training_history()
    plot_rd_curves()

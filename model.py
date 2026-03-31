"""
model.py - Lightweight Neural Post-Processor
============================================
Inspired by the post-processor component in:
"Perceptual Video Compression with Neural Wrapping" (CVPR 2025)

The paper uses large networks trained on video. Here we use a simple
DnCNN-style residual network that runs fast even on CPU.

Architecture:
  Input (JPEG-compressed image)
    --> Conv + ReLU
    --> [N x Conv + BN + ReLU blocks]   <-- learns compression artifacts
    --> Conv (residual output)
    --> Add to input (residual learning)
  Output (artifact-removed, perceptually better image)

Why residual learning?
  The network learns the DIFFERENCE (artifact) between compressed and clean.
  This is easier than learning the whole mapping from scratch.
  Same principle as the paper: the postprocessor recovers information lost in codec.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A single residual block: Conv -> BN -> ReLU -> Conv -> BN -> skip connection
    Batch Norm helps training stability (important for small datasets).
    """
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class NeuralPostProcessor(nn.Module):
    """
    Lightweight neural post-processor for JPEG artifact removal.

    Parameters:
        in_channels  : 3 for RGB images
        num_features : width of the network (64 is enough for CPU training)
        num_blocks   : depth of the network (more blocks = better quality, slower)

    For a CPU-friendly model: num_features=32, num_blocks=4
    For a slightly better model: num_features=64, num_blocks=8
    """
    def __init__(self, in_channels=3, num_features=64, num_blocks=8):
        super(NeuralPostProcessor, self).__init__()

        # Entry convolution: map image channels → feature space
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Stack of residual blocks — the "brain" of the network
        # These learn to detect and remove blocking/ringing artifacts
        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )

        # Exit convolution: map features → residual image (same size as input)
        self.exit = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: compressed image tensor (B, 3, H, W), values in [0, 1]
        returns: enhanced image tensor (B, 3, H, W), values clamped to [0, 1]

        The network predicts the artifact residual, then subtracts it from x.
        This is exactly analogous to the postprocessor O in the paper:
          "O recovers information removed by the codec"
        """
        features = self.entry(x)
        residual = self.exit(self.body(features))
        # Add residual to input: enhanced = compressed + learned_correction
        output = x + residual
        return torch.clamp(output, 0.0, 1.0)


def count_parameters(model):
    """Helper to show how many parameters the model has."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Approximate size    : {total * 4 / 1024 / 1024:.2f} MB (float32)")
    return trainable


if __name__ == "__main__":
    # Quick sanity check
    print("=== NeuralPostProcessor Model Check ===\n")
    model = NeuralPostProcessor(num_features=64, num_blocks=8)
    count_parameters(model)

    # Test with a dummy batch: 2 images, 3 channels, 128x128
    dummy_input = torch.rand(2, 3, 128, 128)
    output = model(dummy_input)
    print(f"\nInput  shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("\nModel looks good!")

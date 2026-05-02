"""Visualize forward diffusion on real images.

Picks 10 random images, noises each at a randomly sampled t from [0, T-1],
and plots: original | noised | noise.
"""

NUM_IMAGES = 10

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import matplotlib.pyplot as plt
from dataset import CelebADataset
from diffusion import LinearDiffusion
import yaml


def to_displayable(x: torch.Tensor) -> torch.Tensor:
    """Convert from [-1,1] to [0,1] and clamp for display."""
    return ((x + 1) / 2).clamp(0, 1)


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = CelebADataset(
        img_dir=os.path.join(os.path.dirname(__file__), "..", "data", "img_align_celeba"),
        image_size=cfg["image_size"],
    )
    diffusion = LinearDiffusion(config_path)

    n = NUM_IMAGES
    T = cfg["num_timesteps"]
    timesteps = torch.randint(0, T, (n,))
    indices = torch.randint(0, len(dataset), (n,))

    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 7.5))
    row_labels = ["Original", "Noised", "Noise"]

    for col, (t_val, img_idx) in enumerate(zip(timesteps, indices)):
        img = dataset[img_idx.item()]              # (C, H, W) in [-1, 1]
        x_0 = img.unsqueeze(0)                     # (1, C, H, W)
        t = t_val.unsqueeze(0)                     # (1,)

        x_t, noise = diffusion.q_sample(x_0, t)

        panels = [
            to_displayable(x_0[0]),
            to_displayable(x_t[0]),
            to_displayable(noise[0]),
        ]

        for row, panel in enumerate(panels):
            axes[row, col].imshow(panel.permute(1, 2, 0).numpy())
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(row_labels[row], fontsize=14)
            if row == 0:
                axes[row, col].set_title(f"t={t_val.item():.0f}", fontsize=11)

    plt.suptitle("Forward Diffusion Visualization", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "diffusion_viz.png"), dpi=150, bbox_inches="tight")
    print("Saved to tools/diffusion_viz.png")
    plt.show()


if __name__ == "__main__":
    main()

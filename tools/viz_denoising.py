"""Visualize trained model's denoising ability.

For 10 random images, noise each at a random t, then show:
  original | noised (x_t) | predicted noise | actual noise | one-step denoised

Also shows a pure generation panel: start from random noise and
iteratively denoise using DDPM sampling for a grid of generated faces.
"""

NUM_IMAGES = 10
NUM_GENERATED = 10

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import matplotlib.pyplot as plt
from dataset import CelebADataset
from diffusion import LinearDiffusion
from ditmodel import DiT
import yaml


def to_displayable(x: torch.Tensor) -> torch.Tensor:
    """Convert from [-1,1] to [0,1] and clamp for display."""
    return ((x + 1) / 2).clamp(0, 1)


@torch.no_grad()
def ddpm_sample(model, diffusion, device, num_samples, channels, image_size):
    """Generate images by iteratively denoising from pure noise."""
    x = torch.randn(num_samples, channels, image_size, image_size, device=device)

    for t_val in reversed(range(diffusion.T)):
        t = torch.full((num_samples,), t_val, device=device, dtype=torch.long)

        beta_t = diffusion.beta[t_val].to(device)
        alpha_t = diffusion.alpha[t_val].to(device)
        alpha_bar_t = diffusion.alpha_bar[t_val].to(device)

        eps_hat = model(x, t)

        # DDPM mean
        x = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_hat
        )

        # Add noise for all steps except t=0
        if t_val > 0:
            z = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * z

    return x


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    weights_path = os.path.join(os.path.dirname(__file__), "..", "dit_weights.pt")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = DiT(config_path).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    diffusion = LinearDiffusion(config_path)

    dataset = CelebADataset(
        img_dir=os.path.join(os.path.dirname(__file__), "..", "data", "img_align_celeba"),
        image_size=cfg["image_size"],
    )

    # --- Panel 1: Denoising quality on real images ---
    n = NUM_IMAGES
    T = cfg["num_timesteps"]
    timesteps = torch.randint(0, T, (n,))
    indices = torch.randint(0, len(dataset), (n,))

    fig, axes = plt.subplots(5, n, figsize=(2.5 * n, 12.5))
    row_labels = ["Original", "Noised", "Pred noise", "True noise", "1-step denoise"]

    for col, (t_val, img_idx) in enumerate(zip(timesteps, indices)):
        img = dataset[img_idx.item()]
        x_0 = img.unsqueeze(0).to(device)
        t = t_val.unsqueeze(0).to(device)

        x_t, noise = diffusion.q_sample(x_0, t)
        x_t, noise = x_t.to(device), noise.to(device)

        with torch.no_grad():
            eps_hat = model(x_t, t)

        # One-step denoise estimate: x_0_hat = (x_t - sqrt(1-a_bar)*eps) / sqrt(a_bar)
        alpha_bar_t = diffusion.alpha_bar[t_val.item()].to(device)
        x_0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)

        panels = [
            to_displayable(x_0[0].cpu()),
            to_displayable(x_t[0].cpu()),
            to_displayable(eps_hat[0].cpu()),
            to_displayable(noise[0].cpu()),
            to_displayable(x_0_hat[0].cpu()),
        ]

        for row, panel in enumerate(panels):
            axes[row, col].imshow(panel.permute(1, 2, 0).numpy())
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(row_labels[row], fontsize=12)
            if row == 0:
                axes[row, col].set_title(f"t={t_val.item():.0f}", fontsize=11)

    plt.suptitle("Denoising Quality", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "denoising_viz.png"),
                dpi=150, bbox_inches="tight")
    print("Saved to tools/denoising_viz.png")
    plt.show()

    # --- Panel 2: Generated samples from pure noise ---
    print("Generating samples (this may take a minute)...")
    samples = ddpm_sample(
        model, diffusion, device,
        num_samples=NUM_GENERATED,
        channels=cfg["image_channels"],
        image_size=cfg["image_size"],
    )

    fig, axes = plt.subplots(1, NUM_GENERATED, figsize=(2.5 * NUM_GENERATED, 2.5))
    for i in range(NUM_GENERATED):
        axes[i].imshow(to_displayable(samples[i].cpu()).permute(1, 2, 0).numpy())
        axes[i].axis("off")

    plt.suptitle("Generated Samples (DDPM)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "generated_viz.png"),
                dpi=150, bbox_inches="tight")
    print("Saved to tools/generated_viz.png")
    plt.show()


if __name__ == "__main__":
    main()

"""Benchmark: compare transformer block counts at 128x128."""

import time
import copy
import torch
import torch.nn as nn
import yaml
from diffusion import LinearDiffusion
from ditmodel import DiT


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def bench_1000_images(model, diffusion, device, image_size, batch_size, T, loss_fn):
    """Run 1000 images through forward + backward + optimizer.step()."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_images = 1000
    num_steps = num_images // batch_size

    # Warmup
    x_0 = torch.randn(batch_size, 3, image_size, image_size, device=device)
    t = torch.randint(0, T, (batch_size,), device=device)
    x_t, noise = diffusion.q_sample(x_0, t)
    x_t, noise = x_t.to(device), noise.to(device)
    loss = loss_fn(model(x_t, t), noise)
    loss.backward()
    optimizer.step()
    sync(device)

    sync(device)
    start = time.perf_counter()

    for step in range(num_steps):
        x_0 = torch.randn(batch_size, 3, image_size, image_size, device=device)
        t = torch.randint(0, T, (batch_size,), device=device)

        x_t, noise = diffusion.q_sample(x_0, t)
        x_t, noise = x_t.to(device), noise.to(device)

        optimizer.zero_grad()
        eps_hat = model(x_t, t)
        loss = loss_fn(eps_hat, noise)
        loss.backward()
        optimizer.step()

    sync(device)
    end = time.perf_counter()

    return end - start, num_steps


def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}\n")

    loss_fn = nn.MSELoss()
    num_images = 202_599
    base_config = "config_64.yaml"

    with open(base_config) as f:
        base_cfg = yaml.safe_load(f)

    print(f"Base: 64x64, patch=16, dim=256, heads=8, mlp=512, bs=32\n")
    print(f"{'Blocks':>6} {'Params':>12} {'1000img':>8} {'ms/step':>8} {'ms/img':>7} {'epoch':>7} {'total':>7}")
    print("-" * 65)

    for num_blocks in [1, 2, 4, 6, 8, 12]:
        # Write a temp config with this block count
        cfg = copy.deepcopy(base_cfg)
        cfg["num_transformer_blocks"] = num_blocks
        tmp_config = f"/tmp/bench_blocks_{num_blocks}.yaml"
        with open(tmp_config, "w") as f:
            yaml.dump(cfg, f)

        model = DiT(tmp_config).to(device)
        diffusion = LinearDiffusion(tmp_config)

        param_count = sum(p.numel() for p in model.parameters())

        elapsed, num_steps = bench_1000_images(
            model, diffusion, device,
            cfg["image_size"], cfg["batch_size"], cfg["num_timesteps"], loss_fn
        )

        per_step = elapsed / num_steps * 1000
        per_img = elapsed / 1000 * 1000
        epoch_min = num_images * (elapsed / 1000) / 60
        total_hr = epoch_min * cfg["num_epochs"] / 60

        print(f"{num_blocks:>6} {param_count:>12,} {elapsed:>7.2f}s {per_step:>7.1f} {per_img:>6.1f} "
              f"{epoch_min:>5.1f}m {total_hr:>5.1f}h")

        del model
        torch.mps.empty_cache() if device == "mps" else None


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import yaml
from diffusion import LinearDiffusion
from ditmodel import DiT
from dataset import get_dataloader


def train(config_path: str = "config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # --- Model & diffusion ---
    model = DiT(config_path).to(device)
    diffusion = LinearDiffusion(config_path)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.MSELoss()

    dataloader = get_dataloader(config_path)
    T = cfg["num_timesteps"]
    accum_steps = cfg.get("grad_accum_steps", 1)

    for epoch in range(cfg["num_epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        optimizer.zero_grad()

        for x_0 in dataloader:
            x_0 = x_0.to(device)

            # Sample random timesteps
            t = torch.randint(0, T, (x_0.shape[0],), device=device)

            # Forward diffusion
            x_t, noise = diffusion.q_sample(x_0, t)
            x_t = x_t.to(device)
            noise = noise.to(device)

            # Predict noise
            epsilon_hat = model(x_t, t)

            loss = loss_fn(epsilon_hat, noise) / accum_steps
            loss.backward()

            num_batches += 1
            epoch_loss += loss.item() * accum_steps

            if num_batches % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{cfg['num_epochs']}  loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "dit_weights.pt")
    print("Saved weights to dit_weights.pt")


if __name__ == "__main__":
    train()

import torch
import yaml


class LinearDiffusion:
    """Forward diffusion process with a linear noise schedule.

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.T = cfg["num_timesteps"]

        # Linear schedule: beta goes from 1e-4 to 0.02 over T steps
        self.beta = torch.linspace(1e-4, 0.02, self.T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Sample x_t given x_0 and timestep t.

        Args:
            x_0: clean images, shape (B, C, H, W)
            t:   timestep indices, shape (B,), values in [0, T-1]
            noise: optional pre-sampled noise, same shape as x_0

        Returns:
            x_t:   noised images, same shape as x_0
            noise: the noise that was added (for the training target)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_bar_t = self.alpha_bar.to(x_0.device)[t]
        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)

        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        return x_t, noise

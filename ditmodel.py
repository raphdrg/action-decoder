import math
import torch
import torch.nn as nn
import yaml


class SinCosPositionalEmbedding(nn.Module):
    """Fixed sin/cos positional embedding for a 2-D grid of patches."""

    def __init__(self, num_patches_h: int, num_patches_w: int, dim: int):
        super().__init__()
        num_patches = num_patches_h * num_patches_w
        pe = torch.zeros(num_patches, dim)

        position = torch.arange(num_patches).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, N, D) so it broadcasts over batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class TimestepEmbedding(nn.Module):
    """Embed scalar timestep t into a vector using sin/cos then an MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer timesteps -> (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device).float() * (-math.log(10000.0) / half)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization: LayerNorm modulated by a conditioning vector.

    Produces per-token scale (gamma) and shift (beta) from the timestep embedding:
        AdaLN(x, t) = gamma(t) * LayerNorm(x) + beta(t)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, 2 * dim)  # predict gamma and beta

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, N, D)
        t_emb: (B, D)
        """
        gamma, beta = self.proj(t_emb).unsqueeze(1).chunk(2, dim=-1)  # each (B, 1, D)
        return gamma * self.norm(x) + beta


class TransformerBlock(nn.Module):
    """AdaLN -> Self-attention -> AdaLN -> MLP, with residual connections.

    Timestep conditioning is injected via Adaptive Layer Normalization.
    """

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int):
        super().__init__()
        self.adaln1 = AdaLN(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.adaln2 = AdaLN(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, N, D) patch tokens
        t_emb: (B, D) timestep embedding
        """
        h = self.adaln1(x, t_emb)
        h, _ = self.attn(h, h, h)
        x = x + h

        h = self.adaln2(x, t_emb)
        h = self.mlp(h)
        x = x + h
        return x


class DiT(nn.Module):
    """Diffusion Transformer: predicts noise epsilon given (x_t, t).

    Pipeline:
        1. Patchify image into non-overlapping patches
        2. Flatten patches to vectors
        3. Linear embedding (affine projection to latent_dim)
        4. Add sin/cos positional embedding
        5. Pass through a stack of TransformerBlocks (conditioned on t)
        6. Output projection back to pixel space
        7. Unpatchify to reconstruct noise estimate
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.image_size = cfg["image_size"]
        self.channels = cfg["image_channels"]
        self.patch_size = cfg["patch_size"]
        self.latent_dim = cfg["latent_dim"]
        num_blocks = cfg["num_transformer_blocks"]
        num_heads = cfg["num_heads"]
        mlp_hidden = cfg["mlp_hidden_dim"]

        assert self.image_size % self.patch_size == 0, (
            "image_size must be divisible by patch_size"
        )

        self.num_patches_h = self.image_size // self.patch_size
        self.num_patches_w = self.image_size // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        patch_dim = self.channels * self.patch_size * self.patch_size

        # 3. Linear embedding (affine: learned weight + bias)
        self.patch_embed = nn.Linear(patch_dim, self.latent_dim)

        # 4. Positional embedding
        self.pos_embed = SinCosPositionalEmbedding(
            self.num_patches_h, self.num_patches_w, self.latent_dim
        )

        # Timestep embedding
        self.time_embed = TimestepEmbedding(self.latent_dim)

        # 5. Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.latent_dim, num_heads, mlp_hidden)
            for _ in range(num_blocks)
        ])

        # 6. Output projection back to pixel space
        self.norm_out = nn.LayerNorm(self.latent_dim)
        self.output_proj = nn.Linear(self.latent_dim, patch_dim)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_patches, patch_dim)"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, nH, nW, C, p, p)
        x = x.reshape(B, self.num_patches, -1)  # (B, N, C*p*p)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, num_patches, patch_dim) -> (B, C, H, W)"""
        B = x.shape[0]
        p = self.patch_size
        C = self.channels
        nH, nW = self.num_patches_h, self.num_patches_w

        x = x.reshape(B, nH, nW, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, nH, p, nW, p)
        x = x.reshape(B, C, nH * p, nW * p)
        return x

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: noisy image (B, C, H, W)
            t:   timestep indices (B,)

        Returns:
            epsilon_hat: predicted noise (B, C, H, W)
        """
        # 1-2. Patchify and flatten
        x = self.patchify(x_t)                  # (B, N, patch_dim)

        # 3. Linear embedding
        x = self.patch_embed(x)                  # (B, N, latent_dim)

        # 4. Add positional embedding
        x = self.pos_embed(x)                    # (B, N, latent_dim)

        # Timestep embedding
        t_emb = self.time_embed(t)               # (B, latent_dim)

        # 5. Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # 6. Output projection and unpatchify
        x = self.norm_out(x)
        x = self.output_proj(x)                  # (B, N, patch_dim)
        epsilon_hat = self.unpatchify(x)          # (B, C, H, W)

        return epsilon_hat

# Action Decoder

A from-scratch Diffusion Transformer (DiT) trained on CelebA faces. Built to understand and experiment with the DDPM diffusion pipeline end-to-end.

## Architecture

The model is a **Diffusion Transformer** that predicts noise (epsilon-prediction) using:

- **Patchify**: non-overlapping 16x16 patches flattened and linearly projected
- **Sin/cos positional embeddings** over the 2D patch grid
- **Timestep conditioning** via sinusoidal embedding + MLP, injected through Adaptive Layer Normalization (AdaLN)
- **Transformer blocks**: self-attention + MLP with residual connections
- **Unpatchify**: linear projection back to pixel space

Forward diffusion uses a standard linear beta schedule (1e-4 to 0.02 over 1000 timesteps).

## Project Structure

```
config.yaml          # Main config (image size, model dims, training params)
config_128.yaml      # 128x128 variant
config_64.yaml       # 64x64 variant
diffusion.py         # LinearDiffusion: forward process q(x_t | x_0)
ditmodel.py          # DiT model: patchify -> transformer -> unpatchify
dataset.py           # CelebA dataset loader with resize/crop/normalize
train.py             # Training loop with gradient accumulation
benchmark.py         # Timing benchmarks across configs
tools/
  unzip_data.py      # Extract CelebA zip into data/
  viz_diffusion.py   # Visualize forward diffusion (original | noised | noise)
  viz_denoising.py   # Visualize model predictions + DDPM generation from noise
```

## Setup

1. Place `img_align_celeba.zip` in `data/` and extract:
   ```
   python tools/unzip_data.py
   ```

2. Train:
   ```
   python train.py
   ```

3. Visualize results:
   ```
   python tools/viz_denoising.py
   ```

## Current Config

- **Image size**: 64x64
- **Patch size**: 16 (4x4 = 16 patches per image)
- **Latent dim**: 256
- **Transformer blocks**: 4
- **Heads**: 8
- **Effective batch size**: 256 (32 x 8 gradient accumulation steps)
- **Optimizer**: Adam, lr=1e-4
- **Epochs**: 100
- **Dataset**: CelebA (~202k images)

## Benchmarking

```
python benchmark.py
```

Compares training throughput across different transformer block counts. Used to find the speed/capacity tradeoff before committing to a long training run.

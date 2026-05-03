# Agent Development Log

Context for future agents working on this project.

## Session: 2026-05-02 / 2026-05-03

### What we built

Built a complete DDPM diffusion pipeline from scratch on CelebA faces:

1. **Forward diffusion** (`diffusion.py`): linear beta schedule, standard `q(x_t | x_0) = N(sqrt(a_bar) * x_0, (1 - a_bar) * I)`. Noise is isotropic Gaussian — independent per R, G, B channel and per pixel.

2. **DiT model** (`ditmodel.py`): Diffusion Transformer with patchify/unpatchify, sin/cos positional embeddings, AdaLN timestep conditioning, and configurable depth. Epsilon-prediction (model predicts the noise that was added).

3. **Training** (`train.py`): standard DDPM training with MSE loss on predicted vs actual noise. Added gradient accumulation (8 steps, effective batch size 256).

4. **Visualization tools**:
   - `tools/viz_diffusion.py` — shows forward noising process
   - `tools/viz_denoising.py` — shows model's denoising predictions + full DDPM reverse sampling to generate faces from scratch

### Key decisions and findings

- **Image size exploration**: started at 256x256, benchmarked down to 128 and 64. At 256x256 training would take ~13h on MPS. 128x128 drops to ~2-4h, 64x64 to ~1.7h. The speedup from 128->64 is marginal because at 16 patches the attention cost is negligible and you hit MPS kernel overhead floors.

- **Transformer depth**: benchmarked 1, 2, 4, 6, 8, 12 blocks at 64x64. With 1 block, loss plateaus at ~0.68 (barely better than trivial baseline of ~0.67-0.68). More blocks are needed for the model to learn meaningful spatial structure.

- **Compute tricks that didn't help on MPS**:
  - Pure fp16 crashes on MPS (LayerNorm dtype mismatch with MPS backend)
  - `torch.autocast` mixed precision works but gives no speedup — model is too small for fp16 matmul gains to matter
  - `torch.compile` fails on Python 3.12 + MPS (`collections.Mapping` import error)

- **Current config**: 64x64, 4 transformer blocks, trained with `dit_weights.pt` saved. Config was briefly set to 12 blocks for benchmarking but weights correspond to 4 blocks — **config.yaml must match the weights file**.

### MSE loss reference for DDPM noise prediction

- ~1.0 = random guess (predicting zeros)
- ~0.67-0.68 = trivial baseline (model learns mean/scale but no spatial structure)
- 0.3-0.5 = learning meaningful denoising
- 0.1-0.3 = good
- <0.1 = strong

### Environment

- macOS, Apple Silicon (MPS backend)
- Python 3.12 (Anaconda)
- PyTorch 2.7.0
- `pin_memory=True` warning on MPS is harmless (just means the flag is ignored)

### What's next

- Train with more blocks (6-12) now that we know the speed tradeoffs
- Try larger images once model architecture is validated at 64x64
- Consider learning rate scheduling, EMA, gradient clipping
- The model currently has no class conditioning or text conditioning — just unconditional face generation

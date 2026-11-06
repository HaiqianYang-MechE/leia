# Architecture

```
Mesh → PerceiverTokenizer.encode → K continuous latents
    → AdaLNContinuousDynamicsTransformer(tokens, action) → next tokens (latent-space AR)
    → PerceiverTokenizer.decode → displacement
                            ↘ stress_proj (Cauchy σ, sym6) OR grad_proj (∂u/∂x) → von Mises in eval
```

LEIA autoregression stays entirely in latent space — encode once at frame 0, predict all subsequent tokens, decode only at evaluation time. Per-step reconstruction error does not compound across the trajectory.

## `leia/` modules

| File | Purpose |
|---|---|
| `tokenizer.py` | `PerceiverTokenizer` — geometry + physics encoders, K continuous latents, decoder, `grad_proj` / `stress_proj` heads |
| `dynamics.py` | `AdaLNContinuousDynamicsTransformer` — FiLM-conditioned next-token predictor |
| `engine.py` | `LeiaEngine` — stateless inference primitives (encode / step / decode); canonical library entry point |
| `datasets.py` | `MmapMultiStructureDataset` (lattice tokenizer), `MmapWholePlateTrajectoryDataset` (dynamics, both regimes), `ViscoLoadReverseDataset` (visco tokenizer, reads HDF5) |
| `stress.py` / `stress_torch.py` | Tet shape function gradients, per-tet stress, boundary reaction force — numpy and differentiable-PyTorch versions |
| `losses.py` | Sobolev and stress-supervision losses used by `train_tokenizer.py` |
| `metrics.py` | Pure metric primitives — arrays in, numbers out (composed by `_eval_pipelines/`) |
| `_eval_pipelines/` | `eval_tokenizer`, `eval_e2e`, `gt_fd` — dispatched by `scripts/eval.py` |
| `training.py` | `save_checkpoint` / `load_checkpoint` (canonical layout, `weights_only=False`) |
| `distributed.py` | DDP setup / teardown / rank queries / all-reduce helpers |
| `regime.py` | Regime constants (μ, λ, action format) for `MICROPLATE_LATTICE` and `MICROPLATE_VISCO` |
| `config.py` | Project-wide paths and W&B project names; not model hyperparams (those live in `configs/*.yaml`) |
| `microgen.py` | Lattice graph features + GPU SDF plate meshing kernels |
| `render.py` | Generic numpy / pyvista / imageio render primitives |

`scripts/` are thin `main()` wrappers around `leia/` (training, eval, dataset prep). `configs/*.yaml` pin hyperparams per paper checkpoint.

"""Project-level constants — paths and W&B project names only.

Model architecture defaults (K, H, layer counts, etc.) are NOT here. Those
are paper-run-specific and live in `configs/*.yaml`. The split:
- This file: stable, project-wide; rarely changes.
- `configs/`: per-run pinned hyperparams; one YAML per paper checkpoint.
"""

# Paths (relative to repo root)
DATA_ROOT = "data"
TOKENIZER_CHECKPOINT_ROOT = "checkpoints/tokenizer"
DYNAMICS_CHECKPOINT_ROOT = "checkpoints/dynamics"
OUTPUT_ROOT = "outputs"

# W&B
WANDB_PROJECT_TOKENIZER = "leia-tokenizer"
WANDB_PROJECT_DYNAMICS = "leia-dynamics"

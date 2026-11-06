"""Single source of truth for the OOD-landscape reproduction layout.

A run root is a directory under ``design_search/outputs/`` that holds every
artifact for one reproduction of ``fig_ood_landscape.pdf``. All pipeline
scripts accept ``--run-root <path>`` and derive their inputs/outputs through
``resolve(run_root, key)``.

Convention:
    <run_root>/
      manifest.json
      pca.json
      heldout_latents.json
      walk/        candidates.json, leia_*, fem_*, npz outputs
      novelty/     same shape as walk/
      consistency/ consistency.jsonl
      ood/         dataset.npz, ckpts/{free,latent,full}/best.pt,
                   predictions_full.jsonl, eval_summary.md, calibration_*.png
      heldout/     held-out characterization (paper §5.4 numbers)
      figures/     fig_ood_landscape.{pdf,png} + 2x4 supplementary
      logs/        stageNN_<name>.log
"""

import os

_KEYS = {
    # Top-level files
    "manifest":               "manifest.json",
    "pca":                    "pca.json",
    "heldout_latents":        "heldout_latents.json",
    # Top-level subdirs
    "walk":                   "walk",
    "novelty":                "novelty",
    "consistency_dir":        "consistency",
    "consistency_jsonl":      "consistency/consistency.jsonl",
    "ood_dir":                "ood",
    "ood_dataset":            "ood/dataset.npz",
    "ood_ckpts_dir":          "ood/ckpts",
    "ood_ckpt_full":          "ood/ckpts/full/best.pt",
    "ood_ckpt_free":          "ood/ckpts/free/best.pt",
    "ood_ckpt_latent":        "ood/ckpts/latent/best.pt",
    "ood_predictions":        "ood/predictions.jsonl",
    "ood_predictions_full":   "ood/predictions_full.jsonl",
    "ood_eval_summary":       "ood/eval_summary.md",
    "heldout_dir":            "heldout",
    "heldout_eval":           "heldout/heldout_eval.md",
    "heldout_predictions":    "heldout/heldout_predictions.jsonl",
    "figures_dir":            "figures",
    "logs_dir":               "logs",
}


def resolve(run_root: str, key: str) -> str:
    """Return the absolute path of ``key`` within ``run_root``.

    Raises KeyError if the key is unknown — this is a tight whitelist so
    typos surface immediately rather than producing silent path bugs.
    """
    if key not in _KEYS:
        raise KeyError(f"unknown layout key: {key!r}; known keys: {sorted(_KEYS)}")
    return os.path.join(run_root, _KEYS[key])


def ensure_dirs(run_root: str) -> None:
    """Create the standard subdir skeleton inside ``run_root``."""
    for sub in ("walk", "novelty", "consistency", "ood", "ood/ckpts",
                "heldout", "figures", "logs"):
        os.makedirs(os.path.join(run_root, sub), exist_ok=True)


def add_run_root_arg(parser, help_text: str | None = None) -> None:
    """Helper: add a standard ``--run-root`` arg to an argparse parser."""
    parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help=help_text or "Reproduction run root; if set, paths are derived "
                          "from layout (overrides per-stage path flags).",
    )

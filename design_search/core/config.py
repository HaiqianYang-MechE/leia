"""Configuration for the LEIA design-search runtime."""

from dataclasses import dataclass, field

from leia.regime import MICROPLATE_LATTICE


@dataclass
class ServingConfig:
    # Production v2: stress-head tokenizer + AdaLN continuous dynamics
    # (velocity_input). Paths are the production_v2 symlink slots that
    # scripts/promote.sh fills (README § 4).
    tokenizer_ckpt: str = "checkpoints/tokenizer/production_v2/best.pt"
    dynamics_ckpt: str = "checkpoints/dynamics/production_v2/best_ar.pt"

    # Material + loading speeds — sourced from
    # data/manifests/microplate_lattice_constants.json via leia.regime.
    mu: float = MICROPLATE_LATTICE.mu
    lmbda: float = MICROPLATE_LATTICE.lmbda
    speeds: tuple = field(default_factory=lambda: MICROPLATE_LATTICE.speeds)

    # Plate geometry — tiles come from the manifest (data invariant);
    # mesh_res is a runtime choice (interactive uses 10, training used 30).
    tiles: tuple = field(default_factory=lambda: MICROPLATE_LATTICE.tiles)
    mesh_res: int = 10

    # Physics
    action_scale: float = 1.0

    # Inference device
    device: str = "cuda:0"

    # OOD detection threshold on tokenizer recon error.
    # See `design_search/landscape/ood/pipeline.py` and `design_search/landscape/score.py`.
    ood_threshold: float = 0.0001

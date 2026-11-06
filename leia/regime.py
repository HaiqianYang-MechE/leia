"""Regime constants — single source of truth for material and loading-protocol
parameters that aren't encoded in the dataset itself.

Two regimes, one manifest each under `data/manifests/`:

- `MICROPLATE_LATTICE` — Neo-Hookean lattice plates. The per-trajectory NPZ
  files don't carry the constitutive constants, so every runtime consumer
  (stress functions, dataset classes, eval pipelines, FEM driver,
  design-search) reads (mu, lmbda, speeds) from the manifest via this module.

- `MICROPLATE_VISCO` — Arruda-Boyce + 3-branch viscoelastic plate. LEIA
  itself uses the stress head on this regime (predicts Cauchy stress
  directly), so the runtime path here doesn't need constitutive constants.
  This constant is exposed for completeness and for evaluators that compute
  stress from predicted internal variables instead of the stress head.
  Names mirror the HDF5 attrs on the `solid_plate_visco` group of
  microplate_visco.h5.
"""
import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitSpec:
    """Dataset partitioning convention for a regime.

    `training_plates` and `heldout_plates` are the published plate-ID rosters.
    `training_cutoff_index` is the per-plate trajectory cut: training used
    `sorted(traj_files)[:training_cutoff_index]`; everything beyond that is a
    proper in-distribution val trajectory.

    `random_split_seed` and `random_val_ratio` describe the within-mmap 90/10
    permutation used by the `Mmap*` dataset classes during training.
    """
    training_plates: tuple
    heldout_plates: tuple
    training_cutoff_index: int
    random_split_seed: int
    random_val_ratio: float


@dataclass(frozen=True)
class LatticeRegime:
    mu: float
    lmbda: float
    speeds: tuple
    dof_names: tuple
    num_steps: int
    tiles: tuple
    splits: SplitSpec


@dataclass(frozen=True)
class ViscoRegime:
    Geq_0: float
    lambdaL: float
    Kbulk: float
    rho: float
    branch_G: tuple
    branch_tau: tuple
    speeds: tuple
    dof_names: tuple
    protocol: str
    num_steps: int
    n_forward_steps: int
    stride: int


_MANIFEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'manifests',
)
MANIFEST_DIR = _MANIFEST_DIR   # public: where the committed manifests live


def _load_lattice_splits() -> SplitSpec:
    with open(os.path.join(_MANIFEST_DIR, 'microplate_lattice_production_plates_v2.json')) as f:
        training_plates = tuple(json.load(f)['plates'])
    with open(os.path.join(_MANIFEST_DIR, 'microplate_lattice_heldout_plates.json')) as f:
        heldout_plates = tuple(json.load(f)['plates'])
    with open(os.path.join(_MANIFEST_DIR, 'microplate_lattice_indist_val_trajs.json')) as f:
        cutoff = int(json.load(f)['training_cutoff_index'])
    # seed=42 / val_ratio=0.1 are the within-mmap permutation convention
    # bound to the published v2 tokenizer + dynamics checkpoints. Not on
    # disk — they're a codebase release constant. Override at the call
    # site when ablating.
    return SplitSpec(
        training_plates=training_plates,
        heldout_plates=heldout_plates,
        training_cutoff_index=cutoff,
        random_split_seed=42,
        random_val_ratio=0.1,
    )


def _load_lattice() -> LatticeRegime:
    with open(os.path.join(_MANIFEST_DIR, 'microplate_lattice_constants.json')) as f:
        d = json.load(f)
    return LatticeRegime(
        mu=float(d['material']['mu']),
        lmbda=float(d['material']['lmbda']),
        speeds=tuple(d['bc_speeds']),
        dof_names=tuple(d['bc_dof_names']),
        num_steps=int(d['num_steps']),
        tiles=tuple(d['tiles']),
        splits=_load_lattice_splits(),
    )


def _load_visco() -> ViscoRegime:
    with open(os.path.join(_MANIFEST_DIR, 'microplate_visco_constants.json')) as f:
        d = json.load(f)
    m = d['material']
    return ViscoRegime(
        Geq_0=float(m['Geq_0']),
        lambdaL=float(m['lambdaL']),
        Kbulk=float(m['Kbulk']),
        rho=float(m['rho']),
        branch_G=tuple(m['branch_G']),
        branch_tau=tuple(m['branch_tau']),
        speeds=tuple(d['bc_speeds']),
        dof_names=tuple(d['bc_dof_names']),
        protocol=str(d['protocol']),
        num_steps=int(d['num_steps']),
        n_forward_steps=int(d['n_forward_steps']),
        stride=int(d['stride']),
    )


MICROPLATE_LATTICE = _load_lattice()
MICROPLATE_VISCO = _load_visco()

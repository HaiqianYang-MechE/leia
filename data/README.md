# Data

What's in git, what's generated locally.

## Committed (small specs, < 1 MB total)

| Path | Purpose |
|---|---|
| `manifests/microplate_lattice_production_plates_v2.json` | 55 training plate IDs (MicroPlate lattice) |
| `manifests/microplate_lattice_heldout_plates.json` | 8 held-out / OOD plate IDs |
| `manifests/microplate_lattice_indist_val_trajs.json` | per-plate val trajectory IDs of the paper build (only `training_cutoff_index` is read) |
| `manifests/microplate_visco_splits.json` | train/val/test split for MicroPlate visco |
| `rve_bank/mg_*/graph.json` | Plate graph skeletons (63 plates, ~60 KB) |

## Generated locally by `data_curation/` (~1.2 TB total)

No bulk artifacts are hosted. Meshes and FEM trajectories are built with the
pipelines in [`../data_curation/`](../data_curation/README.md); checkpoints
are trained from them (see [`../README.md`](../README.md)). Expected on-disk
layout:

| Artifact | Local layout | Size |
|---|---|---|
| Tet meshes | `data/meshes/microplate_lattice/mg_*/mesh.npz` | ~4.3 GB |
| Lattice FEM trajectories (NPZ) | `data/microplate_lattice/npz/` | ~1.2 TB |
| Visco FEM trajectories (HDF5) | `data/microplate_visco/microplate_visco.h5` | ~5 GB |

## Derived caches

Pure derivatives of the FEM core. Run before training (or before stress eval,
which reads the tokenizer mmap):

| Output | Script | Time | Needed for |
|---|---|---|---|
| `data/microplate_lattice/fps/` | `scripts/precompute_fps.py` | ~1 h | tokenizer mmap, eval |
| `data/microplate_lattice/grad_u/` | `scripts/precompute_grad_u.py` | ~1-2 h | tokenizer mmap, e2e GT stress |
| `data/microplate_lattice/plate_tet_geom/` | `scripts/precompute_tet_geom.py` | ~30 min | tet-sobolev ablation row |
| `data/microplate_lattice/plate_dataset_curated_mmap_100k/` | `scripts/build_tokenizer_mmap.py` | ~1 h | tokenizer training, stress eval |
| `data/microplate_lattice/structure_tokens_prod_mmap/` | `scripts/build_dynamics_mmap.py` | ~30 min, after tokenizer training | dynamics training |

See [`../README.md`](../README.md) for the end-to-end build → train → eval flow.

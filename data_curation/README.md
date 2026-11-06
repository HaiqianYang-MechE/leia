# Data Curation

FEM ground-truth generation + canonical mesh build. Run **once** to produce raw
HDF5 / NPZ trajectories and the canonical tet meshes; after that, training and
evaluation operate on the resulting data via `scripts/`.

No bulk artifacts are hosted, so this directory is the first step for every
user: it produces the canonical tet meshes and the FEM trajectories that
`scripts/` train and evaluate against (see [`../data/README.md`](../data/README.md)
for the expected on-disk layout and sizes). The same pipelines serve
regeneration for a different material regime or a refreshed FEM solver.

## Subdirectories

| Subdir | Env | Purpose |
|---|---|---|
| `mesh/` | `leia` | Graph → tet mesh via TetGen + GPU SDF subprocess |
| `fem/`  | `leia` (lattice; FEniCS 2019.1 ships in it) or `fenicsx` (visco; FEniCSx 0.8) | Run FEM on the built meshes |

## Environments

Run from the repo root:

```bash
# 1) Build canonical meshes (main PyTorch env; CUDA GPU for the SDF kernel).
conda activate leia
python data_curation/mesh/build_meshes.py --all

# 2) Run lattice plate FEM (FEniCS / dolfin 2019.1; co-resident in `leia`).
#    Every training plate needs >= 101 converged trajectories (see ../README.md § 1);
#    divergence rates vary by topology, so request many and stop at a target.
PYTHONHASHSEED=0 python data_curation/fem/run_plate_fem.py \
    --n-meshes 0 --trajs-per-mesh 500 --target-converged 110 \
    --mu 1.0 --lmbda 10.0 \
    --workers 100 \
    --output-dir data/microplate_lattice

# 3) Run MicroPlate visco load-reverse FEM (FEniCSx / dolfinx 0.8):
conda env create -f environment-fenicsx.yml   # one-time
conda activate fenicsx
python data_curation/fem/run_visco_load_reverse.py fem --batch 1100 --workers 100

# 4) Package per-seed NPZ → single HDF5 + split JSON (no FEniCSx needed):
conda activate leia
python data_curation/fem/run_visco_load_reverse.py build-hdf5 \
    --source-dir data/visco_load_reverse_production \
    --output-h5 data/microplate_visco/microplate_visco.h5 \
    --split-json data/manifests/microplate_visco_splits.json
```

## Mesh determinism

The plate graphs in `data/rve_bank/mg_*/graph.json` are the canonical topology
spec; `python data_curation/mesh/build_meshes.py --all` builds the tet meshes from them.
TetGen is deterministic for a given input, but the GPU SDF / marching-cubes
stage is float32 and its rounding depends on the CUDA compiler, driver and GPU,
so on a different setup a few plates may mesh with slightly different
connectivity. Nothing downstream depends on an exact match — every later step
derives from the meshes you build.

## Material regimes

**MicroPlate lattice** uses Neo-Hookean elasticity with `μ=1`, `λ=10`, passed
at the command line via `--mu` / `--lmbda`. Wrong values cause a silent
~400× stress error in any consumer that recomputes stress from gradient
(`scripts/eval.py e2e`, design_search FEM validation, etc.).

**MicroPlate visco** uses an Arruda-Boyce 3-branch viscoelastic model with
`Geq=200`, `G=[300, 600, 150]`, `τ=[0.001, 0.2, 3.0]`. These are baked into
`fem/run_visco_load_reverse.py` and the dataset HDF5 attrs; no command-line
flags. LEIA's stress head predicts Cauchy directly, so visco eval has no
equivalent of the lattice μ/λ pitfall.

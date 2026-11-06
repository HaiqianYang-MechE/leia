"""LEIA physics kernel: plate meshing (`physics.meshing`) and the FEM
solvers. `physics.fem.neohookean` (legacy FEniCS) is shared by
`data_curation/fem/run_plate_fem.py` (training data) and
`design_search/core/fem.py` (on-the-fly characterization);
`physics.fem.visco` (FEniCSx) serves `data_curation/fem/run_visco_load_reverse.py`.

Sibling runner package to `data_curation/` and `design_search/`. Excluded from
`pip install -e .` (see `pyproject.toml`); imported with `PYTHONPATH=.` from
the repo root or as `python -m physics.…`.

Heavy FEniCS / FEniCSx imports are confined to `physics.fem.neohookean.*`
and `physics.fem.visco.*`; importing `physics` itself loads neither.
"""

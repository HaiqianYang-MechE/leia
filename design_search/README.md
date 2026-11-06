# `design_search/` — Surrogate-guided design search and OOD detection

This subpackage drives Figure 4 (surrogate-guided beam search over lattice
topologies, paper §5.3) and Figure 5 (OOD-confidence landscape and learned
confidence head, paper §5.4) of the LEIA paper. It builds on the production
v2 tokenizer + dynamics checkpoints to score candidate microstructure designs
in real time, and trains a small MLP head to predict surrogate reliability
on out-of-distribution topologies without FEM.

End-to-end reproduction orchestrators live in [`repro/`](repro/); the
sections below describe the kernel API and per-stage CLIs.

## Layout

```
design_search/
├── core/        # Stable kernel — surrogate-eval primitives
│   ├── config.py        # ServingConfig (paths to v2 ckpts, mu/lmbda, speeds)
│   ├── engine.py        # ModelEngine + Session + step_batch
│   ├── meshing.py       # graph→mesh subprocess (SDF → marching cubes → TetGen)
│   ├── fem.py           # FEMWorkerPool + solve_characterization (Neo-Hookean batch FEM)
│   ├── graph.py         # graph primitives + mutation operators + samplers
│   ├── objectives.py    # volume_fraction, compute_work, anisotropy_score
│   └── characterize.py  # sync_characterize, mesh_candidates_parallel,
│                        # load_seed_graph, save/load_checkpoint
│
├── beam/        # Figure 4 — surrogate-guided design search
│   ├── search.py        # beam-search loop + candidate JSON
│   ├── fem_validate.py  # FEM ground truth on best-per-iter (panel b)
│   ├── bench.py         # per-candidate isolated timing (panel d)
│   └── figure.py        # multi-panel figure assembly
│
├── landscape/   # Figure 5 — OOD-confidence landscape
│   ├── walk.py          # random-walk candidate generation
│   ├── novelty.py       # novelty-search candidate generation
│   ├── score.py         # per-candidate LEIA score / consistency (--metric flag)
│   ├── fem.py           # batch FEM on candidates (panel A ground truth)
│   ├── pca.py           # fit-pca / encode-heldout (subcommands)
│   ├── heldout.py       # held-out characterization sub-pipeline
│   ├── layout.py        # run-root path SoT
│   ├── figure.py        # 4-panel figure (recon · model · consistency · pred)
│   └── ood/             # Confidence-head sub-pipeline
│       ├── head.py      # OODConfidenceHead model
│       └── pipeline.py  # build / train / eval / predict subcommands
│
└── repro/       # Orchestrator shells
    ├── figure4.sh       # 5-stage Figure 4 reproduction
    ├── figure5.sh       # 15-stage Figure 5 reproduction
    └── _fem_watchdog.sh # FEM-pool RAM watchdog (used by figure5.sh)
```

## Public API

`core/` exposes the surrogate-eval primitives consumed by both `beam/` (Figure 4) and `landscape/` (Figure 5). See `core/__init__.py` for the full `__all__`.

## Running individual stages

Each subpackage exposes module-level CLIs:

```bash
# beam/
python -m design_search.beam.search --seed mg_046 --B 5 --K 8 --D 10 --output-dir out/
python -m design_search.beam.fem_validate --beam-results out/results.json --output-dir out/fem/
python -m design_search.beam.bench --beam-results out/results.json --output-dir out/bench/   # LEIA and FEM timed one candidate at a time, at the run's mesh resolution (the paper plots lines across resolutions)
python -m design_search.beam.figure figure --beam-results out/results.json --output out/figure4.pdf

# landscape/
python -m design_search.landscape.walk --run-root out/
PYTHONHASHSEED=0 python -m design_search.landscape.novelty --run-root out/
python -m design_search.landscape.score --metric=score --run-root out/ --stage walk
python -m design_search.landscape.score --metric=consistency --run-root out/
python -m design_search.landscape.fem --run-root out/ --stage walk
python -m design_search.landscape.pca fit-pca --run-root out/
python -m design_search.landscape.pca encode-heldout --run-root out/
python -m design_search.landscape.heldout --run-root out/
python -m design_search.landscape.ood.pipeline build --run-root out/
python -m design_search.landscape.ood.pipeline train --run-root out/
python -m design_search.landscape.ood.pipeline eval --run-root out/
python -m design_search.landscape.ood.pipeline predict --run-root out/
python -m design_search.landscape.figure pca --run-root out/
```

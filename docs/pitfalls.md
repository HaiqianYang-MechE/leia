# Pitfalls

## sym6 ordering — Voigt convention

`stress_sym6 = [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]`. Never reorder.

## `coord_scale` convention (j-axis)

`grad_u_norm[i, j] = phys[i, j] * coord_scale[j] / disp_scale` — j-axis convention is current. GT `grad_u` from mmap is already in physical space — do NOT multiply by `coord_scale`.

## Material constants

For lattice (Neo-Hookean) eval, always pass the material constants matching the FEM dataset (`--stress-mu` / `--stress-lmbda` for `eval.py tokenizer` and `e2e`, `--mu` / `--lmbda` for `eval.py gt-fd`). Canonical values: [`microplate_lattice_constants.json`](../data/manifests/microplate_lattice_constants.json). Visco eval has no such flag — the stress head predicts Cauchy directly.

## Gradient head vs. stress head — never conflate

- `grad_proj` predicts `∂u/∂x` (9 components, Neo-Hookean only).
- `stress_proj` predicts Cauchy sym6 (6 components, universal across constitutive laws).

## Parallel FEM

`OMP_NUM_THREADS` before numpy import. Set `os.environ["OMP_NUM_THREADS"]="1"` at module top, **before** `import numpy`. Otherwise numpy spawns N threads per process and 100-worker FEM pools exhaust the CPU.

`kill <pid>` on a parent process does NOT kill `ProcessPoolExecutor` workers. Use `kill -- -<pgid>` to take down the whole process group. Otherwise zombie FEM workers hold memory and consume RAM.

## Checkpoint selection: use `best_ar.pt`, not `val_loss`

`best.pt` (lowest single-step val MSE) and `best_ar.pt` (lowest autoregressive rollout MSE on the last 5 trajectories of the first training plate — a rollout-stability proxy, not a held-out metric: those trajectories are also in the training windows) can disagree. Production uses `best_ar.pt` — the dynamics model that minimizes one-step error often drifts under long rollouts.
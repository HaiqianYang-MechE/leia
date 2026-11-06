# tests/

Smoke and preview scripts for the release, in two tiers on a cost ladder:

| Prefix | Cost | Output | When to run |
|--------|------|--------|-------------|
| `smoke_*` | seconds | none (pass/fail) | after every change |
| `preview_*` | seconds–minutes | `tests/_artifacts/` (gitignored) | before a release; sanity-check install |

Both write only inside `tests/_artifacts/`, except `preview_train_small.sh`,
which briefly writes `checkpoints/{tokenizer,dynamics}/preview_*` and removes
them on exit. Run `bash tests/clean.sh` to
wipe all preview outputs.

Paper-grade reproduction lives elsewhere — the training / eval recipes in
[`../README.md`](../README.md) and the figure orchestrators in
[`../design_search/repro/`](../design_search/repro/) take hours to days.

## smoke_* — pass/fail (seconds–minutes)

| Script | What it checks |
|--------|----------------|
| `smoke_e2e.sh` | `import leia` + every `scripts/*.py --help` works |
| `smoke_tokenizer.sh` | Tiny PerceiverTokenizer: build, save, reload, forward |
| `smoke_dynamics.sh` | Tiny dynamics model + load production v2 dyn checkpoint |
| `smoke_consumers.sh` | `score.py --metric={score,consistency}` past first inference (catches stale-API bugs that bypass `--help`) |
| `smoke_eval.sh` | `eval.py {e2e,tokenizer,gt-fd}` on 1 plate × 1 frame |

## preview_* — artifact-producing (minutes)

| Script | Output | Inspect |
|--------|--------|---------|
| `preview_meshing.sh` | 3 plates → `.vtu` + `.png` + stats `.json` | `paraview tests/_artifacts/meshing/mg_046.vtu` |
| `preview_fem.sh` | 1 plate → MP4 + 3-frame `.vtu` series, from **stored** trajectory | `mpv tests/_artifacts/fem/*.mp4` |
| `preview_fem_solve.sh` | mesh → live `solve_neo_hookean` (5 steps) → MP4 — exercises the dolfin / FEniCS 2019.1 kernel end-to-end | `mpv tests/_artifacts/fem_solve/*.mp4` |
| `preview_visco.sh` | 1 visco trajectory → MP4 + 3-frame `.vtu`, from **stored** HDF5 | `mpv tests/_artifacts/visco/*.mp4` |
| `preview_visco_solve.sh` | live `solve_arruda_boyce` (3 steps × 2 sub) → MP4 — exercises the dolfinx 0.8 kernel end-to-end | `mpv tests/_artifacts/visco_solve/*.mp4` |
| `preview_dashboard.sh` | 1 plate → GT-vs-LEIA dashboard MP4 | `mpv tests/_artifacts/dashboard/*.mp4` |
| `preview_train_small.sh` | 200-step tokenizer + 5-epoch dynamics, asserts loss decreased | `cat tests/_artifacts/train_small/summary.json` |

`preview_dashboard.sh`, `preview_train_small.sh`, and the smoke scripts
that need them all gate on `checkpoints/{tokenizer,dynamics}/production_v2/`.
Each script skips with a clear message (exit code 77) if missing — train and
promote them first (see [`../README.md § 4. Evaluate`](../README.md#4-evaluate)).
`release_qualify.sh` runs every script even when an earlier one fails, counts
skipped scripts separately, and lists failures and skips at the end; a run in
which everything skipped is not a pass.

`preview_fem_solve.sh` runs in the main `leia` env — FEniCS 2019.1 / dolfin
is bundled with PyTorch (see `environment.yml`).

`preview_visco_solve.sh` needs the separate **`fenicsx` env** (dolfinx 0.8):
`conda env create -f environment-fenicsx.yml`. Skips cleanly if missing.

## Drivers

```bash
bash tests/release_qualify.sh    # all smoke_* + preview_* (~20 min)
bash tests/clean.sh              # wipe tests/_artifacts/
```

Every script activates its conda env through `tests/_conda_activate.sh`, which
uses the `conda` on `PATH` (no hardcoded install location) and is a no-op when
the env is already active.

## Production training

End-to-end retrain from scratch follows
[`../README.md § 3. Train`](../README.md#3-train).
Those are the canonical reproduction paths for the paper checkpoints —
multi-hour runs on 8×H100, run explicitly, not as part of `release_qualify.sh`.

`preview_train_small.sh` is the automated sanity-check for the loss /
dataloader / optimizer wiring on a 3-plate subset; the production
recipes are the same `scripts/launch.sh train-{tokenizer,dynamics}`
plumbing with paper-grade flags.

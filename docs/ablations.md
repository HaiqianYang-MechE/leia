# Table 1 ablation — stress-supervision rows

Paper Table 1 compares six stress-supervision variants on MicroPlate lattice.
Each row is a paired tokenizer + dynamics model evaluated by AR rollout on two
splits: in-distribution (val) and held-out (OOD). Row 6 is production v2 itself
(`configs/microplate_lattice/`); rows 1–5 are trained with the recipe below.

## Rows

| Row | Variant | Tokenizer supervision flags | VM metric key |
|---|---|---|---|
| 1 | no supervision (recon only) | `--sobolev-beta 0.0 --cauchy-stress-weight 0.0` | `disp_vm_corr_median` |
| 2 | tet Sobolev (volume-weighted gradient MSE) | `--sobolev-method tet --sobolev-beta 3.0 --tet-geom-dir data/microplate_lattice/plate_tet_geom --cauchy-stress-weight 0.0` | `disp_vm_corr_median` |
| 3 | autograd Sobolev (Jacobian-based; slowest) | `--sobolev-method autograd --sobolev-beta 3.0 --cauchy-stress-weight 0.0` | `disp_vm_corr_median` |
| 4 | gradient head (Neo-Hookean only) | `--sobolev-method grad-head --sobolev-beta 30.0 --cauchy-stress-weight 0.0` | `grad_head_vm_corr_median` |
| 5 | stress head, no pushforward | `--stress-head --cauchy-stress-weight 1.0 --cauchy-stress-warmup-steps 10000` | `head_vm_corr_median` |
| 6 | stress head + pushforward (production v2: pf=10, velocity_input) | see `configs/microplate_lattice/tokenizer_v2.yaml` | `head_vm_corr_median` |

VM metric per Appendix §A.2 of the paper: rows 1–3 have no head, so von Mises
stress is recovered post hoc by tet-shape-function differentiation of the
predicted displacement (`disp_vm_corr`); row 4 uses the gradient head's ∂u/∂x
through the constitutive law (`grad_head_vm_corr`); rows 5–6 predict σ directly
(`head_vm_corr`).

## Training (rows 1–5)

Pick a row and its supervision flags from the table:

```bash
ROW=row1_no_supervision                                      # row name, used in the run names below
ROW_FLAGS="--sobolev-beta 0.0 --cauchy-stress-weight 0.0"    # that row's flags from the table
```

Tokenizer — shared flags plus the row's supervision flags (~4 hours per row on
8×H100). Row 2 needs the tet-geometry cache from `scripts/precompute_tet_geom.py`
(see [`../README.md § 2. Precompute caches`](../README.md#2-precompute-caches)).

```bash
bash scripts/launch.sh train-tokenizer --gpus 8 --run-name ablation-$ROW -- \
    --mmap-dir data/microplate_lattice/plate_dataset_curated_mmap_100k \
    --num-latents 512 --hidden-dim 1024 \
    --num-encoder-layers 6 --num-decoder-layers 0 \
    --n-input-frames 1 \
    --batch-size 8 --accum-steps 2 \
    --lr 1.0e-4 --weight-decay 1.0e-5 \
    --max-steps 30000 --val-interval 1000 --checkpoint-interval 5000 \
    --lr-schedule cosine \
    --bf16 --wandb \
    $ROW_FLAGS
```

Dynamics — short rollout (`--rollout-steps 2`, against 10 for the production
model), one per row, on a structure-token
mmap built with the row's tokenizer:

```bash
TOK=checkpoints/tokenizer/ablation-${ROW}_*/best.pt
MMAP=data/microplate_lattice/structure_tokens_${ROW}_mmap

python scripts/build_dynamics_mmap.py \
    --tokenizer $TOK \
    --npz-dir data/microplate_lattice/npz \
    --fps-dir data/microplate_lattice/fps \
    --curated-plates data/manifests/microplate_lattice_production_plates_v2.json \
    --output-mmap-dir $MMAP

bash scripts/launch.sh train-dynamics --gpus 8 --run-name ablation-$ROW-dyn -- \
    --tokenizer-checkpoint $TOK \
    --mmap-dir $MMAP \
    --method continuous_pf --rollout-steps 2 --batch-size 8 \
    --epochs 200 --checkpoint-interval 10 --bf16 --wandb
```

## Evaluation (all six rows)

AR rollout per split. In-distribution: the 55 production plates, first
post-training trajectory (`--traj-indices "100" --max-trajs 101`). Held-out:
the 8 OOD plates, first trajectory (`--traj-indices "0" --max-trajs 1`).
For row 6 use `checkpoints/tokenizer/production_v2/best.pt` and
`checkpoints/dynamics/production_v2/best_ar.pt`.

```bash
TOK=checkpoints/tokenizer/ablation-${ROW}_*/best.pt
DYN=checkpoints/dynamics/ablation-${ROW}-dyn_*/best_ar.pt

# in-distribution (val)
python scripts/eval.py e2e \
    --tokenizer $TOK \
    --dynamics  $DYN \
    --npz-dir       data/microplate_lattice/npz \
    --fps-dir       data/microplate_lattice/fps \
    --grad-u-dir    data/microplate_lattice/grad_u \
    --mesh-dir      data/meshes/microplate_lattice \
    --curated-plates data/manifests/microplate_lattice_production_plates_v2.json \
    --max-trajs 101 --traj-indices "100" \
    --eval-frames "5,10,15,20,25,30" \
    --subsample-n 10240 --fps-core-n 8192 \
    --stress-mu 1.0 --stress-lmbda 10.0 \
    --full-mesh \
    --output-dir outputs/table1_ablation/$ROW/val

# held-out (ood): same, with
#   --curated-plates data/manifests/microplate_lattice_heldout_plates.json \
#   --max-trajs 1 --traj-indices "0" \
#   --output-dir outputs/table1_ablation/$ROW/ood
```

Each run writes `*_e2e_results.json`. Table 1 reports, per split, from the
`overall` block: `disp_rel_l2_median` (×100, lower is better) and the row's VM
metric key from the table above (higher is better).

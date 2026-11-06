#!/usr/bin/env python
"""Single-frame gallery snapshot for quick style iteration.

The first run computes the plate rollouts and pickles them under outputs/;
later runs load that cache, so re-rendering after a style tweak takes seconds
instead of minutes. Pass --no-cache (or delete the pickle) after changing the
checkpoints or plates, since the cache is not keyed on them.

Renders both the dynamics gallery and (optionally) the unit-cell gallery
at one frame, saved as PNG.

Usage:
    python scripts/render_snapshot.py                    # last frame
    python scripts/render_snapshot.py --frame 15         # mid-rollout
    python scripts/render_snapshot.py --no-cache         # force fresh
    python scripts/render_snapshot.py --unit-cells       # also render cells
"""

import os
import sys
import argparse
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from leia.engine import LeiaEngine
from leia.render import fit_camera_to_envelope, compose_grid
from render_microplate_gallery import (
    DEFAULT_PLATES,
    precompute_plate_rollout,
    render_dynamics_panel, render_unit_cell_panel,
    precompute_unit_cells,
)

CACHE = 'outputs/_gallery_snapshot_cache.pkl'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--frame', type=int, default=30,
                   help='Frame index to render (default: 30, the last frame)')
    p.add_argument('--no-cache', action='store_true',
                   help='Force fresh rollout instead of loading from cache')
    p.add_argument('--unit-cells', action='store_true',
                   help='Also render the unit-cell snapshot')
    p.add_argument('--out', default='outputs/microplate_gallery/snapshot.png')

    p.add_argument('--tokenizer', default='checkpoints/tokenizer/production_v2/best.pt')
    p.add_argument('--dynamics', default='checkpoints/dynamics/production_v2/best_ar.pt')
    p.add_argument('--npz-dir', default='data/microplate_lattice/npz')
    p.add_argument('--fps-dir', default='data/microplate_lattice/fps')
    p.add_argument('--mesh-dir', default='data/meshes/microplate_lattice')
    p.add_argument('--rve-bank', default='data/rve_bank')
    p.add_argument('--panel-size', type=int, default=720)
    p.add_argument('--text-font-size', type=int, default=10)
    p.add_argument('--subsample-n', type=int, default=10240)
    p.add_argument('--fps-core-n', type=int, default=8192)
    p.add_argument('--stress-mu', type=float, default=1.0)
    p.add_argument('--stress-lmbda', type=float, default=10.0)
    p.add_argument('--smooth-iters', type=int, default=3)
    p.add_argument('--smooth-alpha', type=float, default=0.5)
    p.add_argument('--unit-cell-res', type=int, default=50)
    p.add_argument('--rotation-degrees', type=float, default=360.0)
    p.add_argument('--traj-idx', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def load_or_compute(args, device):
    if os.path.exists(CACHE) and not args.no_cache:
        print(f"[cache] loading {CACHE}")
        with open(CACHE, 'rb') as f:
            return pickle.load(f)

    print("[fresh] rolling out 8 plates...")
    engine = LeiaEngine(tokenizer_ckpt=args.tokenizer,
                        dynamics_ckpt=args.dynamics, device=str(device))
    plate_data = []
    for pid in DEFAULT_PLATES:
        print(f"  {pid}")
        pd = precompute_plate_rollout(pid, args, engine, device)
        plate_data.append(pd)

    cams = [fit_camera_to_envelope(pd['env_min'], pd['env_max'],
                                    args.panel_size, zoom=0.85)
            for pd in plate_data]

    cells = {}
    if args.unit_cells:
        print("[fresh] meshing 8 unit cells...")
        cells = precompute_unit_cells(DEFAULT_PLATES, args.rve_bank,
                                       args.unit_cell_res)

    bundle = {'plate_data': plate_data, 'cams': cams, 'cells': cells}
    print(f"[cache] saving {CACHE}")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(bundle, f)
    return bundle


def main():
    args = parse_args()
    device = torch.device(args.device)
    bundle = load_or_compute(args, device)
    plate_data = bundle['plate_data']
    cams = bundle['cams']
    cells = bundle.get('cells', {})

    fi = args.frame
    ps = args.panel_size

    print(f"[render] dynamics frame {fi}")
    panels = []
    for pi, pd in enumerate(plate_data):
        panels.append(render_dynamics_panel(
            pd['coords'], pd['topo'], pd['disps'][fi],
            pd['vm_nodes'][fi], pd['vmax'], cams[pi], ps,
            pd['action'][fi], args.text_font_size))
    frame = compose_grid(panels, 2, 4)
    out_dyn = args.out.replace('.png', '_dynamics.png')
    os.makedirs(os.path.dirname(out_dyn) or '.', exist_ok=True)
    imageio.imwrite(out_dyn, frame)
    print(f"saved {out_dyn}  ({frame.shape[1]}x{frame.shape[0]})")

    if args.unit_cells and cells:
        n_frames = len(plate_data[0]['disps'])
        theta = args.rotation_degrees * fi / max(n_frames - 1, 1)
        print(f"[render] unit-cell frame {fi} (theta={theta:.0f}°)")
        cell_panels = [render_unit_cell_panel(cells[pid], theta, ps)
                       for pid in DEFAULT_PLATES]
        cell_frame = compose_grid(cell_panels, 2, 4)
        out_cell = args.out.replace('.png', '_unit_cells.png')
        imageio.imwrite(out_cell, cell_frame)
        print(f"saved {out_cell}  ({cell_frame.shape[1]}x{cell_frame.shape[0]})")


if __name__ == '__main__':
    main()

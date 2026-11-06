#!/usr/bin/env python
"""Unified eval entry point for LEIA.

Three subcommands, one shape per paper artifact:

    eval.py tokenizer  --checkpoint A.pt --mmap-dir ...
        Tokenizer-only stress reconstruction (encode GT u → decode → predict
        stress; no dynamics, no AR rollout). Reads from the training mmap.

    eval.py e2e        --tokenizer T.pt --dynamics D.pt --npz-dir ...
        Full pipeline AR rollout — tokenizer + dynamics, decoded each frame,
        compared to GT displacement and stress. Optional --compute-fd for
        predicted-vs-FEM force-displacement comparison.

    eval.py gt-fd      --npz-dir ... --mesh-dir ...
        Precompute GT force-displacement curves from FEM NPZ trajectories.
        No model involved; produces a reference HDF5 for cross-comparison
        with the predicted F-D from `eval.py e2e --compute-fd`.

Each subcommand dispatches to `leia/_eval_pipelines/<module>.py:main()`. Pass
`eval.py <cmd> --help` for the full surface of each.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(prog='leia-eval', add_help=False)
    parser.add_argument('cmd', nargs='?', default=None,
                        choices=['tokenizer', 'e2e', 'gt-fd', None])
    parser.add_argument('-h', '--help', action='store_true')
    args, rest = parser.parse_known_args()

    if args.cmd is None:
        print(__doc__)
        if rest:
            print(f"eval.py: missing subcommand (tokenizer | e2e | gt-fd) before "
                  f"{rest[0]!r}", file=sys.stderr)
            sys.exit(2)
        sys.exit(0)

    # Forward `--help` to the inner subcommand if requested.
    if args.help:
        rest = rest + ['--help']

    sys.argv = [f'leia-eval-{args.cmd}'] + rest

    from leia._eval_pipelines import eval_tokenizer, eval_e2e, gt_fd
    dispatch = {
        'tokenizer': eval_tokenizer.main,
        'e2e':       eval_e2e.main,
        'gt-fd':     gt_fd.main,
    }
    dispatch[args.cmd]()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Stage 1 of the Figure 5 pipeline (figure5.sh): random-walk candidates.

Generates ~3080 candidate graphs by running depth-D random-walk mutations
from each of the 55 training-plate seeds. CPU only and training-safe.

Per seed: emits the seed itself (depth=0) plus depth-1..D mutants, each
mutation applied to the previous mutant (true random walk, not random
expansion from seed). Per-seed RNG is seeded deterministically so a
re-run reproduces identical candidates.

Usage:
    python -m design_search.landscape.walk \\
        --output-dir design_search/outputs/<landscape_run> \\
        --depth 55

    # Pilot:
    python -m design_search.landscape.walk \\
        --output-dir design_search/outputs/<landscape_run>_pilot \\
        --depth 10 \\
        --max-seeds 5

`--seeds-file` defaults to MICROPLATE_LATTICE.splits.training_plates.
"""

import argparse
import json
import os
import time

import numpy as np

from design_search.core.graph import sample_random_mutation
from design_search.core.characterize import load_seed_graph
from leia.regime import MICROPLATE_LATTICE, MANIFEST_DIR


def load_seed_graph_permissive(rve_id: str, rve_bank_dir: str = "data/rve_bank") -> tuple[dict, list]:
    """Load a seed graph; report (graph, validation_errors) without raising.

    Some production training plates have minor validation issues (e.g.,
    duplicate edges) that the LEIA tokenizer has already trained on. We
    accept these as valid seeds for design-space mapping, since the model
    handles them in practice. Returns the graph along with the errors list
    so the caller can log them.
    """
    graph_path = os.path.join(rve_bank_dir, rve_id, "graph.json")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found: {graph_path}")
    with open(graph_path) as f:
        graph = json.load(f)
    # Strip duplicate-edge beams (first occurrence wins), as load_seed_graph
    # does; otherwise every mutation that keeps the duplicate fails validation
    # and the walk stalls on that seed.
    seen, deduped = set(), []
    for beam in graph.get("beams", []):
        a, b = beam["idx"]
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(beam)
    if len(deduped) < len(graph.get("beams", [])):
        print(f"[load_seed_graph_permissive] {rve_id}: removed "
              f"{len(graph['beams']) - len(deduped)} duplicate-edge beam(s)")
        graph["beams"] = deduped
    from design_search.core.graph import validate_graph
    valid, errors = validate_graph(graph)
    return graph, errors


def manifest_dropped_duplicates() -> list[str]:
    """Plate IDs the published training manifest excludes as duplicates
    (recorded in the walk manifest for provenance)."""
    path = os.path.join(MANIFEST_DIR, "microplate_lattice_production_plates_v2.json")
    with open(path) as f:
        return sorted(json.load(f).get("dropped_duplicates", []))


def load_training_seed_ids(seeds_file: str | None, max_seeds: int | None) -> list[str]:
    if seeds_file is None:
        raw = MICROPLATE_LATTICE.splits.training_plates
    else:
        with open(seeds_file) as f:
            raw = json.load(f)["plates"]
    plates = sorted(raw)   # the manifest already excludes its dropped_duplicates
    if max_seeds is not None:
        plates = plates[:max_seeds]
    return plates


def walk_one_seed(
    seed_id: str,
    seed_graph: dict,
    depth: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Depth-`depth` random walk starting from `seed_graph`.

    Returns a list of (depth+1) candidate dicts: depth=0 is the seed,
    depth=k>0 is one mutation applied to the depth=(k-1) graph.

    If a mutation step exhausts retries, that step is skipped and the
    walk continues from the previous valid graph.
    """
    candidates = []
    current = seed_graph
    candidates.append({
        "id": f"{seed_id}_d000",
        "seed_plate": seed_id,
        "depth": 0,
        "parent_id": None,
        "graph": current,
    })

    for d in range(1, depth + 1):
        mutant = sample_random_mutation(current, rng)
        if mutant is None:
            # Walk stalled — re-use previous graph but mark it
            candidates.append({
                "id": f"{seed_id}_d{d:03d}",
                "seed_plate": seed_id,
                "depth": d,
                "parent_id": f"{seed_id}_d{d - 1:03d}",
                "graph": current,
                "mutation_status": "stalled",
            })
            continue

        candidates.append({
            "id": f"{seed_id}_d{d:03d}",
            "seed_plate": seed_id,
            "depth": d,
            "parent_id": f"{seed_id}_d{d - 1:03d}",
            "graph": mutant,
        })
        current = mutant

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output dir; if --run-root is set, "
                             "defaults to <run-root>/walk.")
    parser.add_argument("--seeds-file", type=str, default=None,
                        help="JSON manifest with a 'plates' list. Default: "
                             "MICROPLATE_LATTICE.splits.training_plates.")
    parser.add_argument("--depth", type=int, default=55,
                        help="Random-walk depth per seed (mutations per seed)")
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Cap on number of seeds (for pilots). None = all 55.")
    parser.add_argument("--rng-base", type=int, default=42,
                        help="Per-seed RNG = default_rng(rng_base + seed_idx)")
    parser.add_argument("--rve-bank", type=str, default="data/rve_bank",
                        help="Directory containing seed graphs (per-id subdirs)")
    from design_search.landscape.layout import add_run_root_arg, resolve
    add_run_root_arg(parser)
    args = parser.parse_args()

    if args.output_dir is None:
        if args.run_root is None:
            parser.error("either --output-dir or --run-root is required")
        args.output_dir = resolve(args.run_root, "walk")

    os.makedirs(args.output_dir, exist_ok=True)

    seed_ids = load_training_seed_ids(args.seeds_file, args.max_seeds)
    print(f"Walk: {len(seed_ids)} seeds, depth={args.depth}, "
          f"expected {len(seed_ids) * (args.depth + 1)} candidates")

    t0 = time.time()
    all_candidates = []
    stalled_total = 0
    seeds_with_warnings = []
    for seed_idx, seed_id in enumerate(seed_ids):
        seed_graph, validation_errors = load_seed_graph_permissive(
            seed_id, rve_bank_dir=args.rve_bank,
        )
        if validation_errors:
            seeds_with_warnings.append((seed_id, validation_errors[:3]))
        rng = np.random.default_rng(args.rng_base + seed_idx)
        cands = walk_one_seed(seed_id, seed_graph, args.depth, rng)
        n_stalled = sum(1 for c in cands if c.get("mutation_status") == "stalled")
        stalled_total += n_stalled
        all_candidates.extend(cands)
        warn = " [VALID-WARN]" if validation_errors else ""
        print(f"  [{seed_idx + 1:02d}/{len(seed_ids)}] {seed_id}: "
              f"{len(cands)} candidates ({n_stalled} stalled){warn}")

    elapsed = time.time() - t0

    out_path = os.path.join(args.output_dir, "candidates.json")
    with open(out_path + ".tmp", "w") as f:   # atomic: figure5.sh gates on this file
        json.dump({"candidates": all_candidates}, f)
    os.replace(out_path + ".tmp", out_path)

    meta = {
        "n_seeds": len(seed_ids),
        "depth": args.depth,
        "rng_base": args.rng_base,
        "n_candidates_total": len(all_candidates),
        "n_stalled": stalled_total,
        "elapsed_s": elapsed,
        "seeds_file": args.seeds_file,
        "rve_bank": args.rve_bank,
        "duplicates_dropped": manifest_dropped_duplicates(),
        "seeds_with_validation_warnings": [
            {"seed_id": sid, "errors": errs}
            for sid, errs in seeds_with_warnings
        ],
    }
    meta_path = os.path.join(args.output_dir, "walk_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s. {len(all_candidates)} candidates "
          f"({stalled_total} stalled steps).")
    print(f"  Wrote {out_path}")
    print(f"  Wrote {meta_path}")


if __name__ == "__main__":
    main()

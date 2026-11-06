"""Shared search utilities — characterization, parallel meshing, graph I/O,
crash-recovery checkpoints. Used by both beam search (Figure 4) and landscape
candidate generation (Figure 5).

sync_characterize:        synchronous mechanical characterization (4-direction sweep)
mesh_candidates_parallel: parallel meshing via ThreadPoolExecutor
load_seed_graph:          load and validate an RVE graph from disk
save_checkpoint /         iterative-search crash recovery (JSON)
load_checkpoint:
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from design_search.core.config import ServingConfig
from design_search.core.engine import ModelEngine, Session
from design_search.core.meshing import generate_plate_mesh, MeshingError


# 2-direction characterization: stretch + shear_y
DIRECTIONS_2 = [
    ([1, 0, 0, 0], 0, "stretch"),
    ([0, 0, 1, 0], 2, "shear_y"),
]

# Full 4-direction characterization (for winner validation)
DIRECTIONS_4 = [
    ([1, 0, 0, 0], 0, "stretch"),
    ([0, 1, 0, 0], 1, "twist"),
    ([0, 0, 1, 0], 2, "shear_y"),
    ([0, 0, 0, 1], 3, "shear_z"),
]


def sync_characterize(
    engine: ModelEngine,
    sessions: list[Session],
    directions: list[tuple[list[int], int, str]] = None,
    steps_per_direction: int = 20,
    batch_size: int = 8,
) -> dict[str, dict]:
    """Synchronous mechanical characterization protocol.

    For each direction, steps the sessions forward, collecting F-D data,
    then resets all sessions.

    Returns: {session_id: {direction_label: {"disp": [...], "force": [...]}}}
    """
    if directions is None:
        directions = DIRECTIONS_2

    fd_data = {s.session_id: {} for s in sessions}

    for action, dof_idx, label in directions:
        # Init collectors with (0, 0) starting point
        for s in sessions:
            fd_data[s.session_id][label] = {"disp": [0.0], "force": [0.0]}

        # Step loop
        for step_i in range(steps_per_direction):
            # Sub-batch to limit GPU memory during decode
            for batch_start in range(0, len(sessions), batch_size):
                batch = sessions[batch_start:batch_start + batch_size]
                results = engine.step_batch(batch, action)
                for r in results:
                    fd_data[r.session_id][label]["disp"].append(
                        float(r.gen_disp[dof_idx])
                    )
                    fd_data[r.session_id][label]["force"].append(
                        float(r.gen_force[dof_idx])
                    )

        # Reset all sessions for next direction
        for s in sessions:
            engine.reset_session(s)

    return fd_data


def mesh_candidates_parallel(
    candidates: list[dict],
    config: ServingConfig,
    max_workers: int = 7,
) -> list[tuple[int, "MeshResult"]]:
    """Mesh candidate graphs in parallel via ThreadPoolExecutor.

    Returns list of (original_index, MeshResult) for successful meshes.
    Failed meshes (MeshingError) are silently skipped.
    """
    results = []

    def _mesh_one(idx_graph):
        idx, graph = idx_graph
        mesh = generate_plate_mesh(graph, config)
        return (idx, mesh)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_mesh_one, (i, g)): i
            for i, g in enumerate(candidates)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
            except MeshingError as e:
                print(f"  mesh_failed: cand idx={idx}: {e}")
            except Exception as e:
                print(f"  mesh_error: cand idx={idx}: {type(e).__name__}: {e}")

    # Sort by original index for deterministic ordering
    results.sort(key=lambda x: x[0])
    return results


def load_seed_graph(rve_id: str, rve_bank_dir: str = "data/rve_bank") -> dict:
    """Load and validate an RVE graph from disk.

    Some RVE-bank graphs (e.g. mg_111, mg_289) contain duplicate-edge beams
    that fail `design_search.core.graph.validate_graph`. We strip
    duplicates here — keeping the first occurrence per node-pair — before
    validating, so the seed loads and the search can proceed.
    """
    graph_path = os.path.join(rve_bank_dir, rve_id, "graph.json")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    with open(graph_path) as f:
        graph = json.load(f)

    # Deduplicate edges by canonicalized (min, max) node-pair.
    seen_edges = set()
    deduped_beams = []
    n_dup = 0
    for beam in graph.get("beams", []):
        a, b = beam["idx"]
        key = (min(a, b), max(a, b))
        if key in seen_edges:
            n_dup += 1
            continue
        seen_edges.add(key)
        deduped_beams.append(beam)
    if n_dup > 0:
        print(f"[load_seed_graph] {rve_id}: removed {n_dup} duplicate-edge beam(s)")
        graph["beams"] = deduped_beams

    from design_search.core.graph import validate_graph
    valid, errors = validate_graph(graph)
    if not valid:
        raise ValueError(f"Invalid graph {rve_id}: {errors}")

    return graph


def save_checkpoint(
    output_dir: str,
    iteration: int,
    beam: list[dict],
    all_candidates: list[dict],
) -> str:
    """Save iterative-search state to a JSON checkpoint file.

    Returns the checkpoint file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint_iter{iteration:03d}.json")

    data = {
        "iteration": iteration,
        "beam": beam,
        "n_total_candidates": len(all_candidates),
        "all_candidates": all_candidates,
        "timestamp": time.time(),
    }

    with open(path + ".tmp", "w") as f:   # atomic: resume reads the latest checkpoint
        json.dump(data, f, indent=2)
    os.replace(path + ".tmp", path)

    return path


def load_checkpoint(output_dir: str):
    """Load the latest checkpoint from output_dir.

    Returns (iteration, beam, all_candidates) or None if no checkpoint found.
    """
    if not os.path.isdir(output_dir):
        return None

    checkpoints = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith("checkpoint_iter") and f.endswith(".json")
    ])

    if not checkpoints:
        return None

    latest = os.path.join(output_dir, checkpoints[-1])
    with open(latest) as f:
        data = json.load(f)

    return (data["iteration"], data["beam"], data["all_candidates"])

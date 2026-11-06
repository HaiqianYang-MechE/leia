"""Graph primitives + mutation operators + samplers for the design-search package.

Combines two layers:
  1. **Primitives** (validation + scale/perturb/add/remove/interpolate/apply):
     pure-Python graph manipulators, no FEM dependency.
  2. **Mutation operators + samplers**: topology-changing operations
     (split_beam, remove_node, subgraph_replace) and weighted random
     samplers used by Figure 4 (`design_search.beam.search`) and Figure 5
     novelty search (`design_search.landscape.novelty`).

All functions operate on graph JSON dicts with structure:
    {"nodes": [[x,y,z], ...], "beams": [{"idx": [i,j], "r_start": r, "r_end": r}, ...]}
All functions return NEW dicts (never mutate input).
"""

import copy
import math
from collections import defaultdict

import numpy as np



# ========== VALIDATION ==========

def validate_graph(graph: dict) -> tuple[bool, list[str]]:
    """Validate graph geometry against the capsule-SDF mesh pipeline.

    Returns (is_valid, errors_list).
    """
    errors = []

    # Check nodes
    nodes = graph.get("nodes", [])
    if len(nodes) < 2:
        errors.append(f"Need >= 2 nodes, got {len(nodes)}")

    for i, node in enumerate(nodes):
        if not isinstance(node, (list, tuple)) or len(node) != 3:
            errors.append(f"Node {i}: must be 3-element list, got {node}")
            continue
        for j, val in enumerate(node):
            if not isinstance(val, (int, float)):
                errors.append(f"Node {i}[{j}]: not numeric ({val})")
            elif val < -1.5 or val > 1.5:
                errors.append(f"Node {i}[{j}]: {val} out of [-1.5, 1.5]")

    # Check beams
    beams = graph.get("beams", [])
    if len(beams) < 1:
        errors.append(f"Need >= 1 beam, got {len(beams)}")

    seen_edges = set()
    for i, beam in enumerate(beams):
        if not isinstance(beam, dict):
            errors.append(f"Beam {i}: must be dict, got {type(beam).__name__}")
            continue

        idx = beam.get("idx", [])
        if not isinstance(idx, (list, tuple)) or len(idx) != 2:
            errors.append(f"Beam {i}: idx must be 2-element list")
            continue

        a, b = idx
        if not isinstance(a, int) or not isinstance(b, int):
            errors.append(f"Beam {i}: idx values must be int")
            continue

        if a < 0 or a >= len(nodes) or b < 0 or b >= len(nodes):
            errors.append(f"Beam {i}: idx [{a},{b}] out of range [0, {len(nodes)})")
            continue

        if a == b:
            errors.append(f"Beam {i}: self-loop [{a},{a}]")

        edge_key = (min(a, b), max(a, b))
        if edge_key in seen_edges:
            errors.append(f"Beam {i}: duplicate edge [{a},{b}]")
        seen_edges.add(edge_key)

        # Check radii
        for rname in ("r_start", "r_end"):
            r = beam.get(rname, 0.05)
            if not isinstance(r, (int, float)):
                errors.append(f"Beam {i}: {rname} not numeric ({r})")
            elif r <= 0 or r > 0.5:
                errors.append(f"Beam {i}: {rname}={r} out of (0, 0.5]")

    return (len(errors) == 0, errors)


# ========== MUTATION OPERATIONS ==========

def scale_radii(graph: dict, factor: float) -> dict:
    """Scale all beam radii by factor, clamping to (0, 0.5]."""
    g = copy.deepcopy(graph)
    for beam in g["beams"]:
        for rname in ("r_start", "r_end"):
            r = beam.get(rname, 0.05)
            beam[rname] = max(1e-4, min(0.5, r * factor))
    return g


def scale_radii_selective(graph: dict, beam_indices: list[int], factor: float) -> dict:
    """Scale radii of specific beams by factor."""
    g = copy.deepcopy(graph)
    for i in beam_indices:
        if i < 0 or i >= len(g["beams"]):
            raise IndexError(f"Beam index {i} out of range [0, {len(g['beams'])})")
        beam = g["beams"][i]
        for rname in ("r_start", "r_end"):
            r = beam.get(rname, 0.05)
            beam[rname] = max(1e-4, min(0.5, r * factor))
    return g


def perturb_nodes(graph: dict, node_indices: list[int], deltas: list[list[float]]) -> dict:
    """Add delta vectors to specified node positions, clamping to [-1.5, 1.5]."""
    if len(node_indices) != len(deltas):
        raise ValueError(f"node_indices ({len(node_indices)}) and deltas ({len(deltas)}) must match")
    g = copy.deepcopy(graph)
    for idx, delta in zip(node_indices, deltas):
        if idx < 0 or idx >= len(g["nodes"]):
            raise IndexError(f"Node index {idx} out of range [0, {len(g['nodes'])})")
        if len(delta) != 3:
            raise ValueError(f"Delta must be 3D, got {len(delta)}")
        for j in range(3):
            g["nodes"][idx][j] = max(-1.5, min(1.5, g["nodes"][idx][j] + delta[j]))
    return g


def add_beam(graph: dict, node_i: int, node_j: int, radius: float = 0.05) -> dict:
    """Add a new beam between two existing nodes."""
    g = copy.deepcopy(graph)
    n = len(g["nodes"])

    if node_i < 0 or node_i >= n or node_j < 0 or node_j >= n:
        raise IndexError(f"Node indices [{node_i}, {node_j}] out of range [0, {n})")
    if node_i == node_j:
        raise ValueError(f"Self-loop not allowed: [{node_i}, {node_j}]")
    if radius <= 0 or radius > 0.5:
        raise ValueError(f"Radius {radius} out of (0, 0.5]")

    # Check for duplicate
    edge_key = (min(node_i, node_j), max(node_i, node_j))
    for beam in g["beams"]:
        a, b = beam["idx"]
        if (min(a, b), max(a, b)) == edge_key:
            raise ValueError(f"Edge [{node_i}, {node_j}] already exists")

    g["beams"].append({"idx": [node_i, node_j], "r_start": radius, "r_end": radius})
    return g


def remove_beams(graph: dict, beam_indices: list[int]) -> dict:
    """Remove beams at specified indices."""
    g = copy.deepcopy(graph)
    for i in beam_indices:
        if i < 0 or i >= len(graph["beams"]):
            raise IndexError(f"Beam index {i} out of range [0, {len(graph['beams'])})")

    # Remove in reverse order to preserve indices
    for i in sorted(beam_indices, reverse=True):
        del g["beams"][i]

    if len(g["beams"]) == 0:
        raise ValueError("Cannot remove all beams — graph must have >= 1 beam")
    return g


# ========== ANALYSIS ==========

def graph_stats(graph: dict) -> dict:
    """Compute basic graph statistics."""
    nodes = np.array(graph["nodes"])
    beams = graph["beams"]
    n_nodes = len(nodes)
    n_beams = len(beams)

    # Radii
    radii = []
    for beam in beams:
        radii.append(beam.get("r_start", 0.05))
        radii.append(beam.get("r_end", 0.05))
    radii = np.array(radii)

    # Beam lengths
    lengths = []
    for beam in beams:
        a, b = beam["idx"]
        lengths.append(np.linalg.norm(nodes[a] - nodes[b]))
    lengths = np.array(lengths)

    # Node degree
    degree = defaultdict(int)
    for beam in beams:
        a, b = beam["idx"]
        degree[a] += 1
        degree[b] += 1
    degrees = np.array([degree.get(i, 0) for i in range(n_nodes)])

    return {
        "n_nodes": n_nodes,
        "n_beams": n_beams,
        "mean_radius": float(radii.mean()),
        "std_radius": float(radii.std()),
        "mean_beam_length": float(lengths.mean()),
        "std_beam_length": float(lengths.std()),
        "mean_degree": float(degrees.mean()),
        "bounding_box": {
            "min": nodes.min(axis=0).tolist(),
            "max": nodes.max(axis=0).tolist(),
        },
    }


# ========== BLENDING ==========

def interpolate_graphs(graph_a: dict, graph_b: dict, alpha: float) -> dict:
    """Lerp node positions and radii between two graphs with the same topology.

    Result = (1-alpha)*A + alpha*B. alpha in [0, 1].
    """
    nodes_a = graph_a["nodes"]
    nodes_b = graph_b["nodes"]
    beams_a = graph_a["beams"]
    beams_b = graph_b["beams"]

    if len(nodes_a) != len(nodes_b):
        raise ValueError(f"Node count mismatch: {len(nodes_a)} vs {len(nodes_b)}")
    if len(beams_a) != len(beams_b):
        raise ValueError(f"Beam count mismatch: {len(beams_a)} vs {len(beams_b)}")

    # Verify same connectivity
    for i, (ba, bb) in enumerate(zip(beams_a, beams_b)):
        if sorted(ba["idx"]) != sorted(bb["idx"]):
            raise ValueError(f"Beam {i} connectivity mismatch: {ba['idx']} vs {bb['idx']}")

    alpha = max(0.0, min(1.0, alpha))

    # Lerp nodes
    new_nodes = []
    for na, nb in zip(nodes_a, nodes_b):
        new_nodes.append([
            max(-1.5, min(1.5, (1 - alpha) * na[j] + alpha * nb[j]))
            for j in range(3)
        ])

    # Lerp beam radii
    new_beams = []
    for ba, bb in zip(beams_a, beams_b):
        ra_s = ba.get("r_start", 0.05)
        rb_s = bb.get("r_start", 0.05)
        ra_e = ba.get("r_end", 0.05)
        rb_e = bb.get("r_end", 0.05)
        new_beams.append({
            "idx": list(ba["idx"]),
            "r_start": max(1e-4, min(0.5, (1 - alpha) * ra_s + alpha * rb_s)),
            "r_end": max(1e-4, min(0.5, (1 - alpha) * ra_e + alpha * rb_e)),
        })

    return {"nodes": new_nodes, "beams": new_beams}


# ========== MUTATION DISPATCH ==========

def apply_mutation(graph: dict, mutation: dict) -> dict:
    """Apply a single mutation operation described as a dict.

    Supported ops:
        {"op": "scale_radii", "factor": 1.1}
        {"op": "scale_radii_selective", "beam_indices": [0,2], "factor": 1.3}
        {"op": "perturb_nodes", "node_indices": [0,1], "deltas": [[0.1,0,0], [0,-0.1,0]]}
        {"op": "add_beam", "node_i": 0, "node_j": 3, "radius": 0.08}
        {"op": "remove_beams", "beam_indices": [5]}
    """
    op = mutation.get("op")
    if op == "scale_radii":
        return scale_radii(graph, mutation["factor"])
    elif op == "scale_radii_selective":
        return scale_radii_selective(graph, mutation["beam_indices"], mutation["factor"])
    elif op == "perturb_nodes":
        return perturb_nodes(graph, mutation["node_indices"], mutation["deltas"])
    elif op == "add_beam":
        return add_beam(graph, mutation["node_i"], mutation["node_j"], mutation.get("radius", 0.05))
    elif op == "remove_beams":
        return remove_beams(graph, mutation["beam_indices"])
    else:
        raise ValueError(f"Unknown mutation op: {op}")


def apply_mutations(graph: dict, mutations: list[dict]) -> dict:
    """Apply a sequence of mutations to a graph."""
    result = graph
    for m in mutations:
        result = apply_mutation(result, m)
    return result


# ========== GRAPH CONNECTIVITY ==========

def is_connected(graph: dict) -> bool:
    """Check if all nodes are reachable via BFS from node 0.

    Returns False if any node is unreachable (disconnected graph).
    Disconnected graphs crash the SDF mesher.
    """
    nodes = graph["nodes"]
    beams = graph["beams"]
    n = len(nodes)
    if n == 0:
        return False
    if n == 1:
        return True

    # Build adjacency list
    adj = defaultdict(set)
    for beam in beams:
        a, b = beam["idx"]
        adj[a].add(b)
        adj[b].add(a)

    # BFS from node 0
    visited = set()
    queue = [0]
    visited.add(0)
    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == n


# ========== NEW MUTATION OPERATORS ==========

def split_beam(graph: dict, beam_idx: int) -> dict:
    """Insert a node at the midpoint of a beam, replacing it with two beams.

    Creates a new junction point, enabling topology changes that add_beam alone
    cannot achieve (since add_beam only connects existing nodes).
    """
    g = copy.deepcopy(graph)
    if beam_idx < 0 or beam_idx >= len(g["beams"]):
        raise IndexError(f"Beam index {beam_idx} out of range [0, {len(g['beams'])})")

    beam = g["beams"][beam_idx]
    i, j = beam["idx"]
    r_start = beam.get("r_start", 0.05)
    r_end = beam.get("r_end", 0.05)

    # Compute midpoint
    pi = g["nodes"][i]
    pj = g["nodes"][j]
    mid = [(pi[k] + pj[k]) / 2.0 for k in range(3)]

    # Interpolated radius at midpoint
    r_mid = (r_start + r_end) / 2.0

    # Add new node
    new_node_idx = len(g["nodes"])
    g["nodes"].append(mid)

    # Remove original beam, add two new beams
    del g["beams"][beam_idx]
    g["beams"].append({"idx": [i, new_node_idx], "r_start": r_start, "r_end": r_mid})
    g["beams"].append({"idx": [new_node_idx, j], "r_start": r_mid, "r_end": r_end})

    return g


def remove_node(graph: dict, node_idx: int) -> dict:
    """Remove a node and all its incident beams, reindexing remaining beams.

    Raises ValueError if the result would have zero beams or be disconnected.
    """
    g = copy.deepcopy(graph)
    n = len(g["nodes"])
    if node_idx < 0 or node_idx >= n:
        raise IndexError(f"Node index {node_idx} out of range [0, {n})")

    # Remove all beams incident to node_idx
    g["beams"] = [
        b for b in g["beams"]
        if node_idx not in b["idx"]
    ]

    if len(g["beams"]) == 0:
        raise ValueError("Cannot remove node — would leave zero beams")

    # Remove the node
    del g["nodes"][node_idx]

    # Reindex beam endpoints: any index > node_idx decrements by 1
    for beam in g["beams"]:
        idx = beam["idx"]
        beam["idx"] = [
            (v - 1 if v > node_idx else v) for v in idx
        ]

    # Check connectivity
    if not is_connected(g):
        raise ValueError("Cannot remove node — would disconnect graph")

    return g


def random_radius_per_beam(graph: dict, rng: np.random.Generator, std: float = 0.15) -> dict:
    """Randomize each beam's radius independently with log-normal factors.

    Each beam gets factor = exp(Normal(0, std)), applied to both r_start and r_end.
    """
    g = copy.deepcopy(graph)
    for beam in g["beams"]:
        factor = float(np.exp(rng.normal(0, std)))
        for rname in ("r_start", "r_end"):
            r = beam.get(rname, 0.05)
            beam[rname] = max(1e-4, min(0.5, r * factor))
    return g


# ========== RANDOM MUTATION SAMPLER ==========

# Operation names and sampling probabilities (paper Table 5).
_OPS = [
    ("perturb_nodes", 0.25),
    ("scale_radii_selective", 0.20),
    ("add_beam", 0.15),
    ("remove_beams", 0.15),
    ("split_beam", 0.15),
    ("random_radius_per_beam", 0.10),
]
_OP_NAMES = [op for op, _ in _OPS]
_OP_WEIGHTS = np.array([w for _, w in _OPS])
_OP_WEIGHTS = _OP_WEIGHTS / _OP_WEIGHTS.sum()  # normalize


def _find_unconnected_pair(graph: dict, rng: np.random.Generator):
    """Find a random pair of nodes not already connected by a beam."""
    nodes = graph["nodes"]
    beams = graph["beams"]
    n = len(nodes)

    existing = set()
    for beam in beams:
        a, b_idx = beam["idx"]
        existing.add((min(a, b_idx), max(a, b_idx)))

    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in existing:
                candidates.append((i, j))

    if not candidates:
        return None
    idx = rng.integers(0, len(candidates))
    return candidates[idx]


def _apply_single_op(graph: dict, op_name: str, rng: np.random.Generator) -> dict:
    """Apply a single random mutation operation to the graph."""
    nodes = graph["nodes"]
    beams = graph["beams"]
    n_nodes = len(nodes)
    n_beams = len(beams)

    if op_name == "perturb_nodes":
        k = rng.integers(1, min(4, n_nodes + 1))
        indices = rng.choice(n_nodes, size=k, replace=False).tolist()
        deltas = rng.normal(0, 0.08, size=(k, 3)).tolist()
        return perturb_nodes(graph, indices, deltas)

    elif op_name == "scale_radii_selective":
        k = rng.integers(1, min(5, n_beams + 1))
        indices = rng.choice(n_beams, size=k, replace=False).tolist()
        factor = float(np.exp(rng.normal(0, 0.15)))
        return scale_radii_selective(graph, indices, factor)

    elif op_name == "add_beam":
        pair = _find_unconnected_pair(graph, rng)
        if pair is None:
            return graph  # fully connected, skip
        i, j = pair
        # New-beam radius ~ Uniform(0.005, 0.04) (paper Table 5).
        radius = float(rng.uniform(0.005, 0.04))
        return add_beam(graph, i, j, radius)

    elif op_name == "remove_beams":
        if n_beams <= 1:
            return graph  # can't remove the last beam
        # Try up to 5 beams to find one whose removal keeps connectivity
        beam_order = rng.permutation(n_beams)
        for idx in beam_order[:5]:
            trial = remove_beams(graph, [int(idx)])
            if is_connected(trial):
                return trial
        return graph  # no removable beam found, skip

    elif op_name == "split_beam":
        idx = int(rng.integers(0, n_beams))
        return split_beam(graph, idx)

    elif op_name == "random_radius_per_beam":
        return random_radius_per_beam(graph, rng)

    else:
        raise ValueError(f"Unknown op: {op_name}")


def sample_random_mutation(
    graph: dict,
    rng: np.random.Generator,
    max_retries: int = 20,
) -> dict | None:
    """Generate a random valid mutation of the graph.

    Picks 1-3 random operations, applies them sequentially, validates the result.
    Returns None if max_retries exhausted without producing a valid graph.
    """
    for _ in range(max_retries):
        # Pick number of ops
        n_ops = rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])

        # Pick which ops
        op_indices = rng.choice(len(_OP_NAMES), size=n_ops, p=_OP_WEIGHTS)
        ops = [_OP_NAMES[i] for i in op_indices]

        # Apply sequentially
        result = graph
        try:
            for op in ops:
                result = _apply_single_op(result, op, rng)
        except (ValueError, IndexError):
            continue

        # Validate
        valid, _ = validate_graph(result)
        if not valid:
            continue

        if not is_connected(result):
            continue

        return result

    return None


# ========== AGGRESSIVE OPERATORS (used by the novelty search, Appendix E) ==========
# Larger step magnitudes + a topology-disrupting subgraph_replace operator.
# The conservative operators above are the paper Table 5 set used by the beam
# search (Figure 4) and the random-walk pass of the OOD landscape (Figure 5).

def subgraph_replace(graph: dict, rng: np.random.Generator, k: int = 3,
                     radius_range: tuple = (0.005, 0.04)) -> dict:
    """Replace a random k-node subgraph with k fresh nodes connected to neighbors.

    Picks k random nodes uniformly, deletes them and their incident beams,
    then adds k new nodes near the centroid of the removed nodes and connects
    each to up to 3 nearest surviving nodes.

    Raises ValueError if the result would have zero beams or be disconnected.
    """
    g = copy.deepcopy(graph)
    nodes = np.array(g["nodes"], dtype=np.float64)
    n = len(nodes)
    if k > n - 2:
        raise ValueError(f"Cannot replace {k} of {n} nodes — too few would remain")

    # Pick k random nodes to remove
    to_remove = sorted(rng.choice(n, size=k, replace=False).tolist())
    centroid = nodes[to_remove].mean(axis=0)

    # Drop incident beams
    removed_set = set(to_remove)
    g["beams"] = [
        b for b in g["beams"]
        if b["idx"][0] not in removed_set and b["idx"][1] not in removed_set
    ]

    # Drop nodes (reverse order so indices stay valid during deletion)
    for idx in reversed(to_remove):
        del g["nodes"][idx]

    # Reindex remaining beams: shift each old idx down by count of removed
    # indices < it
    sorted_removed = to_remove  # already sorted ascending
    def _reindex(old_idx: int) -> int:
        shift = sum(1 for r in sorted_removed if r < old_idx)
        return old_idx - shift

    for beam in g["beams"]:
        beam["idx"] = [_reindex(beam["idx"][0]), _reindex(beam["idx"][1])]

    # Spawn k fresh nodes near the centroid (Gaussian offsets, clipped to unit cube)
    new_node_indices = []
    for _ in range(k):
        offset = rng.normal(0, 0.15, size=3)
        new_pt = (centroid + offset).astype(np.float64)
        new_pt = np.clip(new_pt, 0.0, 1.0).tolist()
        new_idx = len(g["nodes"])
        g["nodes"].append(new_pt)
        new_node_indices.append(new_idx)

    # Connect each new node to up to 3 nearest surviving nodes (avoid duplicates)
    surviving_pts = np.array(g["nodes"], dtype=np.float64)
    existing_edges = {tuple(sorted(b["idx"])) for b in g["beams"]}
    for new_i in new_node_indices:
        new_pt = surviving_pts[new_i]
        dists = np.linalg.norm(surviving_pts - new_pt[None, :], axis=1)
        dists[new_i] = np.inf  # exclude self
        n_connect = min(3, len(g["nodes"]) - 1)
        nearest = np.argsort(dists)[:n_connect].tolist()
        for nbr in nearest:
            edge = tuple(sorted([new_i, int(nbr)]))
            if edge in existing_edges:
                continue
            radius = float(rng.uniform(*radius_range))
            g["beams"].append({
                "idx": [new_i, int(nbr)],
                "r_start": radius,
                "r_end": radius,
            })
            existing_edges.add(edge)

    if len(g["beams"]) == 0:
        raise ValueError("Subgraph replace produced zero beams")
    if not is_connected(g):
        raise ValueError("Subgraph replace produced disconnected graph")

    return g


def _apply_single_op_aggressive(graph: dict, op_name: str,
                                rng: np.random.Generator) -> dict:
    """Apply one aggressive mutation. Mirrors `_apply_single_op` but with
    bigger step magnitudes and including `subgraph_replace`."""
    nodes = graph["nodes"]
    beams = graph["beams"]
    n_nodes = len(nodes)
    n_beams = len(beams)

    if op_name == "perturb_nodes_aggressive":
        # 1-5 nodes (vs 1-3), σ=0.3 (vs 0.08)
        k = int(rng.integers(1, min(6, n_nodes + 1)))
        indices = rng.choice(n_nodes, size=k, replace=False).tolist()
        deltas = rng.normal(0, 0.3, size=(k, 3)).tolist()
        return perturb_nodes(graph, indices, deltas)

    elif op_name == "scale_radii_aggressive":
        # 1-8 beams (vs 1-4), σ=0.5 (vs 0.15)
        k = int(rng.integers(1, min(9, n_beams + 1)))
        indices = rng.choice(n_beams, size=k, replace=False).tolist()
        factor = float(np.exp(rng.normal(0, 0.5)))
        return scale_radii_selective(graph, indices, factor)

    elif op_name == "random_radius_per_beam_aggressive":
        # σ=0.5 (vs 0.15)
        return random_radius_per_beam(graph, rng, std=0.5)

    elif op_name == "add_beam":
        # Identical to conservative — operator itself is binary (add or not)
        pair = _find_unconnected_pair(graph, rng)
        if pair is None:
            return graph
        i, j = pair
        radius = float(rng.uniform(0.005, 0.04))
        return add_beam(graph, i, j, radius)

    elif op_name == "remove_beams":
        # Identical to conservative
        if n_beams <= 1:
            return graph
        beam_order = rng.permutation(n_beams)
        for idx in beam_order[:5]:
            trial = remove_beams(graph, [int(idx)])
            if is_connected(trial):
                return trial
        return graph

    elif op_name == "split_beam":
        # Identical to conservative
        idx = int(rng.integers(0, n_beams))
        return split_beam(graph, idx)

    elif op_name == "subgraph_replace":
        # New aggressive operator — k ∈ {2, 3, 4}
        k = int(rng.choice([2, 3, 4]))
        return subgraph_replace(graph, rng, k=k)

    else:
        raise ValueError(f"Unknown aggressive op: {op_name}")


_AGG_OPS = [
    ("perturb_nodes_aggressive",         0.20),
    ("scale_radii_aggressive",            0.18),
    ("random_radius_per_beam_aggressive", 0.10),
    ("add_beam",                          0.15),
    ("remove_beams",                      0.12),
    ("split_beam",                        0.13),
    ("subgraph_replace",                  0.12),
]
_AGG_OP_NAMES = [op for op, _ in _AGG_OPS]
_AGG_OP_WEIGHTS = np.array([w for _, w in _AGG_OPS])
_AGG_OP_WEIGHTS = _AGG_OP_WEIGHTS / _AGG_OP_WEIGHTS.sum()


def sample_aggressive_random_mutation(
    graph: dict,
    rng: np.random.Generator,
    max_retries: int = 20,
) -> dict | None:
    """Aggressive variant of sample_random_mutation.

    Picks 1-3 random ops with weights (0.2, 0.4, 0.4) — average 2.2 ops/step
    vs 1.5 in the conservative sampler. Operator weights bias toward
    perturb/scale/topology with `subgraph_replace` at 12% to occasionally
    inject larger structural changes.

    Returns None if max_retries exhausted without producing a valid +
    connected graph. Higher invalid rate is expected (budget for 30-40%
    retry overhead vs 4% for the conservative sampler).
    """
    for _ in range(max_retries):
        n_ops = int(rng.choice([1, 2, 3], p=[0.2, 0.4, 0.4]))
        op_indices = rng.choice(len(_AGG_OP_NAMES), size=n_ops, p=_AGG_OP_WEIGHTS)
        ops = [_AGG_OP_NAMES[i] for i in op_indices]

        result = graph
        try:
            for op in ops:
                result = _apply_single_op_aggressive(result, op, rng)
        except (ValueError, IndexError):
            continue

        valid, _ = validate_graph(result)
        if not valid:
            continue

        if not is_connected(result):
            continue

        return result

    return None


# ========== SELF-TEST ==========

if __name__ == "__main__":
    import json
    import os

    graph_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "rve_bank", "mg_046", "graph.json",
    )
    with open(graph_path) as f:
        seed = json.load(f)

    print(f"Seed graph: {len(seed['nodes'])} nodes, {len(seed['beams'])} beams")
    valid, errs = validate_graph(seed)
    print(f"Seed valid: {valid}")
    print(f"Seed connected: {is_connected(seed)}")

    # Test split_beam
    split = split_beam(seed, 0)
    print(f"\nAfter split_beam(0): {len(split['nodes'])} nodes, {len(split['beams'])} beams")
    assert len(split["nodes"]) == len(seed["nodes"]) + 1
    assert len(split["beams"]) == len(seed["beams"]) + 1
    assert is_connected(split)

    # Test remove_node on a degree-1 node (if any)
    from collections import Counter
    degree = Counter()
    for b in seed["beams"]:
        degree[b["idx"][0]] += 1
        degree[b["idx"][1]] += 1
    deg1_nodes = [n for n, d in degree.items() if d == 1]
    if deg1_nodes:
        rn = remove_node(seed, deg1_nodes[0])
        print(f"After remove_node({deg1_nodes[0]}): {len(rn['nodes'])} nodes, {len(rn['beams'])} beams")
        assert is_connected(rn)
    else:
        print("No degree-1 nodes to test remove_node")

    # Test random_radius_per_beam
    rng = np.random.default_rng(42)
    rr = random_radius_per_beam(seed, rng)
    v1, _ = validate_graph(rr)
    print(f"\nrandom_radius_per_beam valid: {v1}")

    # Test sample_random_mutation: 200 trials
    print("\n--- Random mutation sampler: 200 trials ---")
    rng = np.random.default_rng(42)
    success = 0
    fail = 0
    node_counts = []
    beam_counts = []
    for i in range(200):
        result = sample_random_mutation(seed, rng)
        if result is not None:
            success += 1
            v, e = validate_graph(result)
            assert v, f"Trial {i} invalid: {e}"
            assert is_connected(result), f"Trial {i} disconnected"
            node_counts.append(len(result["nodes"]))
            beam_counts.append(len(result["beams"]))
        else:
            fail += 1

    rate = success / (success + fail) * 100
    print(f"Success: {success}/200 ({rate:.1f}%)")
    print(f"Failures: {fail}/200")
    print(f"Node count range: {min(node_counts)}-{max(node_counts)} (mean {np.mean(node_counts):.1f})")
    print(f"Beam count range: {min(beam_counts)}-{max(beam_counts)} (mean {np.mean(beam_counts):.1f})")
    print("All valid + connected: OK")

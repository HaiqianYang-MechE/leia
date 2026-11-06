"""TetGen wrappers — inline (fast) and subprocess+kill (robust).

Both call `tetgen.TetGen(surface).tetrahedralize(switches=...)`. They differ
only in whether the call runs in-process (fast for online use) or in a
killable subprocess (robust against TetGen hangs in long batch jobs).

A unified `run_tetgen(...)` dispatches based on `kill_on_timeout`, with
progressive switch-chain fallback. Both modes enforce `timeout`:
  - subprocess+kill: hard SIGKILL via multiprocessing.
  - inline: thread-watchdog (Thread.join with timeout). On timeout the
    native call leaks in a daemon thread until it eventually completes;
    bounded by the ThreadPoolExecutor worker count.

Subprocess uses the `spawn` start method to avoid fork-from-CUDA hazards
when the parent has loaded torch/cuda (e.g., the design_search inference
process). Slower startup (~1s vs ~0.1s for fork) but safe.
"""

import multiprocessing
import os
import tempfile
import threading

import numpy as np


# The TetGen switch chain used to produce the on-disk canonical mesh.npz
# artifacts.
#
# Both data_curation (offline batch) and design_search (online per-candidate)
# use this — they differ only in `kill_on_timeout` (subprocess robustness
# vs inline latency).
CANONICAL_TETGEN = ('pq2.0Y', 'pq2.5Y', 'pq2.5')


# Cached spawn context — picking the start method is process-global, but
# get_context() is thread-safe and idempotent.
_SPAWN_CTX = multiprocessing.get_context("spawn")


def _tetgen_worker(surface_points, surface_faces, switches, result_path):
    """Subprocess worker: TetGen tetrahedralize, save (points, cells) to npz.

    Wrapped in try/except — failures leave the result file empty/missing,
    which the parent treats as None.
    """
    try:
        import pyvista as pv
        import tetgen

        pv_faces = np.column_stack([
            np.full(len(surface_faces), 3, dtype=np.int32),
            surface_faces,
        ]).ravel()
        surface = pv.PolyData(surface_points, pv_faces)
        tet = tetgen.TetGen(surface)
        tet.tetrahedralize(switches=switches, verbose=0)
        grid = tet.grid

        VTK_TETRA = 10
        if grid.n_cells > 0 and np.any(grid.celltypes == VTK_TETRA):
            tet_cells = grid.cells_dict[VTK_TETRA]
            np.savez(result_path,
                     points=np.array(grid.points, dtype=np.float64),
                     cells=tet_cells.astype(np.int64))
    except Exception:
        pass


def tetgen_subprocess(surface, switches, timeout):
    """Run TetGen in a subprocess with hard kill-on-timeout (spawn-based).

    Surface faces are extracted from PyVista's PolyData face buffer (variable-
    length encoding, [n, v0, v1, …, vn] runs); only triangles (n=3) are kept.

    Returns
    -------
    (points, cells) tuple or None on any failure / timeout. Worker exit
    status is intentionally not consulted — the result-file existence and
    content are the gate (the worker swallows exceptions and exits 0).
    """
    surface_points = np.array(surface.points, dtype=np.float64)
    raw_faces = surface.faces
    faces_list = []
    i = 0
    while i < len(raw_faces):
        n = raw_faces[i]
        if n == 3:
            faces_list.append(raw_faces[i + 1:i + 4])
        i += n + 1
    surface_faces = np.array(faces_list, dtype=np.int32)

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        result_path = tmp.name
    try:
        proc = _SPAWN_CTX.Process(
            target=_tetgen_worker,
            args=(surface_points, surface_faces, switches, result_path),
        )
        proc.start()
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.kill()
            proc.join()
            return None
        # Result-file is the gate: worker writes only on success, swallows
        # exceptions on failure → empty/missing file. Native crashes (segfault)
        # also leave the file missing.
        if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
            data = np.load(result_path)
            return data['points'], data['cells']
        return None
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)


def tetgen_inline(surface, switches, timeout=None):
    """Run TetGen in-process with optional thread-watchdog timeout.

    No subprocess startup overhead — TetGen's native `tetrahedralize()` runs
    in the calling process. When `timeout` is set, the call runs in a daemon
    thread; on timeout we return None and let the thread complete in the
    background (Python cannot interrupt native code). Bounded leak: at most
    one wedged thread per ThreadPoolExecutor worker.

    Use `tetgen_subprocess` instead when hard process kill is required (e.g.,
    offline batches over thousands of plates, where a single hang would
    otherwise burn a slot for 24h).
    """
    def _run():
        try:
            import tetgen

            tg = tetgen.TetGen(surface)
            tg.tetrahedralize(switches=switches, verbose=0)
            grid = tg.grid

            VTK_TETRA = 10
            if grid.n_cells > 0 and np.any(grid.celltypes == VTK_TETRA):
                points = np.array(grid.points, dtype=np.float64)
                cells = grid.cells_dict[VTK_TETRA].astype(np.int64)
                state["result"] = (points, cells)
        except Exception as e:
            state["error"] = e

    if timeout is None:
        state = {"result": None, "error": None}
        _run()
        if state["error"] is not None:
            print(f"TetGen ({switches}) failed: {state['error']}")
            return None
        return state["result"]

    state = {"result": None, "error": None}
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Native call still running — leak the thread; it'll terminate when
        # TetGen returns. The slot in the caller's pool is freed immediately.
        print(f"TetGen ({switches}) timed out after {timeout}s; abandoning thread")
        return None
    if state["error"] is not None:
        print(f"TetGen ({switches}) failed: {state['error']}")
        return None
    return state["result"]


def run_tetgen(surface, switches, timeout=300, kill_on_timeout=True):
    """Run TetGen with progressive switch-chain fallback.

    Parameters
    ----------
    surface : pyvista.PolyData — closed surface mesh.
    switches : iterable of str — switch presets to try in order.
        Each string is a TetGen switch sequence (e.g. 'pq1.5Y'). The first
        one to succeed is used.
    timeout : float — per-switch wall-clock limit. Enforced via SIGKILL when
        `kill_on_timeout=True`, else via thread-watchdog (leaks the wedged
        thread on timeout but doesn't block the caller's pool slot).
    kill_on_timeout : bool — if True, run each attempt in a killable
        subprocess (recommended for offline batch). If False, run inline
        (recommended for online use where subprocess startup dominates).

    Returns
    -------
    (points, cells) tuple or None if every switch attempt fails.
    """
    runner = tetgen_subprocess if kill_on_timeout else tetgen_inline
    for sw in switches:
        result = runner(surface, sw, timeout)
        if result is not None:
            return result
    return None

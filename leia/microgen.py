"""Lattice graph features and plate SDF/meshing kernels (a trimmed copy of the
authors' lattice-generation code).

Only the public functions consumed by `design_search/` and
`data_curation/mesh/build_meshes.py` are vendored:

    compute_features(graph)              → 15-d feature vector
    graph_to_mesh(graph, res, symmetry)  → unit-cell tet mesh (RVE bank)
    graph_to_plate_mesh(graph, ...)      → tiled plate surface mesh

`graph_to_*_mesh` requires a CUDA-enabled numba install; `compute_features`
is pure NumPy. Only these entry points are shipped.
"""

import numpy as np

from leia.regime import MICROPLATE_LATTICE


# ============================================================================
# Feature extraction (compute_features)
# ============================================================================

# Per-dimension scale used to normalize the 15-dim feature vector so L2
# distance treats all dimensions equally.
# [log1p(nodes), log1p(beams), mean_x/y/z, std_x/y/z, mean_r, std_r,
#  mean_len, std_len, connectivity, deg_mean, deg_std]
FEATURE_SCALES = np.array(
    [4, 4, 1, 1, 1, 0.5, 0.5, 0.5, 0.3, 0.15, 1.5, 1.0, 5, 10, 5],
    dtype=np.float32,
)


def compute_features(graph):
    """Compute a normalized 15-dim feature vector for diversity measurement."""
    nodes = np.array(graph["nodes"], dtype=np.float32)
    beams = graph["beams"]
    n_nodes, n_beams = len(nodes), len(beams)

    radii, lengths = [], []
    degrees = np.zeros(n_nodes)
    for b in beams:
        if not isinstance(b, dict):
            continue
        radii.extend([b.get("r_start", 0.05), b.get("r_end", 0.05)])
        i, j = b.get("idx", [0, 0])
        if i < n_nodes and j < n_nodes:
            lengths.append(np.linalg.norm(nodes[i] - nodes[j]))
            degrees[i] += 1
            degrees[j] += 1

    raw = np.array([
        np.log1p(n_nodes),
        np.log1p(n_beams),
        *nodes.mean(axis=0),                                # 3 values
        *nodes.std(axis=0),                                  # 3 values
        np.mean(radii) if radii else 0,
        np.std(radii) if radii else 0,
        np.mean(lengths) if lengths else 0,
        np.std(lengths) if lengths else 0,
        n_beams / max(n_nodes, 1),                           # connectivity
        degrees.mean() if n_nodes > 0 else 0,
        degrees.std() if n_nodes > 0 else 0,
    ], dtype=np.float32)
    return raw / FEATURE_SCALES


# ============================================================================
# Symmetry expansion (used by graph_to_mesh + graph_to_plate_mesh)
# ============================================================================

def get_ops_python(mode):
    """Return the symmetry-group rotation/reflection matrices.

    `mode='FULL'` returns all 48 cubic operations; `'CHIRAL'` returns the
    24 with positive determinant.
    """
    ops = []
    perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    for px, py, pz in perms:
        for sx, sy, sz in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                           (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]:
            mat = np.zeros((3, 3))
            mat[0, px], mat[1, py], mat[2, pz] = sx, sy, sz
            if mode == 'FULL' or (mode == 'CHIRAL' and np.linalg.det(mat) > 0):
                ops.append(mat)
    return np.unique(np.array(ops), axis=0)


# ============================================================================
# CUDA SDF kernels + mesh extraction (graph_to_mesh, graph_to_plate_mesh)
#
# These are imported lazily so `compute_features` and `get_ops_python` can
# be used on machines without numba.cuda installed (e.g. CPU-only review
# environments). graph_to_*_mesh requires CUDA at call time.
# ============================================================================

def _cuda_module():
    """Lazy import + kernel definition (numba.cuda is the heavy dep)."""
    from numba import cuda

    @cuda.jit(device=True)
    def smin_cuda(a, b, k):
        h = max(k - abs(a - b), 0.0) / k
        return min(a, b) - h * h * k * 0.25

    @cuda.jit(device=True)
    def sd_box_cuda(px, py, pz, bx, by, bz):
        """Signed Distance to a Box."""
        qx, qy, qz = abs(px) - bx, abs(py) - by, abs(pz) - bz
        out_x, out_y, out_z = max(qx, 0.0), max(qy, 0.0), max(qz, 0.0)
        d_out = (out_x ** 2 + out_y ** 2 + out_z ** 2) ** 0.5
        d_in = max(min(qx, 0.0), max(min(qy, 0.0), min(qz, 0.0)))
        return d_out + d_in

    @cuda.jit(device=True)
    def sd_capsule_graded_cuda(px, py, pz, ax, ay, az, bx, by, bz, r1, r2):
        pax, pay, paz = px - ax, py - ay, pz - az
        bax, bay, baz = bx - ax, by - ay, bz - az
        ba_ba = bax ** 2 + bay ** 2 + baz ** 2
        pa_ba = pax * bax + pay * bay + paz * baz
        h = max(0.0, min(1.0, pa_ba / (ba_ba + 1e-9)))
        dx, dy, dz = pax - bax * h, pay - bay * h, paz - baz * h
        return (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5 - (r1 * (1.0 - h) + r2 * h)

    @cuda.jit
    def rve_kernel_cuda(points, edges, edge_radii, out):
        idx = cuda.grid(1)
        if idx < points.shape[0]:
            px, py, pz = points[idx, 0], points[idx, 1], points[idx, 2]

            d_lattice = 10.0
            for j in range(edges.shape[0]):
                d = sd_capsule_graded_cuda(
                    px, py, pz,
                    edges[j, 0, 0], edges[j, 0, 1], edges[j, 0, 2],
                    edges[j, 1, 0], edges[j, 1, 1], edges[j, 1, 2],
                    edge_radii[j, 0], edge_radii[j, 1])
                d_lattice = smin_cuda(d_lattice, d, 0.15)

            # Box cut at exactly 0.98 — marching cubes generates a valid
            # (slightly aliased) surface along this boundary.
            d_box = sd_box_cuda(px, py, pz, 0.98, 0.98, 0.98)
            out[idx] = max(d_lattice, d_box)  # Boolean intersection

    @cuda.jit
    def rve_plate_kernel_cuda(points, edges, edge_radii,
                              tile_x, tile_y, tile_z,
                              cut_x, cut_y, cut_z, k_blend, out):
        """SDF kernel for tiled plate mesh with modular coordinate wrapping.

        Tiled axes (tile > 1) wrap coordinates back into [-1,1] using
        modular arithmetic. Box cut applies at the plate-scale outer
        boundary. `k_blend` controls smooth-min blending (auto-scaled).
        """
        idx = cuda.grid(1)
        if idx < points.shape[0]:
            px, py, pz = points[idx, 0], points[idx, 1], points[idx, 2]

            if tile_x > 1:
                lx = px - 2.0 * round(px / 2.0)
            else:
                lx = px
            if tile_y > 1:
                ly = py - 2.0 * round(py / 2.0)
            else:
                ly = py
            if tile_z > 1:
                lz = pz - 2.0 * round(pz / 2.0)
            else:
                lz = pz

            d_lattice = 10.0
            for j in range(edges.shape[0]):
                d = sd_capsule_graded_cuda(
                    lx, ly, lz,
                    edges[j, 0, 0], edges[j, 0, 1], edges[j, 0, 2],
                    edges[j, 1, 0], edges[j, 1, 1], edges[j, 1, 2],
                    edge_radii[j, 0], edge_radii[j, 1])
                d_lattice = smin_cuda(d_lattice, d, k_blend)

            d_box = sd_box_cuda(px, py, pz, cut_x, cut_y, cut_z)
            out[idx] = max(d_lattice, d_box)

    return cuda, rve_kernel_cuda, rve_plate_kernel_cuda


def graph_to_mesh(graph_data, res=80, symmetry="FULL"):
    """RVE unit-cell mesh: SDF on a `res`³ grid → marching cubes → PolyData.

    Symmetry-expanded edges are smin-blended on GPU; intersection with a
    [-0.98, 0.98]³ box gives a clean unit-cell crop.
    """
    import pyvista as pv
    from skimage import measure
    cuda, rve_kernel, _ = _cuda_module()

    nodes = np.array(graph_data['nodes'], dtype=np.float32)
    ops = get_ops_python(symmetry)

    exp_e, exp_r = [], []
    for b in graph_data['beams']:
        p1, p2 = nodes[b['idx'][0]], nodes[b['idx'][1]]
        for op in ops:
            exp_e.append([op @ p1, op @ p2])
            exp_r.append([b.get('r_start', 0.05), b.get('r_end', 0.05)])

    e_dev = cuda.to_device(np.array(exp_e, dtype=np.float32))
    r_dev = cuda.to_device(np.array(exp_r, dtype=np.float32))

    # Grid slightly larger than box cut so the implicit function crosses 0.
    pad = 1.02
    lin = np.linspace(-pad, pad, res).astype(np.float32)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
    pts_dev = cuda.to_device(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1))
    out_dev = cuda.device_array(len(X.ravel()), dtype=np.float32)

    threads = 256
    blocks = (len(X.ravel()) + threads - 1) // threads
    rve_kernel[blocks, threads](pts_dev, e_dev, r_dev, out_dev)

    sdf = out_dev.copy_to_host().reshape(X.shape)
    verts, faces, _, _ = measure.marching_cubes(sdf, level=0.0)

    verts[:, 0] = lin[0] + verts[:, 0] * (lin[-1] - lin[0]) / (res - 1)
    verts[:, 1] = lin[0] + verts[:, 1] * (lin[-1] - lin[0]) / (res - 1)
    verts[:, 2] = lin[0] + verts[:, 2] * (lin[-1] - lin[0]) / (res - 1)

    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=int), faces]).flatten()
    # Raw mesh — no snapping; downstream remesh + periodic match handles that.
    return pv.PolyData(verts, faces_pv).clean()


def graph_to_plate_mesh(graph_data, tiles=MICROPLATE_LATTICE.tiles, res_per_cell=50,
                        symmetry="FULL", auto_scale=True):
    """Tile a unit cell into a plate and extract the surface mesh.

    Modular coordinate wrapping evaluates the SDF in unit-cell-local space,
    producing seamless tiling. The box cut applies only at the outer plate
    boundary. With `auto_scale`, edge sizes and blend radius are normalized
    to fit the unit cell and a percolation check inflates `k_blend` until
    the unit cell is a single connected piece.

    Returns
    -------
    pv.PolyData
        Surface mesh at plate scale.
    """
    import pyvista as pv
    from skimage import measure
    cuda, _, rve_plate_kernel = _cuda_module()

    nodes = np.array(graph_data['nodes'], dtype=np.float32)
    ops = get_ops_python(symmetry)

    exp_e, exp_r = [], []
    for b in graph_data['beams']:
        p1, p2 = nodes[b['idx'][0]], nodes[b['idx'][1]]
        for op in ops:
            exp_e.append([op @ p1, op @ p2])
            exp_r.append([b.get('r_start', 0.05), b.get('r_end', 0.05)])

    edges = np.array(exp_e, dtype=np.float32)
    radii = np.array(exp_r, dtype=np.float32)
    k_blend = 0.15
    if auto_scale:
        max_extent = np.max(np.abs(edges))
        if 0 < max_extent < 1.0:
            scale = 1.0 / max_extent
            edges *= scale
            radii *= scale
            k_blend *= scale
    e_dev = cuda.to_device(edges)
    r_dev = cuda.to_device(radii)

    # Percolation check: increase k_blend until unit cell is a single piece.
    if auto_scale:
        check_res = 40
        lin_c = np.linspace(-1.02, 1.02, check_res).astype(np.float32)
        Xc, Yc, Zc = np.meshgrid(lin_c, lin_c, lin_c, indexing='ij')
        pts_c = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=1)
        pts_c_dev = cuda.to_device(pts_c)
        out_c_dev = cuda.device_array(pts_c.shape[0], dtype=np.float32)
        thr_c = 256
        blk_c = (pts_c.shape[0] + thr_c - 1) // thr_c
        for _attempt in range(6):
            rve_plate_kernel[blk_c, thr_c](
                pts_c_dev, e_dev, r_dev,
                np.int32(1), np.int32(1), np.int32(1),
                np.float32(0.98), np.float32(0.98), np.float32(0.98),
                np.float32(k_blend), out_c_dev)
            sdf_c = out_c_dev.copy_to_host().reshape(Xc.shape)
            try:
                vc, fc, _, _ = measure.marching_cubes(sdf_c, level=0.0)
            except ValueError:
                break  # empty mesh
            faces_pv_c = np.hstack([np.full((fc.shape[0], 1), 3, dtype=int), fc])
            mc = pv.PolyData(vc, faces_pv_c).clean()
            if mc.n_points == 0:
                break
            n_regions = mc.connectivity(largest=False)["RegionId"].max() + 1
            if n_regions <= 1:
                break
            k_blend *= 1.3

    tx, ty, tz = tiles
    pad_x, pad_y, pad_z = tx * 1.02, ty * 1.02, tz * 1.02
    cut_x, cut_y, cut_z = tx * 0.98, ty * 0.98, tz * 0.98
    res_x, res_y, res_z = tx * res_per_cell, ty * res_per_cell, tz * res_per_cell

    lin_x = np.linspace(-pad_x, pad_x, res_x).astype(np.float32)
    lin_y = np.linspace(-pad_y, pad_y, res_y).astype(np.float32)
    lin_z = np.linspace(-pad_z, pad_z, res_z).astype(np.float32)
    X, Y, Z = np.meshgrid(lin_x, lin_y, lin_z, indexing='ij')
    pts_flat = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    pts_dev = cuda.to_device(pts_flat)
    out_dev = cuda.device_array(pts_flat.shape[0], dtype=np.float32)

    threads = 256
    blocks = (pts_flat.shape[0] + threads - 1) // threads
    rve_plate_kernel[blocks, threads](
        pts_dev, e_dev, r_dev,
        np.int32(tx), np.int32(ty), np.int32(tz),
        np.float32(cut_x), np.float32(cut_y), np.float32(cut_z),
        np.float32(k_blend), out_dev)

    sdf = out_dev.copy_to_host().reshape(X.shape)
    verts, faces, _, _ = measure.marching_cubes(sdf, level=0.0)

    verts[:, 0] = lin_x[0] + verts[:, 0] * (lin_x[-1] - lin_x[0]) / (res_x - 1)
    verts[:, 1] = lin_y[0] + verts[:, 1] * (lin_y[-1] - lin_y[0]) / (res_y - 1)
    verts[:, 2] = lin_z[0] + verts[:, 2] * (lin_z[-1] - lin_z[0]) / (res_z - 1)

    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=int), faces]).flatten()
    return pv.PolyData(verts, faces_pv).clean()

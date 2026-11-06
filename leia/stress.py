"""Pure-numpy stress post-processing from decoded displacement fields.

Computes stress on the physical plate mesh, matching FEniCS:
  decoded displacement at the plate nodes -> per-tet F = I + grad(u)
  -> per-tet P (Neo-Hookean) -> Cauchy -> von Mises

Material model: compressible Neo-Hookean. The MicroPlate lattice constants
(mu, lmbda) come from `data/manifests/microplate_lattice_constants.json` via
`leia.regime.MICROPLATE_LATTICE`. The `neo_hookean_*` helpers default to
them; `stress_from_disps*` and `compute_stress_on_merged_mesh` default to
the legacy solid-plate values (mu=370, lmbda=1200) and every caller in this
repo passes mu/lmbda explicitly — do the same. Visco uses Arruda-Boyce,
which these helpers do not cover.

Stress must be computed on the physical mesh: the deformation gradient
depends on the coordinate frame, and FEniCS computes F = I + grad(u) on the
physical mesh.

Reference: matches the FEniCS symbolic stress assembly used to generate the FEM
data — see `data_curation/fem/run_plate_fem.py` for the FEniCS side.
"""

import numpy as np

from leia.regime import MICROPLATE_LATTICE


# ── Tet mesh geometry ────────────────────────────────────────────────────────

def tet4_dNdx(points, cells):
    """Compute shape function gradients and volumes for linear tet4 elements.

    Linear tet: constant gradient per element.
    Reference shape functions: N1=1-xi-eta-zeta, N2=xi, N3=eta, N4=zeta
    dN/dxi = [[-1,-1,-1], [1,0,0], [0,1,0], [0,0,1]]

    Jacobian J[i,j] = dx_i/dxi_j, with edge vectors as COLUMNS:
      J = [e1 | e2 | e3]  where e_k = x_{k+1} - x_0
    dN/dx = dN/dxi @ J^{-1}

    Parameters
    ----------
    points : (N, 3) array — node coordinates
    cells : (M, 4) array — tet connectivity (node indices)

    Returns
    -------
    dNdx : (M, 4, 3) — shape function gradients in physical coords
    volumes : (M,) — element volumes (positive)
    """
    # Edge vectors: x2-x1, x3-x1, x4-x1
    x = points[cells]  # (M, 4, 3)
    e1 = x[:, 1] - x[:, 0]  # (M, 3)
    e2 = x[:, 2] - x[:, 0]  # (M, 3)
    e3 = x[:, 3] - x[:, 0]  # (M, 3)

    # Jacobian: J[m, i, j] = dx_i/dxi_j — edge vectors as COLUMNS
    J = np.stack([e1, e2, e3], axis=2)  # (M, 3, 3)
    detJ = np.linalg.det(J)  # (M,)
    volumes = np.abs(detJ) / 6.0

    # Guard against exactly-singular Jacobians (degenerate tets with
    # zero volume). Replace them with identity so inv() doesn't fail;
    # the quality filter downstream will exclude them anyway.
    singular = np.abs(detJ) < 1e-30
    if singular.any():
        J[singular] = np.eye(3)

    Jinv = np.linalg.inv(J)  # (M, 3, 3)

    # dN/dxi for linear tet (constant)
    dNdxi = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])  # (4, 3)

    # dN/dx = dN/dxi @ J^{-1}  — (4, 3) @ (M, 3, 3) -> (M, 4, 3)
    dNdx = np.einsum('ij,mjk->mik', dNdxi, Jinv)

    return dNdx, volumes


# ── Constitutive law (micro-scale Neo-Hookean) ──────────────────────────────

def neo_hookean_P(F, mu=MICROPLATE_LATTICE.mu, lmbda=MICROPLATE_LATTICE.lmbda):
    """First Piola-Kirchhoff stress for compressible Neo-Hookean material.

    P = mu * (F - F^{-T}) + lambda * ln(J) * F^{-T}

    Parameters
    ----------
    F : (..., 3, 3) array — deformation gradient
    mu, lmbda : float — material parameters

    Returns
    -------
    P : (..., 3, 3) — first Piola-Kirchhoff stress
    """
    J = np.linalg.det(F)  # (...)
    Finv = np.linalg.inv(F)  # (..., 3, 3)
    FinvT = np.swapaxes(Finv, -2, -1)  # (..., 3, 3)
    lnJ = np.log(np.maximum(J, 1e-30))  # clamp to avoid log(0)
    P = mu * (F - FinvT) + lmbda * lnJ[..., None, None] * FinvT
    return P


def cauchy_from_P(P, F):
    """Cauchy stress from first Piola-Kirchhoff stress.

    sigma = (1/J) * P @ F^T

    Parameters
    ----------
    P : (..., 3, 3)
    F : (..., 3, 3)

    Returns
    -------
    sigma : (..., 3, 3) — Cauchy stress
    """
    J = np.linalg.det(F)  # (...)
    sigma = np.einsum('...ij,...kj->...ik', P, F) / J[..., None, None]
    return sigma


def neohookean_cauchy_sym6_from_gradu(grad_u, mu=MICROPLATE_LATTICE.mu, lmbda=MICROPLATE_LATTICE.lmbda):
    """Closed-form Neo-Hookean Cauchy stress (sym6 Voigt) from displacement gradient.

    sigma = (mu/J) * (b - I) + (lambda/J) * ln(J) * I
    where F = I + grad_u, b = F F^T, J = det(F).

    Avoids the tet shape-function projection path when grad_u is already available
    per node — lets tokenizer training compute stress head ground truth from the
    existing mmap grad_u field with no mesh geometry needed.

    Parameters
    ----------
    grad_u : (..., 3, 3) array — displacement gradient d u_i / d x_j (physical units).
    mu, lmbda : float — Neo-Hookean parameters. Hires lattice default: mu=1, lmbda=10.

    Returns
    -------
    sym6 : (..., 6) array — Cauchy stress in Voigt order [xx, yy, zz, xy, xz, yz].
    """
    I = np.eye(3, dtype=grad_u.dtype)
    F = grad_u + I
    J = np.linalg.det(F)                              # (...,)
    b = np.einsum('...ij,...kj->...ik', F, F)         # left Cauchy-Green b = F F^T
    J_safe = np.maximum(J, 1e-30)
    inv_J = 1.0 / J_safe
    lnJ = np.log(J_safe)
    sigma = (mu * inv_J)[..., None, None] * (b - I) \
          + (lmbda * inv_J * lnJ)[..., None, None] * I
    sym6 = np.stack([
        sigma[..., 0, 0],
        sigma[..., 1, 1],
        sigma[..., 2, 2],
        sigma[..., 0, 1],
        sigma[..., 0, 2],
        sigma[..., 1, 2],
    ], axis=-1)
    return sym6


def von_mises(sigma):
    """Von Mises equivalent stress from Cauchy stress tensor.

    sigma_VM = sqrt(3/2 * s:s)  where s = sigma - (1/3)*tr(sigma)*I

    Parameters
    ----------
    sigma : (..., 3, 3) — Cauchy stress

    Returns
    -------
    vm : (...) — von Mises scalar stress
    """
    tr = np.trace(sigma, axis1=-2, axis2=-1)  # (...)
    I = np.eye(3)
    s = sigma - (tr / 3.0)[..., None, None] * I
    # s:s = sum of s_ij * s_ij
    ss = np.sum(s * s, axis=(-2, -1))
    vm = np.sqrt(1.5 * np.maximum(ss, 0.0))
    return vm


def cauchy_to_sym6(stress_33):
    """Extract 6 symmetric components from a (..., 3, 3) Cauchy tensor.

    Returns (..., 6) in Voigt order [xx, yy, zz, xy, xz, yz].
    """
    return np.stack([
        stress_33[..., 0, 0],
        stress_33[..., 1, 1],
        stress_33[..., 2, 2],
        stress_33[..., 0, 1],
        stress_33[..., 0, 2],
        stress_33[..., 1, 2],
    ], axis=-1)


def sym6_to_cauchy(sym6):
    """Expand a (..., 6) Voigt sym6 tensor to a (..., 3, 3) Cauchy tensor."""
    shape = sym6.shape[:-1]
    out = np.zeros((*shape, 3, 3), dtype=sym6.dtype)
    out[..., 0, 0] = sym6[..., 0]
    out[..., 1, 1] = sym6[..., 1]
    out[..., 2, 2] = sym6[..., 2]
    out[..., 0, 1] = out[..., 1, 0] = sym6[..., 3]
    out[..., 0, 2] = out[..., 2, 0] = sym6[..., 4]
    out[..., 1, 2] = out[..., 2, 1] = sym6[..., 5]
    return out


# ── Stress on physical merged mesh ──────────────────────────────────────────

def tet_quality(points, cells):
    """Compute tet quality metric: volume / max_edge_length^3.

    Good tets have quality ~0.12 (regular tet). Sliver tets at cell
    interfaces have quality < 1e-4 and produce spurious stress gradients.

    Parameters
    ----------
    points : (N, 3) — node coordinates
    cells : (M, 4) — tet connectivity

    Returns
    -------
    quality : (M,) — quality metric per tet
    """
    x = points[cells]  # (M, 4, 3)
    # All 6 edge lengths
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append(np.linalg.norm(x[:, i] - x[:, j], axis=1))
    max_edge = np.maximum(np.max(edges, axis=0), 1e-30)

    e1 = x[:, 1] - x[:, 0]
    e2 = x[:, 2] - x[:, 0]
    e3 = x[:, 3] - x[:, 0]
    J = np.stack([e1, e2, e3], axis=1)
    vol = np.abs(np.linalg.det(J)) / 6.0

    return vol / (max_edge ** 3)


def precompute_stress_geometry(merged_coords, merged_cells,
                               quality_threshold=1e-4):
    """Precompute mesh geometry for stress computation (call once per shape).

    Returns dNdx, volumes, and quality mask that can be reused across frames.

    Parameters
    ----------
    merged_coords : (N, 3) — reference node coordinates (physical frame)
    merged_cells : (M, 4) — tet connectivity
    quality_threshold : float — minimum tet quality

    Returns
    -------
    dNdx : (M, 4, 3) — shape function gradients
    volumes : (M,) — tet volumes
    good : (M,) bool — quality mask
    """
    dNdx, volumes = tet4_dNdx(merged_coords, merged_cells)
    quality = tet_quality(merged_coords, merged_cells)
    good = quality >= quality_threshold
    return dNdx, volumes, good


def stress_from_disps(merged_cells, u_merged, dNdx, volumes, good,
                      mu=370.0, lmbda=1200.0):
    """Compute stress using precomputed geometry (fast, call per frame).

    Parameters
    ----------
    merged_cells : (M, 4) — tet connectivity
    u_merged : (N, 3) — displacement at each node
    dNdx : (M, 4, 3) — precomputed shape function gradients
    volumes : (M,) — precomputed tet volumes
    good : (M,) bool — precomputed quality mask
    mu, lmbda : float — material parameters

    Returns
    -------
    vm_per_tet, P, F, vm_per_node — same as compute_stress_on_merged_mesh
    """
    N = int(merged_cells.max()) + 1
    M = len(merged_cells)

    u_nodes = u_merged[merged_cells]  # (M, 4, 3)
    grad_u = np.einsum('mki,mkj->mij', u_nodes, dNdx)
    F = np.eye(3) + grad_u

    P = np.zeros((M, 3, 3), dtype=np.float64)
    vm_per_tet = np.zeros(M, dtype=np.float64)

    if good.any():
        P[good] = neo_hookean_P(F[good], mu=mu, lmbda=lmbda)
        sigma = cauchy_from_P(P[good], F[good])
        vm_per_tet[good] = von_mises(sigma)

    vm_node_accum = np.zeros(N, dtype=np.float64)
    vol_node_accum = np.zeros(N, dtype=np.float64)
    good_cells = merged_cells[good]
    good_vm = vm_per_tet[good]
    good_vol = volumes[good]
    for j in range(4):
        np.add.at(vm_node_accum, good_cells[:, j], good_vm * good_vol)
        np.add.at(vol_node_accum, good_cells[:, j], good_vol)
    vol_node_accum = np.maximum(vol_node_accum, 1e-30)
    vm_per_node = (vm_node_accum / vol_node_accum).astype(np.float32)

    return vm_per_tet, P, F, vm_per_node


def stress_from_disps_nodal_cauchy(merged_cells, u_merged, dNdx, volumes, good,
                                    mu=370.0, lmbda=1200.0):
    """Compute per-node Cauchy stress sym6 via volume-weighted scatter.

    Same as stress_from_disps but returns full Cauchy tensor at nodes
    (not just scalar von Mises). Used for stress head GT supervision.

    Returns
    -------
    cauchy_sym6_node : (N, 6) float32 — Cauchy stress [xx,yy,zz,xy,xz,yz]
    vm_per_node : (N,) float32 — von Mises (for verification)
    """
    N = int(merged_cells.max()) + 1
    M = len(merged_cells)

    u_nodes = u_merged[merged_cells]  # (M, 4, 3)
    grad_u = np.einsum('mki,mkj->mij', u_nodes, dNdx)
    F = np.eye(3) + grad_u

    # Compute Cauchy stress on good tets
    sigma_all = np.zeros((M, 3, 3), dtype=np.float64)
    if good.any():
        P_good = neo_hookean_P(F[good], mu=mu, lmbda=lmbda)
        sigma_all[good] = cauchy_from_P(P_good, F[good])

    # Extract sym6 from per-tet Cauchy
    sym6_tet = np.zeros((M, 6), dtype=np.float64)
    sym6_tet[:, 0] = sigma_all[:, 0, 0]  # xx
    sym6_tet[:, 1] = sigma_all[:, 1, 1]  # yy
    sym6_tet[:, 2] = sigma_all[:, 2, 2]  # zz
    sym6_tet[:, 3] = sigma_all[:, 0, 1]  # xy
    sym6_tet[:, 4] = sigma_all[:, 0, 2]  # xz
    sym6_tet[:, 5] = sigma_all[:, 1, 2]  # yz

    # Volume-weighted scatter to nodes
    sym6_node_accum = np.zeros((N, 6), dtype=np.float64)
    vol_node_accum = np.zeros(N, dtype=np.float64)
    good_cells = merged_cells[good]
    good_sym6 = sym6_tet[good]
    good_vol = volumes[good]
    for j in range(4):
        np.add.at(sym6_node_accum, good_cells[:, j],
                  good_sym6 * good_vol[:, None])
        np.add.at(vol_node_accum, good_cells[:, j], good_vol)
    vol_node_accum = np.maximum(vol_node_accum, 1e-30)
    cauchy_sym6_node = (sym6_node_accum / vol_node_accum[:, None]).astype(np.float32)

    # Also compute von Mises at nodes for verification
    vm_tet = von_mises(sigma_all[good]) if good.any() else np.zeros(0)
    vm_node_accum = np.zeros(N, dtype=np.float64)
    if good.any():
        for j in range(4):
            np.add.at(vm_node_accum, good_cells[:, j], vm_tet * good_vol)
    vm_per_node = (vm_node_accum / vol_node_accum).astype(np.float32)

    return cauchy_sym6_node, vm_per_node


def compute_stress_on_merged_mesh(merged_coords, merged_cells, u_merged,
                                  mu=370.0, lmbda=1200.0,
                                  quality_threshold=1e-4):
    """Compute stress on the physical merged mesh.

    This matches the FEniCS computation: F = I + grad(u) where u is the
    displacement field on the physical mesh and grad is w.r.t. physical coords.

    Sliver tets (quality < quality_threshold) are excluded from stress
    computation to avoid spurious gradients. FEniCS handles these via
    CG1 L2 projection (consistent mass matrix); we filter them instead.

    Parameters
    ----------
    merged_coords : (N, 3) — reference node coordinates (physical frame)
    merged_cells : (M, 4) — tet connectivity
    u_merged : (N, 3) — displacement at each node (physical frame)
    mu, lmbda : float — Neo-Hookean material parameters
    quality_threshold : float — minimum tet quality (volume/max_edge^3)

    Returns
    -------
    vm_per_tet : (M,) — von Mises stress per tet (0 for filtered tets)
    P_per_tet : (M, 3, 3) — PK1 stress per tet
    F_per_tet : (M, 3, 3) — deformation gradient per tet
    dNdx : (M, 4, 3) — shape function gradients
    volumes : (M,) — tet volumes
    vm_per_node : (N,) — von Mises averaged to nodes
    """
    N = len(merged_coords)
    M = len(merged_cells)

    # Shape function gradients on reference physical mesh
    dNdx, volumes = tet4_dNdx(merged_coords, merged_cells)  # (M,4,3), (M,)

    # Filter sliver tets
    quality = tet_quality(merged_coords, merged_cells)
    good = quality >= quality_threshold
    n_filtered = M - good.sum()

    # Displacement gradient: grad(u)_ij = sum_k u_nodes[k,i] * dNdx[k,j]
    u_nodes = u_merged[merged_cells]  # (M, 4, 3)
    grad_u = np.einsum('mki,mkj->mij', u_nodes, dNdx)  # (M, 3, 3)

    # Deformation gradient: F = I + grad(u)
    F = np.eye(3) + grad_u  # (M, 3, 3)

    # Initialize outputs
    P = np.zeros((M, 3, 3), dtype=np.float64)
    vm_per_tet = np.zeros(M, dtype=np.float64)

    if good.any():
        # Neo-Hookean stress only on good tets
        P[good] = neo_hookean_P(F[good], mu=mu, lmbda=lmbda)

        # Cauchy stress and von Mises per tet
        sigma = cauchy_from_P(P[good], F[good])
        vm_per_tet[good] = von_mises(sigma)

    # Volume-weighted average tet VM to nodes (mimics CG1 L2 projection).
    # Only good tets contribute. This prevents sliver tets with spurious
    # stress from corrupting the node-averaged field.
    vm_node_accum = np.zeros(N, dtype=np.float64)
    vol_node_accum = np.zeros(N, dtype=np.float64)
    good_cells = merged_cells[good]
    good_vm = vm_per_tet[good]
    good_vol = volumes[good]
    for j in range(4):
        np.add.at(vm_node_accum, good_cells[:, j], good_vm * good_vol)
        np.add.at(vol_node_accum, good_cells[:, j], good_vol)
    vol_node_accum = np.maximum(vol_node_accum, 1e-30)
    vm_per_node = (vm_node_accum / vol_node_accum).astype(np.float32)

    return vm_per_tet, P, F, dNdx, volumes, vm_per_node


# ── Reaction force assembly ─────────────────────────────────────────────────

def assemble_internal_force(P_tets, dNdx, volumes, cells, n_nodes):
    """Assemble internal force vector from per-tet PK1 stress.

    f_int[node, i] = sum over elements containing node:
        P[elem, i, j] * dN_node/dx_j * volume[elem]

    Parameters
    ----------
    P_tets : (M, 3, 3) — first Piola-Kirchhoff stress per tet
    dNdx : (M, 4, 3) — shape function gradients
    volumes : (M,) — element volumes
    cells : (M, 4) — tet connectivity
    n_nodes : int — total number of nodes

    Returns
    -------
    f_int : (n_nodes, 3) — internal force vector
    """
    # Nodal force contribution: f[node] = P @ dN/dx * V
    # For each tet, for each local node j:
    #   f_contrib[m, j, i] = sum_k P[m, i, k] * dNdx[m, j, k] * volumes[m]
    f_contrib = np.einsum('mik,mjk,m->mji', P_tets, dNdx, volumes)  # (M, 4, 3)

    f_int = np.zeros((n_nodes, 3), dtype=np.float64)
    for j in range(4):
        np.add.at(f_int, cells[:, j], f_contrib[:, j, :])

    return f_int


# ── Plate force-displacement utilities ────────────────────────────────────────

def precompute_boundary_geometry(cells, dNdx, volumes, right_node_ids):
    """Precompute geometry for boundary-only reaction force computation.

    Identifies tets that have at least one node on the right boundary, so
    stress can be computed on the boundary subset instead of the whole mesh.

    Parameters
    ----------
    cells : (M, 4) — tet connectivity
    dNdx : (M, 4, 3) — shape function gradients
    volumes : (M,) — element volumes
    right_node_ids : (K,) int — right boundary node indices

    Returns
    -------
    dict with keys:
        tet_indices : (B,) int — indices of boundary tets
        cells_bdy : (B, 4) — connectivity of boundary tets
        dNdx_bdy : (B, 4, 3) — shape function gradients for boundary tets
        volumes_bdy : (B,) — volumes of boundary tets
    """
    is_right = np.zeros(cells.max() + 1, dtype=bool)
    is_right[right_node_ids] = True
    has_right_node = is_right[cells].any(axis=1)
    tet_indices = np.where(has_right_node)[0]
    return {
        'tet_indices': tet_indices,
        'cells_bdy': cells[tet_indices],
        'dNdx_bdy': dNdx[tet_indices],
        'volumes_bdy': volumes[tet_indices],
    }


def boundary_reaction_force(u_disp, coords, n_nodes, right_node_ids,
                             cy, cz, bdy_geom, mu=MICROPLATE_LATTICE.mu, lmbda=MICROPLATE_LATTICE.lmbda):
    """Compute generalized reaction force using only boundary tets.

    Much faster than the full-mesh approach (stress_from_disps +
    assemble_internal_force) because it only computes stress for
    the ~1-2% of tets that touch the right boundary.

    Parameters
    ----------
    u_disp : (N, 3) — displacement field at all nodes
    coords : (N, 3) — reference coordinates
    n_nodes : int — total number of nodes
    right_node_ids : (K,) int — right boundary node indices
    cy, cz : float — rotation center for torque
    bdy_geom : dict — from precompute_boundary_geometry
    mu, lmbda : float — material parameters

    Returns
    -------
    gen_force : (4,) float64 — [Fx, Mx, Fy, Fz]
    """
    cells_bdy = bdy_geom['cells_bdy']
    dNdx_bdy = bdy_geom['dNdx_bdy']
    volumes_bdy = bdy_geom['volumes_bdy']

    # Compute stress only for boundary tets
    u_nodes = u_disp[cells_bdy]  # (B, 4, 3)
    grad_u = np.einsum('mki,mkj->mij', u_nodes, dNdx_bdy)
    F = np.eye(3) + grad_u
    P_bdy = neo_hookean_P(F, mu, lmbda)

    # Assemble internal forces from boundary tets only
    f_int = assemble_internal_force(
        P_bdy, dNdx_bdy, volumes_bdy, cells_bdy, n_nodes)

    return generalized_force_from_f_int(f_int, right_node_ids, coords, cy, cz)


def identify_right_boundary_nodes(coords, eps=1e-2):
    """Find nodes on the right face (x = x_max) of a plate mesh.

    Matches FEniCS boundary detection: ``df.near(x[0], x_max, eps)``.

    Parameters
    ----------
    coords : (N, 3) — reference node coordinates
    eps : float — tolerance for boundary detection

    Returns
    -------
    node_ids : (K,) int array — right boundary node indices
    x_max : float — maximum x coordinate
    """
    x = coords[:, 0]
    x_max = x.max()
    node_ids = np.where(x >= x_max - eps)[0]
    return node_ids, float(x_max)


def generalized_force_from_f_int(f_int, right_node_ids, coords, cy, cz):
    """Compute 4D generalized force from internal force vector at right boundary.

    The four components are work-conjugate to the generalized displacement
    (stretch, twist, shear_y, shear_z):

    - Fx: axial force (conjugate to stretch dx)
    - Mx: torque about x-axis (conjugate to twist angle theta)
    - Fy: shear y force (conjugate to shear dy)
    - Fz: shear z force (conjugate to shear dz)

    Parameters
    ----------
    f_int : (N, 3) — assembled internal force vector
    right_node_ids : (K,) int — indices of right-boundary nodes
    coords : (N, 3) — reference node coordinates
    cy : float — y-coordinate of twist rotation center
    cz : float — z-coordinate of twist rotation center

    Returns
    -------
    gen_force : (4,) float64 — [Fx, Mx, Fy, Fz]
    """
    f_bdy = f_int[right_node_ids]  # (K, 3)
    y = coords[right_node_ids, 1]
    z = coords[right_node_ids, 2]

    Fx = -f_bdy[:, 0].sum()
    Fy = -f_bdy[:, 1].sum()
    Fz = -f_bdy[:, 2].sum()
    # Torque about x-axis: Mx = -sum( (y-cy)*fz - (z-cz)*fy )
    Mx = -((y - cy) * f_bdy[:, 2] - (z - cz) * f_bdy[:, 1]).sum()

    return np.array([Fx, Mx, Fy, Fz], dtype=np.float64)


def boundary_reaction_force_from_cauchy(u_disp, cauchy_nodal, coords, n_nodes,
                                        right_node_ids, cy, cz, bdy_geom):
    """Compute generalized reaction force from displacement + Cauchy stress.

    Unlike boundary_reaction_force(), this does NOT apply a constitutive law.
    The Cauchy stress is provided directly (e.g., from FEM GT or stress head
    prediction). Only kinematic operations (F from u, Cauchy→Piola conversion)
    and geometry (shape function integration) are used.

    Parameters
    ----------
    u_disp : (N, 3) — displacement field at all nodes
    cauchy_nodal : (N, 3, 3) or (N, 6) — Cauchy stress at all nodes
        If (N, 6): sym6 Voigt order [xx, yy, zz, xy, xz, yz]
    coords : (N, 3) — reference coordinates
    n_nodes : int — total number of nodes
    right_node_ids : (K,) int — right boundary node indices
    cy, cz : float — rotation center for torque
    bdy_geom : dict — from precompute_boundary_geometry

    Returns
    -------
    gen_force : (4,) float64 — [Fx, Mx, Fy, Fz]
    """
    cells_bdy = bdy_geom['cells_bdy']
    dNdx_bdy = bdy_geom['dNdx_bdy']
    volumes_bdy = bdy_geom['volumes_bdy']

    # Convert sym6 to 3x3 if needed
    # Voigt order: [xx, yy, zz, xy, xz, yz] = indices [0, 1, 2, 3, 4, 5]
    if cauchy_nodal.ndim == 2 and cauchy_nodal.shape[1] == 6:
        s = cauchy_nodal
        cauchy_33 = np.zeros((len(s), 3, 3), dtype=s.dtype)
        cauchy_33[:, 0, 0] = s[:, 0]                              # xx
        cauchy_33[:, 1, 1] = s[:, 1]                              # yy
        cauchy_33[:, 2, 2] = s[:, 2]                              # zz
        cauchy_33[:, 0, 1] = cauchy_33[:, 1, 0] = s[:, 3]         # xy
        cauchy_33[:, 0, 2] = cauchy_33[:, 2, 0] = s[:, 4]         # xz
        cauchy_33[:, 1, 2] = cauchy_33[:, 2, 1] = s[:, 5]         # yz
    else:
        cauchy_33 = cauchy_nodal

    # Compute F at boundary tets from displacement
    u_nodes = u_disp[cells_bdy]  # (B, 4, 3)
    grad_u = np.einsum('mki,mkj->mij', u_nodes, dNdx_bdy)  # (B, 3, 3)
    F = np.eye(3) + grad_u  # (B, 3, 3)

    # Average Cauchy stress at tet centers from node values
    cauchy_tet = cauchy_33[cells_bdy].mean(axis=1)  # (B, 3, 3)

    # Cauchy → Piola: P = J * σ * F^(-T)
    J = np.linalg.det(F)  # (B,)
    F_inv_T = np.linalg.inv(F).transpose(0, 2, 1)  # (B, 3, 3)
    P_bdy = J[:, None, None] * np.einsum('mij,mjk->mik', cauchy_tet, F_inv_T)

    # Assemble internal forces from boundary tets
    f_int = assemble_internal_force(P_bdy, dNdx_bdy, volumes_bdy, cells_bdy, n_nodes)

    return generalized_force_from_f_int(f_int, right_node_ids, coords, cy, cz)


# MicroPlate lattice 4-DOF action speeds: (stretch, twist, shear_y, shear_z).
# Sourced from data/manifests/microplate_lattice_constants.json via
# leia.regime — re-exported here for callers that import the constant
# directly from leia.stress.
MICROPLATE_LATTICE_SPEEDS = MICROPLATE_LATTICE.speeds


def compute_gen_disp(inputs, speeds=MICROPLATE_LATTICE_SPEEDS):
    """Compute cumulative generalized displacement from discrete actions.

    Parameters
    ----------
    inputs : (T, 4) int — per-step actions [stretch, twist, shear_y, shear_z]
    speeds : tuple of 4 floats — speed constants per DOF

    Returns
    -------
    gen_disp : (T+1, 4) float64 — cumulative [dx, theta, dy, dz], frame 0 = zeros
    """
    speeds = np.array(speeds, dtype=np.float64)
    T = inputs.shape[0]
    gen_disp = np.zeros((T + 1, 4), dtype=np.float64)
    gen_disp[1:] = np.cumsum(inputs * speeds[None, :], axis=0)
    return gen_disp


def compute_cumulative_bc(inputs, speeds=MICROPLATE_LATTICE_SPEEDS):
    """Cumulative 4-DOF boundary condition from discrete per-step actions.

    Parameters
    ----------
    inputs : (n_traj, n_inc, 4) — per-step actions in {-1, 0, +1}
    speeds : tuple of 4 floats — speed constants per DOF

    Returns
    -------
    cumulative : (n_traj, n_inc, 4) float32 — cumulative BC per trajectory
    """
    speeds_arr = np.array(speeds, dtype=np.float32)
    return np.cumsum(inputs * speeds_arr[None, None, :], axis=1).astype(np.float32)


def boundary_force_from_grad_head(grad_u_phys, coords, n_nodes,
                                  right_node_ids, cy, cz, bdy_geom,
                                  mu, lmbda):
    """Boundary reaction force using per-node grad_u from the gradient head.

    Averages 4 node grad_u values per tet to get per-tet grad_u, computes
    Neo-Hookean PK1, assembles internal force, projects onto generalized
    coordinates at the right boundary.
    """
    cells_bdy = bdy_geom['cells_bdy']
    dNdx_bdy = bdy_geom['dNdx_bdy']
    volumes_bdy = bdy_geom['volumes_bdy']

    grad_u_tet = grad_u_phys[cells_bdy].mean(axis=1)
    F_bdy = np.eye(3) + grad_u_tet
    P_bdy = neo_hookean_P(F_bdy, mu=mu, lmbda=lmbda)

    f_int = assemble_internal_force(
        P_bdy, dNdx_bdy, volumes_bdy, cells_bdy, n_nodes)
    return generalized_force_from_f_int(f_int, right_node_ids, coords, cy, cz)


def boundary_force_from_stress_head(sym6_nodes, u_decoded, coords, n_nodes,
                                    right_node_ids, cy, cz, bdy_geom):
    """Boundary reaction force using per-node Cauchy sym6 from the stress head.

    Stress-head VALUES are accurate (trained directly on FEM ground truth),
    but per-tet GRADIENTS of decoded displacement are noisy. Using
    P = NeoHookean(F_decoded) here would give wildly wrong reaction forces
    (sign flips, order-of-magnitude errors). Instead, we use Piola = J σ F^{-T}
    with F from the decoded u — F enters only first-order, so noise there is
    tolerated.
    """
    cells_bdy = bdy_geom['cells_bdy']
    dNdx_bdy = bdy_geom['dNdx_bdy']
    volumes_bdy = bdy_geom['volumes_bdy']

    # Per-tet Cauchy: average sym6 over 4 tet nodes, then expand to 3x3.
    sym6_tet = sym6_nodes[cells_bdy].mean(axis=1)                  # (M_bdy, 6)
    sigma_tet = np.zeros((sym6_tet.shape[0], 3, 3), dtype=sym6_tet.dtype)
    sigma_tet[:, 0, 0] = sym6_tet[:, 0]
    sigma_tet[:, 1, 1] = sym6_tet[:, 1]
    sigma_tet[:, 2, 2] = sym6_tet[:, 2]
    sigma_tet[:, 0, 1] = sigma_tet[:, 1, 0] = sym6_tet[:, 3]
    sigma_tet[:, 0, 2] = sigma_tet[:, 2, 0] = sym6_tet[:, 4]
    sigma_tet[:, 1, 2] = sigma_tet[:, 2, 1] = sym6_tet[:, 5]

    # Per-tet F from decoded u (used only in F^{-T}; gradient noise is fine).
    grad_u_tet = np.einsum('mki,mkj->mij',
                           u_decoded[cells_bdy], dNdx_bdy)         # (M_bdy, 3, 3)
    F_tet = np.eye(3) + grad_u_tet
    J_tet = np.linalg.det(F_tet)
    F_inv_T = np.linalg.inv(F_tet).transpose(0, 2, 1)
    # P = J σ F^{-T}
    P_tet = J_tet[:, None, None] * np.einsum(
        'mij,mjk->mik', sigma_tet, F_inv_T)

    f_int = assemble_internal_force(
        P_tet, dNdx_bdy, volumes_bdy, cells_bdy, n_nodes)
    return generalized_force_from_f_int(f_int, right_node_ids, coords, cy, cz)

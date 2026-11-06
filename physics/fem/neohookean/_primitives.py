"""dolfin primitives for the Neo-Hookean solver.

All values and conventions match data_curation/fem/run_plate_fem.py exactly
(byte-equivalence is the gate). Import dolfin at module-top — errors clearly
under the wrong conda env.
"""

import numpy as np
import dolfin as df


# Boundary detection tolerance — matches df.near(x[0], x_*, eps=1e-2).
BOUNDARY_EPS = 1e-2

# Newton solver parameters — matches both data_curation and design_search.
NEWTON_MAX_ITER = 15
NEWTON_RELATIVE_TOLERANCE = 1e-3
NEWTON_LINEAR_SOLVER = "mumps"


def build_dolfin_mesh(coords, topo):
    """Build a dolfin Mesh from numpy arrays via MeshEditor.

    MeshEditor preserves vertex ordering — this is load-bearing for
    byte-equivalence with the canonical training-data NPZs.
    """
    mesh = df.Mesh()
    editor = df.MeshEditor()
    editor.open(mesh, "tetrahedron", 3, 3)
    editor.init_vertices(len(coords))
    editor.init_cells(len(topo))
    for i, pt in enumerate(coords):
        editor.add_vertex(i, pt)
    for i, cell in enumerate(topo):
        editor.add_cell(i, cell)
    editor.close()
    return mesh


def get_mesh_geometry(coords):
    """Return (x_min, x_max, cy, cz) — left/right face x positions and y/z centers."""
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    z_coords = coords[:, 2]
    x_min = x_coords.min()
    x_max = x_coords.max()
    cy = (y_coords.min() + y_coords.max()) / 2.0
    cz = (z_coords.min() + z_coords.max()) / 2.0
    return x_min, x_max, cy, cz


class ControlExpression(df.UserExpression):
    """4-DOF parameterized boundary expression for the right face.

    Parameterized by (stretch, twist, shear_y, shear_z). For each material
    point on the right face, the prescribed displacement is:

      u_x = stretch
      u_y = (cy + Yc·cos θ − Zc·sin θ + sy) − Y
      u_z = (cz + Yc·sin θ + Zc·cos θ + sz) − Z

    where Yc = Y − cy, Zc = Z − cz, θ = twist.

    The kernel mutates the four scalar attributes between Newton solves
    to advance the load increment.
    """

    def __init__(self, cy, cz, **kwargs):
        self.stretch = 0.0
        self.twist = 0.0
        self.sy = 0.0
        self.sz = 0.0
        self.cy = cy
        self.cz = cz
        super().__init__(**kwargs)

    def eval(self, values, x):
        Y, Z = x[1], x[2]
        Yc, Zc = Y - self.cy, Z - self.cz
        c, s = np.cos(self.twist), np.sin(self.twist)
        Yn = self.cy + Yc * c - Zc * s + self.sy
        Zn = self.cz + Yc * s + Zc * c + self.sz
        values[0] = self.stretch
        values[1] = Yn - Y
        values[2] = Zn - Z

    def value_shape(self):
        return (3,)


def build_dirichlet_bcs(V, x_min, x_max, control_expr):
    """Left face (x=x_min) clamped; right face (x=x_max) driven by control_expr."""

    def left_boundary(x, on_boundary):
        return on_boundary and df.near(x[0], x_min, eps=BOUNDARY_EPS)

    def right_boundary(x, on_boundary):
        return on_boundary and df.near(x[0], x_max, eps=BOUNDARY_EPS)

    return [
        df.DirichletBC(V, df.Constant((0, 0, 0)), left_boundary),
        df.DirichletBC(V, control_expr, right_boundary),
    ]


def build_neohookean_problem(V, u, bcs, mu, lmbda):
    """Build a Neo-Hookean NonlinearVariationalProblem + Newton solver.

    Energy density (per unit reference volume):

        ψ = (μ/2)(I_C − 3) − μ ln J + (λ/2)(ln J)²

    with J = det F, I_C = tr(F^T F), F = I + ∇u.

    Returns (problem, solver). The solver is configured for MUMPS Newton
    with max_iter=15, relative_tolerance=1e-3 — matches today's exact
    settings on both data_curation and design_search sides.
    """
    v = df.TestFunction(V)
    d = u.geometric_dimension()
    I = df.Identity(d)
    F_def = I + df.grad(u)
    J = df.det(F_def)
    C_tensor = F_def.T * F_def
    Ic = df.tr(C_tensor)
    psi = ((mu / 2) * (Ic - 3) - mu * df.ln(J)
           + (lmbda / 2) * (df.ln(J)) ** 2)
    F_form = df.derivative(psi * df.dx, u, v)
    J_form = df.derivative(F_form, u, df.TrialFunction(V))

    problem = df.NonlinearVariationalProblem(F_form, u, bcs, J_form)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["linear_solver"] = NEWTON_LINEAR_SOLVER
    solver.parameters["newton_solver"]["maximum_iterations"] = NEWTON_MAX_ITER
    solver.parameters["newton_solver"]["relative_tolerance"] = NEWTON_RELATIVE_TOLERANCE
    return problem, solver


def vertex_displacements(u, mesh):
    """Extract (N, 3) float32 displacement at vertex order from a dolfin Function."""
    flat_vals = u.compute_vertex_values(mesh)
    return flat_vals.reshape(3, -1).T.astype(np.float32)

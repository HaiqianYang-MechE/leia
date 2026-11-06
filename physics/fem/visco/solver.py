"""Arruda-Boyce 3-branch viscoelastic FEM solver (dolfinx / FEniCSx 0.8).

Quasi-static Arruda-Boyce equilibrium + 3-branch neo-Hookean non-equilibrium
hyperelasticity, with Newmark-β time integration on a uniform-box plate mesh.

Lifted from data_curation/fem/run_visco_load_reverse.py:run_simulation.
The driver retains action generation, NPZ writing, and HDF5 consolidation;
this module is the parameterized kernel.

Conda env: `fenicsx` (see environment-fenicsx.yml).
"""

import time

import numpy as np


def solve_arruda_boyce(
    actions,
    *,
    material,
    box_min=(-4.9, -4.9, -0.98),
    box_max=(4.9, 4.9, 0.98),
    mesh_resolution=(10, 10, 2),
    speeds=(0.10, 0.06, 0.10),
    n_substeps=5,
    dt_load=0.002,
    dt_release=0.0002,
    p1_steps=None,
    p2_steps=0,
    p3_steps=0,
    release_ramp=0,
    newmark=(0.0, 0.5, 0.25),
    log_progress=True,
):
    """Run the 3-phase Arruda-Boyce viscoelastic plate simulation.

    Parameters
    ----------
    actions : (n_action_steps, 4) int — per-step discrete {-1, 0, +1} actions
        for [stretch, twist, shear_y, shear_z]. Phase 1 ramps these into BCs.
    material : dict — {'Geq_0', 'lambdaL', 'Kbulk', 'rho',
                       'branches': [(G, tau), ...]}.
    box_min, box_max : 3-tuples — plate box extents.
    mesh_resolution : 3-tuple — uniform tet mesh resolution per axis.
    speeds : 3-tuple — (speed_s, speed_t, speed_sh) per action step.
    n_substeps : int — FEM sub-steps per action step (Phase 1).
    dt_load : float — timestep during Phase 1-2.
    dt_release : float — timestep during Phase 3.
    p1_steps : int or None — Phase 1 step count; default = n_action_steps * n_substeps.
    p2_steps : int — Phase 2 (hold) step count.
    p3_steps : int — Phase 3 (release with ramped traction) step count.
    release_ramp : int — Phase 3 ramp duration; 0 → ramp = p3_steps.
    newmark : 3-tuple — (alpha, gamma, beta) for Newmark-β.
    log_progress : bool — print per-step progress to stdout.

    Returns
    -------
    dict with keys:
      'status'         : 'ok' | 'diverged' | 'failed'
      'error'          : str (only if not 'ok')
      'time_seconds'   : float
      'coords'         : (N, 3) float32
      'topology'       : (M, 4) int32
      'displacements'  : (T+1, N, 3) float32
      'velocities'     : (T+1, N, 3) float32
      'cauchy_stress'  : (T+1, N, 3, 3) float32
      'Cv'             : (T+1, n_branches, N, 6) float32
      'pressure'       : (T+1, N) float32
      'bc_state'       : (T+1, 4) float32
      'phase_labels'   : (T+1,) int8
      'times'          : (T+1,) float64
      'actions_5dof'   : (T+1, 5) float32  — [d_s, d_t, d_sy, d_sz, is_free]
      'traction_scale' : (T+1,) float32
      'reaction_force' : (T+1, 3) float32  — Fx/Fy/Fz on right face
      'n_substeps'     : int
      'p1_steps'       : int
      'p2_steps'       : int
      'p3_steps'       : int
      'actions_expanded': (p1_steps, 4) int  — actions repeated by n_substeps
    """
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import fem, mesh, log
    from dolfinx.fem import (Constant, dirichletbc, Function, functionspace,
                             Expression)
    from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
    from dolfinx.nls.petsc import NewtonSolver
    import ufl
    from ufl import (TestFunctions, TrialFunction, Identity, grad, det, inv,
                     tr, sqrt, conditional, gt, inner, derivative, split, dev,
                     dot)
    import basix
    from basix.ufl import element, mixed_element, quadrature_element

    log.set_log_level(log.LogLevel.WARNING)

    actions = np.asarray(actions, dtype=np.int8)
    n_action_steps = actions.shape[0]
    speed_s, speed_t, speed_sh = speeds
    alpha_nm, gamma_nm, beta_nm = newmark
    nx, ny, nz = mesh_resolution
    xmin, ymin, zmin = box_min
    xmax, ymax, zmax = box_max

    if p1_steps is None:
        p1_steps = n_action_steps * n_substeps
    total_steps = p1_steps + p2_steps + p3_steps
    if p1_steps % n_action_steps != 0 and log_progress:
        print(f"  WARNING: p1_steps={p1_steps} not divisible by "
              f"n_action_steps={n_action_steps}.")

    t_start = time.time()
    n_branches = len(material['branches'])

    if log_progress:
        print("=" * 60)
        print("Arruda-Boyce 3-branch viscoelastic plate")
        print("=" * 60)
        print(f"  Material: equilibrium + {n_branches} branches")
        for i, (G, tau) in enumerate(material['branches']):
            print(f"    Branch {i}: G={G} kPa, tau={tau} s")
        print(f"  dt Phase 1-2 = {dt_load} s, Phase 3 = {dt_release} s")
        print(f"  Phase 1 (Load):    {p1_steps} steps = {p1_steps*dt_load:.3f}s")
        print(f"  Phase 2 (Hold):    {p2_steps} steps = {p2_steps*dt_load:.3f}s")
        print(f"  Phase 3 (Release): {p3_steps} steps = {p3_steps*dt_release:.3f}s")
        print()

    # ------------------------------------------------------------------
    # 1. Mesh
    # ------------------------------------------------------------------
    domain = mesh.create_box(
        MPI.COMM_SELF,
        [[xmin, ymin, zmin], [xmax, ymax, zmax]],
        [nx, ny, nz],
        mesh.CellType.tetrahedron)

    coords = domain.geometry.x.copy()
    n_vertices = coords.shape[0]
    topo_map = domain.topology.connectivity(3, 0)
    n_cells = domain.topology.index_map(3).size_local
    topology = np.array([topo_map.links(i) for i in range(n_cells)])

    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0
    if log_progress:
        print(f"  Mesh: {n_vertices} vertices, {n_cells} cells")

    # ------------------------------------------------------------------
    # 2. Function spaces
    # ------------------------------------------------------------------
    U2 = element("Lagrange", domain.basix_cell(), 2, shape=(3,))
    P1 = element("Lagrange", domain.basix_cell(), 1)
    TH = mixed_element([U2, P1])
    ME = functionspace(domain, TH)

    P0 = quadrature_element(domain.basix_cell(), degree=2, scheme="default")
    T0 = basix.ufl.blocked_element(P0, shape=(3, 3))
    V3 = functionspace(domain, T0)

    V2 = functionspace(domain, U2)

    U1_out = element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    P1_out = element("Lagrange", domain.basix_cell(), 1)
    T1_out = element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
    V_u1 = functionspace(domain, U1_out)
    V_p1 = functionspace(domain, P1_out)
    V_t1 = functionspace(domain, T1_out)

    n_dofs = ME.dofmap.index_map.size_local * ME.dofmap.index_map_bs
    if log_progress:
        print(f"  DOFs: {n_dofs}")

    # ------------------------------------------------------------------
    # 3. Solution + state functions
    # ------------------------------------------------------------------
    w = Function(ME)
    u, p = split(w)
    w_old = Function(ME)
    u_old, p_old = split(w_old)
    u_test, p_test = TestFunctions(ME)
    dw = TrialFunction(ME)

    Cv_old = [Function(V3) for _ in range(n_branches)]
    Cv_temp = [Function(V3) for _ in range(n_branches)]

    def identity_func(x):
        values = np.zeros((9, x.shape[1]), dtype=np.float64)
        values[0] = 1.0
        values[4] = 1.0
        values[8] = 1.0
        return values

    for cv in Cv_old:
        cv.interpolate(identity_func)

    v_old = Function(V2)
    a_old = Function(V2)

    # ------------------------------------------------------------------
    # 4. Material constants
    # ------------------------------------------------------------------
    Geq_0 = Constant(domain, PETSc.ScalarType(material['Geq_0']))
    Kbulk = Constant(domain, PETSc.ScalarType(material['Kbulk']))
    lambdaL = Constant(domain, PETSc.ScalarType(material['lambdaL']))
    rho = Constant(domain, PETSc.ScalarType(material['rho']))

    Gneq_list = []
    tau_list = []
    for G_i, tau_i in material['branches']:
        Gneq_list.append(Constant(domain, PETSc.ScalarType(G_i)))
        tau_list.append(Constant(domain, PETSc.ScalarType(tau_i)))

    alpha_c = Constant(domain, PETSc.ScalarType(alpha_nm))
    gamma_c = Constant(domain, PETSc.ScalarType(gamma_nm))
    beta_c = Constant(domain, PETSc.ScalarType(beta_nm))
    dk = Constant(domain, PETSc.ScalarType(dt_load))

    # ------------------------------------------------------------------
    # 5. Constitutive model
    # ------------------------------------------------------------------
    def safe_sqrt(x):
        return sqrt(x + 1.0e-16)

    def F_calc(u_):
        return Identity(3) + grad(u_)

    def dist_part(A):
        return A / (det(A) ** (1.0 / 3.0))

    def Cv_update(u_, Cv_old_i, tau_i):
        F_ = F_calc(u_)
        J_ = det(F_)
        C_ = F_.T * F_
        return dist_part(Cv_old_i + (dk / tau_i) * J_ ** (-2.0 / 3.0) * C_)

    def zeta_calc(u_):
        F_ = F_calc(u_)
        J_ = det(F_)
        Fbar = J_ ** (-1.0 / 3.0) * F_
        Cbar = Fbar.T * Fbar
        lb = safe_sqrt(tr(Cbar) / 3.0)
        z = lb / lambdaL
        z = conditional(gt(z, 0.99), 0.99, z)
        beta_L = z * (3.0 - z ** 2.0) / (1.0 - z ** 2.0)
        return (lambdaL / (3.0 * lb)) * beta_L

    def T_eq_calc(u_, p_):
        F_ = F_calc(u_)
        J_ = det(F_)
        Fbar = J_ ** (-1.0 / 3.0) * F_
        Bbar = Fbar * Fbar.T
        Geq = Geq_0 * zeta_calc(u_)
        return (1.0 / J_) * Geq * dev(Bbar) - p_ * Identity(3)

    def T_neq_calc(u_, Cv_i, Gneq_i):
        F_ = F_calc(u_)
        J_ = det(F_)
        C_ = F_.T * F_
        return (J_ ** (-5.0 / 3.0) * Gneq_i *
                (F_ * inv(Cv_i) * F_.T -
                 (1.0 / 3.0) * inner(C_, inv(Cv_i)) * Identity(3)))

    def Piola_calc(u_, p_, Cv_list_, Gneq_list_):
        F_ = F_calc(u_)
        J_ = det(F_)
        T_total = T_eq_calc(u_, p_)
        for Cv_i, G_i in zip(Cv_list_, Gneq_list_):
            T_total = T_total + T_neq_calc(u_, Cv_i, G_i)
        return J_ * T_total * inv(F_.T)

    # ------------------------------------------------------------------
    # 6. Newmark-beta time integration
    # ------------------------------------------------------------------
    def update_a(u_, u_old_, v_old_, a_old_):
        return ((u_ - u_old_ - dk * v_old_) / beta_c / dk ** 2
                - (1 - 2 * beta_c) / (2 * beta_c) * a_old_)

    def update_v(a_, v_old_, a_old_):
        return v_old_ + dk * ((1 - gamma_c) * a_old_ + gamma_c * a_)

    a_new = update_a(u, u_old, v_old, a_old)
    v_new = update_v(a_new, v_old, a_old)
    u_avg = alpha_c * u_old + (1 - alpha_c) * u
    p_avg = alpha_c * p_old + (1 - alpha_c) * p

    Cv_new = [Cv_update(u_avg, Cv_old[i], tau_list[i]) for i in range(n_branches)]
    Piola = Piola_calc(u_avg, p_avg, Cv_new, Gneq_list)

    # ------------------------------------------------------------------
    # 7. Boundary facet tags + measures (needed for traction BC)
    # ------------------------------------------------------------------
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)

    left_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], xmin))
    right_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], xmax))

    boundaries = [
        (1, lambda x: np.isclose(x[0], xmin)),
        (2, lambda x: np.isclose(x[0], xmax)),
    ]
    facet_indices, facet_markers = [], []
    for (marker, locator) in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tags = mesh.meshtags(domain, fdim,
                               facet_indices[sorted_facets],
                               facet_markers[sorted_facets])

    dx_q = ufl.Measure('dx', domain=domain,
                       metadata={'quadrature_degree': 2,
                                 'quadrature_rule': 'default'})
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags,
                     metadata={'quadrature_degree': 2})
    n_ref = ufl.FacetNormal(domain)

    traction_scale = Constant(domain, PETSc.ScalarType(0.0))
    Piola_frozen = Function(V_t1)

    # ------------------------------------------------------------------
    # 8. Weak form
    # ------------------------------------------------------------------
    Res = (inner(Piola, grad(u_test)) * dx_q
           + inner(rho * a_new, u_test) * dx_q
           + inner((det(F_calc(u_avg)) - 1) + p_avg / Kbulk, p_test) * dx_q
           - inner(traction_scale * dot(Piola_frozen, n_ref), u_test) * ds(2))
    Jac = derivative(Res, w, dw)

    # ------------------------------------------------------------------
    # 9. Dirichlet BCs + solvers
    # ------------------------------------------------------------------
    V0, _ = ME.sub(0).collapse()
    left_dofs = fem.locate_dofs_topological((ME.sub(0), V0), fdim, left_facets)
    right_dofs = fem.locate_dofs_topological((ME.sub(0), V0), fdim, right_facets)

    u_zero = Function(V0)
    u_right = Function(V0)
    bc_left = dirichletbc(u_zero, left_dofs, ME.sub(0))
    bc_right = dirichletbc(u_right, right_dofs, ME.sub(0))

    def update_right_bc(stretch, twist, sy, sz):
        def bc_values(x):
            Y, Z = x[1], x[2]
            Yc, Zc = Y - cy, Z - cz
            c, s = np.cos(twist), np.sin(twist)
            Yn = cy + Yc * c - Zc * s + sy
            Zn = cz + Yc * s + Zc * c + sz
            vals = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)
            vals[0] = stretch
            vals[1] = Yn - Y
            vals[2] = Zn - Z
            return vals
        u_right.interpolate(bc_values)

    def make_solver(bcs):
        prob = NonlinearProblem(Res, w, bcs, Jac)
        slv = NewtonSolver(MPI.COMM_SELF, prob)
        slv.atol = 1e-8
        slv.rtol = 1e-8
        slv.max_it = 50
        slv.convergence_criterion = "incremental"
        ksp = slv.krylov_solver
        opts = PETSc.Options()
        pfx = ksp.getOptionsPrefix()
        opts[f"{pfx}ksp_type"] = "preonly"
        opts[f"{pfx}pc_type"] = "lu"
        opts[f"{pfx}pc_factor_mat_solver_type"] = "mumps"
        ksp.setFromOptions()
        return slv

    solver_constrained = make_solver([bc_left, bc_right])
    solver_traction = make_solver([bc_left])

    # ------------------------------------------------------------------
    # 10. Expression objects + output projectors
    # ------------------------------------------------------------------
    Cv_expr = [Expression(Cv_new[i], V3.element.interpolation_points())
               for i in range(n_branches)]
    a_expr = Expression(a_new, V2.element.interpolation_points())
    v_expr = Expression(v_new, V2.element.interpolation_points())

    a_func = Function(V2)
    v_func = Function(V2)

    trial_t = ufl.TrialFunction(V_t1)
    test_t = ufl.TestFunction(V_t1)
    a_proj_t = inner(trial_t, test_t) * ufl.dx

    F_avg = Identity(3) + grad(u_avg)
    J_avg = det(F_avg)
    Cauchy = (1.0 / J_avg) * Piola * F_avg.T
    L_cauchy = inner(Cauchy, test_t) * dx_q
    cauchy_projector = LinearProblem(
        a_proj_t, L_cauchy, [],
        petsc_options={"ksp_type": "cg", "ksp_rtol": 1e-12,
                       "ksp_max_it": 1000})

    Cv_projectors = []
    for i in range(n_branches):
        L_cv = inner(Cv_new[i], test_t) * dx_q
        Cv_projectors.append(LinearProblem(
            a_proj_t, L_cv, [],
            petsc_options={"ksp_type": "cg", "ksp_rtol": 1e-12,
                           "ksp_max_it": 1000}))

    L_piola = inner(Piola, test_t) * dx_q
    piola_projector = LinearProblem(
        a_proj_t, L_piola, [],
        petsc_options={"ksp_type": "cg", "ksp_rtol": 1e-12,
                       "ksp_max_it": 1000})

    u_out = Function(V_u1)
    v_out = Function(V_u1)
    p_out = Function(V_p1)

    def extract_sym6(tensor_array, n_nodes):
        T = tensor_array.reshape(n_nodes, 3, 3)
        return np.stack([T[:, 0, 0], T[:, 0, 1], T[:, 0, 2],
                         T[:, 1, 1], T[:, 1, 2], T[:, 2, 2]], axis=-1)

    actions_expanded = np.repeat(actions, n_substeps, axis=0)

    # ------------------------------------------------------------------
    # 11. Storage
    # ------------------------------------------------------------------
    all_displacements = []
    all_velocities = []
    all_cauchy = []
    all_pressure = []
    all_Cv = []
    all_bc_state = []
    all_phase_labels = []
    all_times = []
    all_actions = []
    all_traction_scale = []
    all_reaction_force = []

    piola_proj_func = Function(V_t1)
    rxn_form_components = []
    for comp in range(3):
        e_i = ufl.as_vector([1.0 if j == comp else 0.0 for j in range(3)])
        rxn_form_components.append(
            fem.form(dot(dot(piola_proj_func, n_ref), e_i) * ds(2)))

    def compute_reaction_force():
        piola_proj_func.x.array[:] = piola_projector.solve().x.array[:]
        return np.array(
            [fem.assemble_scalar(rxn_form_components[i]) for i in range(3)],
            dtype=np.float32)

    # Initial state (frame 0)
    u_out.interpolate(w.sub(0))
    all_displacements.append(
        u_out.x.array.copy().reshape(-1, 3).astype(np.float32))
    all_velocities.append(np.zeros((n_vertices, 3), dtype=np.float32))
    all_cauchy.append(np.zeros((n_vertices, 3, 3), dtype=np.float32))
    all_pressure.append(np.zeros(n_vertices, dtype=np.float32))
    all_Cv.append(np.zeros((n_branches, n_vertices, 6), dtype=np.float32))
    all_bc_state.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    all_phase_labels.append(0)
    all_times.append(0.0)
    all_actions.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    all_traction_scale.append(0.0)
    all_reaction_force.append(np.zeros(3, dtype=np.float32))

    # ------------------------------------------------------------------
    # 12. Main simulation loop
    # ------------------------------------------------------------------
    bc_state = {'stretch': 0.0, 'twist': 0.0, 'sy': 0.0, 'sz': 0.0}
    t_sim = 0.0
    error_msg = None
    final_status = 'ok'

    s_act = t_act = sy_act = sz_act = 0  # last action (used in actions_5dof)

    for step in range(total_steps):
        if step < p1_steps:
            phase = 1
            slv = solver_constrained

            action_idx = min(step // n_substeps, n_action_steps - 1)
            sub_in_action = step % n_substeps
            frac = (sub_in_action + 1) / n_substeps

            s_act = int(actions[action_idx, 0])
            t_act = int(actions[action_idx, 1])
            sy_act = int(actions[action_idx, 2])
            sz_act = int(actions[action_idx, 3])

            cur_stretch = bc_state['stretch'] + frac * s_act * speed_s
            cur_twist = bc_state['twist'] + frac * t_act * speed_t
            cur_sy = bc_state['sy'] + frac * sy_act * speed_sh
            cur_sz = bc_state['sz'] + frac * sz_act * speed_sh

            update_right_bc(cur_stretch, cur_twist, cur_sy, cur_sz)

            if sub_in_action == n_substeps - 1:
                bc_state['stretch'] += s_act * speed_s
                bc_state['twist'] += t_act * speed_t
                bc_state['sy'] += sy_act * speed_sh
                bc_state['sz'] += sz_act * speed_sh

        elif step < p1_steps + p2_steps:
            phase = 2
            slv = solver_constrained
            if step == p1_steps:
                update_right_bc(bc_state['stretch'], bc_state['twist'],
                                bc_state['sy'], bc_state['sz'])
                if log_progress:
                    print(f"\n  Phase 2 (Hold): BC frozen at "
                          f"stretch={bc_state['stretch']:.3f}, "
                          f"twist={bc_state['twist']:.3f}, "
                          f"sy={bc_state['sy']:.3f}, sz={bc_state['sz']:.3f}")

        else:
            phase = 3
            steps_into_p3 = step - (p1_steps + p2_steps)

            ramp_steps = release_ramp if release_ramp > 0 else p3_steps

            if steps_into_p3 == 0:
                piola_proj = piola_projector.solve()
                Piola_frozen.x.array[:] = piola_proj.x.array[:]
                traction_scale.value = 1.0
                dk.value = dt_release
                if log_progress:
                    print(f"\n  Phase 3 (Release): traction extracted, "
                          f"ramp={ramp_steps} steps, dt={dt_release}")

            if steps_into_p3 < ramp_steps:
                traction_scale.value = 1.0 - (steps_into_p3 + 1) / ramp_steps
            else:
                traction_scale.value = 0.0

            slv = solver_traction

        # --- Solve ---
        try:
            n_iters, converged = slv.solve(w)
            if not converged:
                error_msg = f"Newton diverged at step {step} (phase {phase})"
                final_status = 'diverged'
                if log_progress:
                    print(f"  DIVERGED at step {step} (phase {phase})")
                break
        except RuntimeError as e:
            error_msg = f"step {step} (phase {phase}): {e}"
            final_status = 'diverged'
            if log_progress:
                print(f"  ERROR at step {step} (phase {phase}): {e}")
            break

        # --- Update internal state ---
        for i in range(n_branches):
            Cv_temp[i].interpolate(Cv_expr[i])
            Cv_old[i].x.array[:] = Cv_temp[i].x.array[:]

        a_func.interpolate(a_expr)
        v_func.interpolate(v_expr)
        a_old.x.array[:] = a_func.x.array[:]
        v_old.x.array[:] = v_func.x.array[:]
        w_old.x.array[:] = w.x.array[:]
        dt_current = dt_release if phase == 3 else dt_load
        t_sim += dt_current

        # --- Store fields ---
        u_out.interpolate(w.sub(0))
        disp_vals = u_out.x.array.copy().reshape(-1, 3).astype(np.float32)
        all_displacements.append(disp_vals)

        v_out.interpolate(v_func)
        all_velocities.append(
            v_out.x.array.copy().reshape(-1, 3).astype(np.float32))

        cauchy_proj = cauchy_projector.solve()
        all_cauchy.append(
            cauchy_proj.x.array.copy().reshape(-1, 3, 3).astype(np.float32))

        p_out.interpolate(w.sub(1))
        all_pressure.append(p_out.x.array.copy().astype(np.float32))

        cv_frame = np.zeros((n_branches, n_vertices, 6), dtype=np.float32)
        for i in range(n_branches):
            cv_proj = Cv_projectors[i].solve()
            cv_frame[i] = extract_sym6(
                cv_proj.x.array.copy().astype(np.float32), n_vertices)
        all_Cv.append(cv_frame)

        bc_vals = np.array([bc_state['stretch'], bc_state['twist'],
                            bc_state['sy'], bc_state['sz']], dtype=np.float32)
        all_bc_state.append(bc_vals)
        all_phase_labels.append(phase)
        all_times.append(t_sim)

        if phase == 1:
            d_s = s_act * speed_s / n_substeps
            d_t = t_act * speed_t / n_substeps
            d_sy = sy_act * speed_sh / n_substeps
            d_sz = sz_act * speed_sh / n_substeps
            all_actions.append(
                np.array([d_s, d_t, d_sy, d_sz, 0.0], dtype=np.float32))
        elif phase == 2:
            all_actions.append(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        else:
            all_actions.append(
                np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        all_traction_scale.append(float(traction_scale.value))
        all_reaction_force.append(compute_reaction_force())

        if log_progress and (step % 500 == 0 or step == total_steps - 1
                             or step in (p1_steps, p1_steps + p2_steps)):
            phase_names = {1: "LOAD", 2: "HOLD", 3: "RELEASE"}
            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed
            eta = (total_steps - step - 1) / rate if rate > 0 else 0
            print(f"  step {step+1:5d}/{total_steps} [{phase_names[phase]:7s}] "
                  f"t={t_sim:.4f}s  |u|_max={np.abs(disp_vals).max():.4f}  "
                  f"iters={n_iters}  [{elapsed:.0f}s, ETA {eta:.0f}s]")

    elapsed = time.time() - t_start
    if log_progress:
        print(f"\n  Total simulation time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return {
        'status': final_status,
        'error': error_msg,
        'time_seconds': elapsed,
        'coords': coords.astype(np.float32),
        'topology': topology.astype(np.int32),
        'displacements': np.array(all_displacements),
        'velocities': np.array(all_velocities),
        'cauchy_stress': np.array(all_cauchy),
        'Cv': np.array(all_Cv),
        'pressure': np.array(all_pressure),
        'bc_state': np.array(all_bc_state),
        'phase_labels': np.array(all_phase_labels, dtype=np.int8),
        'times': np.array(all_times, dtype=np.float64),
        'actions_5dof': np.array(all_actions),
        'traction_scale': np.array(all_traction_scale, dtype=np.float32),
        'reaction_force': np.array(all_reaction_force),
        'n_substeps': n_substeps,
        'p1_steps': p1_steps,
        'p2_steps': p2_steps,
        'p3_steps': p3_steps,
        'actions_expanded': actions_expanded,
        'final_bc_state': dict(bc_state),
    }

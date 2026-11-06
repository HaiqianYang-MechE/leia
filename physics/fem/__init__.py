"""FEM solver kernel — Neo-Hookean (dolfin / FEniCS 2019.1) and
Arruda-Boyce 3-branch viscoelastic (dolfinx / FEniCSx 0.8).

Public API stays flat via lazy dispatch: this module does NOT import
dolfin or dolfinx at top level. The first call to a solver triggers the
appropriate import — wrong conda env errors clearly at that point.

Neo-Hookean dolfin is bundled in the main `leia` env (see
`environment.yml`). Visco dolfinx ships in the separate `fenicsx` env
(`environment-fenicsx.yml`).
"""


def solve_neo_hookean(*args, **kwargs):
    """Quasi-static Neo-Hookean solver (dolfin / FEniCS 2019.1).

    See physics.fem.neohookean.solver.solve_neo_hookean for full signature.
    """
    from .neohookean import solve_neo_hookean as _f
    return _f(*args, **kwargs)


def characterize_directions(*args, **kwargs):
    """4-DOF direction-sweep characterization (stretch / twist / shear_y / shear_z).

    See physics.fem.neohookean.characterize.characterize_directions.
    """
    from .neohookean import characterize_directions as _f
    return _f(*args, **kwargs)


def solve_arruda_boyce(*args, **kwargs):
    """3-phase Arruda-Boyce 3-branch viscoelastic solver (dolfinx / FEniCSx 0.8).

    See physics.fem.visco.solver.solve_arruda_boyce for full signature.
    """
    from .visco import solve_arruda_boyce as _f
    return _f(*args, **kwargs)


__all__ = [
    "solve_neo_hookean",
    "characterize_directions",
    "solve_arruda_boyce",
]

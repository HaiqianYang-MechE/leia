"""Differentiable stress utilities (PyTorch).

`neo_hookean_P_torch(F, mu, lmbda)` is the only exposed function: it computes
the first Piola-Kirchhoff stress for a compressible Neo-Hookean material from
a deformation gradient. Used by the eval pipelines and the design-search
runtime to convert tokenizer-decoded gradients into stress without dropping
out of the autograd graph.
"""

import torch

from leia.regime import MICROPLATE_LATTICE


def neo_hookean_P_torch(F, mu=MICROPLATE_LATTICE.mu, lmbda=MICROPLATE_LATTICE.lmbda):
    """First Piola-Kirchhoff stress for a compressible Neo-Hookean material.

        P = mu * (F - F^{-T}) + lambda * ln(J) * F^{-T}

    Defaults match the MicroPlate lattice regime; pass `mu` / `lmbda` for
    other materials.

    Parameters
    ----------
    F : (M, 3, 3) tensor — deformation gradient
    mu, lmbda : float — material parameters

    Returns
    -------
    P : (M, 3, 3) — PK1 stress
    """
    J = torch.linalg.det(F)                           # (M,)
    J_safe = torch.clamp(J, min=1e-6)
    Finv = torch.linalg.inv(F)                        # (M, 3, 3)
    FinvT = Finv.transpose(-2, -1)                    # (M, 3, 3)
    P = mu * (F - FinvT) + lmbda * torch.log(J_safe)[..., None, None] * FinvT
    return P

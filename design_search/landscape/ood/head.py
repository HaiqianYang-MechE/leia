"""AlphaFold-pLDDT-style learned OOD confidence head.

Reads candidate features (latent + free signals + graph stats), predicts
a categorical distribution over `vm_corr_mean` bins, and exposes:
  - logits             (raw pre-softmax)
  - predict()          → (predicted_scalar, predicted_entropy, probs)
where `predicted_scalar` is the expectation under the softmax and
`predicted_entropy` is the entropy of the distribution (uncertainty).

Training loss: cross_entropy(logits, true_bin).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OODConfidenceHead(nn.Module):
    """Small MLP: LayerNorm → 256 → 128 → n_bins logits.

    Args:
        d_in: input feature dimension (e.g. 11 for free, 1024 for latent, 1050 for full).
        d_hidden: width of the first hidden layer; second is d_hidden // 2.
        n_bins: number of bins for vm_corr_mean discretization.
        dropout: dropout probability after each hidden layer.
        bin_centers: (n_bins,) tensor of bin center values. Registered as
            buffer so .predict() works without external state.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 256,
        n_bins: int = 20,
        dropout: float = 0.2,
        bin_centers: torch.Tensor | None = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_bins = n_bins

        self.backbone = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(d_hidden // 2, n_bins)

        if bin_centers is None:
            bin_centers = torch.linspace(0.0, 1.0, n_bins)
        else:
            bin_centers = torch.as_tensor(bin_centers, dtype=torch.float32)
        assert bin_centers.shape == (n_bins,), \
            f"bin_centers must have shape ({n_bins},), got {tuple(bin_centers.shape)}"
        self.register_buffer("bin_centers", bin_centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, n_bins)."""
        return self.head(self.backbone(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """Inference convenience: returns (scalar, entropy, probs).

        - scalar (B,) : Σ_k probs[:, k] · bin_centers[k]
        - entropy (B,): −Σ_k probs[:, k] · log(probs[:, k])
        - probs (B, n_bins): softmax distribution
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        scalar = (probs * self.bin_centers[None, :]).sum(dim=-1)
        entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)
        return scalar, entropy, probs

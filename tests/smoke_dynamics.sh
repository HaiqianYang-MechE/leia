#!/usr/bin/env bash
# Smoke test the dynamics model: build tiny, forward, then round-trip the
# production v2 checkpoint if it is present (skips otherwise).

set -euo pipefail

cd "$(dirname "$0")/.."

source tests/_conda_activate.sh leia

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python - <<'PYEOF'
import os
import torch
from leia.dynamics import AdaLNContinuousDynamicsTransformer
from leia.tokenizer import PerceiverTokenizer

# Tiny dynamics
dyn = AdaLNContinuousDynamicsTransformer(
    num_latents=16, embed_dim=64,
    action_dim=4, heads=4, layers=2,
    velocity_input=True,
)
print(f"params: {sum(p.numel() for p in dyn.parameters()):,}")

# Forward
B, K, H = 2, 16, 64
state = torch.randn(B, K, H)
prev = torch.randn(B, K, H)
action = torch.randn(B, 4)
out = dyn(state, action, prev_state=prev)
assert out.shape == (B, K, H), f"out: {out.shape}"
print(f"forward OK: {out.shape}")

# Production round-trip if checkpoints are linked in
tok_path = "checkpoints/tokenizer/production_v2/best.pt"
dyn_path = "checkpoints/dynamics/production_v2/best_ar.pt"
if os.path.exists(tok_path) and os.path.exists(dyn_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, tok_stats = PerceiverTokenizer.from_checkpoint(tok_path, device=device)
    print(f"tokenizer loaded: K={tok.latents.shape[0]}, H={tok.latents.shape[1]}, "
          f"stress_head={tok.stress_head_enabled}")
    dyn, dyn_stats = AdaLNContinuousDynamicsTransformer.from_checkpoint(
        dyn_path, tokenizer_stats=tok_stats, device=device)
    print(f"dynamics loaded: K={dyn.num_latents}, layers={len(dyn.layers)}, "
          f"velocity_input={dyn.velocity_input}, action_dim={dyn_stats['action_dim']}")
else:
    print("(production checkpoints not linked — skipping round-trip)")

print("PASS smoke_dynamics")
PYEOF

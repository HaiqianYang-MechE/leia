#!/usr/bin/env bash
# Smoke test the tokenizer kernel: build, save, reload, forward.
# Does NOT touch GPU data — runs in ~10 seconds.

set -euo pipefail

cd "$(dirname "$0")/.."

source tests/_conda_activate.sh leia

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python - <<'PYEOF'
import os, tempfile
import numpy as np
import torch
from leia.tokenizer import PerceiverTokenizer
from leia.training import save_checkpoint, load_checkpoint

# Tiny tokenizer
tok = PerceiverTokenizer(
    input_dim=3, hidden_dim=64, num_latents=16,
    num_encoder_layers=2, num_decoder_layers=0,
    n_input_frames=1, stress_head=True,
)
print(f"params: {sum(p.numel() for p in tok.parameters()):,}")

# Forward
B, N = 2, 100
x = torch.randn(B, N, 3)
ref = torch.randn(B, N, 3)
recon, gp, sp = tok(x, ref)
assert recon.shape == (B, N, 3), f"recon: {recon.shape}"
assert gp.shape == (B, N, 3, 3), f"grad_pred: {gp.shape}"
assert sp.shape == (B, N, 6), f"stress_pred: {sp.shape}"
print(f"forward OK: recon {recon.shape}, grad {gp.shape}, stress {sp.shape}")

# Save → load round-trip
opt = torch.optim.AdamW(tok.parameters(), lr=1e-4)
cfg = dict(hidden_dim=64, num_latents=16, num_encoder_layers=2,
           num_decoder_layers=0, n_input_frames=1, stress_head=True)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, 'test.pt')
    save_checkpoint(p, tok, optimizer=opt, epoch=1, config=cfg,
                    extras=dict(disp_scale=1.0,
                                stress_mean=np.zeros(6, np.float32),
                                stress_std=np.ones(6, np.float32)))
    tok2, stats = PerceiverTokenizer.from_checkpoint(p, device='cpu')
    same = all((a.data == b.data).all().item() for a, b in zip(tok.parameters(), tok2.parameters()))
    assert same, "weights differ after round-trip"
    print(f"round-trip OK: disp_scale={stats['disp_scale']}, has_stress={'stress_mean' in stats}")

print("PASS smoke_tokenizer")
PYEOF

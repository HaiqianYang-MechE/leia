"""Loss primitives for tokenizer training.

Two Sobolev variants on the displacement gradient. Pure functions: tensors
in, scalar tensor out. Caller composes recon MSE + Sobolev + stress weights.

Prefer the gradient head + Sobolev (Table 1 row 4) or the stress head + Cauchy
MSE (row 5, production). Autograd Sobolev (row 3) is the slowest path and is
reserved for the ablation table.
"""

import torch


def compute_whole_plate_tet_sobolev_loss(x_recon, P_gt, tet_sample,
                                         coord_scale, disp_scale):
    """Volume-weighted tet shape function Sobolev loss for whole-plate training.

    Each sample carries its own (cells, dNdx, volumes) bundle, computed over a
    Monte Carlo subset of tets, so different plates with different meshes can
    train together in a single batch.

    Both prediction and target are in PHYSICAL displacement-gradient space
    (∂u_phys/∂x_phys, dimensionless). u_pred is normalized (model output) so
    multiply by disp_scale to convert.

    Parameters
    ----------
    x_recon : (B, max_v, 3) — decoded displacement at FPS ∪ extras (normalized)
    P_gt    : (B, max_v, 3, 3) — GT displacement gradient (physical units)
    tet_sample : dict
        cells_remapped : (B, K, 4) long — indices into x_recon's mesh axis
        dNdx           : (B, K, 4, 3) — shape function gradients in physical coords
        volumes        : (B, K) — tet volumes (physical)
    coord_scale : (B, 3) — kept for signature parity (unused here)
    disp_scale  : float — tokenizer displacement normalization scale

    Returns
    -------
    loss : scalar tensor — volume-weighted physical-space MSE on grad_u
    """
    cells_remapped = tet_sample['cells_remapped']
    dNdx = tet_sample['dNdx']
    volumes = tet_sample['volumes']
    B, K, _ = cells_remapped.shape
    device = x_recon.device

    if cells_remapped.device != device:
        cells_remapped = cells_remapped.to(device)
        dNdx = dNdx.to(device)
        volumes = volumes.to(device)

    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, K, 4)
    u_pred_nodes = x_recon[batch_idx, cells_remapped]
    u_pred_phys = u_pred_nodes * disp_scale

    grad_pred_phys = torch.einsum('bkmi,bkmj->bkij', u_pred_phys, dNdx)

    P_at_nodes = P_gt[batch_idx, cells_remapped]
    grad_gt_phys = P_at_nodes.mean(dim=2)

    diff = grad_pred_phys - grad_gt_phys
    frob_sq = (diff * diff).sum(dim=(-2, -1))
    vol_total = volumes.sum(dim=1).clamp_min(1e-12)
    per_sample = (frob_sq * volumes).sum(dim=1) / vol_total
    return per_sample.mean()


def compute_autograd_stress_loss(model, x, ref, grad_u_gt,
                                 coord_scale, disp_scale):
    """Sobolev loss via autograd Jacobian (dec=0 only).

    Supervises the displacement gradient ∂u/∂x directly — the quantity that
    determines stress — without backpropagating through Neo-Hookean.

    IMPORTANT: Encodes with no_grad (detaching z_q from ref), then decodes
    with grad on ref, so the Jacobian captures only the decoder's spatial
    variation (the physical displacement gradient).

    Parameters
    ----------
    model : PerceiverTokenizer or DDP-wrapped
    x : (B, N, 3) — input displacements (normalized)
    ref : (B, N, 3) — normalized query coordinates
    grad_u_gt : (B, N, 3, 3) — GT displacement gradient from FEM (physical units)
    coord_scale : (B, 3) — (c_max - c_min) / 2 per mesh axis
    disp_scale : float — tokenizer displacement normalization scale

    Returns
    -------
    loss : scalar — MSE on ∂u_pred/∂x vs ∂u_gt/∂x (physical units)
    """
    raw = model.module if hasattr(model, 'module') else model
    B, N, _ = x.shape

    # Step 1: Encode with no grad — z_q must NOT depend on ref
    with torch.no_grad():
        mesh_features = raw.input_proj(x) + raw.pos_encoding(ref)
        latents = raw.latents.repeat(B, 1, 1)
        z_enc = raw.cross_attn_encoder(query=latents, key=mesh_features)
        z_enc = raw.encoder_self_attn(z_enc)
        z_enc = raw.pre_vq_norm(z_enc)
        z_q = z_enc

    # Step 2: Decode with grad on ref (decoder-only spatial gradient)
    ref_g = ref.detach().requires_grad_(True)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        queries = raw.pos_encoding(ref_g)
        features = raw.cross_attn_decoder(query=queries, key=z_q)
        if raw.decoder_self_attn is not None:
            features = raw.decoder_self_attn(features)
        u_pred = raw.final_proj(raw.final_norm(features))

    # Step 3: Autograd Jacobian (3 backward passes)
    u_f = u_pred.float()
    grad_u = []
    for i in range(3):
        g = torch.autograd.grad(u_f[..., i].sum(), ref_g,
                                create_graph=True, retain_graph=True)[0]
        grad_u.append(g.float())
    grad_u = torch.stack(grad_u, dim=-2)

    # Convert to physical displacement gradient
    inv_coord_scale = 1.0 / coord_scale.float()
    scale_factor = disp_scale * inv_coord_scale
    grad_u_phys = grad_u * scale_factor[:, None, None, :]

    return torch.nn.functional.mse_loss(grad_u_phys, grad_u_gt.float())

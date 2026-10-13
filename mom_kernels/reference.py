# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""Eager reference implementation of the causal MoM layer math.

Single source of truth for WHAT the MoM layer computes: the kernel paths
(xla_tpu.py on TPU, triton_gpu.py on CUDA) are validated against it by
tests/test_parity.py. Any change to MoM math lands here first, then in the
kernels, then parity is re-run.
"""
import torch
import torch.nn.functional as F


def build_gate_bias(top_idx, top_gates, num_blocks, block_size, mask_dtype):
    """Per-token gate bias over the slots: (B, S, num_blocks * block_size).

    log(gate) on the blocks each token selected, finfo(mask_dtype).min
    elsewhere — softmax over exactly the top_blocks * block_size active
    slots of each token. Built by a one-hot x log-gates matmul, so the LM
    gradient reaches the router through the gates. fp32 output (router
    gates are fp32 by construction).
    """
    log_gates = top_gates.clamp_min(1e-9).log().float()
    onehot = F.one_hot(top_idx, num_blocks)
    sel = onehot.any(dim=2)
    bias_blocks = torch.einsum(
        "bsk,bske->bse", log_gates, onehot.to(log_gates.dtype))
    bias_blocks = bias_blocks.masked_fill(~sel, float(torch.finfo(mask_dtype).min))
    return bias_blocks.repeat_interleave(block_size, dim=-1)


def prefix_mean_reference(x):
    """Causal prefix means h_t = mean(x_0..x_t), fp32 — the router summary."""
    cs = torch.cumsum(x.float(), dim=1)
    denom = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=cs.dtype)
    return cs / denom.view(1, -1, 1)


def mom_attention_reference(q_sh, k, v, bias):
    """Multi-head attention over the memory bank with the per-token bias.

    q_sh: (B, S, H, hd) shared query stream; k/v: shared bank projections
    (L, D) or batched (B, L, D); bias: (B, S, L) additive bias on the
    logits (see build_gate_bias). Not causal over the sequence: the
    attention targets are memory slots, not positions; causality lives in
    the router's prefix mean.
    """
    B, S, H, hd = q_sh.shape
    L = k.shape[0] if k.dim() == 2 else k.shape[1]

    if k.dim() == 2:
        kk = k.view(1, L, H, hd).transpose(1, 2)             # [1, H, L, hd]
        vv = v.view(1, L, H, hd).transpose(1, 2)             # [1, H, L, hd]
    else:
        kk = k.view(B, L, H, hd).transpose(1, 2)              # [B, H, L, hd]
        vv = v.view(B, L, H, hd).transpose(1, 2)

    q = q_sh.transpose(1, 2)                                  # [B, H, S, hd]
    attn_mask = bias.to(q_sh.dtype).unsqueeze(1)              # [B, 1, S, L]
    out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=attn_mask, is_causal=False)
    return out.transpose(1, 2)


def mom_full_reference(x, layer, M_blocks):
    """Full causal MoMLayer forward, mirroring model.MoMLayer step by step.

    Kept separate from the nn.Module so the parity suite can drive the
    exact same math without kernel dispatch or cached mode. Returns
    (output, router_probs, top_idx, top_gates) — router stats at token
    scale, N = B*S rows flattened, matching the training-time stash.
    """
    B, S, D = x.shape
    q_sh = layer.q_proj(x).view(B, S, layer.n_heads, layer.head_dim)

    h = prefix_mean_reference(x).to(layer.router_down.weight.dtype)
    router_logits = layer.router_up(torch.tanh(layer.router_down(h))).float()
    if layer.training and layer.noise_scale > 0:
        gumbel = -torch.log(-torch.log(torch.rand_like(router_logits) + 1e-9) + 1e-9)
        noisy_logits = router_logits + layer.noise_scale * gumbel
    else:
        noisy_logits = router_logits
    router_probs = F.softmax(router_logits, dim=-1)
    top_values, top_idx = noisy_logits.topk(layer.top_blocks, dim=-1)
    top_gates = F.softmax(top_values, dim=-1)

    L = layer.mem_slots
    slots = M_blocks.view(L, D)
    k = layer.k_proj(slots)
    v = layer.v_proj(slots)

    bias = build_gate_bias(top_idx, top_gates, layer.num_blocks,
                           layer.block_size, q_sh.dtype)
    out = mom_attention_reference(q_sh, k, v, bias)
    out = layer.mom_scale * layer.o_proj(out.reshape(B, S, D))
    return out, router_probs, top_idx, top_gates

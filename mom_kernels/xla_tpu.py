# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""Fused-shape MoM attention for XLA/TPU (causal, per-token routing).

Two XLA-friendly eager implementations of the same math: "chunked"
(default — softmax over the full slot axis per query chunk, fp32 scores,
bounded score memory) and "online" (flash-style running max/sum over slot
blocks). Both take the routing in structured form (top_idx, top_gates) —
the gate bias is built per chunk inside, so the dense (B, S, L) bias is
never fully materialised in the chunked path (4x less bias memory at
S=1024, _CHUNK=256).

Shape stability is the design constraint: fixed chunk/block constants
(_CHUNK, _BLOCK_N) keep the HLO graphs stable across steps, which is what
matters for XLA compilation on the TPU.
"""
import math
import os

import torch
import torch.nn.functional as F

_CHUNK = 256
_BLOCK_N = 64


def _attn_mode(requested=None):
    """Resolve the attention variant: "chunked" (default) or "online"."""
    mode = requested or os.environ.get("NAYLIS_XLA_ATTN", "chunked")
    mode = mode.lower().strip()
    if mode not in ("chunked", "online"):
        raise RuntimeError(f"unknown xla attention mode {mode!r} (chunked|online)")
    return mode


def _prep_kv(k, v, H):
    """Normalise K/V to head layout: shared (L, D) -> [1, H, L, hd],
    batched (B, L, D) -> [B, H, L, hd]."""
    if k.dim() == 2:
        L = k.shape[0]
        return k.view(1, L, H, -1).transpose(1, 2), v.view(1, L, H, -1).transpose(1, 2)
    B, L, _ = k.shape
    return k.view(B, L, H, -1).transpose(1, 2), v.view(B, L, H, -1).transpose(1, 2)


def _bias_blocks(top_idx, top_gates, num_blocks, mask_dtype):
    """Block-level bias (B, S, num_blocks): log(gate) on the selected
    blocks, finfo.min elsewhere. The small (65K at run shapes) tensor the
    slot-level bias is expanded from, per chunk or per slot block."""
    log_gates = top_gates.clamp_min(1e-9).log().float()
    onehot = F.one_hot(top_idx, num_blocks)
    sel = onehot.any(dim=2)
    bias_blocks = torch.einsum(
        "bsk,bske->bse", log_gates, onehot.to(log_gates.dtype))
    return bias_blocks.masked_fill(~sel, float(torch.finfo(mask_dtype).min))


def mom_read(q_sh, k, v, top_idx, top_gates, *, num_blocks, block_size, mode=None):
    """MoM memory read with the per-token gate bias, on the XLA path.

    Args:
        q_sh: (B, S, H, hd) shared query stream.
        k, v: (L, D) shared bank projections or (B, L, D) batched.
        top_idx: (B, S, top) selected memory blocks per token.
        top_gates: (B, S, top) softmax router gates per token (fp32).
        num_blocks, block_size: bank geometry.
        mode: "chunked" | "online" (default: NAYLIS_XLA_ATTN or chunked).

    Returns: (B, S, H, hd) — query layout.
    """
    m = _attn_mode(mode)
    if m == "online":
        return _mom_read_online(q_sh, k, v, top_idx, top_gates, num_blocks, block_size)
    return _mom_read_chunked(q_sh, k, v, top_idx, top_gates, num_blocks, block_size)


def _mom_read_chunked(q_sh, k, v, top_idx, top_gates, num_blocks, block_size):
    """Chunked variant: fp32 softmax over all slots, per _CHUNK of queries.

    The slot-level bias is expanded per query chunk — [B, Sc, L] alive at
    a time instead of the full [B, S, L].
    """
    B, S, H, HD = q_sh.shape

    q = q_sh.transpose(1, 2)                                 # [B, H, S, HD]
    kk, vv = _prep_kv(k, v, H)
    bias_blocks = _bias_blocks(top_idx, top_gates, num_blocks, q_sh.dtype)
    scale = 1.0 / math.sqrt(HD)

    outs = []
    for s0 in range(0, S, _CHUNK):
        qc = q[:, :, s0:s0 + _CHUNK]                         # [B, H, Sc, HD]
        # slot bias for this chunk only: [B, Sc, nb] -> [B, 1, Sc, L]
        bias_c = bias_blocks[:, s0:s0 + _CHUNK].repeat_interleave(block_size, dim=-1)
        bias_c = bias_c.unsqueeze(1)
        scores = (qc @ kk.transpose(-1, -2)).to(torch.float32) * scale + bias_c
        probs = torch.softmax(scores, dim=-1)
        outs.append(probs.to(k.dtype) @ vv)
    out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
    return out.transpose(1, 2)


def _mom_read_online(q_sh, k, v, top_idx, top_gates, num_blocks, block_size):
    """Online variant: running max/sum over _BLOCK_N slot blocks
    (flash-style).

    When _BLOCK_N divides block_size each slot block lies inside a single
    memory block, so the bias of the block is a (B, S) column gather; the
    fallback builds the dense bias once (same values, more memory).
    """
    B, S, H, HD = q_sh.shape

    q = q_sh.transpose(1, 2)                                 # [B, H, S, HD]
    kk, vv = _prep_kv(k, v, H)
    bias_blocks = _bias_blocks(top_idx, top_gates, num_blocks, q_sh.dtype)
    scale = 1.0 / math.sqrt(HD)

    m_i = torch.full((B, H, S, 1), float("-inf"), device=q_sh.device,
                     dtype=torch.float32)
    l_i = torch.zeros((B, H, S, 1), device=q_sh.device, dtype=torch.float32)
    acc = torch.zeros((B, H, S, HD), device=q_sh.device, dtype=torch.float32)

    L = kk.shape[-2]
    aligned = block_size % _BLOCK_N == 0
    if not aligned:
        # dense fallback: expand the bias once (values identical)
        bias_full = bias_blocks.repeat_interleave(block_size, dim=-1).unsqueeze(1)

    for n0 in range(0, L, _BLOCK_N):
        kc = kk[:, :, n0:n0 + _BLOCK_N]
        vc = vv[:, :, n0:n0 + _BLOCK_N]
        scores = (q @ kc.transpose(-1, -2)).to(torch.float32) * scale
        if aligned:
            # one slot block == one memory block: bias is a column gather
            blk = n0 // block_size
            scores = scores + bias_blocks[:, :, blk].unsqueeze(-1).unsqueeze(1)
        else:
            scores = scores + bias_full[..., n0:n0 + _BLOCK_N]
        m_new = torch.maximum(m_i, scores.amax(dim=-1, keepdim=True))
        alpha = torch.exp(m_i - m_new)
        p = torch.exp(scores - m_new)
        l_i = l_i * alpha + p.sum(dim=-1, keepdim=True)
        acc = acc * alpha + p.to(q_sh.dtype) @ vc
        m_i = m_new

    out = (acc / l_i).to(q_sh.dtype)
    return out.transpose(1, 2)

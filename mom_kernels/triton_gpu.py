# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""Triton kernels for the causal MoM memory read on CUDA (opt-in, bf16).

The causal router reads the full bank with a per-token gate bias: log(gate)
on the selected blocks, a hard mask elsewhere. The eager path materialises
that bias densely — (B, S, L) floats per layer — and SDPA with an
arbitrary additive mask cannot use the flash backend. These kernels never
build the dense bias: the flash-style forward takes (top_idx, top_gates)
directly (small (B, S, top) tensors) and reconstructs the bias in-register
per tile — a block-id comparison against the token's top-k ids — which
removes both the mask materialisation and the SDPA mask fallback.

Autograd: ``mom_read`` wraps ``_MomReadFunction`` — the forward saves the
online-softmax m/l stats (fp32); the backward runs a delta/dq kernel
(which also reduces the gate gradients dlog_gates) and a dk/dv kernel
that accumulates over the batch for the shared bank projections. The
same decomposition as the retired per-sequence kernels, adapted to the
structured bias.

Numerics: the mask constant is -1e6 (finite, so online-softmax
rescaling annuls fully-masked tiles exactly) and every row always has its
top blocks active. Launch config: BLOCK_M x BLOCK_N tiles with
NAYLIS_TRITON_BLOCK_M / _BLOCK_N overrides; bf16 activations only (the
repo AMP dtype — fp16 is not supported anywhere in this codebase).

Status: experimental opt-in (use_kernel=true on CUDA). The default GPU
path remains eager SDPA — the path that trained the causal reference
run. Validate on the box (tests/test_parity.py GPU sections,
mom_kernels/probe.py) before arming it in any A/B arm.
"""
import os

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

_MASK = -1e6


def _block_sizes():
    """BLOCK_M/BLOCK_N from the environment (defaults 128 x 64)."""
    bm = int(os.environ.get("NAYLIS_TRITON_BLOCK_M", "128"))
    bn = int(os.environ.get("NAYLIS_TRITON_BLOCK_N", "64"))
    for name, v in (("BLOCK_M", bm), ("BLOCK_N", bn)):
        if v < 16 or v & (v - 1) != 0:
            raise ValueError(f"NAYLIS_TRITON_{name}={v} must be a power of two >= 16")
    return bm, bn


def _num_warps():
    return int(os.environ.get("NAYLIS_TRITON_WARPS", "4"))


if HAS_TRITON:

    @triton.jit
    def _mom_attn_fwd_kernel(
        Q, K, V, TOP, LG, O, M, L,
        sm_scale, mask_neg,
        S, H, L_slots, TOP_K, block_size,
        stride_qb, stride_qs, stride_qh,
        stride_kl, stride_kh,
        stride_vl, stride_vh,
        stride_tb, stride_ts,
        stride_lb, stride_ls,
        stride_ob, stride_os, stride_oh,
        stride_mb,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        """Flash forward: online softmax over the slot axis, gate bias
        reconstructed in-register from (top_idx, log_gates)."""
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        valid_m = offs_m < S

        q = tl.load(
            Q + b * stride_qb + offs_m[:, None] * stride_qs + h * stride_qh
            + offs_d[None, :],
            mask=valid_m[:, None], other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for n0 in range(0, L_slots, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            valid_n = offs_n < L_slots
            k = tl.load(
                K + offs_n[:, None] * stride_kl + h * stride_kh + offs_d[None, :],
                mask=valid_n[:, None], other=0.0)
            v = tl.load(
                V + offs_n[:, None] * stride_vl + h * stride_vh + offs_d[None, :],
                mask=valid_n[:, None], other=0.0)

            scores = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale

            # structured bias: log(gate) on the token's selected blocks,
            # mask elsewhere (and on out-of-range slots)
            blk = offs_n // block_size
            bias = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            sel_any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
            for t in range(TOP_K):
                top_t = tl.load(
                    TOP + b * stride_tb + offs_m * stride_ts + t,
                    mask=valid_m, other=0)
                lg_t = tl.load(
                    LG + b * stride_lb + offs_m * stride_ls + t,
                    mask=valid_m, other=0.0)
                sel = blk[None, :] == top_t[:, None]
                bias += tl.where(sel, lg_t[:, None], 0.0)
                sel_any += sel.to(tl.int32)
            bias = tl.where((sel_any > 0) & valid_n[None, :], bias, mask_neg)
            scores = scores + bias

            m_new = tl.maximum(m_i, tl.max(scores, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
            m_i = m_new

        o = acc / l_i[:, None]
        tl.store(
            O + b * stride_ob + offs_m[:, None] * stride_os + h * stride_oh
            + offs_d[None, :],
            o.to(O.dtype.element_ty), mask=valid_m[:, None])
        tl.store(M + pid_bh * stride_mb + offs_m, m_i, mask=valid_m)
        tl.store(L + pid_bh * stride_mb + offs_m, l_i, mask=valid_m)

    @triton.jit
    def _mom_attn_bwd_dq_kernel(
        Q, DO, K, V, TOP, LG, M, L, DELTA, DQ, DG,
        sm_scale, mask_neg,
        S, H, L_slots, TOP_K, block_size,
        stride_qb, stride_qs, stride_qh,
        stride_kl, stride_kh,
        stride_vl, stride_vh,
        stride_tb, stride_ts,
        stride_lb, stride_ls,
        stride_dqb, stride_dqs, stride_dqh,
        stride_mb,
        stride_dgb, stride_dgs,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        """Backward dq (+ dlog_gates, atomically reduced over heads): p is
        rebuilt from the saved m/l stats, ds = p * (dp - delta)."""
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        valid_m = offs_m < S

        q = tl.load(
            Q + b * stride_qb + offs_m[:, None] * stride_qs + h * stride_qh
            + offs_d[None, :],
            mask=valid_m[:, None], other=0.0)
        do = tl.load(
            DO + b * stride_qb + offs_m[:, None] * stride_qs + h * stride_qh
            + offs_d[None, :],
            mask=valid_m[:, None], other=0.0)
        m_i = tl.load(M + pid_bh * stride_mb + offs_m, mask=valid_m, other=0.0)
        l_i = tl.load(L + pid_bh * stride_mb + offs_m, mask=valid_m, other=1.0)
        delta = tl.load(DELTA + pid_bh * stride_mb + offs_m, mask=valid_m, other=0.0)

        dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for n0 in range(0, L_slots, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            valid_n = offs_n < L_slots
            k = tl.load(
                K + offs_n[:, None] * stride_kl + h * stride_kh + offs_d[None, :],
                mask=valid_n[:, None], other=0.0)
            v = tl.load(
                V + offs_n[:, None] * stride_vl + h * stride_vh + offs_d[None, :],
                mask=valid_n[:, None], other=0.0)

            scores = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
            blk = offs_n // block_size
            bias = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            sel_any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
            for t in range(TOP_K):
                top_t = tl.load(
                    TOP + b * stride_tb + offs_m * stride_ts + t,
                    mask=valid_m, other=0)
                lg_t = tl.load(
                    LG + b * stride_lb + offs_m * stride_ls + t,
                    mask=valid_m, other=0.0)
                sel = blk[None, :] == top_t[:, None]
                bias += tl.where(sel, lg_t[:, None], 0.0)
                sel_any += sel.to(tl.int32)
            bias = tl.where((sel_any > 0) & valid_n[None, :], bias, mask_neg)
            scores = scores + bias

            p = tl.exp(scores - m_i[:, None]) / l_i[:, None]
            dp = tl.dot(do, tl.trans(v)).to(tl.float32)
            ds = p * (dp - delta[:, None])
            dq_acc += tl.dot(ds.to(q.dtype), k)

            # dlog_gates[b, s, t] = sum over the slots of block top[b, s, t]
            # (reduced over heads by the atomic across this b's programs)
            for t in range(TOP_K):
                top_t = tl.load(
                    TOP + b * stride_tb + offs_m * stride_ts + t,
                    mask=valid_m, other=0)
                sel_t = blk[None, :] == top_t[:, None]
                contrib = tl.sum(
                    tl.where(sel_t & valid_n[None, :], ds, 0.0), axis=1)
                tl.atomic_add(
                    DG + b * stride_dgb + offs_m * stride_dgs + t,
                    contrib, mask=valid_m)

        dq = dq_acc * sm_scale
        tl.store(
            DQ + b * stride_dqb + offs_m[:, None] * stride_dqs + h * stride_dqh
            + offs_d[None, :],
            dq.to(DQ.dtype.element_ty), mask=valid_m[:, None])


    @triton.jit
    def _mom_attn_bwd_dkdv_kernel(
        Q, DO, K, V, TOP, LG, M, L, DELTA, DK, DV,
        sm_scale, mask_neg,
        B, S, H, L_slots, TOP_K, block_size,
        stride_qb, stride_qs, stride_qh,
        stride_kl, stride_kh,
        stride_vl, stride_vh,
        stride_tb, stride_ts,
        stride_lb, stride_ls,
        stride_mb,
        stride_dkl, stride_dkh,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        """Backward dk/dv for the SHARED bank projections: one program per
        (slot tile, head), accumulating over the batch and all query tiles."""
        pid_n = tl.program_id(0)
        h = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        valid_n = offs_n < L_slots

        k = tl.load(
            K + offs_n[:, None] * stride_kl + h * stride_kh + offs_d[None, :],
            mask=valid_n[:, None], other=0.0)
        v = tl.load(
            V + offs_n[:, None] * stride_vl + h * stride_vh + offs_d[None, :],
            mask=valid_n[:, None], other=0.0)
        blk = offs_n // block_size

        dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

        for b in range(B):
            for m0 in range(0, S, BLOCK_M):
                offs_m = m0 + tl.arange(0, BLOCK_M)
                valid_m = offs_m < S
                q = tl.load(
                    Q + b * stride_qb + offs_m[:, None] * stride_qs
                    + h * stride_qh + offs_d[None, :],
                    mask=valid_m[:, None], other=0.0)
                do = tl.load(
                    DO + b * stride_qb + offs_m[:, None] * stride_qs
                    + h * stride_qh + offs_d[None, :],
                    mask=valid_m[:, None], other=0.0)
                m_i = tl.load(
                    M + (b * H + h) * stride_mb + offs_m,
                    mask=valid_m, other=0.0)
                l_i = tl.load(
                    L + (b * H + h) * stride_mb + offs_m,
                    mask=valid_m, other=1.0)
                delta = tl.load(
                    DELTA + (b * H + h) * stride_mb + offs_m,
                    mask=valid_m, other=0.0)

                scores = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
                bias = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                sel_any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
                for t in range(TOP_K):
                    top_t = tl.load(
                        TOP + b * stride_tb + offs_m * stride_ts + t,
                        mask=valid_m, other=0)
                    lg_t = tl.load(
                        LG + b * stride_lb + offs_m * stride_ls + t,
                        mask=valid_m, other=0.0)
                    sel = blk[None, :] == top_t[:, None]
                    bias += tl.where(sel, lg_t[:, None], 0.0)
                    sel_any += sel.to(tl.int32)
                bias = tl.where((sel_any > 0) & valid_n[None, :], bias, mask_neg)
                scores = scores + bias

                p = tl.exp(scores - m_i[:, None]) / l_i[:, None]
                dp = tl.dot(do, tl.trans(v)).to(tl.float32)
                ds = p * (dp - delta[:, None])

                dv_acc += tl.dot(tl.trans(p.to(q.dtype)), do)
                dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q)

        dk = dk_acc * sm_scale
        tl.store(
            DK + offs_n[:, None] * stride_dkl + h * stride_dkh + offs_d[None, :],
            dk.to(DK.dtype.element_ty), mask=valid_n[:, None])
        tl.store(
            DV + offs_n[:, None] * stride_dkl + h * stride_dkh + offs_d[None, :],
            dv_acc.to(DV.dtype.element_ty), mask=valid_n[:, None])


class _MomReadFunction(torch.autograd.Function):
    """Autograd wrapper of the Triton MoM read (see module docstring)."""

    @staticmethod
    def forward(ctx, q_sh, k, v, top_idx, log_gates, num_blocks, block_size,
                block_m, block_n):
        B, S, H, HD = q_sh.shape
        L = k.shape[0]
        BH = B * H
        topk = top_idx.shape[-1]
        scale = 1.0 / (HD ** 0.5)

        o = torch.empty_like(q_sh)
        m_stats = torch.empty((BH, S), device=q_sh.device, dtype=torch.float32)
        l_stats = torch.empty((BH, S), device=q_sh.device, dtype=torch.float32)

        grid = (triton.cdiv(S, block_m), BH)
        _mom_attn_fwd_kernel[grid](
            q_sh, k, v, top_idx, log_gates, o, m_stats, l_stats,
            scale, _MASK,
            S, H, L, topk, block_size,
            q_sh.stride(0), q_sh.stride(1), q_sh.stride(2),
            k.stride(0), HD,
            v.stride(0), HD,
            top_idx.stride(0), top_idx.stride(1),
            log_gates.stride(0), log_gates.stride(1),
            o.stride(0), o.stride(1), o.stride(2),
            S,
            BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=HD,
            num_warps=_num_warps(),
        )
        ctx.save_for_backward(q_sh, k, v, top_idx, log_gates, o, m_stats, l_stats)
        ctx.meta = (block_size, block_m, block_n)
        return o

    @staticmethod
    def backward(ctx, do):
        q_sh, k, v, top_idx, log_gates, o, m_stats, l_stats = ctx.saved_tensors
        block_size, block_m, block_n = ctx.meta
        B, S, H, HD = q_sh.shape
        L = k.shape[0]
        BH = B * H
        topk = top_idx.shape[-1]
        scale = 1.0 / (HD ** 0.5)

        do = do.contiguous()
        delta = (o.float() * do.float()).sum(-1)               # [B, S, H]
        delta = delta.permute(0, 2, 1).reshape(BH, S).contiguous()

        dq = torch.empty_like(q_sh)
        dg = torch.zeros((B, S, topk), device=q_sh.device, dtype=torch.float32)
        grid_dq = (triton.cdiv(S, block_m), BH)
        _mom_attn_bwd_dq_kernel[grid_dq](
            q_sh, do, k, v, top_idx, log_gates, m_stats, l_stats, delta, dq, dg,
            scale, _MASK,
            S, H, L, topk, block_size,
            q_sh.stride(0), q_sh.stride(1), q_sh.stride(2),
            k.stride(0), HD,
            v.stride(0), HD,
            top_idx.stride(0), top_idx.stride(1),
            log_gates.stride(0), log_gates.stride(1),
            dq.stride(0), dq.stride(1), dq.stride(2),
            S,
            dg.stride(0), dg.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=HD,
            num_warps=_num_warps(),
        )

        dk = torch.empty((L, H, HD), device=q_sh.device, dtype=k.dtype)
        dv = torch.empty_like(dk)
        grid_kv = (triton.cdiv(L, block_n), H)
        _mom_attn_bwd_dkdv_kernel[grid_kv](
            q_sh, do, k, v, top_idx, log_gates, m_stats, l_stats, delta, dk, dv,
            scale, _MASK,
            B, S, H, L, topk, block_size,
            q_sh.stride(0), q_sh.stride(1), q_sh.stride(2),
            k.stride(0), HD,
            v.stride(0), HD,
            top_idx.stride(0), top_idx.stride(1),
            log_gates.stride(0), log_gates.stride(1),
            S,
            H * HD, HD,
            BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=HD,
            num_warps=_num_warps(),
        )

        return dq, dk.view(L, -1), dv.view(L, -1), None, dg, None, None, None, None


def mom_read(q_sh, k, v, top_idx, top_gates, *, num_blocks, block_size):
    """Flash MoM read with the in-register gate bias (CUDA, bf16).

    Same contract as mom_kernels.xla_tpu.mom_read: q_sh (B, S, H, hd),
    shared k/v (L, D), per-token (top_idx, top_gates); returns
    (B, S, H, hd). K/V must be shared (2-D); gradients flow back to the
    projections, the queries and the router gates (dlog_gates).
    """
    if not HAS_TRITON:
        raise RuntimeError("triton is not installed — the Triton MoM read cannot run.")
    if k.dim() != 2 or v.dim() != 2:
        raise RuntimeError("triton mom_read: shared K/V (L, D) only, got batched inputs")
    if q_sh.dtype != torch.bfloat16:
        raise RuntimeError(f"triton mom_read requires bf16 activations, got {q_sh.dtype}")
    if k.shape[0] != num_blocks * block_size:
        raise RuntimeError(
            f"triton mom_read: bank is {k.shape[0]} slots but num_blocks "
            f"({num_blocks}) x block_size ({block_size}) disagrees")

    q_sh = q_sh.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    top_idx = top_idx.to(torch.int32).contiguous()
    log_gates = top_gates.clamp_min(1e-9).log().float().contiguous()

    block_m, block_n = _block_sizes()
    return _MomReadFunction.apply(
        q_sh, k, v, top_idx, log_gates, num_blocks, block_size, block_m, block_n)

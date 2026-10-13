# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""Pallas (Mosaic) kernels for the causal MoM memory read on TPU.

Causal edition: the router selects blocks PER TOKEN, so the read attends
over the full bank with a per-token gate bias — log(gate) on the selected
blocks, a hard mask elsewhere. The kernels never materialise the dense
(B, S, L) bias: it is reconstructed in-register per tile from
(top_idx, log_gates), a block-id comparison against the token's top-k
ids.

Structure (unchanged from the per-sequence edition): every accelerated
entry point is split into a ``*_plan`` function — which builds the
kernel, grid, BlockSpecs and arguments without executing anything — and
a thin ``*_pallas`` runner that either calls ``pl.pallas_call`` or
delegates to an injected ``runner``. That split is what lets
mom_kernels/pallas_tpu/simulate.py execute the exact same kernels on CPU
as a numpy simulator (math validation without a TPU), and lets
bench_tpu.py time the real thing on-box.

Kernels: ``mom_read_plan`` (flash-style online-softmax forward, saves
the m/l stats), ``mom_read_bwd_plan`` (dq + dlog-gates kernel, and the
dk/dv kernel that accumulates over the batch for the shared bank
projections). ``mom_read_reference_jax`` is the fp32 oracle the bench
and the parity suite check against.

Shape constraints: S must divide by BLOCK_M and L by BLOCK_N — the bench
picks tiles that divide (fail-loud ValueError otherwise, never a silent
fallback).
"""
import jax
import jax.numpy as jnp

try:
    import jax.experimental.pallas as pl
    HAS_PALLAS = True
except ImportError:
    pl = None
    HAS_PALLAS = False

_NEG = -1e6


def _cdiv(a, b):
    """Ceiling division."""
    return -(-a // b)


def mom_read_plan(q_sh, k, v, top_idx, log_gates, *, block_size,
                  BLOCK_M=128, BLOCK_N=64):
    """Build (without running) the flash-style MoM read forward kernel.

    One program per (query tile, batch*head). The gate bias is rebuilt
    in-register per slot tile from (top_idx, log_gates). Returns the
    (kernel, grid, in_specs, out_specs, out_shapes, args) plan tuple.
    """
    B, S, H, HD = q_sh.shape
    L = k.shape[0]
    T = top_idx.shape[-1]
    if S % BLOCK_M != 0:
        raise ValueError(f"S={S} must be a multiple of BLOCK_M={BLOCK_M}")
    if L % BLOCK_N != 0:
        raise ValueError(f"L={L} must be a multiple of BLOCK_N={BLOCK_N}")
    scale = 1.0 / (HD ** 0.5)
    BH = B * H
    dt = q_sh.dtype

    # layouts: q [BH, S, HD]; k/v per head [H, L, HD]; routing per token
    q = q_sh.transpose(0, 2, 1, 3).reshape(BH, S, HD)
    kk = k.reshape(L, H, HD).transpose(1, 0, 2)
    vv = v.reshape(L, H, HD).transpose(1, 0, 2)

    def kernel(q_ref, k_ref, v_ref, top_ref, lg_ref, o_ref, m_ref, l_ref):
        q = q_ref[0]
        k = k_ref[0]
        v = v_ref[0]
        top = top_ref[0]
        lg = lg_ref[0]
        m_i = jnp.full((BLOCK_M,), _NEG, jnp.float32)
        l_i = jnp.zeros((BLOCK_M,), jnp.float32)
        acc = jnp.zeros((BLOCK_M, HD), jnp.float32)
        for n0 in range(0, L, BLOCK_N):
            kc = k[n0:n0 + BLOCK_N]
            vc = v[n0:n0 + BLOCK_N]
            s = jnp.einsum("md,nd->mn", q, kc).astype(jnp.float32) * scale
            # structured gate bias, in-register
            blk = (n0 + jnp.arange(BLOCK_N)) // block_size
            bias = jnp.zeros((BLOCK_M, BLOCK_N), jnp.float32)
            sel_any = jnp.zeros((BLOCK_M, BLOCK_N), jnp.int32)
            for t in range(T):
                sel = blk[None, :] == top[:, t][:, None]
                bias = bias + jnp.where(sel, lg[:, t][:, None], 0.0)
                sel_any = sel_any + sel.astype(jnp.int32)
            s = s + jnp.where(sel_any > 0, bias, _NEG)
            m_new = jnp.maximum(m_i, s.max(axis=1))
            alpha = jnp.exp(m_i - m_new)
            p = jnp.exp(s - m_new[:, None])
            l_i = l_i * alpha + p.sum(axis=1)
            acc = acc * alpha[:, None] + (p.astype(dt) @ vc).astype(jnp.float32)
            m_i = m_new
        o_ref[0] = (acc / l_i[:, None]).astype(o_ref.dtype)
        m_ref[0] = m_i
        l_ref[0] = l_i

    out_shapes = (jax.ShapeDtypeStruct((BH, S, HD), dt),
                  jax.ShapeDtypeStruct((BH, S), jnp.float32),
                  jax.ShapeDtypeStruct((BH, S), jnp.float32))
    grid = (_cdiv(S, BLOCK_M), BH)
    in_specs = [
        pl.BlockSpec((1, BLOCK_M, HD), lambda pm, pb: (pb, pm * BLOCK_M, 0)),
        pl.BlockSpec((1, L, HD), lambda pm, pb: (pb % H, 0, 0)),
        pl.BlockSpec((1, L, HD), lambda pm, pb: (pb % H, 0, 0)),
        pl.BlockSpec((1, BLOCK_M, T), lambda pm, pb: (pb // H, pm * BLOCK_M, 0)),
        pl.BlockSpec((1, BLOCK_M, T), lambda pm, pb: (pb // H, pm * BLOCK_M, 0)),
    ]
    out_specs = (
        pl.BlockSpec((1, BLOCK_M, HD), lambda pm, pb: (pb, pm * BLOCK_M, 0)),
        pl.BlockSpec((1, BLOCK_M), lambda pm, pb: (pb, pm * BLOCK_M)),
        pl.BlockSpec((1, BLOCK_M), lambda pm, pb: (pb, pm * BLOCK_M)),
    )
    args = [q, kk, vv, top_idx.astype(jnp.int32), log_gates.astype(jnp.float32)]
    return kernel, grid, in_specs, out_specs, out_shapes, args


def mom_read_pallas(q_sh, k, v, top_idx, log_gates, *, block_size,
                     BLOCK_M=128, BLOCK_N=64, runner=None, return_stats=False):
    """Run the MoM read kernel (pallas_call, or the injected runner).

    Returns the read output (B, S, H, hd); with ``return_stats`` also the
    online-softmax m/l stats (fp32) the backward consumes.
    """
    if not HAS_PALLAS:
        raise RuntimeError("jax.experimental.pallas is unavailable")
    B, S, H, HD = q_sh.shape
    kernel, grid, in_specs, out_specs, out_shapes, args = mom_read_plan(
        q_sh, k, v, top_idx, log_gates, block_size=block_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    if runner is not None:
        o, m, l = runner(kernel, grid, in_specs, out_specs, args, out_shapes)
    else:
        fun = pl.pallas_call(
            kernel, out_shape=out_shapes, grid=grid,
            in_specs=in_specs, out_specs=out_specs, name="mom_read",
        )
        o, m, l = fun(*args)
    out = o.reshape(B, H, S, HD).transpose(0, 2, 1, 3)
    return (out, m, l) if return_stats else out


def mom_read_bwd_plan(q_sh, k, v, top_idx, log_gates, o, do, m, l, *,
                      block_size, BLOCK_M=128, BLOCK_N=64):
    """Build (without running) the backward kernels: dq + dlog_gates, and
    dk/dv for the shared bank (accumulated over the batch).

    Reuses the saved m/l stats and delta = rowsum(o * do) — the same
    decomposition as the Triton backward. dlog_gates comes out per
    (batch*head) program and is summed over heads afterwards.
    """
    B, S, H, HD = q_sh.shape
    L = k.shape[0]
    T = top_idx.shape[-1]
    if S % BLOCK_M != 0:
        raise ValueError(f"S={S} must be a multiple of BLOCK_M={BLOCK_M}")
    if L % BLOCK_N != 0:
        raise ValueError(f"L={L} must be a multiple of BLOCK_N={BLOCK_N}")
    scale = 1.0 / (HD ** 0.5)
    BH = B * H
    dt = q_sh.dtype

    q = q_sh.transpose(0, 2, 1, 3).reshape(BH, S, HD)
    do_bh = do.transpose(0, 2, 1, 3).reshape(BH, S, HD)
    o_bh = o.transpose(0, 2, 1, 3).reshape(BH, S, HD)
    kk = k.reshape(L, H, HD).transpose(1, 0, 2)
    vv = v.reshape(L, H, HD).transpose(1, 0, 2)
    delta = jnp.einsum("bsd,bsd->bs", o_bh.astype(jnp.float32),
                       do_bh.astype(jnp.float32))

    def dq_kernel(q_ref, k_ref, v_ref, top_ref, lg_ref, do_ref, m_ref,
                  l_ref, delta_ref, dq_ref, dgp_ref):
        q = q_ref[0]
        k = k_ref[0]
        v = v_ref[0]
        top = top_ref[0]
        lg = lg_ref[0]
        do = do_ref[0]
        m_i = m_ref[0]
        l_i = l_ref[0]
        delta = delta_ref[0]
        l_safe = jnp.where(l_i > 0.0, l_i, 1.0)
        dq_acc = jnp.zeros((BLOCK_M, HD), jnp.float32)
        dg_acc = jnp.zeros((BLOCK_M, T), jnp.float32)
        for n0 in range(0, L, BLOCK_N):
            kc = k[n0:n0 + BLOCK_N]
            vc = v[n0:n0 + BLOCK_N]
            s = jnp.einsum("md,nd->mn", q, kc).astype(jnp.float32) * scale
            blk = (n0 + jnp.arange(BLOCK_N)) // block_size
            bias = jnp.zeros((BLOCK_M, BLOCK_N), jnp.float32)
            sel_any = jnp.zeros((BLOCK_M, BLOCK_N), jnp.int32)
            for t in range(T):
                sel = blk[None, :] == top[:, t][:, None]
                bias = bias + jnp.where(sel, lg[:, t][:, None], 0.0)
                sel_any = sel_any + sel.astype(jnp.int32)
            s = s + jnp.where(sel_any > 0, bias, _NEG)
            p = jnp.exp(s - m_i[:, None]) / l_safe[:, None]
            dp = jnp.einsum("md,nd->mn", do, vc).astype(jnp.float32)
            ds = p * (dp - delta[:, None])
            dq_acc = dq_acc + (ds.astype(dt) @ kc).astype(jnp.float32)
            for t in range(T):
                sel = blk[None, :] == top[:, t][:, None]
                dg_acc = dg_acc.at[:, t].add((ds * sel).sum(axis=1))
        dq_ref[0] = (dq_acc * scale).astype(dq_ref.dtype)
        dgp_ref[0] = dg_acc

    dq_out = (jax.ShapeDtypeStruct((BH, S, HD), dt),
              jax.ShapeDtypeStruct((BH, S, T), jnp.float32))
    dq_grid = (_cdiv(S, BLOCK_M), BH)
    dq_in = [
        pl.BlockSpec((1, BLOCK_M, HD), lambda pm, pb: (pb, pm * BLOCK_M, 0)),
        pl.BlockSpec((1, L, HD), lambda pm, pb: (pb % H, 0, 0)),
        pl.BlockSpec((1, L, HD), lambda pm, pb: (pb % H, 0, 0)),
        pl.BlockSpec((1, BLOCK_M, T), lambda pm, pb: (pb // H, pm * BLOCK_M, 0)),
        pl.BlockSpec((1, BLOCK_M, T), lambda pm, pb: (pb // H, pm * BLOCK_M, 0)),
        pl.BlockSpec((1, BLOCK_M, HD), lambda pm, pb: (pb, pm * BLOCK_M, 0)),
        pl.BlockSpec((1, BLOCK_M), lambda pm, pb: (pb, pm * BLOCK_M)),
        pl.BlockSpec((1, BLOCK_M), lambda pm, pb: (pb, pm * BLOCK_M)),
        pl.BlockSpec((1, BLOCK_M), lambda pm, pb: (pb, pm * BLOCK_M)),
    ]
    dq_specs = (
        pl.BlockSpec((1, BLOCK_M, HD), lambda pm, pb: (pb, pm * BLOCK_M, 0)),
        pl.BlockSpec((1, BLOCK_M, T), lambda pm, pb: (pb, pm * BLOCK_M, 0)),
    )
    dq_args = [q, kk, vv, top_idx.astype(jnp.int32),
               log_gates.astype(jnp.float32), do_bh, m, l, delta]

    def dkdv_kernel(q_ref, do_ref, k_ref, v_ref, top_ref, lg_ref, m_ref,
                    l_ref, delta_ref, dk_ref, dv_ref):
        pid_n = pl.program_id(0)
        q = q_ref[0]
        do = do_ref[0]
        k = k_ref[0]
        v = v_ref[0]
        top = top_ref[0]
        lg = lg_ref[0]
        m_i = m_ref[0]
        l_i = l_ref[0]
        delta = delta_ref[0]
        l_safe = jnp.where(l_i > 0.0, l_i, 1.0)
        offs_n = pid_n * BLOCK_N + jnp.arange(BLOCK_N)
        blk = offs_n // block_size
        dk_acc = jnp.zeros((BLOCK_N, HD), jnp.float32)
        dv_acc = jnp.zeros((BLOCK_N, HD), jnp.float32)
        for m0 in range(0, S, BLOCK_M):
            qc = q[m0:m0 + BLOCK_M]
            doc = do[m0:m0 + BLOCK_M]
            mc = m_i[m0:m0 + BLOCK_M]
            lc = l_safe[m0:m0 + BLOCK_M]
            dc = delta[m0:m0 + BLOCK_M]
            topc = top[m0:m0 + BLOCK_M]
            lgc = lg[m0:m0 + BLOCK_M]
            s = jnp.einsum("md,nd->mn", qc, k).astype(jnp.float32) * scale
            bias = jnp.zeros((BLOCK_M, BLOCK_N), jnp.float32)
            sel_any = jnp.zeros((BLOCK_M, BLOCK_N), jnp.int32)
            for t in range(T):
                sel = blk[None, :] == topc[:, t][:, None]
                bias = bias + jnp.where(sel, lgc[:, t][:, None], 0.0)
                sel_any = sel_any + sel.astype(jnp.int32)
            s = s + jnp.where(sel_any > 0, bias, _NEG)
            p = jnp.exp(s - mc[:, None]) / lc[:, None]
            dp = jnp.einsum("md,nd->mn", doc, v).astype(jnp.float32)
            ds = p * (dp - dc[:, None])
            dv_acc = dv_acc + (p.astype(dt).T @ doc).astype(jnp.float32)
            dk_acc = dk_acc + (ds.astype(dt).T @ qc).astype(jnp.float32)
        dk_ref[0] = (dk_acc * scale).astype(dk_ref.dtype)
        dv_ref[0] = dv_acc.astype(dv_ref.dtype)

    dkdv_out = (jax.ShapeDtypeStruct((BH, L, HD), dt),
                jax.ShapeDtypeStruct((BH, L, HD), dt))
    dkdv_grid = (_cdiv(L, BLOCK_N), BH)
    dkdv_in = [
        pl.BlockSpec((1, S, HD), lambda pn, pb: (pb, 0, 0)),
        pl.BlockSpec((1, S, HD), lambda pn, pb: (pb, 0, 0)),
        pl.BlockSpec((1, BLOCK_N, HD), lambda pn, pb: (pb % H, pn * BLOCK_N, 0)),
        pl.BlockSpec((1, BLOCK_N, HD), lambda pn, pb: (pb % H, pn * BLOCK_N, 0)),
        pl.BlockSpec((1, S, T), lambda pn, pb: (pb // H, 0, 0)),
        pl.BlockSpec((1, S, T), lambda pn, pb: (pb // H, 0, 0)),
        pl.BlockSpec((1, S), lambda pn, pb: (pb, 0)),
        pl.BlockSpec((1, S), lambda pn, pb: (pb, 0)),
        pl.BlockSpec((1, S), lambda pn, pb: (pb, 0)),
    ]
    dkdv_specs = (
        pl.BlockSpec((1, BLOCK_N, HD), lambda pn, pb: (pb, pn * BLOCK_N, 0)),
        pl.BlockSpec((1, BLOCK_N, HD), lambda pn, pb: (pb, pn * BLOCK_N, 0)),
    )
    dkdv_args = [q, do_bh, kk, vv, top_idx.astype(jnp.int32),
                 log_gates.astype(jnp.float32), m, l, delta]

    return {
        "dq": (dq_kernel, dq_grid, dq_in, dq_specs, dq_out, dq_args),
        "dkdv": (dkdv_kernel, dkdv_grid, dkdv_in, dkdv_specs, dkdv_out,
                 dkdv_args),
    }


def mom_read_bwd_pallas(q_sh, k, v, top_idx, log_gates, o, do, m, l, *,
                        block_size, BLOCK_M=128, BLOCK_N=64, runner=None):
    """Run both backward kernels; returns (dq, dk, dv, dlog_gates).

    dk/dv are summed over batch*head into the shared (L, D) layout;
    dlog_gates is summed over heads.
    """
    if not HAS_PALLAS:
        raise RuntimeError("jax.experimental.pallas is unavailable")
    B, S, H, HD = q_sh.shape
    L = k.shape[0]
    plans = mom_read_bwd_plan(
        q_sh, k, v, top_idx, log_gates, o, do, m, l, block_size=block_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    def _run(plan):
        kernel, grid, in_specs, out_specs, out_shapes, args = plan
        if runner is not None:
            return runner(kernel, grid, in_specs, out_specs, args, out_shapes)
        fun = pl.pallas_call(
            kernel, out_shape=out_shapes, grid=grid,
            in_specs=in_specs, out_specs=out_specs, name="mom_read_bwd",
        )
        return fun(*args)

    dq_bh, dgp = _run(plans["dq"])
    dk_bh, dv_bh = _run(plans["dkdv"])

    dq = dq_bh.reshape(B, H, S, HD).transpose(0, 2, 1, 3)
    # shared bank: sum over batch, per-head slices reassembled into (L, D)
    dk = dk_bh.reshape(B, H, L, HD).sum(axis=0).transpose(1, 0, 2).reshape(L, H * HD)
    dv = dv_bh.reshape(B, H, L, HD).sum(axis=0).transpose(1, 0, 2).reshape(L, H * HD)
    dlog_gates = dgp.reshape(B, H, S, top_idx.shape[-1]).sum(axis=1)
    return dq, dk, dv, dlog_gates


def mom_read_reference_jax(q_sh, k, v, top_idx, log_gates, block_size):
    """JAX reference of the causal MoM read: fp32 scores, dense bias,
    softmax attention. Returns (o, p) — the output and the probabilities
    (the oracle the Pallas kernels are checked against)."""
    B, S, H, HD = q_sh.shape
    L = k.shape[0]
    nb = L // block_size

    lg = log_gates.astype(jnp.float32)
    sel = (top_idx[..., None] == jnp.arange(nb)).any(axis=-2)     # [B, S, nb]
    bias_blocks = jnp.einsum(
        "bsk,bske->bse", lg, jax.nn.one_hot(top_idx, nb, dtype=jnp.float32))
    bias_blocks = jnp.where(sel, bias_blocks, _NEG)
    bias = jnp.repeat(bias_blocks, block_size, axis=-1)           # [B, S, L]

    q = q_sh.transpose(0, 2, 1, 3).astype(jnp.float32)             # [B, H, S, HD]
    kk = k.reshape(1, L, H, HD).transpose(0, 2, 1, 3).astype(jnp.float32)
    vv = v.reshape(1, L, H, HD).transpose(0, 2, 1, 3)

    s = jnp.matmul(q, kk.transpose(0, 1, 3, 2)) / (HD ** 0.5) \
        + bias[:, None, :, :]
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.matmul(p.astype(q_sh.dtype), vv)
    return o.transpose(0, 2, 1, 3), p

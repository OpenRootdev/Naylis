#!/usr/bin/env python3
# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""TPU bench for the causal Pallas MoM read: parity first, then timings.

Compares the Pallas kernels (flash-style forward with the in-register
gate bias, plus the dq/dkdv/dlog-gates backward) against the jit'd JAX
reference — forward outputs, gradients, all as scaled errors with
dtype-dependent tolerances — and times both sides per iteration.
``--simulate`` swaps pallas_call for the CPU numpy simulator (maths only,
no timing — that is the parity gate usable on any box); on a real TPU
the table reports the fwd / fwd+bwd speedups and the ideal minimum HBM
traffic per layer, the sanity ceiling for the measured numbers.

Usage: python mom_kernels/pallas_tpu/bench_tpu.py --simulate --shape tiny
       python mom_kernels/pallas_tpu/bench_tpu.py --shape 300m --dtype bf16
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

try:
    from mom_kernels.pallas_tpu import mom_pallas, simulate
except ImportError:
    import mom_pallas
    import simulate

SHAPES = {
    "tiny": dict(B=2, S=64, H=2, HD=16, D=32, nb=4, bs=16, T=2),
    "300m": dict(B=16, S=1024, H=16, HD=64, D=1024, nb=8, bs=128, T=2),
    "500m": dict(B=16, S=1024, H=20, HD=64, D=1280, nb=8, bs=128, T=2),
    "1b": dict(B=16, S=1024, H=16, HD=128, D=2048, nb=8, bs=128, T=2),
}


def _pick_block(n, candidates):
    """First tile size in ``candidates`` that divides ``n`` (else raise)."""
    for c in candidates:
        if n % c == 0:
            return c
    raise ValueError(f"no block in {candidates} divides axis of length {n}")


def make_inputs(shp, dt):
    """Deterministic inputs (rng seed 0) at the given shape/dtype."""
    rng = np.random.default_rng(0)
    B, S, H, HD, D, nb, bs, T = (shp["B"], shp["S"], shp["H"], shp["HD"],
                                 shp["D"], shp["nb"], shp["bs"], shp["T"])
    L = nb * bs
    top = jnp.asarray(rng.integers(0, nb, size=(B, S, T)), jnp.int32)
    k = jnp.asarray((rng.standard_normal((L, D)) * 0.4).astype(np.float32)).astype(dt)
    v = jnp.asarray((rng.standard_normal((L, D)) * 0.4).astype(np.float32)).astype(dt)
    q = jnp.asarray((rng.standard_normal((B, S, H, HD)) * 0.7).astype(np.float32)).astype(dt)
    do = jnp.asarray((rng.standard_normal((B, S, H, HD)) * 0.5).astype(np.float32)).astype(dt)
    gates = jax.nn.softmax(
        jnp.asarray(rng.standard_normal((B, S, T)).astype(np.float32)), axis=-1)
    log_gates = jnp.log(jnp.clip(gates, 1e-9))
    return dict(top=top, k=k, v=v, q=q, do=do, log_gates=log_gates, L=L)


def timeit(fn, iters, warmup=10):
    """Mean ms/iter of ``fn`` on device (block_until_ready'd)."""
    for _ in range(warmup):
        out = fn()
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e3


def scaled_err(a, b):
    """Max absolute error scaled by the magnitude of b (the parity metric)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = max(float(np.abs(b).max()), 1e-6)
    return float(np.max(np.abs(a - b))) / scale


grad_err = scaled_err


def _mb(x):
    """Format a byte count as MB/KB."""
    return f"{x / 1e6:.0f} MB" if x >= 1e7 else f"{x / 1e3:.0f} KB"


def _simulate_runner(kernel, grid, in_specs, out_specs, inputs, out_shapes):
    """Inject the CPU numpy simulator as the pallas runner."""
    return simulate.simulate(kernel, grid, in_specs, out_specs, inputs,
                             list(out_shapes))


def main():
    """Run parity (always) and timings (unless --simulate), then report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=sorted(SHAPES), default="300m")
    ap.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--simulate", action="store_true",
                    help="CPU numpy kernel simulator — parity only, no timing")
    args = ap.parse_args()

    shp = SHAPES[args.shape]
    dt = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    tol = 2e-2 if args.dtype == "bf16" else 1e-5
    inp = make_inputs(shp, dt)
    B, S, H, HD, D, nb, bs, T = (shp["B"], shp["S"], shp["H"], shp["HD"],
                                  shp["D"], shp["nb"], shp["bs"], shp["T"])
    L = inp["L"]

    print(f"[bench] jax={jax.__version__} devices={jax.devices()}")
    print(f"[bench] shape={args.shape} dtype={args.dtype} "
          f"B={B} S={S} H={H} HD={HD} D={D} L={L} (nb={nb}, bs={bs}, top={T})")
    if args.simulate:
        print("[bench] SIMULATE MODE: numpy kernel simulator, MATHS ONLY —")
        print("[bench] no performance value (measured > expected, project rule).")
        if args.shape != "tiny":
            print("[bench] WARNING: --simulate at prod shapes is VERY slow")
            print("[bench] (host numpy + per-program full-tensor copies).")
            print("[bench] Use: python bench_tpu.py --simulate --shape tiny")

    runner = None
    if args.simulate:
        runner = _simulate_runner
    maybe_jit = (lambda f: f) if args.simulate else jax.jit

    BM = _pick_block(S, (128, 64, 32))
    BN = _pick_block(L, (64, 32, 16))
    print(f"[bench] tiles BLOCK_M={BM} BLOCK_N={BN}")

    def ref_fn():
        return mom_pallas.mom_read_reference_jax(
            inp["q"], inp["k"], inp["v"], inp["top"], inp["log_gates"], bs)[0]
    ref_out = ref_fn()
    ref_jit = maybe_jit(ref_fn)
    ms_ref = None if args.simulate else timeit(ref_jit, args.iters)

    def ref_bwd_fn():
        def fwd(q_, k_, v_, lg_):
            return mom_pallas.mom_read_reference_jax(
                q_, k_, v_, inp["top"], lg_, bs)[0]
        o, vjp_fn = jax.vjp(fwd, inp["q"], inp["k"], inp["v"], inp["log_gates"])
        grads = vjp_fn(inp["do"])
        return o, grads
    _, ref_grads = ref_bwd_fn()
    r_dq, r_dk, r_dv, r_dlg = ref_grads
    ref_bwd_jit = maybe_jit(ref_bwd_fn)
    ms_ref_bwd = None if args.simulate else timeit(ref_bwd_jit, args.iters)

    def pallas_fn():
        return mom_pallas.mom_read_pallas(
            inp["q"], inp["k"], inp["v"], inp["top"], inp["log_gates"],
            block_size=bs, BLOCK_M=BM, BLOCK_N=BN, runner=runner)
    pallas_jit = maybe_jit(pallas_fn)
    p_out = pallas_jit()

    e_o = scaled_err(p_out.astype(jnp.float32), ref_out.astype(jnp.float32))
    print(f"[parity] out  scaled={e_o:.2e} (tol={tol:g})")
    ok_fwd = e_o < tol
    print(f"[parity] fwd {'PASS' if ok_fwd else 'FAIL'}")
    ms_pallas = None if args.simulate else timeit(pallas_jit, args.iters)

    def pallas_bwd_fn():
        o, m, l = mom_pallas.mom_read_pallas(
            inp["q"], inp["k"], inp["v"], inp["top"], inp["log_gates"],
            block_size=bs, BLOCK_M=BM, BLOCK_N=BN, runner=runner,
            return_stats=True)
        return mom_pallas.mom_read_bwd_pallas(
            inp["q"], inp["k"], inp["v"], inp["top"], inp["log_gates"],
            o, inp["do"], m, l, block_size=bs, BLOCK_M=BM, BLOCK_N=BN,
            runner=runner)
    pallas_bwd_jit = maybe_jit(pallas_bwd_fn)
    b_dq, b_dk, b_dv, b_dlg = pallas_bwd_jit()

    g_dq = grad_err(b_dq, r_dq)
    g_dk = grad_err(b_dk, r_dk)
    g_dv = grad_err(b_dv, r_dv)
    g_dlg = grad_err(b_dlg, r_dlg)
    print(f"[parity] bwd dq          err={g_dq:.2e} (scaled, tol={tol:g})")
    print(f"[parity] bwd dk (shared) err={g_dk:.2e}")
    print(f"[parity] bwd dv (shared) err={g_dv:.2e}")
    print(f"[parity] bwd dlog_gates  err={g_dlg:.2e}")
    ok_bwd = max(g_dq, g_dk, g_dv, g_dlg) < tol
    print(f"[parity] bwd {'PASS' if ok_bwd else 'FAIL'}")
    ms_pallas_bwd = None if args.simulate else timeit(pallas_bwd_jit,
                                                      args.iters)

    bpe = 2 if dt == jnp.bfloat16 else 4
    qo_bytes = 2 * B * S * H * HD * bpe
    kv_bytes = 2 * L * D * bpe                # shared bank projections
    routing_bytes = B * S * T * (4 + 4)
    stats_bytes = 2 * B * H * S * 4
    fwd_total = kv_bytes + qo_bytes + routing_bytes + stats_bytes
    bwd_extra = qo_bytes + kv_bytes + qo_bytes + routing_bytes + stats_bytes

    print()
    print("=" * 74)
    if args.simulate:
        print(f"{'variant':<22}{'parity':>10}   (simulate mode — no timing)")
        print("-" * 74)
        print(f"{'pallas fwd':<22}{'PASS' if ok_fwd else 'FAIL':>10}")
        print(f"{'pallas fwd+bwd':<22}{'PASS' if ok_bwd else 'FAIL':>10}")
        print("-" * 74)
        print(f"ideal min traffic ~fwd {_mb(fwd_total)} + bwd extra "
              f"~{_mb(bwd_extra)} per layer")
        print("=" * 74)
        if not (ok_fwd and ok_bwd):
            raise SystemExit("parity FAILED — see numbers above")
        print("[bench] SIMULATE OK: kernel maths validated on CPU. "
              "Run on the TPU for the numbers that matter.")
        return

    print(f"{'variant':<30}{'ms/iter':>10}{'speedup':>10}")
    print("-" * 74)
    print(f"{'fwd: jit reference (XLA)':<30}{ms_ref:>10.3f}{1.0:>10.1f}")
    print(f"{'fwd: pallas flash read':<30}{ms_pallas:>10.3f}"
          f"{ms_ref / ms_pallas:>10.2f}")
    print(f"{'fwd+bwd: jit reference (vjp)':<30}{ms_ref_bwd:>10.3f}{1.0:>10.1f}")
    print(f"{'fwd+bwd: pallas chain':<30}{ms_pallas_bwd:>10.3f}"
          f"{ms_ref_bwd / ms_pallas_bwd:>10.2f}")
    print("-" * 74)
    print(f"ideal min traffic ~fwd {_mb(fwd_total)} + bwd extra "
          f"~{_mb(bwd_extra)} per layer "
          f"(kv shared {_mb(kv_bytes)} + q/o {_mb(qo_bytes)} + "
          f"routing {_mb(routing_bytes)})")
    print("=" * 74)
    if not (ok_fwd and ok_bwd):
        raise SystemExit("parity FAILED — see numbers above")


if __name__ == "__main__":
    main()

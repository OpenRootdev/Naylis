# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""MoM preflight: fail fast, before anything heavy, on any box.

Run before a training or eval session: builds the causal MoMLayer at the
REAL run shapes (B=8, S=1024, D=1024, bank 1024 slots), checks the eager
forward against mom_kernels/reference.py, checks strict causality at the
cut position, checks the XLA fused path against the same reference, and —
on a CUDA box — the Triton flash read (forward and gradients) when the
triton package is present. Exits non-zero on any failure.

Usage: python mom_kernels/probe.py  (no arguments)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import MoMLayer
from mom_kernels import reference

B, S, D = 8, 1024, 1024
if not torch.cuda.is_available():
    # CPU boxes (4GB) cannot hold the backward at B=8; the parity and
    # causality semantics are batch-independent, the expensive dims stay.
    B = 2
H, HD = 16, 64
NB, BS, T = 8, 128, 2

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"probe | torch={torch.__version__} | device={device} | "
          f"shapes B={B} S={S} D={D} bank={NB}x{BS}")

    torch.manual_seed(7)
    layer = MoMLayer(
        d_model=D, n_heads=H, mem_size=NB * BS, block_size=BS,
        top_blocks=T, router_rank=32,
    ).to(device)
    layer.mom_scale.data.fill_(0.5)
    layer.eval()

    torch.manual_seed(11)
    M = (torch.randn(NB, BS * D, device=device) * 0.02).requires_grad_(True)
    torch.manual_seed(13)
    x = torch.randn(B, S, D, device=device)

    # A. eager forward == reference, bitwise in fp32
    out = layer(x, M)
    out_ref, probs, top_idx, gates = reference.mom_full_reference(x, layer, M)
    diff = float((out.detach() - out_ref.detach()).abs().max())
    check("A1. eager causal MoMLayer == reference (fp32 bitwise)",
          torch.equal(out, out_ref), f"max_abs={diff:.2e}")

    # B. strict causality at the cut position: perturb the future only
    cut = S // 2
    x2 = x.clone()
    x2[:, cut + 1:] += 1.0
    out2 = layer(x2, M)
    same = torch.equal(out[:, :cut + 1], out2[:, :cut + 1])
    diff = float((out[:, :cut + 1].detach() - out2[:, :cut + 1].detach()).abs().max())
    check("B1. causality: outputs at <= cut bit-identical under future perturbation",
          same, f"max_abs={diff:.2e}")

    # C. backward: gradients finite and alive on router + bank + gates path
    loss = out.float().pow(2).mean()
    loss.backward()
    finite = all(torch.isfinite(p.grad).all().item() for p in layer.parameters())
    router_alive = float(layer.router_up.weight.grad.abs().sum()) > 0
    bank_alive = M.grad is not None and float(M.grad.abs().sum()) > 0
    check("C1. backward: all grads finite", finite)
    check("C2. router grad alive (LM reaches the router through the gates)", router_alive)
    check("C3. memory bank grad alive", bank_alive)

    # D. XLA fused path == reference at the same shapes (per-token bias)
    for p in layer.parameters():
        p.grad = None
    layer.zero_grad()
    q_sh = layer.q_proj(x).view(B, S, H, HD)
    k = layer.k_proj(M.view(NB * BS, D))
    v = layer.v_proj(M.view(NB * BS, D))
    bias = reference.build_gate_bias(top_idx, gates, NB, BS, q_sh.dtype)
    ref_att = reference.mom_attention_reference(q_sh, k, v, bias)
    from mom_kernels import xla_tpu
    got_ch = xla_tpu.mom_read(q_sh, k, v, top_idx, gates,
                              num_blocks=NB, block_size=BS, mode="chunked")
    got_on = xla_tpu.mom_read(q_sh, k, v, top_idx, gates,
                              num_blocks=NB, block_size=BS, mode="online")
    check("D1. xla chunked == reference (fp32)",
          torch.allclose(got_ch, ref_att, atol=1e-4),
          f"max_abs={float((got_ch - ref_att).abs().max()):.2e}")
    check("D2. xla online == reference (fp32)",
          torch.allclose(got_on, ref_att, atol=1e-4),
          f"max_abs={float((got_on - ref_att).abs().max()):.2e}")

    # E. Triton flash read (CUDA boxes with triton): forward + grads vs eager
    if device == "cuda":
        try:
            from mom_kernels import triton_gpu
        except Exception:
            triton_gpu = None
        if triton_gpu is not None and triton_gpu.HAS_TRITON:
            layer_bf = MoMLayer(
                d_model=D, n_heads=H, mem_size=NB * BS, block_size=BS,
                top_blocks=T, router_rank=32,
            ).to(device).to(torch.bfloat16)
            layer_bf.eval()
            M_bf = (torch.randn(NB, BS * D, device=device) * 0.02).to(torch.bfloat16)
            x_bf = x.to(torch.bfloat16)
            _, _, top_idx_bf, gates_bf = reference.mom_full_reference(
                x_bf, layer_bf, M_bf)
            q_bf = layer_bf.q_proj(x_bf).view(B, S, H, HD)
            k_bf = layer_bf.k_proj(M_bf.view(NB * BS, D))
            v_bf = layer_bf.v_proj(M_bf.view(NB * BS, D))
            bias_bf = reference.build_gate_bias(top_idx_bf, gates_bf, NB, BS, q_bf.dtype)
            ref_bf = reference.mom_attention_reference(q_bf, k_bf, v_bf, bias_bf)
            got_tr = triton_gpu.mom_read(q_bf, k_bf, v_bf, top_idx_bf, gates_bf,
                                         num_blocks=NB, block_size=BS)
            err = float((got_tr.float() - ref_bf.float()).abs().max())
            check("E1. triton flash read == eager reference (bf16)",
                  err < 2e-2, f"max_abs={err:.2e}")

            # gradients: full read path vs the eager autograd oracle
            for tgt in (q_bf, k_bf, v_bf):
                tgt.requires_grad_(True)
            gates_bf.requires_grad_(True)
            out_tr = triton_gpu.mom_read(q_bf, k_bf, v_bf, top_idx_bf, gates_bf,
                                         num_blocks=NB, block_size=BS)
            out_ea = reference.mom_attention_reference(
                q_bf, k_bf, v_bf, bias_bf)
            g = torch.randn_like(out_tr)
            out_tr.backward(g)
            tr_grads = [q_bf.grad.clone(), k_bf.grad.clone(), v_bf.grad.clone(),
                        gates_bf.grad.clone()]
            for tgt in (q_bf, k_bf, v_bf, gates_bf):
                tgt.grad = None
            out_ea.backward(g)
            ea_grads = [q_bf.grad.clone(), k_bf.grad.clone(), v_bf.grad.clone(),
                        gates_bf.grad.clone()]
            worst = max(float((a.float() - b.float()).abs().max())
                        for a, b in zip(tr_grads, ea_grads))
            check("E2. triton backward (dq/dk/dv/dlog_gates) == eager autograd",
                  worst < 2e-1, f"max_abs={worst:.2e} (bf16 grads)")
        else:
            print("[probe] E skipped: triton not importable on this CUDA box")
    else:
        print("[probe] E skipped: no CUDA device (triton section)")

    # F. pure-fp16 evaluation smoke (the dtype crash class)
    try:
        layer_h = MoMLayer(
            d_model=D, n_heads=H, mem_size=NB * BS, block_size=BS,
            top_blocks=T, router_rank=32,
        ).half().to(device)
        layer_h.eval()
        _ = layer_h(x.half().to(device), M.half())
        check("F1. pure fp16 forward runs (router dtype guard)", True)
    except RuntimeError as e:
        check("F1. pure fp16 forward runs (router dtype guard)", False, str(e)[:120])

    print()
    failed = [n for n, ok in RESULTS if not ok]
    print(f"probe: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        for n in failed:
            print(f"  - {n}")
        sys.exit(1)
    print("PROBE OK")


if __name__ == "__main__":
    main()

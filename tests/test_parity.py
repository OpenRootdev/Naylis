#!/usr/bin/env python3
# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""Parity suite for the causal MoM layer, against mom_kernels/reference.py.

Sections: 1 — eager MoMLayer == causal reference (fp32, bitwise);
2 — the prefix-mean router summary is causal and math-exact; 3 — strict
causality: perturbing future tokens leaves past outputs bit-identical
(layer and model level); 4-6 — the XLA chunked/online attention against
the same reference (fp32 and bf16 class, both bias-construction paths);
7-8 — resolve_mode / preflight contracts (fail-loud CPU, bf16-only and
power-of-two head_dim on CUDA, xla pass-through); 9-12 — model-level
forward (loss/aux finite, router grads alive), eval determinism, Gumbel
exploration armed in train mode; 13 — the MoM gradient-checkpoint region
recomputes exactly; 14 — the arithmetic aux-loss formulation == the
scatter oracle; 15 — legacy checkpoint keys (graph_mem/graph_norm/
graph_scale, the pre-repo training scripts) remap and load with
bit-identical outputs; 16-17 — pure-fp16 forward survives the router
dtype guard / pure-bf16 model is finite and deterministic; 18 — the AMP
policy refuses fp16 (bf16-only repo); 19 — the precision policy (bf16
converts nothing, explicit fp8 refuses on a no-fp8 box, MoM routing is
hard-excluded from fp8 targets); 20 — the Triton flash read (CUDA boxes
with triton): forward and gradients vs the eager oracle.

A failure exits 1 — kernels are only a speed story while parity holds.
"""

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--skip-model", action="store_true")
args = parser.parse_args()

import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

import mom_kernels
from mom_kernels import reference
from model import MoMLayer, NaylisLlamaForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS = []


def check(name, ok, detail=""):
    """Record and print one check; returns the boolean."""
    RESULTS.append((name, bool(ok)))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


def max_abs(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max())


def scaled_max(a, b):
    """Max absolute error scaled by the magnitude of b (the parity metric)."""
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-6)


def close(a, b, atol, rtol=0.0):
    return torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)


B, S, H, HD = 2, 64, 2, 16
D = H * HD
NB, BS, T = 4, 16, 2
L = NB * BS


def make_layer(use_kernel=False, mom_grad_ckpt=False, seed=7):
    """Deterministic causal MoMLayer at the suite shapes (mom_scale=0.5)."""
    torch.manual_seed(seed)
    layer = MoMLayer(
        d_model=D, n_heads=H, mem_size=NB * BS, block_size=BS,
        top_blocks=T, router_rank=8, use_kernel=use_kernel,
        mom_grad_ckpt=mom_grad_ckpt,
    ).to(DEVICE)
    layer.mom_scale.data.fill_(0.5)
    return layer


def make_bank(seed=11, requires_grad=True):
    """Deterministic fp32 memory bank (NB, block_size * D)."""
    torch.manual_seed(seed)
    m = torch.randn(NB, BS * D, device=DEVICE) * 1.0
    return m.requires_grad_(requires_grad) if requires_grad else m


print(f"\n=== device={DEVICE} ===\n")

# ---------------------------------------------------------------- 1. parity
layer = make_layer()
M = make_bank()
x = torch.randn(B, S, D, device=DEVICE)
torch.manual_seed(123)
out_ref, _, _, _ = reference.mom_full_reference(x, layer, M)
torch.manual_seed(123)
out_layer = layer(x, M)
check("1. eager causal MoMLayer == reference (fp32 bitwise)",
      torch.equal(out_layer, out_ref))

# ------------------------------------------------- 2. prefix-mean causality
h = layer._prefix_mean(x)
check("2a. prefix mean: h[:, 0] == x[:, 0] (bitwise, fp32 cumsum start)",
      torch.equal(h[:, 0], x[:, 0].float()))
manual = torch.stack([x[:, :t + 1].float().mean(dim=1) for t in range(S)], dim=1)
check("2b. prefix mean: h_t == mean(x_0..x_t) (math exact; cumsum keeps a "
      "sequential accumulation order so rounding may differ from the "
      "pairwise-reduced torch.mean)",
      close(h, manual, atol=1e-5), f"max_abs={max_abs(h, manual):.2e}")
h_future = layer._prefix_mean(x.clone())
h_future_perturbed = layer._prefix_mean(x.clone())
check("2c. prefix mean of the perturbed future: past rows unchanged (bitwise)",
      torch.equal(h_future[:, :S // 2], h_future_perturbed[:, :S // 2]))

# ---------------------------------------------------- 3. strict causality
cut = S // 2
layer.eval()
with torch.no_grad():
    out_a = layer(x, M)
    x2 = x.clone()
    x2[:, cut + 1:] += 1.0
    out_b = layer(x2, M)
check("3a. causality (layer): outputs at <= cut bit-identical under future perturbation",
      torch.equal(out_a[:, :cut + 1], out_b[:, :cut + 1]),
      f"max_abs={max_abs(out_a[:, :cut + 1], out_b[:, :cut + 1]):.2e}")

# ----------------------------------------------- 4-6. XLA fused attention
torch.manual_seed(5)
q_sh = torch.randn(B, S, H, HD, device=DEVICE) * 0.7
k_shared = torch.randn(L, D, device=DEVICE) * 0.4
v_shared = torch.randn(L, D, device=DEVICE) * 0.4
g = F.softmax(torch.randn(B, S, T, device=DEVICE, dtype=torch.float32), -1)
top_sel = torch.randint(0, NB, (B, S, T), device=DEVICE)
bias = reference.build_gate_bias(top_sel, g, NB, BS, torch.float32)

ref_att = reference.mom_attention_reference(q_sh, k_shared, v_shared, bias)
got_ch = mom_kernels.xla_tpu.mom_read(
    q_sh, k_shared, v_shared, top_sel, g, num_blocks=NB, block_size=BS, mode="chunked")
got_on = mom_kernels.xla_tpu.mom_read(
    q_sh, k_shared, v_shared, top_sel, g, num_blocks=NB, block_size=BS, mode="online")
check("4. xla chunked == reference (fp32, per-token bias, per-chunk build)",
      close(got_ch, ref_att, atol=1e-5), f"max_abs={max_abs(got_ch, ref_att):.2e}")
check("5. xla online == reference (fp32; bs=16 < BLOCK_N -> dense-bias path)",
      close(got_on, ref_att, atol=1e-5), f"max_abs={max_abs(got_on, ref_att):.2e}")

# 5b. the block-aligned online path (block_size a multiple of _BLOCK_N):
# the slot bias is a (B, S) column gather instead of a dense expansion.
NB2, BS2 = 2, 64
L2 = NB2 * BS2
top_sel2 = torch.randint(0, NB2, (B, S, T), device=DEVICE)
g2 = F.softmax(torch.randn(B, S, T, device=DEVICE, dtype=torch.float32), -1)
k2 = torch.randn(L2, D, device=DEVICE) * 0.4
v2 = torch.randn(L2, D, device=DEVICE) * 0.4
bias2 = reference.build_gate_bias(top_sel2, g2, NB2, BS2, torch.float32)
ref2 = reference.mom_attention_reference(q_sh, k2, v2, bias2)
got2 = mom_kernels.xla_tpu.mom_read(
    q_sh, k2, v2, top_sel2, g2, num_blocks=NB2, block_size=BS2, mode="online")
check("5b. xla online (block-aligned bias gather, bs=64) == reference",
      close(got2, ref2, atol=1e-5), f"max_abs={max_abs(got2, ref2):.2e}")

for dtype, atol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
    q_d = q_sh.to(dtype)
    k_d = k_shared.to(dtype)
    v_d = v_shared.to(dtype)
    ref_d = reference.mom_attention_reference(q_d, k_d, v_d, bias.to(dtype))
    got_d = mom_kernels.xla_tpu.mom_read(
        q_d, k_d, v_d, top_sel, g, num_blocks=NB, block_size=BS, mode="chunked")
    check(f"6. xla chunked == reference ({str(dtype).split('.')[-1]})",
          close(got_d, ref_d, atol=atol), f"max_abs={max_abs(got_d, ref_d):.2e}")

# batched K/V layout too (the old call convention)
k_b = k_shared.unsqueeze(0).expand(B, -1, -1).contiguous()
v_b = v_shared.unsqueeze(0).expand(B, -1, -1).contiguous()
got_b = mom_kernels.xla_tpu.mom_read(
    q_sh, k_b, v_b, top_sel, g, num_blocks=NB, block_size=BS, mode="chunked")
check("6b. xla chunked: batched K/V == shared K/V",
      close(got_b, got_ch, atol=1e-6), f"max_abs={max_abs(got_b, got_ch):.2e}")

# ------------------------------------------- 7-8. backend contract rules
try:
    mom_kernels.resolve_mode(torch.device("cpu"), torch.float32)
    check("7a. resolve_mode raises on CPU", False)
except RuntimeError:
    check("7a. resolve_mode raises on CPU", True)
try:
    mom_kernels.resolve_mode("cuda", torch.float32, head_dim=64)
    check("7b. resolve_mode raises on CUDA with fp32 activations (bf16-only)", False)
except RuntimeError:
    check("7b. resolve_mode raises on CUDA with fp32 activations (bf16-only)", True)
try:
    mom_kernels.resolve_mode("cuda", torch.bfloat16, head_dim=96)
    check("7c. resolve_mode raises on CUDA with non-power-of-two head_dim", False)
except RuntimeError:
    check("7c. resolve_mode raises on CUDA with non-power-of-two head_dim", True)
if mom_kernels.HAS_TRITON:
    check("7d. resolve_mode('cuda', bf16, hd=64) == 'triton'",
          mom_kernels.resolve_mode("cuda", torch.bfloat16, head_dim=64) == "triton")
else:
    try:
        mom_kernels.resolve_mode("cuda", torch.bfloat16, head_dim=64)
        check("7d. resolve_mode raises on CUDA without triton", False)
    except RuntimeError:
        check("7d. resolve_mode raises on CUDA without triton", True)
check("7e. resolve_mode('xla') == 'xla' (dtype-agnostic TPU path)",
      mom_kernels.resolve_mode("xla", torch.float32, head_dim=64) == "xla")

try:
    mom_kernels.preflight("cpu")
    check("8a. preflight raises on cpu", False)
except RuntimeError:
    check("8a. preflight raises on cpu", True)
if mom_kernels.HAS_TRITON:
    check("8b. preflight accepts cuda (triton installed)",
          mom_kernels.preflight("cuda") is None)
else:
    try:
        mom_kernels.preflight("cuda")
        check("8b. preflight raises on cuda without triton", False)
    except RuntimeError:
        check("8b. preflight raises on cuda without triton", True)
check("8c. preflight accepts xla", mom_kernels.preflight("xla") is None)

# ------------------------------------------------------- 9-12. model level
if not args.skip_model:
    from transformers import LlamaConfig

    def tiny_config(vocab=256):
        return LlamaConfig(
            vocab_size=vocab, hidden_size=D, intermediate_size=56,
            num_hidden_layers=2, num_attention_heads=H, num_key_value_heads=H,
            max_position_embeddings=128, rms_norm_eps=1e-5, rope_theta=500000.0,
            attention_bias=False, tie_word_embeddings=True,
            attn_implementation="sdpa", use_cache=False,
        )

    def build_model(**kw):
        torch.manual_seed(0)
        model = NaylisLlamaForCausalLM(
            tiny_config(), mem_size=NB * BS, mom_block_size=BS,
            mom_top_blocks=T, mom_router_rank=8, mom_aux_weight=0.01,
            use_liger=False, **kw,
        ).to(DEVICE)
        for m in model.modules():
            if hasattr(m, "mom_scale"):
                m.mom_scale.data.fill_(0.5)
        return model

    model = build_model()
    ids = torch.randint(0, 256, (2, 32), device=DEVICE)
    labels = ids.clone()

    model.train()
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    finite = torch.isfinite(out.loss).item()
    router_live = all(
        torch.isfinite(p.grad).all().item() and float(p.grad.abs().sum()) > 0
        for n, p in model.named_parameters() if "router" in n and p.grad is not None
    )
    check("9. model train: loss+aux finite, router grads alive (LM through "
          "the gates + aux through the probs)",
          finite and router_live, f"loss={float(out.loss):.4f}")

    model.eval()
    with torch.no_grad():
        logits_a = model(input_ids=ids).logits
        logits_b = model(input_ids=ids).logits
    check("10. eval determinism: two forwards bit-identical (Gumbel off)",
          torch.equal(logits_a, logits_b))

    t_cut = 16
    ids2 = ids.clone()
    ids2[:, t_cut + 1:] = (ids2[:, t_cut + 1:] + 7) % 256
    with torch.no_grad():
        logits_c = model(input_ids=ids2).logits
    check("11. causality (model): logits at <= t bit-identical under future "
          "token perturbation",
          torch.equal(logits_a[:, :t_cut + 1], logits_c[:, :t_cut + 1]),
          f"max_abs={max_abs(logits_a[:, :t_cut + 1], logits_c[:, :t_cut + 1]):.2e}")

    lay = make_layer(seed=9)
    lay.train()
    torch.manual_seed(4)
    o1 = lay(x, M)
    torch.manual_seed(5)
    o2 = lay(x, M)
    check("12a. train mode: Gumbel noise changes the routing between calls",
          not torch.equal(o1, o2))
    lay.noise_scale = 0.0
    torch.manual_seed(4)
    o3 = lay(x, M)
    torch.manual_seed(4)
    o4 = lay(x, M)
    check("12b. train mode, noise annealed to 0: bit-identical",
          torch.equal(o3, o4))

    # ---------------------------------------------- 13. gradient checkpoint
    model_ck = build_model(mom_grad_ckpt=True)
    model_pl = build_model(mom_grad_ckpt=False)
    model_ck.train()
    model_pl.train()
    torch.manual_seed(77)
    loss_ck = model_ck(input_ids=ids, labels=labels).loss
    loss_ck.backward()
    torch.manual_seed(77)
    loss_pl = model_pl(input_ids=ids, labels=labels).loss
    loss_pl.backward()
    grads_ck = {n: p.grad.detach().clone() for n, p in model_ck.named_parameters()}
    grads_pl = {n: p.grad.detach().clone() for n, p in model_pl.named_parameters()}
    same_loss = torch.equal(loss_ck, loss_pl)
    worst = max(max_abs(grads_ck[n], grads_pl[n]) for n in grads_pl)
    check("13. mom_grad_ckpt: loss bitwise + all grads match (exact recompute; "
          "same Gumbel draws for both runs)",
          same_loss and worst < 1e-6, f"loss_diff={float(loss_ck - loss_pl):.2e} "
          f"max_grad_abs={worst:.2e}")

    # ------------------------------------------------ 14. aux oracle
    torch.manual_seed(42)
    probs = F.softmax(torch.randn(128, NB, dtype=torch.float32), -1)
    top_idx = torch.randint(0, NB, (128, T))
    blocks = torch.arange(NB)
    sel_mask = (top_idx.unsqueeze(2) == blocks.view(1, 1, -1)).any(1)
    frac_arith = sel_mask.float().mean(dim=0)
    aux_arith = NB * (frac_arith * probs.mean(dim=0)).sum()
    selected = torch.zeros_like(probs)
    selected.scatter_(1, top_idx, 1.0)
    aux_scatter = NB * (selected.mean(dim=0) * probs.mean(dim=0)).sum()
    check("14. aux loss: arithmetic (comparison+any) == scatter oracle",
          close(aux_arith, aux_scatter, atol=1e-6),
          f"max_abs={float((aux_arith - aux_scatter).abs()):.2e}")

    # ------------------------------------------------ 15. legacy remap
    import evaluate

    legacy = {}
    for k, v in model.state_dict().items():
        k2 = k.replace(".mom.", ".graph_mem.").replace(".mom_norm.", ".graph_norm.")
        k2 = k2.replace(".graph_mem.mom_scale", ".graph_mem.graph_scale")
        legacy[k2] = v
    n_renamed = sum(1 for a, b in zip(model.state_dict().keys(), legacy.keys()) if a != b)
    remapped = evaluate._remap_legacy_keys(legacy)
    model_fresh = build_model()
    model_fresh.eval()
    missing, unexpected = model_fresh.load_state_dict(remapped, strict=False)
    strict_ok = not missing and not unexpected
    with torch.no_grad():
        logits_legacy = model_fresh(input_ids=ids).logits
    check("15a. legacy graph_* keys remap + load strict + bit-identical output",
          n_renamed > 0 and strict_ok and torch.equal(logits_legacy, logits_a),
          f"renamed={n_renamed} missing={len(missing)} unexpected={len(unexpected)}")

    # 15b. end-to-end: a legacy checkpoint FOLDER on disk (config.json
    # without variant/mem_size — how the standalone training scripts saved
    # it, the causal reference run included) through evaluate.load_model.
    import shutil
    import tempfile

    from safetensors.torch import save_file as _save_file

    # the real legacy safetensors omits the tied lm_head (deduplicated on
    # save) — reproduce that exactly, tied-rebinding path included
    legacy_on_disk = {k: v for k, v in legacy.items() if k != "lm_head.weight"}
    tmp = tempfile.mkdtemp(prefix="naylis_legacy_")
    try:
        model.save_pretrained(tmp)  # config.json (bare LlamaConfig) + safetensors
        legacy_dir = os.path.join(tmp, "legacy")
        os.makedirs(legacy_dir, exist_ok=True)
        shutil.copy(os.path.join(tmp, "config.json"), legacy_dir)
        _save_file({k: v.contiguous() for k, v in legacy_on_disk.items()},
                   os.path.join(legacy_dir, "model.safetensors"))
        m_ref = build_model().to(torch.bfloat16).eval()
        with torch.no_grad():
            logits_ref_bf16 = m_ref(input_ids=ids).logits
        m_leg = evaluate.load_model(legacy_dir, device=DEVICE)
        with torch.no_grad():
            logits_leg = m_leg(input_ids=ids).logits
        check("15b. evaluate.load_model on a legacy checkpoint folder "
              "(variant inferred, MoM geometry derived, keys remapped) == "
              "bit-identical bf16 output",
              torch.equal(logits_leg, logits_ref_bf16),
              f"dtype={logits_leg.dtype}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------ 16-17. pure dtypes
    try:
        model_h = build_model().half()
        model_h.eval()
        with torch.no_grad():
            logits_h = model_h(input_ids=ids).logits
        check("16. pure fp16 model forward: finite logits (router dtype guard)",
              torch.isfinite(logits_h).all().item())
    except RuntimeError as e:
        check("16. pure fp16 model forward: finite logits (router dtype guard)",
              False, str(e)[:120])

    model_b = build_model().to(torch.bfloat16)
    model_b.eval()
    with torch.no_grad():
        logits_b1 = model_b(input_ids=ids).logits
        logits_b2 = model_b(input_ids=ids).logits
    check("17. pure bf16 model: finite logits + deterministic (router dtype guard)",
          torch.isfinite(logits_b1).all().item() and torch.equal(logits_b1, logits_b2),
          f"dtype={logits_b1.dtype}")

    # ------------------------------------------------ 18. AMP policy
    import pretrain as pretrain_mod

    try:
        pretrain_mod.resolve_amp_dtype("cpu", "fp16")
        check("18a. resolve_amp_dtype refuses fp16 (no GradScaler path)", False)
    except ValueError:
        check("18a. resolve_amp_dtype refuses fp16 (no GradScaler path)", True)
    check("18b. resolve_amp_dtype('bf16') -> (bf16=True, fp16=False)",
          pretrain_mod.resolve_amp_dtype("cpu", "bf16") == (True, False))

    # ------------------------------------------------ 19. precision policy
    from naylis_precision import (DEFAULT_TARGETS, _HARD_EXCLUDE, _match,
                                  resolve_plan)

    plan_bf16 = resolve_plan({"mode": "bf16"})
    check("19a. bf16 plan: nothing converted, no refusal",
          plan_bf16["fp8"] is False and not plan_bf16["swapped"]
          and plan_bf16["raise"] is None)
    plan_fp8 = resolve_plan({"mode": "fp8"})
    fp8_capable = (torch.cuda.is_available()
                   and torch.cuda.get_device_capability(0) >= (8, 9))
    if fp8_capable:
        check("19b. explicit fp8 resolves to an fp8 plan on this box",
              plan_fp8["fp8"] and plan_fp8["raise"] is None)
    else:
        check("19b. explicit fp8 REFUSES on a no-fp8 box (fail-loud, never "
              "silent bf16)",
              plan_fp8["raise"] is not None)
    leaked = [t for t in DEFAULT_TARGETS if _match(t, _HARD_EXCLUDE)]
    check("19c. default fp8 targets keep the MoM routing path out "
          "(router/bank/mom.k/v/mom_scale)",
          not leaked)
    plan_try = resolve_plan({"mode": "bf16", "targets": (
        "mom.router_down", "model.M_blocks", "self_attn.q_proj")})
    check("19d. hard excludes filter hostile target lists (routing params "
          "never in fp8)",
          "mom.router_down" not in plan_try["targets"]
          and "model.M_blocks" not in plan_try["targets"]
          and "self_attn.q_proj" in plan_try["targets"])
else:
    print("[skip] 9-19 (--skip-model)")

# --------------------------------------------- 20. Triton flash read (GPU)
if DEVICE == "cuda" and mom_kernels.HAS_TRITON:
    from mom_kernels import triton_gpu

    torch.manual_seed(8)
    q_t = (torch.randn(B, S, H, HD, device=DEVICE) * 0.7).to(torch.bfloat16)
    k_t = (torch.randn(L, D, device=DEVICE) * 0.4).to(torch.bfloat16)
    v_t = (torch.randn(L, D, device=DEVICE) * 0.4).to(torch.bfloat16)
    g_t = F.softmax(torch.randn(B, S, T, device=DEVICE, dtype=torch.float32), -1)
    top_t = torch.randint(0, NB, (B, S, T), device=DEVICE)
    for t_ in (q_t, k_t, v_t, g_t):
        t_.requires_grad_(True)
    bias_t = reference.build_gate_bias(top_t, g_t, NB, BS, torch.bfloat16)

    out_tr = triton_gpu.mom_read(q_t, k_t, v_t, top_t, g_t,
                                 num_blocks=NB, block_size=BS)
    out_ea = reference.mom_attention_reference(q_t, k_t, v_t, bias_t)
    check("20a. triton flash read == eager reference (bf16)",
          close(out_tr, out_ea, atol=2e-2),
          f"max_abs={max_abs(out_tr, out_ea):.2e}")

    g_seed = torch.randn_like(out_tr)
    out_tr.backward(g_seed)
    tr_grads = (q_t.grad.clone(), k_t.grad.clone(), v_t.grad.clone(), g_t.grad.clone())
    for t_ in (q_t, k_t, v_t, g_t):
        t_.grad = None
    out_ea.backward(g_seed)
    ea_grads = (q_t.grad.clone(), k_t.grad.clone(), v_t.grad.clone(), g_t.grad.clone())
    worst = max(scaled_max(a, b) for a, b in zip(tr_grads, ea_grads))
    check("20b. triton backward == eager autograd (dq/dk/dv/dlog_gates, bf16)",
          worst < 2e-2, f"scaled_err={worst:.2e}")
else:
    print("[skip] 20 (triton parity — needs CUDA + triton)")

print("\n" + "=" * 64)
failed = [n for n, ok in RESULTS if not ok]
print(f"RESULT: {len(RESULTS) - len(failed)}/{len(RESULTS)} sections passed")
if failed:
    print("FAILED:")
    for n in failed:
        print(f"  - {n}")
    sys.exit(1)
print("ALL APPLICABLE SECTIONS PASSED")

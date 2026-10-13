#!/usr/bin/env python3
# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""On-box validator of the fp8 precision stack (naylis_precision.py).

Sections: A — plan resolution for auto/bf16/fp8 (explicit fp8 must either
run fp8 or REFUSE — never a silent bf16 degradation); B — module
targeting on a real small Naylis model (trunk GEMMs swapped, MoM
routing/bank/lm_head untouched, mom_qo opt-in); C — forward/backward
finite under bf16 autocast (+TE fp8 context when armed); D — AdamW
micro-train with loss decreasing; E — checkpoint round-trip: the
converted model's state_dict must strict-load into a PLAIN model (the
autopsy/evaluate contract); F — TE Float8CurrentScaling vs torchao
tensorwise logits (the only same-class pair; informational tolerance).

Exit codes: 1 = failures (do NOT arm fp8 — fix first or pin bf16),
2 = no CUDA device (section A only ran). run_biggpu.sh gates the run
on this probe.
"""

import argparse
import contextlib
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=6, help="micro-train steps in section D")
args = parser.parse_args()

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from naylis_precision import apply_precision, resolve_plan

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


def finite(t):
    return t is None or bool(torch.isfinite(t.detach()).all().item())


print("=== A. plan resolution on this box ===")
for mode in ("auto", "bf16"):
    plan = resolve_plan({"mode": mode, "backend": "auto"})
    summary = f"mode={plan['mode']} backend={plan['backend']} fp8={plan['fp8']}"
    if plan["recipe_name"]:
        summary += f" recipe=[{plan['recipe_name']}]"
    print(f"  [precision mode={mode!r}] -> {summary}")
    for line in plan["notes"]:
        print(f"      {line}")
    if plan["raise"] is not None:
        check(f"A-{mode}: unexpected refusal for a fallback-capable mode", False,
              str(plan["raise"])[:160])
    else:
        check(f"A-{mode}: resolves without refusal", True)

plan_fp8 = resolve_plan({"mode": "fp8", "backend": "auto"})
summary = f"backend={plan_fp8['backend']} fp8={plan_fp8['fp8']}"
if plan_fp8["recipe_name"]:
    summary += f" recipe=[{plan_fp8['recipe_name']}]"
print(f"  [precision mode='fp8'] -> {summary}")
for line in plan_fp8["notes"]:
    print(f"      {line}")
if plan_fp8["fp8"] and plan_fp8["raise"] is None:
    check("A3. explicit mode='fp8' resolves to fp8=ON (named recipe above)", True, summary)
elif plan_fp8["raise"] is not None:
    check("A3. explicit mode='fp8' REFUSES on this box (fail loud)", True,
          str(plan_fp8["raise"])[:160])
else:
    check("A3. explicit mode='fp8' SILENTLY degraded to bf16", False,
          "explicit fp8 must raise or run fp8, never bf16")

AUTO_PLAN = resolve_plan({"mode": "auto", "backend": "auto"})
FP8_ARM = AUTO_PLAN["fp8"]

if not torch.cuda.is_available():
    print("\nprobe_fp8 needs a CUDA device for sections B-F (this box: CPU).")
    failed = [n for n, ok in RESULTS if not ok]
    print(f"RESULT: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed (A only)")
    sys.exit(2)

DEV = "cuda"
print(f"\nprobe_fp8 | torch={torch.__version__} device={torch.cuda.get_device_name(0)} "
      f"cap=SM{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]} "
      f"fp8_arm={'ON' if FP8_ARM else 'OFF (bf16 fallback path)'}")


def build_small_model():
    from transformers import LlamaConfig
    from model import NaylisLlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=2048, hidden_size=256, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256, rms_norm_eps=1e-5, rope_theta=500000.0,
        attention_bias=False, tie_word_embeddings=True,
        attn_implementation="sdpa", use_cache=False,
    )
    torch.manual_seed(11)
    return NaylisLlamaForCausalLM(
        cfg, mem_size=1024, mom_block_size=128, mom_top_blocks=2,
        mom_router_rank=32, mom_aux_weight=0.001,
        use_liger=False, use_kernel=False,
    )


def te_ctx(plan):
    if plan["fp8"] and plan["backend"] == "te":
        from transformer_engine.pytorch import fp8_autocast
        return fp8_autocast(enabled=True, fp8_recipe=plan["te_recipe"])
    return contextlib.nullcontext()


print("\n=== B. module targeting on a real (small) Naylis model ===")
model = build_small_model().to(DEV)
model, plan = apply_precision(model, {"mode": "auto", "backend": "auto"}, device=DEV)

expected = [
    f"model.layers.{i}.{t}"
    for i in range(2)
    for t in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
]
if plan["fp8"]:
    check("B1. swap list == trunk GEMMs exactly",
          sorted(plan["swapped"]) == sorted(expected),
          f"swapped={len(plan['swapped'])} expected={len(expected)}")
else:
    check("B1. bf16 fallback: nothing swapped", not plan["swapped"])


def is_fp8_module(mod):
    return ("Float8" in type(mod).__name__
            or type(mod).__module__.startswith("transformer_engine"))


mom0 = model.model.layers[0].mom
mom_qo = plan.get("_mom_qo", False)
excluded_ok = (
    not is_fp8_module(mom0.q_proj) or mom_qo
) and (
    not is_fp8_module(mom0.o_proj) or mom_qo
) and not any(
    is_fp8_module(m) for m in (mom0.k_proj, mom0.v_proj,
                               mom0.router_down, mom0.router_up)
)
check("B2. MoM layer untouched by fp8 (router/k/v + q/o by default)",
      excluded_ok,
      "router/k_proj/v_proj (and q_proj/o_proj unless mom_qo) stay nn.Linear")
check("B3. mom_scale / M_blocks are plain fp32 params",
      isinstance(model.model.layers[0].mom.mom_scale, torch.nn.Parameter)
      and model.model.M_blocks.dtype == torch.float32)
check("B4. lm_head NOT converted (tied embeddings + liger FLCE path)",
      not is_fp8_module(model.lm_head))

print("\n=== C. forward/backward finite (bf16 autocast, no GradScaler) ===")
ids = torch.randint(0, 2048, (4, 128), device=DEV)
labels = ids.clone()
model.train()
with torch.autocast(DEV, dtype=torch.bfloat16), te_ctx(plan):
    out = model(input_ids=ids, labels=labels)
out.loss.backward()
grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
check("C1. loss finite", finite(out.loss), f"loss={float(out.loss):.4f}")
check("C2. all grads finite", all(finite(g) for g in grads.values()),
      f"{len(grads)} grads")
check("C3. MoM autopsy params still carry grads (mom_scale, M_blocks, router)",
      finite(model.model.layers[0].mom.mom_scale.grad)
      and finite(model.model.M_blocks.grad)
      and finite(model.model.layers[0].mom.router_up.weight.grad))
model.zero_grad(set_to_none=True)

print(f"\n=== D. micro-train: {args.steps} AdamW steps, lr 3e-4 ===")
model = build_small_model().to(DEV)
model, plan = apply_precision(model, {"mode": "auto", "backend": "auto"}, device=DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
losses = []
ok_steps = True
for step in range(1, args.steps + 1):
    with torch.autocast(DEV, dtype=torch.bfloat16), te_ctx(plan):
        out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    gn = torch.norm(torch.stack(
        [p.grad.float().norm() for p in model.parameters() if p.grad is not None]))
    opt.step()
    opt.zero_grad(set_to_none=True)
    weights_finite = all(finite(p) for p in model.parameters())
    losses.append(float(out.loss))
    ok = finite(out.loss) and finite(gn) and weights_finite
    ok_steps = ok_steps and ok
    print(f"    step {step}: loss={float(out.loss):.4f} grad_norm={float(gn):.3f} finite={ok}")
    if not ok:
        break
check(f"D1. {args.steps}-step micro-train stays finite (fp8={'on' if plan['fp8'] else 'off'})",
      ok_steps)
check("D2. loss decreased over the micro-train", losses and losses[-1] < losses[0],
      f"{losses[0]:.4f} -> {losses[-1]:.4f}")

print("\n=== E. checkpoint round-trip: converted -> plain model, strict load ===")
sd = model.state_dict()
plain = build_small_model()
try:
    plain.load_state_dict(sd, strict=True)
    check("E1. state_dict loads into the PLAIN model (strict=True)",
          True, f"{len(sd)} tensors — autopsy/evaluate contract preserved")
except RuntimeError as e:
    check("E1. state_dict loads into the PLAIN model (strict=True)", False, str(e)[:200])

try:
    from naylis_precision import _te_availability, _torchao_availability
    te_ok, _ = _te_availability()
    ao_ok, _ = _torchao_availability()
except Exception:
    te_ok = ao_ok = False

if te_ok and ao_ok and torch.cuda.get_device_capability(0) >= (8, 9):
    print("\n=== F. cross-backend parity: TE Float8CurrentScaling vs torchao tensorwise ===")
    print("    (the only same-class pair alive — v4.1: TE has NO rowwise float8 recipe;")
    print("     both sides are per-tensor current scaling. Informational tolerance.)")
    try:
        m_te = build_small_model().to(DEV)
        m_ao = build_small_model().to(DEV)
        m_ao.load_state_dict(m_te.state_dict())
        m_te, plan_te = apply_precision(m_te, {"mode": "fp8", "backend": "te", "recipe": "current"}, device=DEV)
        # torchao converts modules directly — no context manager to keep.
        m_ao, _ = apply_precision(m_ao, {"mode": "fp8", "backend": "torchao", "recipe": "tensorwise"}, device=DEV)
        m_te.eval(); m_ao.eval()
        with torch.no_grad():
            with torch.autocast(DEV, dtype=torch.bfloat16), te_ctx(plan_te):
                l_te = m_te(input_ids=ids).logits
            with torch.autocast(DEV, dtype=torch.bfloat16):
                l_ao = m_ao(input_ids=ids).logits
        diff = (l_te.float() - l_ao.float()).abs().max().item()
        check("F1. TE current vs torchao tensorwise logits finite + close",
              finite(l_te) and finite(l_ao) and diff < 2.0,
              f"max|Δlogits|={diff:.4f} (per-tensor current scaling class — informational)")
    except Exception as e:
        check("F1. TE vs torchao cross-check", False, f"{type(e).__name__}: {e}")
else:
    print("\n[F skipped: needs BOTH backends installed and SM >= 8.9]")

print("\n" + "=" * 64)
failed = [n for n, ok in RESULTS if not ok]
print(f"RESULT: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
if failed:
    print("FAILED:")
    for n in failed:
        print(f"  - {n}")
    print("\nDo NOT start the fp8 run. Fix the failing section first "
          "(or set [precision] mode='bf16' to train without fp8).")
    sys.exit(1)
print("ALL GREEN — the precision stack is sound on this box. Arm "
      "[precision] mode='fp8' (or 'auto') for the run.")

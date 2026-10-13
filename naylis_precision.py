# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""Precision policy: bf16 native, fp8 opt-in.

``resolve_plan`` turns the ``[precision]`` table of an experiment toml
(mode/backend/recipe/targets/mom_qo) into an explicit, printable plan;
``apply_precision`` executes it on a built model. The policy is fail-loud
by design: an explicit ``mode="fp8"`` that cannot run (no SM89+, no
backend, missing recipe) raises instead of silently degrading to bf16 —
a silent fallback would corrupt an A/B arm.

Backends: transformer_engine (``te``) swaps the targeted nn.Linear for
te.Linear and the trainer wraps compute_loss in fp8_autocast with the
resolved recipe; torchao (``torchao``) drives convert_to_float8_training
with a module filter. Recipes are NAMED and pinnable — "current" and
"tensorwise" are the same per-tensor current-scaling class on both
backends (the only cross-backend pair); rowwise lives on torchao only,
block/delayed on TE only. TE has no rowwise float8 recipe at all.

Targeting: fp8 covers the trunk GEMMs (attention q/k/v/o, mlp
 gate/up/down) and optionally MoM q/o (``mom_qo = true``). The MoM
routing path (router_up/router_down, mom_scale, M_blocks, mom.k/v_proj),
norms, embeddings and the tied lm_head are hard-excluded — they stay
fp32/autocast so the routing signal and the memory bank keep full
precision. fp8 needs CUDA (SM89+); TPU runs stay bf16.

probe_fp8.py is the on-box validator: plan resolution, swap list,
forward/backward finiteness, micro-train, checkpoint round-trip and the
cross-backend sanity check.
"""
import inspect

import torch
import torch.nn as nn


DEFAULT_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

_MOM_QO_TARGETS = ("mom.q_proj", "mom.o_proj")

_HARD_EXCLUDE = (
    "router_down",
    "router_up",
    "mom_scale",
    "M_blocks",
    "mom.k_proj",
    "mom.v_proj",
    "lm_head",
    "embed_tokens",
    "mom_norm",
    "input_layernorm",
    "post_attention_layernorm",
    "norm",
)

FP8_MIN_CAPABILITY = (8, 9)

_VALID_MODES = ("auto", "fp8", "bf16")
_VALID_BACKENDS = ("auto", "te", "torchao")
_VALID_RECIPES = ("auto", "current", "tensorwise", "rowwise", "block", "delayed")

_TE_RECIPES = {
    "current": (
        "Float8CurrentScaling",
        "Float8CurrentScaling — E4M3 per-tensor, current scaling, no EMA",
    ),
    "block": (
        "Float8BlockScaling",
        "Float8BlockScaling — E4M3, 1x32 blocks, current scaling (SM90+ rec.)",
    ),
    "delayed": (
        "DelayedScaling",
        "DelayedScaling — E4M3 per-tensor EMA — not recipe-equivalent to any "
        "torchao recipe; pin ONE backend for the whole A/B",
    ),
}
_TE_PREF_ORDER = {
    "auto": ("current", "block", "delayed"),
    "current": ("current",),
    "tensorwise": ("current",),
    "block": ("block",),
    "delayed": ("delayed",),
}

_AO_RECIPES = {
    "rowwise": "rowwise",
    "tensorwise": "tensorwise",
    "current": "tensorwise",
}


def _cuda_capability(device=None) -> tuple | None:
    """(major, minor) of the CUDA device (or None when not applicable)."""
    if not torch.cuda.is_available():
        return None
    try:
        idx = 0
        if device is not None:
            dev = torch.device(device)
            if dev.type == "cuda" and dev.index is not None:
                idx = dev.index
        return torch.cuda.get_device_capability(idx)
    except Exception:
        return None


def _te_availability() -> tuple[bool, str]:
    """Is transformer_engine importable, and which recipes does it expose?"""
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.pytorch import fp8_autocast
        from transformer_engine.common import recipe
        _ = (te.Linear, fp8_autocast)
        have = [key for key in _TE_RECIPES if hasattr(recipe, _TE_RECIPES[key][0])]
        ver = getattr(__import__("transformer_engine"), "__version__", "?")
        return True, (
            f"transformer_engine {ver} "
            f"(recipes present: {', '.join(have) if have else 'NONE of current/block/delayed — old or broken install'})"
        )
    except Exception as e:
        return False, f"transformer_engine unusable ({type(e).__name__}: {e})"


def _te_make_recipe(recipe_pref: str) -> tuple[object, str, str]:
    """Instantiate the first TE recipe class matching the preference order."""
    from transformer_engine.common import recipe as te_recipe_mod

    order = _TE_PREF_ORDER[recipe_pref]
    for key in order:
        cls_name, human = _TE_RECIPES[key]
        cls = getattr(te_recipe_mod, cls_name, None)
        if cls is None:
            continue
        try:
            return cls(), key, human
        except TypeError as e:
            raise RuntimeError(
                f"TE recipe {cls_name}() rejected no-arg construction ({e}) — "
                f"unexpected TE API shape on this box; run probe_fp8.py and pin "
                f"a known-good version."
            ) from e
    wanted = ", ".join(_TE_RECIPES[k][0] for k in order)
    raise RuntimeError(
        f"This TE install exposes none of: {wanted}. "
        f"TE has no rowwise float8 recipe on main. "
        f"pip install -U transformer_engine, or set backend='torchao'."
    )


def _torchao_resolver():
    """Resolve torchao: (ok, note, factory) with a recipe-name factory."""
    try:
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig

        def factory_modern(recipe_key):
            if recipe_key not in _AO_RECIPES:
                raise RuntimeError(
                    f"torchao.float8 serves rowwise|tensorwise only, got {recipe_key!r}"
                )
            name = _AO_RECIPES[recipe_key]
            if hasattr(Float8LinearConfig, "from_recipe_name"):
                return convert_to_float8_training, Float8LinearConfig.from_recipe_name(name)
            try:
                return convert_to_float8_training, Float8LinearConfig(recipe_name=name)
            except TypeError as e:
                raise RuntimeError(
                    f"Float8LinearConfig exposes neither from_recipe_name nor "
                    f"recipe_name= ({e}) — torchao API shape not recognised; "
                    f"pin a torchao matching the documented torchao.float8 API."
                ) from e

        ver = getattr(__import__("torchao"), "__version__", "?")
        return True, f"torchao {ver} (torchao.float8 — current API)", factory_modern
    except ImportError:
        pass
    except Exception as e:
        return False, f"torchao.float8 import raised {type(e).__name__}: {e}", None

    return False, "torchao not installed (or torchao.float8 not importable)", None


def _torchao_availability() -> tuple[bool, str]:
    """Availability note for the plan log (factory discarded)."""
    ok, note, _factory = _torchao_resolver()
    return ok, note


def _torchao_convert(model, targets, recipe_key):
    """Convert the targeted Linears via torchao convert_to_float8_training.

    Uses the module filter the torchao API exposes (module_filter_fn or
    filter_fn depending on version) so the hard exclusions hold; returns
    the names of the swapped modules.
    """
    ok, note, factory = _torchao_resolver()
    if not ok:
        raise RuntimeError(f"torchao became unavailable at convert time ({note})")
    convert_fn, config = factory(recipe_key)

    def _filter(mod, fqn):
        return isinstance(mod, nn.Linear) and _match(fqn, targets)

    params = inspect.signature(convert_fn).parameters
    if "module_filter_fn" in params:
        convert_fn(model, module_filter_fn=_filter, config=config)
    elif "filter_fn" in params:
        convert_fn(model, filter_fn=_filter, config=config)
    else:
        raise RuntimeError(
            "convert_to_float8_training exposes no module filter argument — "
            "refusing to convert the WHOLE model (MoM exclusions are hard). "
            "Pin a torchao version with module_filter_fn/filter_fn support."
        )
    swapped = [
        name for name, mod in model.named_modules()
        if "Float8" in type(mod).__name__ and _match(name, targets)
    ]
    return swapped


def resolve_plan(prec_cfg: dict | None, device=None) -> dict:
    """Turn the ``[precision]`` config into an explicit plan dict.

    The plan carries mode/backend/recipe resolution, the final target
    list, human-readable notes (one [precision] line each, printed by
    apply_precision) and, when the request cannot run as asked, a
    ``raise`` entry the caller must honour — resolve_plan never converts
    and never silently falls back on explicit fp8.
    """
    cfg = prec_cfg or {}
    mode = str(cfg.get("mode", "bf16")).strip().lower()
    backend = str(cfg.get("backend", "auto")).strip().lower()
    recipe = str(cfg.get("recipe", "auto")).strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"[precision] mode must be one of {_VALID_MODES}, got {mode!r}")
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"[precision] backend must be one of {_VALID_BACKENDS}, got {backend!r}")
    if recipe not in _VALID_RECIPES:
        raise ValueError(
            f"[precision] recipe must be one of {_VALID_RECIPES}, got {recipe!r} "
            f"(TE has no rowwise float8 — current/tensorwise are the same class "
            f"on both backends)"
        )

    mom_qo = bool(cfg.get("mom_qo", False))
    targets = tuple(cfg.get("targets", None) or DEFAULT_TARGETS)
    if mom_qo:
        targets = tuple(targets) + _MOM_QO_TARGETS
    targets = tuple(t for t in targets if not _match(t, _HARD_EXCLUDE))

    notes = []
    plan = {
        "mode": mode,
        "backend": None,
        "fp8": False,
        "recipe_key": None,
        "recipe_name": None,
        "recipe_class": None,
        "te_recipe": None,
        "targets": targets,
        "_mom_qo": mom_qo,
        "swapped": [],
        "notes": notes,
        "raise": None,
    }

    cap = _cuda_capability(device)
    dev_note = f"device capability = SM{cap[0]}.{cap[1]}" if cap else "no CUDA device visible"
    notes.append(f"[precision] requested mode={mode!r} backend={backend!r} recipe={recipe!r} ({dev_note})")

    if mode == "bf16":
        notes.append("[precision] mode=bf16 -> pure phase-2 numerics, nothing converted.")
        plan["backend"] = "none"
        return plan

    hw_ok = bool(cap) and cap >= FP8_MIN_CAPABILITY
    if not hw_ok:
        msg = (
            f"[precision] fp8 needs SM >= {FP8_MIN_CAPABILITY[0]}.{FP8_MIN_CAPABILITY[1]} "
            f"(fp8 tensor cores); this device is "
            f"{'SM%d.%d' % cap if cap else 'not CUDA'}."
        )
        if mode == "fp8":
            plan["raise"] = RuntimeError(msg + " Set mode='bf16' (or 'auto') in [precision].")
            notes.append(msg + " EXPLICIT mode=fp8 -> refusing to run.")
            return plan
        notes.append(msg + " auto -> falling back to bf16.")
        plan["backend"] = "none"
        return plan

    te_ok, te_note = _te_availability()
    ao_ok, ao_note = _torchao_availability()
    notes.append(f"[precision] TE      : {te_note}")
    notes.append(f"[precision] torchao : {ao_note}")

    if backend == "te":
        if not te_ok:
            plan["raise"] = RuntimeError(
                f"[precision] backend='te' requested but {te_note}. "
                f"pip install transformer_engine, or set backend='torchao'."
            )
            return plan
        chosen = "te"
    elif backend == "torchao":
        if not ao_ok:
            plan["raise"] = RuntimeError(
                f"[precision] backend='torchao' requested but {ao_note}. "
                f"pip install torchao (torchao.float8 API recommended), or set backend='te'."
            )
            return plan
        chosen = "torchao"
    else:
        if te_ok:
            chosen = "te"
        elif ao_ok:
            chosen = "torchao"
        elif mode == "fp8":
            plan["raise"] = RuntimeError(
                "[precision] mode='fp8' but NEITHER transformer_engine NOR torchao is "
                f"usable (TE: {te_note} | torchao: {ao_note}). Install a backend "
                "or set mode='bf16' deliberately — a silent bf16 fallback here "
                "would corrupt an fp8 A/B arm."
            )
            notes.append("[precision] auto: no usable fp8 backend -> EXPLICIT mode=fp8 -> refusing to run.")
            return plan
        else:
            notes.append(
                "[precision] auto: neither transformer_engine nor torchao is usable -> bf16. "
                "(Both are optional: bf16 needs neither. mode='auto' semantics.)"
            )
            plan["backend"] = "none"
            return plan
        notes.append(f"[precision] auto -> backend '{chosen}' (TE primary, torchao fallback).")

    plan["backend"] = chosen
    plan["fp8"] = True

    if chosen == "te":
        if recipe == "rowwise":
            plan["raise"] = RuntimeError(
                "[precision] recipe='rowwise' + backend='te': TE has NO rowwise float8 "
                "recipe — Float8RowwiseRecipe does not exist on TE main. Rowwise "
                "lives on torchao only. Use recipe='auto' (-> current), "
                "recipe='current', or backend='torchao'."
            )
            return plan
        recipe_obj, recipe_key, recipe_name = _te_make_recipe(recipe)
        plan["te_recipe"] = recipe_obj
        plan["recipe_key"] = recipe_key
        plan["recipe_name"] = f"TE {recipe_name}"
        plan["recipe_class"] = {
            "current": "per-tensor current scaling",
            "block": "1x32 block current scaling",
            "delayed": "per-tensor EMA (DelayedScaling)",
        }[recipe_key]
        notes.append(f"[precision] TE recipe: {plan['recipe_name']}")
        if recipe_key == "delayed":
            notes.append(
                "[precision] WARNING: DelayedScaling is the OLD per-tensor EMA recipe. "
                "Prefer Float8CurrentScaling/Float8BlockScaling; do not mix backends "
                "across an A/B while on DelayedScaling."
            )
    else:
        if recipe in ("block", "delayed"):
            plan["raise"] = RuntimeError(
                f"[precision] recipe={recipe!r} is a TE-side recipe; torchao float8 serves "
                f"'rowwise' | 'tensorwise'. Pin backend='te' for {recipe!r}, or pick a "
                f"torchao recipe."
            )
            return plan
        if recipe == "auto":
            ao_recipe = "rowwise"
            origin = "auto -> rowwise (torchao-recommended)"
        else:
            ao_recipe = _AO_RECIPES[recipe]
            origin = f"requested {recipe!r} -> {ao_recipe!r}"
        _, _, factory = _torchao_resolver()
        try:
            factory(ao_recipe)
        except RuntimeError as e:
            plan["raise"] = RuntimeError(f"[precision] {e}")
            return plan
        plan["recipe_key"] = ao_recipe
        plan["recipe_name"] = {
            "rowwise": (
                "torchao Float8LinearConfig.from_recipe_name('rowwise') — E4M3, "
                "per-row act / per-col weight, current scaling"
            ),
            "tensorwise": (
                "torchao Float8LinearConfig.from_recipe_name('tensorwise') — E4M3 "
                "per-tensor, current scaling (== TE Float8CurrentScaling class)"
            ),
        }[ao_recipe]
        plan["recipe_class"] = {
            "rowwise": "rowwise/colwise current scaling",
            "tensorwise": "per-tensor current scaling",
        }[ao_recipe]
        notes.append(f"[precision] torchao recipe: {plan['recipe_name']} [{origin}]")

    if plan["recipe_class"] == "per-tensor current scaling":
        notes.append(
            "[precision] cross-backend parity: TE Float8CurrentScaling <=> torchao "
            "'tensorwise' (same class). Switching backend mid-A/B is acceptable ONLY "
            "within this pair."
        )
    else:
        notes.append(
            "[precision] cross-backend WARNING: this recipe class has NO equivalent on "
            "the other backend (TE dropped rowwise float8). Pin ONE backend for the "
            "whole A/B."
        )

    notes.append(
        "[precision] fp8 targets: " + ", ".join(targets) +
        (" (+ MoM q/o)" if mom_qo else "") +
        " — router/mom_scale/M_blocks/mom.k/v_proj/lm_head stay out of fp8."
    )
    return plan


def _match(fqn: str, targets) -> bool:
    """Does a fully-qualified module name match a target suffix?"""
    return any(fqn == t or fqn.endswith("." + t) for t in targets)


def _parent_and_attr(model, fqn: str):
    """Resolve 'a.b.c' to (object holding 'c', 'c') for attribute swaps."""
    parts = fqn.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _convert_with_te(model, targets):
    """Swap the targeted nn.Linear modules for te.Linear (weights copied)."""
    import transformer_engine.pytorch as te

    swapped = []
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if not _match(name, targets):
            continue
        parent, attr = _parent_and_attr(model, name)
        new = te.Linear(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
        with torch.no_grad():
            new.weight.copy_(mod.weight)
            if mod.bias is not None:
                new.bias.copy_(mod.bias)
        setattr(parent, attr, new)
        swapped.append(name)
    return swapped


def apply_precision(model, prec_cfg: dict | None, *, device=None):
    """Apply the plan to ``model``; returns (model, plan).

    Raises the plan's refusal when the config cannot run as asked, prints
    the notes, and records the swapped module names in plan["swapped"]
    (empty for bf16 — nothing is converted).
    """
    plan = resolve_plan(prec_cfg, device=device)

    if plan["raise"] is not None:
        raise plan["raise"]

    if plan["fp8"]:
        if plan["backend"] == "te":
            plan["swapped"] = _convert_with_te(model, plan["targets"])
        elif plan["backend"] == "torchao":
            plan["swapped"] = _torchao_convert(model, plan["targets"], plan["recipe_key"])
        plan["notes"].append(
            f"[precision] converted {len(plan['swapped'])} module(s) to fp8 "
            f"[{plan['recipe_name']}]: " + ", ".join(plan["swapped"])
        )
    else:
        plan["notes"].append("[precision] bf16 native (fp32 master weights, no GradScaler).")

    for line in plan["notes"]:
        print(line)
    return model, plan

# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""Benchmark the ablation checkpoints with lm-evaluation-harness.

Downloads each arm (final checkpoint first, seed-suffixed intermediate
checkpoint as fallback) from the Hub — or uses a local ./model_<name>
with --skip-download — loads it in bf16 (no liger, no kernels at eval
time; the MoM layer runs its eager path) and runs the zero-shot suite
(plus 5-shot MMLU) grouped by shot count. Results land in a comparison
table and a JSON file; a task that fails or is absent is reported as
N/A instead of aborting the other arms.

Usage: python evaluate.py [--models naylis_mom_causal vanilla vanilla_thin_ffn ...]
                          [--device auto|cuda|cpu|xla] [--skip-download]
"""
import argparse
import glob
import json
import logging
import os
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, PreTrainedModel

from model import NaylisLlamaForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "TheRealSkyline/naylis_ablation_300M"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_TOKENIZER = "HuggingFaceTB/cosmo2-tokenizer"
# Folder keys of the consolidated HF dataset (TheRealSkyline/naylis_ablation_300M).
# The deprecated leaky arm (DEPRECATED_naylis_graph_moe_leaky) is excluded on
# purpose: its numbers are invalidated and not faithfully evaluated here.
DEFAULT_MODELS = ["naylis_mom_causal", "vanilla", "vanilla_thin_ffn"]
DEFAULT_SEED = 257

TASKS: dict[str, int] = {
    "hellaswag": 0,
    "arc_easy": 0,
    "arc_challenge": 0,
    "piqa": 0,
    "boolq": 0,
    "copa": 0,
    "winogrande": 0,
    "sciq": 0,
    "openbookqa": 0,
    "mmlu": 5,
}

_TOKENIZERS: dict[str, AutoTokenizer] = {}


def is_xla(device) -> bool:
    """True when ``device`` is an XLA (TPU) device."""
    try:
        return torch.device(device).type == "xla"
    except Exception:
        return str(device).lower().startswith("xla")


def resolve_device() -> str:
    """Device autodetection: CUDA -> XLA/TPU -> CPU."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        try:
            kind = str(xr.device_type())
        except Exception:
            kind = "unknown"
        if kind.upper() not in ("TPU", "UNKNOWN"):
            logger.warning("torch_xla runtime is %s, not TPU — using CPU", kind)
            return "cpu"
        device = str(xm.xla_device())
        logger.info("XLA device available: %s (runtime=%s)", device, kind)
        return device
    except ImportError:
        return "cpu"


def _resolve_requested(device: str) -> str:
    """Map the --device argument (auto/cpu/xla/...) to a concrete device."""
    if device == "auto":
        return resolve_device()
    if device == "cpu":
        return "cpu"
    if is_xla(device):
        if ":" in str(device):
            return device
        resolved = resolve_device()
        if not is_xla(resolved):
            logger.warning("--device xla requested but no TPU available, using %s", resolved)
        return resolved
    return device if torch.cuda.is_available() else "cpu"


def _eval_dtype() -> torch.dtype:
    """Evaluation dtype: bf16, matching the training numerics."""
    return torch.bfloat16


def get_tokenizer(name: str) -> AutoTokenizer:
    """Cached tokenizer lookup by name."""
    if name not in _TOKENIZERS:
        logger.info("Loading tokenizer: %s", name)
        _TOKENIZERS[name] = AutoTokenizer.from_pretrained(name)
    return _TOKENIZERS[name]


def download_model_from_hf(model_name: str, repo_id: str, repo_type: str, seed: int = DEFAULT_SEED) -> str | None:
    """Local path of the arm's weights: final folder, else the seed checkpoint."""
    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or None
    logger.info("Downloading %s from %s (%s)", model_name, repo_id, repo_type)
    try:
        local_dir = snapshot_download(
            repo_id=repo_id, repo_type=repo_type, token=token,
            allow_patterns=f"{model_name}/*",
        )
        model_path = os.path.join(local_dir, model_name)
        if os.path.isdir(model_path) and os.listdir(model_path):
            logger.info("Downloaded %s -> %s", model_name, model_path)
            return model_path
        logger.warning("Empty directory for %s, trying fallback checkpoint", model_name)
    except Exception:
        logger.exception("Failed to download final checkpoint for %s", model_name)

    try:
        key = f"{model_name}_seed{seed}"
        local_dir = snapshot_download(
            repo_id=repo_id, repo_type=repo_type, token=token,
            allow_patterns=f"{key}/checkpoint/*",
        )
        model_path = os.path.join(local_dir, key, "checkpoint")
        if os.path.isdir(model_path) and os.listdir(model_path):
            logger.info("Downloaded intermediate checkpoint for %s -> %s", model_name, model_path)
            return model_path
    except Exception:
        logger.exception("Fallback download also failed for %s", model_name)

    return None


def _find_weights_file(model_path: str) -> str:
    """Weights file (or shard index) inside a checkpoint folder, or raise."""
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = os.path.join(model_path, name)
        if os.path.isfile(candidate):
            return candidate
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        candidate = os.path.join(model_path, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No weights file found in {model_path}")


def _load_state_dict(weights_path: str) -> dict[str, torch.Tensor]:
    """Load a state dict: safetensors (incl. shards) or torch .bin."""
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(weights_path)

    if weights_path.endswith(".index.json"):
        model_dir = os.path.dirname(weights_path)
        state_dict: dict[str, torch.Tensor] = {}
        if weights_path.endswith(".safetensors.index.json"):
            from safetensors.torch import load_file

            for shard in sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors"))):
                state_dict.update(load_file(shard))
        else:
            for shard in sorted(glob.glob(os.path.join(model_dir, "pytorch_model-*.bin"))):
                state_dict.update(torch.load(shard, map_location="cpu"))
        return state_dict

    return torch.load(weights_path, map_location="cpu")


def _load_weights_strict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], model_name: str) -> None:
    """strict=False load, then raise if anything was missing/unexpected."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected state_dict mismatch for model={model_name!r}: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _remap_legacy_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename pre-repo checkpoint keys to this codebase's layout.

    The causal reference run (and the earlier ablation arms) were trained
    by the standalone scripts, which named the MoM submodules graph_mem /
    graph_norm / graph_scale; this repo names them mom / mom_norm /
    mom_scale with identical shapes and semantics. Returns the state dict
    unchanged when no legacy key is present.
    """
    remapped = {}
    renamed = 0
    for key, value in state_dict.items():
        new_key = key
        if ".graph_mem." in new_key:
            new_key = new_key.replace(".graph_mem.", ".mom.")
        if ".graph_norm." in new_key:
            new_key = new_key.replace(".graph_norm.", ".mom_norm.")
        if new_key.endswith(".mom.graph_scale"):
            new_key = new_key[: -len("graph_scale")] + "mom_scale"
        if new_key != key:
            renamed += 1
        remapped[new_key] = value
    if renamed:
        logger.info(
            "legacy checkpoint layout detected: %d keys remapped "
            "(graph_mem/graph_norm/graph_scale -> mom/mom_norm/mom_scale). "
            "NOTE: checkpoints trained with the PRE-CAUSAL (leaky) router are "
            "not faithfully evaluated by this codebase; the causal reference "
            "run loads and evaluates faithfully.",
            renamed,
        )
    return remapped


_LEGACY_TOP_BLOCKS = 2  # protocol value of every run so far; not recoverable from weights
_LEGACY_AUX_WEIGHT = 0.01  # training-only knob, inert at eval


def _resolve_mom_kwargs(config: LlamaConfig, state_dict: dict[str, torch.Tensor]) -> dict | None:
    """MoM constructor kwargs for a naylis checkpoint, config or legacy.

    Repo checkpoints carry mem_size / mom_* in their config.json. The
    checkpoints trained by the standalone scripts (the causal reference run
    included) saved a bare LlamaConfig — those derive the bank geometry
    from the weights themselves: M_blocks gives num_blocks and block_size,
    router_down gives router_rank. top_blocks is a routing hyperparameter,
    invisible in the weights: it defaults to the protocol value (2).
    Returns None when the checkpoint has no MoM layers at all.
    """
    has_mom = any(".mom." in k for k in state_dict)
    if not has_mom:
        return None

    mem_size = getattr(config, "mem_size", None)
    if mem_size is not None:
        return {
            "mem_size": mem_size,
            "mom_block_size": config.mom_block_size,
            "mom_top_blocks": config.mom_top_blocks,
            "mom_router_rank": config.mom_router_rank,
            "mom_aux_weight": config.mom_aux_weight,
        }

    mb = state_dict.get("model.M_blocks")
    if mb is None:
        raise RuntimeError(
            "MoM checkpoint without model.M_blocks — cannot derive the bank "
            "geometry (legacy config carries no [naylis] keys)."
        )
    rd = next((v for k, v in state_dict.items()
               if k.endswith(".mom.router_down.weight")), None)
    if rd is None:
        raise RuntimeError("MoM checkpoint without a router — cannot derive router_rank.")

    num_blocks = mb.shape[0]
    block_size = mb.shape[1] // config.hidden_size
    router_rank = int(rd.shape[0])
    logger.info(
        "legacy config without MoM keys — derived from the checkpoint: "
        "mem_size=%d (bank %d x %d), router_rank=%d, top_blocks=%d "
        "(protocol default, not stored in the weights), aux_weight=%.3f "
        "(training-only, inert at eval)",
        num_blocks * block_size, num_blocks, block_size, router_rank,
        _LEGACY_TOP_BLOCKS, _LEGACY_AUX_WEIGHT,
    )
    return {
        "mem_size": num_blocks * block_size,
        "mom_block_size": block_size,
        "mom_top_blocks": _LEGACY_TOP_BLOCKS,
        "mom_router_rank": router_rank,
        "mom_aux_weight": _LEGACY_AUX_WEIGHT,
    }


def load_model(model_path: str, device: str = "cuda") -> PreTrainedModel:
    """Build the variant from the saved config, load weights, eval mode.

    config.variant picks Llama vs NaylisLlamaForCausalLM; when the config
    predates the repo (no variant key — the standalone training scripts),
    the variant is inferred from the weights and the MoM geometry derived
    from the bank shapes. The tied lm_head is re-bound from the embeddings
    when the checkpoint omits it; the model is cast to the bf16 eval dtype
    before moving to the device.
    """
    config = LlamaConfig.from_pretrained(model_path)
    state_dict = _load_state_dict(_find_weights_file(model_path))
    state_dict = _remap_legacy_keys(state_dict)

    mom_kwargs = _resolve_mom_kwargs(config, state_dict)
    variant = getattr(config, "variant", None)
    if variant is None:
        variant = "naylis" if mom_kwargs else "vanilla"
        logger.info("config.variant absent — resolved from the weights: %r", variant)

    if variant == "naylis":
        model = NaylisLlamaForCausalLM(config, use_liger=False, **mom_kwargs)
    else:
        model = LlamaForCausalLM(config)

    if (
        config.tie_word_embeddings
        and "lm_head.weight" not in state_dict
        and "model.embed_tokens.weight" in state_dict
    ):
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    _load_weights_strict(model, state_dict, os.path.basename(model_path))

    model = model.to(_eval_dtype())
    model.eval()
    return model.to(device)


def benchmark_model(
    model_path: str,
    model_name: str,
    tasks: dict[str, int],
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, dict[str, Any]] | None:
    """Run the lm-eval suite on one arm; {task: {accuracy, metric}} or None.

    Tasks are grouped by shot count for one evaluator call per group;
    acc_norm is preferred over acc when the task reports it. A failed
    task group degrades to N/A entries for that group only.
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    logger.info("Benchmarking %s (%s) on: %s", model_name, model_path, tasks)

    try:
        model = load_model(model_path, device=device)
        tokenizer_name = getattr(model.config, "tokenizer_name", None) or DEFAULT_TOKENIZER
        tokenizer = get_tokenizer(tokenizer_name)
    except Exception:
        logger.exception("Failed to load model for %s", model_name)
        return None

    try:
        lm = HFLM(
            pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
            device=device,
        )
    except Exception:
        logger.exception("Failed to initialize HFLM for %s", model_name)
        return None

    results_dict: dict[str, dict[str, Any]] = {}

    shot_groups: dict[int, list[str]] = {}
    for task_name, shots in tasks.items():
        shot_groups.setdefault(shots, []).append(task_name)

    for shots, task_names in shot_groups.items():
        try:
            results = evaluator.simple_evaluate(model=lm, tasks=task_names, num_fewshot=shots, batch_size=batch_size)
        except Exception:
            logger.exception("lm-eval run failed for %s tasks=%s", model_name, task_names)
            for task_name in task_names:
                results_dict[task_name] = {"accuracy": None, "metric": "N/A"}
            continue

        for task_name in task_names:
            task_results = results["results"].get(task_name)
            if task_results is None:
                results_dict[task_name] = {"accuracy": None, "metric": "N/A"}
                logger.warning("Task %s not present in results for %s", task_name, model_name)
                continue

            if "acc_norm,none" in task_results:
                acc, metric_name = task_results["acc_norm,none"], "acc_norm"
            elif "acc,none" in task_results:
                acc, metric_name = task_results["acc,none"], "acc"
            else:
                acc, metric_name = None, "N/A"

            results_dict[task_name] = {"accuracy": acc, "metric": metric_name}
            logger.info("  %-20s %s", task_name, f"{acc:.4f} ({metric_name})" if acc is not None else "N/A")

    return results_dict


def print_results_table(all_results: dict[str, dict[str, dict[str, Any]]], tasks: list[str]) -> None:
    """Print the arms x tasks accuracy table with an AVERAGE row."""
    model_names = list(all_results.keys())
    col_width = 18

    header = f"{'Task':<20}" + "".join(f"{name:<{col_width}}" for name in model_names)
    print(header)
    print("-" * len(header))

    for task in tasks:
        row = f"{task:<20}"
        for name in model_names:
            acc = all_results.get(name, {}).get(task, {}).get("accuracy")
            row += f"{acc:<{col_width}.4f}" if acc is not None else f"{'N/A':<{col_width}}"
        print(row)

    print("-" * len(header))
    row = f"{'AVERAGE':<20}"
    for name in model_names:
        accs = [r["accuracy"] for r in all_results.get(name, {}).values() if r["accuracy"] is not None]
        row += f"{np.mean(accs):<{col_width}.4f}" if accs else f"{'N/A':<{col_width}}"
    print(row)


def save_results(all_results: dict[str, Any], output_path: str) -> None:
    """Dump the full results dict to JSON."""
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI: models, repo, tasks/shots, batch size, device, output."""
    parser = argparse.ArgumentParser(description="Benchmark Naylis checkpoints with lm-evaluation-harness")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model names to benchmark (repo folder keys, e.g. naylis_mom_causal vanilla)")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face repo hosting the checkpoints")
    parser.add_argument("--repo_type", type=str, default=DEFAULT_REPO_TYPE, help="Repo type (dataset or model)")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), help="lm-eval task names to run")
    parser.add_argument(
        "--num_fewshot", type=int, default=None,
        help="Override the per-task shot count in TASKS and apply it to every task in this run",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, xla, or auto (default: cuda -> xla -> cpu)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed suffix used for fallback checkpoint lookup")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Use local checkpoints at ./model_<name> instead of downloading from the Hub",
    )
    return parser


def main() -> None:
    """Resolve args, benchmark every retrievable arm, print and save."""
    args = build_arg_parser().parse_args()

    resolved_tasks = {task: (args.num_fewshot if args.num_fewshot is not None else TASKS.get(task, 0)) for task in args.tasks}
    resolved_device = _resolve_requested(args.device)

    logger.info("Models: %s", args.models)
    logger.info("Tasks: %s", resolved_tasks)
    logger.info("Repo: %s (%s)", args.repo_id, args.repo_type)
    logger.info("Device: %s | dtype: %s | batch_size: %d", resolved_device, _eval_dtype(), args.batch_size)

    try:
        import lm_eval

        logger.info("lm-eval version: %s", lm_eval.__version__)
    except ImportError:
        logger.error("lm-eval is not installed. Run: pip install lm-eval")
        return

    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    for model_name in args.models:
        logger.info("Processing model: %s", model_name)

        if args.skip_download:
            model_path = f"./model_{model_name}"
            if not os.path.exists(model_path):
                logger.warning("Local checkpoint not found: %s", model_path)
                continue
        else:
            model_path = download_model_from_hf(model_name, repo_id=args.repo_id, repo_type=args.repo_type, seed=args.seed)
            if model_path is None:
                logger.warning("Could not retrieve checkpoint for %s", model_name)
                continue

        results = benchmark_model(
            model_path=model_path, model_name=model_name, tasks=resolved_tasks,
            batch_size=args.batch_size, device=resolved_device,
        )
        if results is not None:
            all_results[model_name] = results

    if not all_results:
        logger.warning("No results to report.")
        return

    print_results_table(all_results, list(resolved_tasks))
    save_results(all_results, args.output)


if __name__ == "__main__":
    main()

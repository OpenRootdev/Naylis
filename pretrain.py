# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""Pretraining harness for the Naylis ablation study.

One experiment = one TOML: vanilla.toml (compute-matched dense
baseline), vanilla_thin.toml (trunk-matched dense baseline),
naylisMoM.toml / naylisMoM_v3.toml
(the MoM arms), bench_*.toml (precision/timing benches). The harness
downloads the token bin from the Hub, builds the variant (Llama or
NaylisLlamaForCausalLM — causal router, per-token routing), applies the
[precision] policy, then drives a HF Trainer with Naylis-specific
callbacks: token-speed/ETA telemetry, Gumbel noise annealing on the
router, hourly checkpoints uploaded to the Hub for crash resume, and
post-run training curves.

Protocol guards, all fail-loud: tokens_per_step must equal world_size x
batch_size x grad_accum x seq_len (the A/B invariant — the classic
multi-GPU footgun), liger must be armed explicitly (it is a numeric
fork), AMP is bf16-only (no GradScaler, no fp16), and use_kernel=true
requires a kernel backend (XLA/TPU fused path or the experimental
Triton read on CUDA; eager SDPA is the default).

Rank 0 owns the Hub uploads (final model, curves); other ranks exit
after train().

Usage: python pretrain.py --config naylisMoM.toml --hf_token <token>
"""
import argparse
import glob
import json
import os
import random
import threading
import time
import tomllib
from collections import defaultdict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if os.environ.get("PJRT_DEVICE", "").upper() != "TPU":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_callback import PrinterCallback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import HAS_LIGER, NaylisLlamaForCausalLM
import mom_kernels
from naylis_precision import apply_precision

ALLOWED_VARIANTS = frozenset({"vanilla", "naylis"})


def _warmup_kwargs(ratio: float) -> dict:
    """TrainingArguments warmup kwarg across transformers versions."""
    import inspect
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": float(ratio)}


def is_xla(device) -> bool:
    """True when ``device`` is an XLA (TPU) device."""
    try:
        return torch.device(device).type == "xla"
    except Exception:
        return str(device).lower().startswith("xla")


def resolve_device():
    """Pick the training device: CUDA (torchrun-aware) -> XLA/TPU -> CPU."""
    if torch.cuda.is_available():
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank:
            torch.cuda.set_device(int(local_rank))
            print(f"[Device] cuda:{local_rank} (torchrun rank device)")
            return f"cuda:{local_rank}"
        return "cuda"
    try:
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        kind = None
        try:
            kind = xr.device_type()
        except Exception:
            pass
        if kind and str(kind).upper() != "TPU":
            raise RuntimeError(f"PJRT device is {kind!r}, not TPU")
        device = xm.xla_device()
        print(f"[Device] XLA {device} (runtime={kind or 'unknown'})")
        return device
    except ImportError:
        print("[Device] No CUDA / torch_xla, using CPU")
        return "cpu"


def get_world_size(device) -> int:
    """World size: XLA runtime, WORLD_SIZE env, or torch.distributed."""
    if is_xla(device):
        try:
            import torch_xla.runtime as xr

            return xr.world_size()
        except Exception:
            return 1
    ws_env = os.environ.get("WORLD_SIZE")
    if ws_env:
        try:
            return max(int(ws_env), 1)
        except ValueError:
            pass
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def resolve_amp_dtype(device, requested: str):
    """AMP policy: bf16 only. Returns the (bf16, fp16) pair
    TrainingArguments expects; fp16/none raise — no GradScaler in this
    codebase, the target hardware is bf16-native (A100/H100/TPU)."""
    requested = (requested or "bf16").lower().strip()
    if requested in ("fp16", "none"):
        raise ValueError(
            f"[train] amp_dtype={requested!r} is not supported: fp16 and "
            f"GradScaler are absent from this repo (bf16-native target). "
            f"Remove the key or set amp_dtype = \"bf16\"."
        )
    if requested not in ("auto", "bf16"):
        raise ValueError(f"[train] amp_dtype must be auto|bf16, got {requested!r}")
    return True, False


def _check_tokens_per_step(tokens_per_step, target, *, world_size, batch, accum, seq):
    """Guard the tokens/step protocol invariant (default 32 768).

    Multi-GPU multiplies the GLOBAL batch by the process count; a mismatch
    raises with the rescaling recipe instead of letting an A/B arm quietly
    train at a different batch.
    """
    if not target:
        print("[Init] WARNING: tokens_per_step guard DISABLED ([train] tokens_per_step = 0) "
              "— the 32 768 tok/step protocol invariant is no longer verified.")
        return
    if tokens_per_step != int(target):
        raise ValueError(
            f"tokens/step = world_size x batch_size x grad_accum x seq_len "
            f"= {world_size} x {batch} x {accum} x {seq} = {tokens_per_step}, "
            f"expected {int(target)} (A/B protocol invariant). Multi-GPU footgun: "
            f"NGPU=N multiplies the GLOBAL batch by N — rescale the toml "
            f"(8 GPU -> batch_size=4, grad_accum=1 ; 2 GPU -> grad_accum=1), or set "
            f"[train] tokens_per_step = {tokens_per_step} deliberately."
        )
    print(f"[Init] tokens/step = {tokens_per_step} == {int(target)} (protocol guard OK)")


def apply_tf32(enabled: bool, device) -> None:
    """Enable TF32 for fp32 matmuls on CUDA (no-op elsewhere)."""
    if not enabled:
        return
    if torch.cuda.is_available() and not is_xla(device) and device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
            print("[Init] TF32 enabled for fp32 matmuls (torch.backends + 'high')")
        except Exception:
            pass


def get_vocab_size(tokenizer_name: str) -> int:
    """Vocabulary size of the experiment tokenizer."""
    return len(AutoTokenizer.from_pretrained(tokenizer_name))


def build_llama_config(arch: dict, vocab_size: int, variant: str, naylis: dict | None) -> LlamaConfig:
    """LlamaConfig from the [arch] table (+ the MoM keys when naylis)."""
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=arch["d_model"],
        intermediate_size=arch["intermediate_size"],
        num_hidden_layers=arch["n_layers"],
        num_attention_heads=arch["n_heads"],
        num_key_value_heads=arch["n_heads"],
        max_position_embeddings=arch["seq_len"],
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        attention_bias=False,
        tie_word_embeddings=True,
        attn_implementation="sdpa",
        use_cache=False,
    )
    config.variant = variant
    config.tokenizer_name = arch["tokenizer_name"]
    if naylis is not None:
        config.mem_size = naylis["mem_size"]
        config.mom_block_size = naylis["mom_block_size"]
        config.mom_top_blocks = naylis["mom_top_blocks"]
        config.mom_router_rank = naylis["mom_router_rank"]
        config.mom_aux_weight = naylis["mom_aux_weight"]
        config.use_kernel = bool(naylis.get("use_kernel", False))
        config.mom_grad_ckpt = bool(naylis.get("mom_grad_ckpt", False))
    return config


def build_model(variant: str, arch: dict, naylis: dict, vocab_size: int, device,
                use_liger: bool = False):
    """Build the variant on ``device`` and return (model, config).

    Liger (CUDA-only) is applied here when explicitly armed — fused
    RMSNorm/SwiGLU/RoPE + fused-linear cross-entropy, a NUMERIC fork vs
    the plain path that needs its own dedicated A/B.
    """
    config = build_llama_config(arch, vocab_size, variant, naylis)
    device_type = "xla" if is_xla(device) else torch.device(device).type
    use_kernel = bool((naylis or {}).get("use_kernel", False))
    if use_liger:
        if not HAS_LIGER:
            raise RuntimeError(
                "[train] use_liger=true but liger-kernel is not installed "
                "(pip install liger-kernel)."
            )
        if device_type != "cuda":
            raise RuntimeError(
                f"[train] use_liger=true on device={device_type!r} — the liger "
                f"path is CUDA-only in this harness."
            )
        from liger_kernel.transformers import apply_liger_kernel_to_llama

        apply_liger_kernel_to_llama()
        print(
            "[Init] Liger ON (apply_liger_kernel_to_llama: fused RMSNorm/SwiGLU/RoPE "
            "+ LigerFusedLinearCrossEntropyLoss — no (B,S,V) logits "
            "materialized). NUMERIC fork vs OFF: dedicated A/B."
        )
    else:
        print(
            "[Init] Liger OFF — standard RMSNorm + CE (logits (B,S,V) materialized)."
        )

    if use_kernel and variant != "naylis":
        print("[Init] note: [naylis] use_kernel=true ignored — vanilla variant has no MoM layer")

    if variant == "vanilla":
        model = LlamaForCausalLM(config)
    else:
        model = NaylisLlamaForCausalLM(
            config,
            mem_size=naylis["mem_size"],
            mom_block_size=naylis["mom_block_size"],
            mom_top_blocks=naylis["mom_top_blocks"],
            mom_router_rank=naylis["mom_router_rank"],
            mom_aux_weight=naylis["mom_aux_weight"],
            use_liger=use_liger,
            use_kernel=use_kernel,
            mom_grad_ckpt=bool((naylis or {}).get("mom_grad_ckpt", False)),
        )

    return model.to(device), config


class BinDataset(torch.utils.data.Dataset):
    """Memory-mapped uint16 token bin cut into seq_len rows.

    Row i spans tokens [i*S, (i+1)*S); input_ids and labels carry the
    same S tokens (the shift happens inside the loss). ``__getitems__``
    serves whole index batches so the dataloader gathers once per step.
    """

    def __init__(self, file_path, seq_len, total_tokens_to_read=None):
        self.seq_len = seq_len
        self.tokens = np.memmap(file_path, dtype=np.uint16, mode="r")
        if total_tokens_to_read:
            self.tokens = self.tokens[:total_tokens_to_read]
        self.n_examples = len(self.tokens) // seq_len

    def __len__(self):
        return self.n_examples

    def __getitems__(self, indices):
        starts = np.asarray(indices, dtype=np.int64) * self.seq_len
        rows = starts[:, None] + np.arange(self.seq_len, dtype=np.int64)
        input_ids = torch.from_numpy(self.tokens[rows].astype(np.int64))
        return {"input_ids": input_ids, "labels": input_ids}

    def __getitem__(self, idx):
        batch = self.__getitems__([idx])
        return {"input_ids": batch["input_ids"][0], "labels": batch["labels"][0]}


def bin_collate(batch):
    """Stack pre-collated rows (or pass a __getitems__ dict through)."""
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], dict):
        return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
    raise TypeError(f"bin_collate: unexpected batch type {type(batch)!r}")


class SpeedCallback(TrainerCallback):
    """Training telemetry on every log line: tok/s, progress, ETA.

    Also surfaces the MoM diagnostics the NaylisTrainer injects into the
    logs: mom_scale avg/max, M_blocks norm, router entropy and dead-block
    fraction.
    """

    def __init__(self, tokens_per_step, total_tokens):
        self.tokens_per_step = tokens_per_step
        self.total_tokens = total_tokens
        self.t0 = None
        self.step0 = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        self.step0 = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs or self.t0 is None:
            return
        elapsed = time.time() - self.t0
        if elapsed <= 0:
            return
        steps_this_session = max(state.global_step - self.step0, 0)
        tok_per_sec = (self.tokens_per_step * steps_this_session) / elapsed
        tokens_done = self.tokens_per_step * state.global_step
        remaining = max(self.total_tokens - tokens_done, 0)
        eta_min = (remaining / tok_per_sec) / 60 if tok_per_sec > 0 else float("inf")

        pieces = [f"loss={logs['loss']:.4f}"]
        if (grad_norm := logs.get("grad_norm")) is not None:
            pieces.append(f"grad_norm={grad_norm:.2f}")
        if (lr := logs.get("learning_rate")) is not None:
            pieces.append(f"lr={lr:.2e}")
        pieces.extend([
            f"tok/s={tok_per_sec:,.0f}",
            f"tokens={tokens_done/1e6:.0f}M/{self.total_tokens/1e6:.0f}M",
            f"ETA={eta_min:.0f}min",
        ])
        mom_avg = logs.get("mom_scale_mean")
        mom_max = logs.get("mom_scale_max")
        if mom_avg is not None and mom_max is not None:
            pieces.append(f"mom_scale(avg/max)={mom_avg:.4f}/{mom_max:.4f}")
        blocks_norm = logs.get("mom_blocks_norm")
        if blocks_norm is not None:
            pieces.append(f"M_blocks={blocks_norm:.3f}")
        diag = []
        if "mom_entropy_mean" in logs:
            diag.append(f"entropy(avg)={logs['mom_entropy_mean']:.3f}")
        if "mom_dead_frac_mean" in logs:
            diag.append(f"dead_blocks={logs['mom_dead_frac_mean']*100:.1f}%")
        if diag:
            pieces.append(", ".join(diag))
        tqdm.write(f"[Step {state.global_step}] " + " | ".join(pieces))


class NaylisTrainer(Trainer):
    """HF Trainer with the TE fp8 autocast and the MoM stat logging.

    compute_loss wraps super() in fp8_autocast when the [precision] plan
    resolved a TE recipe (torchao converts modules directly and needs no
    context manager); log() enriches every line with the router/bank
    stats collected from the live model.
    """

    def __init__(self, *args, te_fp8_recipe=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._te_fp8_recipe = te_fp8_recipe

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._te_fp8_recipe is not None:
            from transformer_engine.pytorch import fp8_autocast
            with fp8_autocast(enabled=True, fp8_recipe=self._te_fp8_recipe):
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)

    def log(self, logs, *args, **kwargs):
        logs.update(self._collect_mom_stats())
        super().log(logs, *args, **kwargs)

    def _collect_mom_stats(self):
        """One CPU vector of router entropy / dead frac / scales / bank norm."""
        entropies, dead_fracs, scales = [], [], []
        bank = None
        for module in self.model.modules():
            e = getattr(module, "last_entropy", None)
            d = getattr(module, "last_dead_frac", None)
            if e is not None:
                entropies.append(e.detach())
            if d is not None:
                dead_fracs.append(d.detach())
        for name, param in self.model.named_parameters():
            if name.endswith("mom_scale"):
                scales.append(param.detach().abs())
            elif name.endswith("M_blocks"):
                bank = param.detach()

        flat = entropies + dead_fracs + scales
        if bank is not None:
            flat = flat + [bank.norm()]
        if not flat:
            return {}
        vec = torch.stack([t.float().reshape(()) for t in flat]).cpu()

        i = 0

        def seg(n):
            nonlocal i
            out = vec[i:i + n]
            i += n
            return out

        stats = {}
        if entropies:
            e = seg(len(entropies))
            stats["mom_entropy_mean"] = e.mean().item()
            stats["mom_entropy_min"] = e.min().item()
        if dead_fracs:
            d = seg(len(dead_fracs))
            stats["mom_dead_frac_mean"] = d.mean().item()
            stats["mom_dead_frac_max"] = d.max().item()
        if scales:
            s = seg(len(scales))
            stats["mom_scale_mean"] = s.mean().item()
            stats["mom_scale_max"] = s.max().item()
        if bank is not None:
            stats["mom_blocks_norm"] = vec[i].item()
        return stats


def merge_log_histories(*histories):
    """Merge log histories by step (last writer wins) for resumed runs."""
    by_step = {}
    for history in histories:
        for entry in history or []:
            step = entry.get("step")
            if step is not None:
                by_step[step] = entry
    return [by_step[s] for s in sorted(by_step)]


def load_log_history_from_checkpoint(checkpoint_dir):
    """log_history from a checkpoint's trainer_state.json ([] if absent)."""
    if not checkpoint_dir:
        return []
    path = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            return json.load(f).get("log_history", [])
    except Exception as e:
        print(f"[Curves] Could not read {path}: {e}")
        return []


def generate_training_curves(log_history, output_dir, run_label):
    """One PNG per numeric metric of the (merged) log history."""
    os.makedirs(output_dir, exist_ok=True)
    series = defaultdict(list)
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        for key, value in entry.items():
            if key in ("step", "epoch"):
                continue
            if isinstance(value, (int, float)):
                series[key].append((step, value))

    for metric_name, points in series.items():
        if not points:
            continue
        points = sorted(points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.figure(figsize=(9, 5))
        plt.plot(xs, ys, linewidth=1.2)
        plt.xlabel("step")
        plt.ylabel(metric_name)
        plt.title(f"{run_label} - {metric_name}")
        plt.grid(alpha=0.3)
        out_path = os.path.join(output_dir, f"{metric_name}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f"[Curves]   -> {out_path}")


def push_curves_to_hub(curves_dir, model_name, seed, hf_token, repo_id, repo_type):
    """Upload the curves folder under <model>_seed<seed>/curves (best-effort)."""
    try:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        api.upload_folder(
            folder_path=curves_dir,
            path_in_repo=f"{model_name}_seed{seed}/curves",
            repo_id=repo_id,
            repo_type=repo_type,
            token=hf_token,
        )
        print(f"[Curves] Uploaded to {repo_id}/{model_name}_seed{seed}/curves")
    except Exception as e:
        print(f"[Curves] Warning: failed to upload curves: {e}")


class NoiseAnnealCallback(TrainerCallback):
    """Disables the router Gumbel noise at cutoff_frac of the run.

    Exploration noise on the router logits anneals to 0 once global_step
    reaches cutoff_frac * max_steps (one-shot, then inert).
    """

    def __init__(self, model, cutoff_frac):
        self.model = model
        self.cutoff_frac = cutoff_frac
        self._disabled = False

    def on_step_end(self, args, state, control, **kwargs):
        if self._disabled or state.max_steps <= 0:
            return control
        if state.global_step >= self.cutoff_frac * state.max_steps:
            for module in self.model.modules():
                if hasattr(module, "noise_scale"):
                    module.noise_scale = 0.0
            self._disabled = True
            print(
                f"\n[NoiseAnneal] Gumbel noise disabled at step {state.global_step} "
                f"(cutoff={self.cutoff_frac:.0%} of {state.max_steps} steps)"
            )
        return control


class HourlyCheckpointCallback(TrainerCallback):
    """Save + upload a checkpoint every interval_sec of wall clock.

    Flags control.should_save on the first step after each interval, then
    rank 0 uploads the latest checkpoint-* to the Hub — optionally in a
    daemon thread; an upload still running skips that cycle (the local
    checkpoint is safe, the next cycle catches up).
    """

    def __init__(self, model_name, seed, hf_token, repo_id, repo_type, interval_sec,
                 async_upload=False):
        self.model_name = model_name
        self.seed = seed
        self.hf_token = hf_token
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.interval_sec = interval_sec
        self.last_save = None
        self.async_upload = async_upload
        self._upload_thread = None

    def _upload(self, latest, tag):
        try:
            api = HfApi(token=self.hf_token)
            api.create_repo(repo_id=self.repo_id, repo_type=self.repo_type, exist_ok=True)
            api.upload_folder(
                folder_path=latest,
                path_in_repo=f"{self.model_name}_seed{self.seed}/checkpoint",
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                token=self.hf_token,
            )
            print(
                f"\n[Checkpoint] {tag} uploaded to "
                f"{self.repo_id}/{self.model_name}_seed{self.seed}/checkpoint"
            )
        except Exception as e:
            print(f"\n[Checkpoint] Warning: upload failed: {e}")

    def _launch_upload(self, latest, tag):
        if not self.async_upload:
            self._upload(latest, tag)
            return
        if self._upload_thread is not None and self._upload_thread.is_alive():
            print(f"\n[Checkpoint] previous upload still running — skipping this one "
                  f"(local checkpoint {latest} is safe; it uploads on the next cycle)")
            return
        self._upload_thread = threading.Thread(
            target=self._upload, args=(latest, tag), daemon=True, name="ckpt-upload"
        )
        self._upload_thread.start()

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_save = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.last_save >= self.interval_sec:
            control.should_save = True
            self.last_save = time.time()
        return control

    def on_save(self, args, state, control, **kwargs):
        if os.environ.get("RANK", "0") not in ("0", ""):
            return control
        ckpts = sorted(
            glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
            key=lambda p: int(p.split("-")[-1]),
        )
        if not ckpts:
            return control
        latest = ckpts[-1]
        self._launch_upload(latest, f"step {state.global_step}")
        return control


def try_download_checkpoint(model_name, seed, hf_token, repo_id, repo_type):
    """Resume path from the Hub's <model>_seed<seed>/checkpoint (or None)."""
    key = f"{model_name}_seed{seed}"
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            token=hf_token,
            allow_patterns=f"{key}/checkpoint/*",
        )
        resume_path = os.path.join(local_dir, key, "checkpoint")
        if os.path.isdir(resume_path) and os.listdir(resume_path):
            print(f"[Resume] Found checkpoint: {resume_path}")
            return resume_path
    except Exception as e:
        print(f"[Resume] No usable checkpoint on the Hub ({e}), starting from scratch.")
    return None


def load_experiment(config_path: str):
    """Load + validate a TOML experiment; returns (cfg, arch, naylis, name, variant)."""
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    arch = cfg["arch"]
    missing = [k for k in ("model_name", "variant", "intermediate_size", "tokenizer_name") if k not in arch]
    if missing:
        raise ValueError(f"config [arch] missing required keys: {missing}")

    model_name = arch["model_name"]
    variant = arch["variant"]
    if not model_name or not isinstance(model_name, str):
        raise ValueError("arch.model_name must be a non-empty string")
    if variant not in ALLOWED_VARIANTS:
        raise ValueError(f"arch.variant must be one of {sorted(ALLOWED_VARIANTS)}, got {variant!r}")

    naylis = cfg.get("naylis")
    if variant == "naylis" and naylis is None:
        raise ValueError("variant='naylis' requires a [naylis] section")

    return cfg, arch, naylis, model_name, variant


def main() -> None:
    """Wire everything (config, data, model, precision, callbacks) and train."""
    parser = argparse.ArgumentParser(description="Pretrain a Naylis ablation run")
    parser.add_argument("--config", type=str, required=True, help="TOML experiment file (vanilla.toml, vanilla_thin.toml, naylisMoM.toml)")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face write token (overrides HF_TOKEN env)")
    args = parser.parse_args()

    cfg, arch, naylis, model_name, variant = load_experiment(args.config)
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    hub_cfg = cfg["hub"]

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing Hugging Face token. Pass --hf_token or set HF_TOKEN.")

    seed = train_cfg["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = resolve_device()
    world_size = get_world_size(device)

    if "use_liger" in train_cfg:
        use_liger = bool(train_cfg["use_liger"])
        liger_origin = "[train] use_liger explicit"
    else:
        use_liger = False
        liger_origin = "key absent -> default false"
    print(f"[Init] use_liger={'true' if use_liger else 'false'} ({liger_origin})")
    if not use_liger and HAS_LIGER:
        print("[Init] NOTE: liger-kernel IS importable but use_liger is false/absent — "
              "it must be armed explicitly (numeric fork).")

    if variant == "naylis" and naylis.get("use_kernel", False):
        dev_type = "xla" if is_xla(device) else torch.device(device).type
        mom_kernels.preflight(dev_type)
        print(f"[Init] use_kernel=true on {dev_type} — kernel MoM read armed "
              f"(xla fused-shape on TPU / triton flash on CUDA, experimental)")

    vocab_size = get_vocab_size(arch["tokenizer_name"])
    print(f"[Tokenizer] {arch['tokenizer_name']} -> vocab_size={vocab_size}")

    print(f"Downloading {data_cfg['hf_filename']}...")
    data_path = hf_hub_download(
        repo_id=data_cfg["hf_repo_id"],
        filename=data_cfg["hf_filename"],
        repo_type=data_cfg["hf_repo_type"],
        token=hf_token,
    )
    train_dataset = BinDataset(data_path, arch["seq_len"], total_tokens_to_read=train_cfg["total_tokens"])

    model, llama_config = build_model(variant, arch, naylis, vocab_size, device,
                                      use_liger=use_liger)

    if variant == "naylis":
        model.mom_aux_normalize = bool(train_cfg.get("mom_aux_normalize", False))
        model.mom_aux_accum_steps = int(train_cfg["grad_accum"])
        w_aux = naylis["mom_aux_weight"]
        accum = int(train_cfg["grad_accum"])
        if model.mom_aux_normalize:
            print(f"[Init] mom_aux_normalize=true: aux/{accum} per micro-batch — effective "
                  f"weight {w_aux} INVARIANT to the GPU split.")
        else:
            print(f"[Init] mom_aux_normalize=false (default, reference baseline): aux added "
                  f"per micro-batch while CE is cycle-averaged -> effective weight "
                  f"~ {w_aux} x grad_accum = {w_aux * accum} [expected under DDP grad "
                  f"averaging; UNTESTED]. Flip mom_aux_normalize=true for split "
                  f"invariance.")

    prec_cfg = cfg.get("precision") or {}
    model, prec_info = apply_precision(model, prec_cfg, device=device)

    # TE fp8 + activation checkpointing: the recompute pass runs OUTSIDE
    # the fp8_autocast context (use_reentrant=False restores torch autocast
    # state, not custom contexts), so te.Linear modules would recompute in
    # high precision — a silent forward/recompute numeric fork. Fail loud:
    # torchao converts modules directly (consistent recompute), or drop
    # checkpointing for the TE fp8 arm.
    ckpt_armed = bool((naylis or {}).get("mom_grad_ckpt", False)) or bool(
        train_cfg.get("gradient_checkpointing", False))
    if prec_info.get("fp8") and prec_info.get("backend") == "te" and ckpt_armed:
        raise RuntimeError(
            "[precision] backend='te' (fp8) combined with activation "
            "checkpointing (mom_grad_ckpt / gradient_checkpointing): the "
            "recompute pass runs outside fp8_autocast, so te.Linear would "
            "recompute in high precision — a silent numeric fork. Use "
            "backend='torchao' (module-level conversion, consistent "
            "recompute) or disable checkpointing for this arm."
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Run: {model_name} | variant={variant} | Params: {n_params/1e6:.1f}M | "
        f"intermediate_size={llama_config.intermediate_size} | world_size={world_size}"
    )

    tokens_per_step = world_size * train_cfg["batch_size"] * train_cfg["grad_accum"] * arch["seq_len"]
    _check_tokens_per_step(
        tokens_per_step,
        train_cfg.get("tokens_per_step", 32_768),
        world_size=world_size,
        batch=train_cfg["batch_size"],
        accum=train_cfg["grad_accum"],
        seq=arch["seq_len"],
    )
    speed_cb = SpeedCallback(
        tokens_per_step=tokens_per_step,
        total_tokens=train_cfg["total_tokens"],
    )

    output_dir = f"./ablation_{model_name}"
    resume_path = try_download_checkpoint(model_name, seed, hf_token, hub_cfg["repo_id"], hub_cfg["repo_type"])
    on_tpu = is_xla(device)
    on_cuda = (torch.device(device).type == "cuda") if device != "cpu" and not on_tpu else False


    use_bf16, use_fp16 = resolve_amp_dtype(device, train_cfg.get("amp_dtype", "bf16"))
    print("[Init] AMP dtype: bf16 (fp16 unsupported; fp8 is handled by [precision])")

    optim_cfg = train_cfg.get("optimizer", "auto")
    if optim_cfg == "auto":
        optim_cfg = "adamw_torch_fused" if on_cuda else "adamw_torch"
    print(f"[Init] optimizer: {optim_cfg}")

    apply_tf32(bool(train_cfg.get("tf32", True)), device)

    torch_compile = bool(train_cfg.get("torch_compile", False))
    if torch_compile:
        print("[Init] torch_compile=true — A/B it separately first "
              "(the XLA-side smoke test measured ~5e-7 drift)")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["grad_accum"],
        num_train_epochs=1,
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        **_warmup_kwargs(train_cfg["warmup_ratio"]),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=use_bf16,
        fp16=use_fp16,
        optim=optim_cfg,
        torch_compile=torch_compile,
        report_to="none",
        dataloader_num_workers=0 if on_tpu else 4,
        dataloader_drop_last=on_tpu,
        dataloader_pin_memory=on_cuda,
        seed=seed,
        data_seed=seed,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    checkpoint_cb = HourlyCheckpointCallback(
        model_name, seed, hf_token, hub_cfg["repo_id"], hub_cfg["repo_type"],
        interval_sec=train_cfg["checkpoint_interval_sec"],
        async_upload=bool(cfg.get("hub", {}).get("async_upload", False)),
    )
    callbacks = [speed_cb, checkpoint_cb]
    if variant == "naylis":
        callbacks.append(NoiseAnnealCallback(model, cutoff_frac=train_cfg["noise_anneal_cutoff_frac"]))

    trainer = NaylisTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=bin_collate,
        callbacks=callbacks,
        te_fp8_recipe=prec_info.get("te_recipe"),
    )
    trainer.remove_callback(PrinterCallback)

    trainer.train(resume_from_checkpoint=resume_path)

    is_rank0 = os.environ.get("RANK", "0") in ("0", "")
    if is_rank0:
        trainer.save_model(f"./model_{model_name}")

        old_history = load_log_history_from_checkpoint(resume_path)
        merged_history = merge_log_histories(old_history, trainer.state.log_history)
        curves_dir = f"./curves_{model_name}"
        print(f"\n[Curves] Generating plots ({len(merged_history)} merged log points)...")
        generate_training_curves(merged_history, curves_dir, run_label=model_name)
        push_curves_to_hub(curves_dir, model_name, seed, hf_token, hub_cfg["repo_id"], hub_cfg["repo_type"])

        try:
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hub_cfg["repo_id"], repo_type=hub_cfg["repo_type"], exist_ok=True)
            api.upload_folder(
                folder_path=f"./model_{model_name}",
                path_in_repo=model_name,
                repo_id=hub_cfg["repo_id"],
                repo_type=hub_cfg["repo_type"],
                token=hf_token,
            )
            print(f"Uploaded: {hub_cfg['repo_id']}/{model_name}")
        except Exception as e:
            print(f"[Warning] Upload failed: {e}")
    else:
        print(f"[Rank {os.environ.get('RANK')}] training done — rank 0 handles save/curves/upload.")


if __name__ == "__main__":
    main()

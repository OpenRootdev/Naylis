# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""Naylis MoM kernels: public API and backend selection (causal edition).

Backend selection is fail-loud by design: XLA/TPU resolves to the
fused-shape attention path, CUDA to the Triton flash read (bf16
activations, head_dim <= 128), anything else raises — ``use_kernel=true``
means "accelerated", and a silent eager fallback would corrupt bench data.
``mom_read`` dispatches the memory read in structured form (top_idx +
top_gates, never a dense (B, S, L) bias across the API boundary);
``mom_kernels.reference`` holds the math mirror the parity suite checks
against.
"""
import torch

try:
    from . import triton_gpu
    from .triton_gpu import HAS_TRITON
except Exception:
    triton_gpu = None
    HAS_TRITON = False

from . import reference, xla_tpu

__all__ = ["reference", "xla_tpu", "triton_gpu", "preflight", "resolve_mode",
           "mom_read", "announce"]

_ANNOUNCED = {"backend": None}


def _device_type(device) -> str:
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":")[0].lower()


def preflight(device_type: str) -> None:
    """Fail fast at startup if the requested device has no kernel backend."""
    if device_type == "xla":
        return
    if device_type == "cuda":
        if not HAS_TRITON:
            raise RuntimeError(
                "use_kernel=true on CUDA requires the triton package "
                "(pip install triton)."
            )
        return
    raise RuntimeError(
        f"use_kernel=true requires an XLA/TPU device (fused-shape path) or "
        f"a CUDA device (Triton read); this process resolved to "
        f"{device_type!r}. Set use_kernel=false (eager SDPA, the parity "
        f"reference) or run on the accelerator."
    )


def resolve_mode(device, dtype, *, head_dim=None) -> str:
    """Pick the kernel backend ("xla" | "triton") and check its contract.

    The Triton path requires bf16 activations (the repo AMP dtype; fp16 is
    not supported anywhere in this codebase) and head_dim <= 128, power of
    two; violations raise instead of falling back to eager.
    """
    device_type = _device_type(device)
    if device_type == "xla":
        return "xla"
    if device_type == "cuda":
        if not HAS_TRITON:
            raise RuntimeError("use_kernel=true: triton is not installed on this CUDA box.")
        if dtype != torch.bfloat16:
            raise RuntimeError(
                f"use_kernel=true (triton) requires bf16 activations "
                f"(the repo AMP dtype; fp16 is unsupported); got {dtype}."
            )
        if head_dim is not None:
            if head_dim > 128 or head_dim & (head_dim - 1) != 0:
                raise RuntimeError(
                    f"use_kernel=true (triton): head_dim {head_dim} must be "
                    f"a power of two <= 128."
                )
        return "triton"
    raise RuntimeError(
        f"use_kernel=true: no kernel backend for device {device_type!r} "
        f"(XLA/TPU -> fused-shape path, CUDA -> Triton read)."
    )


def mom_read(q_sh, k, v, top_idx, top_gates, *, num_blocks, block_size, mode: str):
    """MoM memory read with the per-token gate bias, on the selected backend.

    Takes the routing in structured form — (top_idx, top_gates) per token —
    so no backend needs the dense (B, S, L) bias materialised by the caller.
    Returns (B, S, H, hd), query layout.
    """
    if mode == "xla":
        return xla_tpu.mom_read(q_sh, k, v, top_idx, top_gates,
                                num_blocks=num_blocks, block_size=block_size)
    if mode == "triton":
        if triton_gpu is None:
            raise RuntimeError("triton backend unavailable")
        return triton_gpu.mom_read(q_sh, k, v, top_idx, top_gates,
                                   num_blocks=num_blocks, block_size=block_size)
    raise RuntimeError(f"unknown kernel mode {mode!r}")


def announce(backend: str) -> None:
    """Print the backend banner once per process (on first forward)."""
    if _ANNOUNCED["backend"] != backend:
        _ANNOUNCED["backend"] = backend
        if backend == "xla":
            print("[MoM] use_kernel=true -> backend 'xla': chunked attention, "
                  "fp32 softmax, per-chunk gate bias, bounded score memory "
                  "(NAYLIS_XLA_ATTN=online for the flash-style variant).")
        elif backend == "triton":
            print("[MoM] use_kernel=true -> backend 'triton': flash read with "
                  "in-register gate bias (no dense mask), custom backward. "
                  "Experimental — keep it out of A/B arms until "
                  "tests/test_parity.py GPU sections pass on the box.")

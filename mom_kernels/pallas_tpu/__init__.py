# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""Pallas kernels for the causal MoM read: plan/runner + CPU simulator.

``mom_pallas`` holds the kernels (plan/runner split), ``simulate`` runs
the exact same kernel code on CPU as a numpy simulator (math validation
without a TPU), ``bench_tpu`` is the on-box parity + timing bench:

    python mom_kernels/pallas_tpu/bench_tpu.py --simulate --shape tiny
    python mom_kernels/pallas_tpu/bench_tpu.py --shape 300m --dtype bf16
"""
from . import mom_pallas, simulate

__all__ = ["mom_pallas", "simulate"]

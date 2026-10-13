#!/usr/bin/env bash
# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
#
# Launcher for a big-GPU ablation run. Fails fast: both probes (causal MoM
# parity/causality + Triton read parity at run shapes, precision stack)
# must be green before the run starts. The MoM read runs eager SDPA by
# default — the path that trained the causal reference run; use_kernel
# arms the Triton flash read on CUDA (experimental) or the XLA fused path
# on TPU.
# Usage: ./run_biggpu.sh [config.toml] [hf_token]
#   NGPU=8 ./run_biggpu.sh naylisMoM.toml   -> torchrun, 8 processes
set -euo pipefail

CONFIG="${1:-naylisMoM.toml}"
NGPU="${NGPU:-1}"
HF_TOKEN="${HF_TOKEN:-${2:-}}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "${NGPU}" -gt 1 ]; then
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
fi

python mom_kernels/probe.py || { echo "MoM preflight KO — do NOT start the run."; exit 1; }
python probe_fp8.py        || { echo "precision probe KO — see sections above."; exit 1; }

# Optional, TPU boxes only: math validation of the Pallas kernels
#   python mom_kernels/pallas_tpu/bench_tpu.py --simulate --shape tiny
#   python mom_kernels/pallas_tpu/bench_tpu.py --shape 300m --dtype bf16

if [ -z "${HF_TOKEN}" ]; then
  echo "HF_TOKEN missing: export HF_TOKEN=... (or pass it as the 2nd argument)."
  exit 1
fi

if [ "${NGPU}" -eq 1 ]; then
  exec python pretrain.py --config "${CONFIG}" --hf_token "${HF_TOKEN}"
else
  exec torchrun --nproc_per_node="${NGPU}" pretrain.py \
    --config "${CONFIG}" --hf_token "${HF_TOKEN}"
fi

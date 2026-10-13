# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""CPU simulator for Pallas kernels: run the exact kernel code, no TPU.

Executes a Pallas plan (kernel, grid, in/out specs) with numpy blocks on
the host: for every program id in the grid, it slices the input/output
blocks the BlockSpecs describe, calls the kernel with ``_Ref`` views and
writes the results back. ``pl.program_id`` is monkey-patched per program
so kernels read the same API as on device.

This validates KERNEL MATHS only — performance numbers from the simulator
would be meaningless (host numpy, full-tensor copies per program); the
project rule is measured > expected, so timings come from bench_tpu.py on
a real TPU.
"""
import itertools

import jax.numpy as jnp
import numpy as np


class _Ref:
    """Block view handed to a kernel: reads slice through, writes copy back."""

    __slots__ = ("block",)

    def __init__(self, block):
        self.block = block

    def __getitem__(self, idx):
        return self.block[idx]

    def __setitem__(self, idx, value):
        arr = np.array(self.block, copy=True)
        arr[idx] = np.asarray(value)
        self.block = jnp.asarray(arr)

    @property
    def shape(self):
        return self.block.shape

    @property
    def dtype(self):
        return self.block.dtype


def _slice_of(tensor, block_shape, base):
    """Slice ``tensor`` at ``base`` with the given block shape."""
    sl = tuple(slice(int(b), int(b) + int(bs))
               for b, bs in zip(base, block_shape))
    return tensor[sl]


def simulate(kernel, grid, in_specs, out_specs, inputs, out_shapes_dtypes):
    """Run a Pallas plan on CPU, block by block, over the full grid.

    Returns the list of output arrays (jnp) after every program has written
    its block — the same result pallas_call would produce on device.
    """
    import jax.experimental.pallas as pl

    if not isinstance(out_specs, (tuple, list)):
        out_specs = (out_specs,)

    def _shape_dtype(s):
        if hasattr(s, "shape") and hasattr(s, "dtype"):
            return s.shape, s.dtype
        return s

    outs = [jnp.full(shape, np.nan, dtype)
            for shape, dtype in (_shape_dtype(s) for s in out_shapes_dtypes)]

    real_program_id = pl.program_id
    try:
        for pids in itertools.product(*(range(g) for g in grid)):
            in_blocks = [
                _Ref(_slice_of(x, spec.block_shape, spec.index_map(*pids)))
                for x, spec in zip(inputs, in_specs)
            ]
            out_blocks = [
                _Ref(_slice_of(o, spec.block_shape, spec.index_map(*pids)))
                for o, spec in zip(outs, out_specs)
            ]

            def fake_program_id(axis, _pids=pids):
                return jnp.int32(_pids[axis])

            pl.program_id = fake_program_id
            kernel(*in_blocks, *out_blocks)
            pl.program_id = real_program_id

            for o_idx, ref in enumerate(out_blocks):
                spec = out_specs[o_idx]
                sl = tuple(slice(int(b), int(b) + int(bs))
                           for b, bs in zip(spec.index_map(*pids), spec.block_shape))
                cur = np.array(outs[o_idx], copy=True)
                cur[sl] = np.asarray(ref.block)
                outs[o_idx] = jnp.asarray(cur)
    finally:
        pl.program_id = real_program_id
    return outs

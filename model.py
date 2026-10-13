# Copyright (c) 2026 Silyan Larak
# SPDX-License-Identifier: MIT
"""Naylis: a Llama backbone extended with a Mixture-of-Memory (MoM) layer.

The MoM sublayer is the single architectural delta against the vanilla
Llama baselines of the ablation (vanilla.toml = compute-matched
dense control, vanilla_thin.toml = trunk-matched dense control — the
MoM arm's exact backbone with the memory removed). Every decoder block
keeps stock causal
self-attention and MLP and adds a memory read between the two,
residual-wired:

    x = x + attn(input_layernorm(x))           # stock Llama self-attention
    x = x + mom(mom_norm(x))                   # <- the delta: read memory
    x = x + mlp(post_attention_layernorm(x))   # stock Llama MLP

MoMLayer routes a *learnable, model-level* memory bank ``M_blocks`` —
``num_blocks`` blocks of ``block_size * d_model`` each, owned once by
NaylisLlamaModel and shared by all layers — instead of key/value tensors
derived from the input sequence. A low-rank router scores every block
from the causal prefix mean of the hidden states (per token), the top
``top_blocks`` blocks per token are read, and the query stream attends
over those slots with the router gates added as a bias on the attention
logits.

The router is CAUSAL: h_t = mean(x_0..x_t), so the blocks serving
position t depend only on positions 0..t — the same context the model
has when predicting token t+1. The original ablation router averaged the
full sequence, which leaked future tokens into the routing decision at
train time; the leak ablation (same seed, same tokens) showed the causal
router matches then beats the leaky one at eval — those numbers live in
the README.

Read mechanism (per-token routing without a per-token gather): K/V
projections are computed once per layer over ALL bank slots and shared
across the batch; the attention reads all ``mem_size`` slots with a
per-token additive bias — log(gate) on the chosen blocks, a hard mask
elsewhere — so the softmax normalises over exactly the ``top_blocks *
block_size`` active slots of each token. The bias is differentiable end
to end (one-hot x log-gates), so the LM gradient reaches the router
through the gates. Cost: the MoM attention spans mem_size slots instead
of the gathered subset — ≈ +58% per-token MACs over the thin trunk and
≈ −7% under the wide control (per-layer MAC recount in METHODOLOGY.md) —
the price of strict per-token causality, measured in the reported MFU.

Numerics: training is bf16-only (fp32 master weights, no GradScaler);
fp8 (Transformer Engine or torchao) is an opt-in [precision] policy that
never touches the routing path. With ``use_kernel=true`` the memory read
dispatches to a kernel backend — XLA/TPU (fused-shape) or CUDA (Triton
flash read, experimental); eager SDPA is the default and the parity
reference, the path that trained the causal reference run.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, LlamaRMSNorm
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import mom_kernels
from mom_kernels import reference

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    from liger_kernel.transformers import LigerRMSNorm
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False


class MoMLayer(nn.Module):
    """Mixture-of-Memory attention over a shared learnable bank, causal.

    One routing decision PER TOKEN: the low-rank router scores the blocks
    from the prefix mean h_t = mean(x_0..x_t), the top ``top_blocks``
    blocks of each token are read with softmax gates, and the query
    stream attends to them with the gates injected as an additive bias on
    the attention logits. Router logits get Gumbel noise during training
    (exploration); the harness anneals ``noise_scale`` to 0 over the run.

    The output is gated by ``mom_scale`` — zero-initialised, LayerScale
    style, so the memory path starts silent and the residual stream is
    stable at init.

    Execution path, resolved once from the first forward and cached in
    ``_kernel_mode``: eager SDPA (default, the parity reference) or the
    backend ``use_kernel`` resolves to ("xla" on TPU, "triton" on CUDA).

    With ``mom_grad_ckpt`` the read is wrapped in a gradient-checkpoint
    region (recompute is exact — no RNG inside ``_read``); the ROUTER
    stays outside so the aux loss keeps its gradient toward the trunk.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mem_size: int,
        block_size: int,
        top_blocks: int,
        router_rank: int,
        use_kernel: bool = False,
        mom_grad_ckpt: bool = False,
    ):
        """Build the MoM layer.

        Args:
            d_model: model hidden size (must equal n_heads * head_dim).
            n_heads: heads of the shared query stream.
            mem_size: bank size in tokens; the bank holds
                mem_size // block_size blocks.
            block_size: tokens per memory block — the routing granularity.
            top_blocks: blocks selected per token (top-k routing).
            router_rank: bottleneck of the low-rank router
                (d_model -> router_rank -> num_blocks).
            use_kernel: dispatch the memory read to the kernel backend
                (XLA/TPU fused-shape path, or the experimental Triton
                flash read on CUDA; raises on CPU at resolve time).
            mom_grad_ckpt: checkpoint the memory read during training
                (exact recompute, ~3x less activation memory, ~25-35%
                slower). Inert on TPU, where XLA rematerialises on its own.
        """
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if mem_size % block_size != 0:
            raise ValueError(f"mem_size ({mem_size}) must be a multiple of block_size ({block_size})")
        if top_blocks > mem_size // block_size:
            raise ValueError(
                f"top_blocks ({top_blocks}) exceeds the number of memory blocks "
                f"({mem_size // block_size} = mem_size {mem_size} // block_size {block_size})"
            )

        self.num_blocks = mem_size // block_size
        self.block_size = block_size
        self.top_blocks = top_blocks
        self.mem_slots = mem_size

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.router_down = nn.Linear(d_model, router_rank, bias=False)
        self.router_up = nn.Linear(router_rank, self.num_blocks, bias=False)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.mom_scale = nn.Parameter(torch.zeros(1))

        self.last_router_probs = None
        self.last_top_idx = None
        self.last_entropy = None
        self.last_dead_frac = None

        self.noise_scale = 1.0

        self.use_kernel = use_kernel
        self.mom_grad_ckpt = mom_grad_ckpt
        self._kernel_mode = None

    def _resolve_kernel(self, x):
        mode = mom_kernels.resolve_mode(x.device, x.dtype, head_dim=self.head_dim)
        mom_kernels.announce(mode)
        return mode

    def _prefix_mean(self, x):
        """Causal prefix means h_t = mean(x_0..x_t), fp32 (cumsum keeps
        the running-sum order; a 1024-step fp16 sum would lose ~1e-3
        relative). In generation with a KV cache the running sum is kept
        O(1) per token (h = sum / (t + 1)) — values identical to training."""
        cs = torch.cumsum(x.float(), dim=1)
        denom = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=cs.dtype)
        return cs / denom.view(1, -1, 1)

    def _router_logits(self, x):
        """Prefix-mean causal router logits, per token: (B, S, num_blocks).

        fp32 accumulation for the cumsum, then a cast to the router weight
        dtype before the linears — a no-op under autocast training (fp32
        master weights) that also keeps pure-bf16/fp16 evaluation from
        crashing on a dtype mismatch.
        """
        h = self._prefix_mean(x).to(self.router_down.weight.dtype)
        return self.router_up(torch.tanh(self.router_down(h))).float()

    def forward(self, x, M_blocks):
        """Read from the shared memory bank.

        Args:
            x: hidden states, shape (B, S, d_model).
            M_blocks: the shared memory bank, shape (num_blocks,
                block_size * d_model) — owned by NaylisLlamaModel.

        Returns:
            Tensor of shape (B, S, d_model): mom_scale * o_proj(attention
            over the selected memory slots of each token).
        """
        B, S, D = x.shape

        # Router first, outside any checkpoint region: the aux loss keeps
        # its gradient toward the trunk, and q_proj consumes no RNG draws
        # so the Gumbel falls at the same point of the stream as the
        # reference runs.
        router_logits = self._router_logits(x)

        if self.training and self.noise_scale > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(router_logits) + 1e-9) + 1e-9)
            noisy_logits = router_logits + self.noise_scale * gumbel
        else:
            noisy_logits = router_logits

        router_probs = F.softmax(router_logits, dim=-1)
        top_values, top_idx = noisy_logits.topk(self.top_blocks, dim=-1)
        # Differentiable gate: topk keeps the gradient on the selected
        # VALUES, so the LM loss reaches the router through the additive
        # bias built from these gates (standard MoE mechanism).
        top_gates = F.softmax(top_values, dim=-1)

        if self.training:
            self.last_router_probs = router_probs.reshape(-1, self.num_blocks)
            self.last_top_idx = top_idx.reshape(-1, self.top_blocks)

        if self.use_kernel:
            if self._kernel_mode is None:
                self._kernel_mode = self._resolve_kernel(x)
            mode = self._kernel_mode
        else:
            mode = None

        if self.mom_grad_ckpt and self.training and torch.is_grad_enabled():
            out = torch.utils.checkpoint.checkpoint(
                self._read, x, top_idx, top_gates, M_blocks, mode,
                use_reentrant=False, preserve_rng_state=False)
        else:
            out = self._read(x, top_idx, top_gates, M_blocks, mode)
        return self.mom_scale * out

    def _read(self, x, top_idx, top_gates, M_blocks, mode=None):
        """Memory read: K/V over all slots, per-token gate bias, attention.

        Checkpointable as-is: no RNG inside. ``mode`` selects the kernel
        backend (None = eager SDPA); kernels receive the routing in
        structured form (top_idx, top_gates) and build the bias
        themselves. K/V are projected once per layer over the whole bank
        and shared across the batch.
        """
        B, S, D = x.shape
        q_sh = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)

        L = self.mem_slots
        slots = M_blocks.view(L, D)
        k_shared = self.k_proj(slots)
        v_shared = self.v_proj(slots)

        if mode is not None:
            out = mom_kernels.mom_read(
                q_sh, k_shared, v_shared, top_idx, top_gates,
                num_blocks=self.num_blocks, block_size=self.block_size, mode=mode)
        else:
            # Eager path (the parity reference): dense per-token bias +
            # SDPA. The mask value is finite so exp() underflows cleanly,
            # and every row always has its top blocks active.
            bias = reference.build_gate_bias(
                top_idx, top_gates, self.num_blocks, self.block_size, q_sh.dtype)
            q = q_sh.transpose(1, 2)
            k = k_shared.view(1, L, self.n_heads, self.head_dim) \
                            .transpose(1, 2).expand(B, -1, -1, -1)
            v = v_shared.view(1, L, self.n_heads, self.head_dim) \
                            .transpose(1, 2).expand(B, -1, -1, -1)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias.unsqueeze(1).to(q.dtype), is_causal=False)
            out = out.transpose(1, 2)
        return self.o_proj(out.reshape(B, S, D))


class NaylisDecoderLayer(LlamaDecoderLayer):
    """Stock Llama decoder block plus the MoM memory read, between
    attention and MLP, residual-wired.

    ``M_blocks`` is threaded through the stack by NaylisLlamaModel — the
    bank is model-level state shared across layers — hence the extra
    forward argument and the fail-loud guard when it is missing. With
    ``grad_ckpt`` the attention and MLP calls wrap in checkpoint regions
    (exact recompute, no RNG inside either).
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        mem_size: int,
        mom_block_size: int,
        mom_top_blocks: int,
        mom_router_rank: int,
        use_liger: bool,
        use_kernel: bool = False,
        mom_grad_ckpt: bool = False,
    ):
        super().__init__(config, layer_idx)

        norm_cls = LigerRMSNorm if use_liger else LlamaRMSNorm
        self.mom_norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.mom = MoMLayer(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            mem_size=mem_size,
            block_size=mom_block_size,
            top_blocks=mom_top_blocks,
            router_rank=mom_router_rank,
            use_kernel=use_kernel,
            mom_grad_ckpt=mom_grad_ckpt,
        )
        self._grad_ckpt = mom_grad_ckpt

    def _maybe_ckpt(self, fn, *tensors):
        """Checkpoint region when armed (exact recompute, no RNG inside)."""
        if self._grad_ckpt and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                fn, *tensors, use_reentrant=False, preserve_rng_state=False)
        return fn(*tensors)

    def forward(self, hidden_states, M_blocks, position_embeddings=None, **kwargs):
        if M_blocks is None:
            raise ValueError("M_blocks is required: the MoM layer cannot run without the shared memory bank")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self._maybe_ckpt(
            lambda hs: self.self_attn(
                hidden_states=hs, position_embeddings=position_embeddings, **kwargs),
            hidden_states)
        hidden_states = attn_outputs[0] if isinstance(attn_outputs, tuple) else attn_outputs
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mom_norm(hidden_states)
        hidden_states = self.mom(hidden_states, M_blocks)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._maybe_ckpt(self.mlp, hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class NaylisLlamaModel(LlamaModel):
    """Llama backbone that owns the shared memory bank ``M_blocks``.

    A single bank for the whole model: cross-layer sharing is deliberate —
    memory is model-level state, not per-layer state, and it keeps the MoM
    parameter budget independent of depth. The custom forward threads
    ``M_blocks`` and rotary position embeddings through the
    NaylisDecoderLayer stack.
    """

    def __init__(
        self,
        config: LlamaConfig,
        mem_size: int,
        mom_block_size: int,
        mom_top_blocks: int,
        mom_router_rank: int,
        use_liger: bool,
        use_kernel: bool = False,
        mom_grad_ckpt: bool = False,
    ):
        super().__init__(config)

        num_blocks = mem_size // mom_block_size
        self.M_blocks = nn.Parameter(torch.randn(num_blocks, mom_block_size * config.hidden_size) * 0.02)

        self.layers = nn.ModuleList([
            NaylisDecoderLayer(
                config,
                layer_idx=i,
                mem_size=mem_size,
                mom_block_size=mom_block_size,
                mom_top_blocks=mom_top_blocks,
                mom_router_rank=mom_router_rank,
                use_liger=use_liger,
                use_kernel=use_kernel,
                mom_grad_ckpt=mom_grad_ckpt,
            )
            for i in range(config.num_hidden_layers)
        ])

        # No post_init() here: the outer NaylisLlamaForCausalLM.post_init()
        # initialises every registered submodule (incl. these layers and
        # M_blocks' in-module Linears); calling it here as well just re-drew
        # every weight a second time at build.

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        seq_len = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                M_blocks=self.M_blocks,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                **kwargs
            )[0]

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class NaylisLlamaForCausalLM(LlamaForCausalLM):
    """Causal LM wrapper: LM head, optional Liger fused loss, MoM aux loss.

    Training-only plumbing: each MoMLayer stashes ``last_router_probs`` /
    ``last_top_idx`` (token scale, N = B*S) during its forward, and
    ``_collect_mom_aux_loss`` consumes them right after the LM loss,
    adding the load-balancing aux with weight ``mom_aux_weight``. With
    ``use_liger`` the head loss runs through Liger fused-linear
    cross-entropy (no logits materialisation).
    """

    def __init__(
        self,
        config: LlamaConfig,
        mem_size: int,
        mom_block_size: int,
        mom_top_blocks: int,
        mom_router_rank: int,
        mom_aux_weight: float,
        use_liger: bool = False,
        use_kernel: bool = False,
        mom_grad_ckpt: bool = False,
    ):
        super().__init__(config)

        if use_liger and not HAS_LIGER:
            raise RuntimeError("liger_kernel is not installed but use_liger=True was requested.")
        self.use_liger = use_liger

        self.model = NaylisLlamaModel(
            config,
            mem_size=mem_size,
            mom_block_size=mom_block_size,
            mom_top_blocks=mom_top_blocks,
            mom_router_rank=mom_router_rank,
            use_liger=use_liger,
            use_kernel=use_kernel,
            mom_grad_ckpt=mom_grad_ckpt,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        if self.use_liger:
            self.liger_lce = LigerFusedLinearCrossEntropyLoss(reduction="sum")

        self.mom_aux_weight = mom_aux_weight

        self.mom_aux_normalize = False
        self.mom_aux_accum_steps = None

        self.post_init()

    def _collect_mom_aux_loss(self, device):
        """MoM load-balancing aux loss, averaged over MoM layers.

        num_blocks * mean_b(frac_selected_b * mean_prob_b): pushes each
        block's mean router probability toward how often it is actually
        selected — balancing without clone tokens. The selection mask is
        built arithmetically (comparison + any, no scatter): identical
        values and gradients, and it compiles on every backend including
        XLA. Also records per-layer router entropy and dead-block fraction
        for the logs, then clears the stashed routing stats.
        """
        aux = torch.zeros((), device=device, dtype=torch.float32)
        count = 0

        for module in self.modules():
            if isinstance(module, MoMLayer):
                if module.last_router_probs is None or module.last_top_idx is None:
                    continue

                probs = module.last_router_probs.float()
                top_idx = module.last_top_idx

                blocks = torch.arange(module.num_blocks, device=top_idx.device)
                sel_mask = (top_idx.unsqueeze(2) == blocks.view(1, 1, -1)).any(1)
                frac_selected = sel_mask.to(probs.dtype).mean(dim=0)
                mean_prob = probs.mean(dim=0)

                aux = aux + module.num_blocks * (frac_selected * mean_prob).sum()

                entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
                module.last_entropy = entropy.detach()
                module.last_dead_frac = (~sel_mask.any(0)).to(probs.dtype).mean().detach()

                module.last_router_probs = None
                module.last_top_idx = None

                count += 1

        if count > 0:
            aux = aux / count

        return aux

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None,
                num_items_in_batch=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )
        hidden_states = outputs.last_hidden_state

        if labels is not None:
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.view(-1)

            if self.use_liger:
                loss = self.liger_lce(
                    self.lm_head.weight,
                    shift_hidden.view(-1, shift_hidden.size(-1)),
                    shift_labels,
                )
            else:
                shift_logits = self.lm_head(shift_hidden).view(-1, self.config.vocab_size)
                loss = F.cross_entropy(shift_logits.float(), shift_labels, reduction="sum")

            if num_items_in_batch is not None:
                loss = loss / num_items_in_batch
            else:
                n_valid = (shift_labels != -100).sum().clamp_min(1)
                loss = loss / n_valid

            if self.training:
                aux_loss = self._collect_mom_aux_loss(hidden_states.device)
                if self.mom_aux_normalize:
                    if not self.mom_aux_accum_steps or self.mom_aux_accum_steps < 1:
                        raise RuntimeError(
                            "mom_aux_normalize=true but mom_aux_accum_steps was not set "
                            "by the harness (pretrain.py sets it from [train] grad_accum) "
                            "— refusing to guess the accumulation cycle length."
                        )
                    aux_loss = aux_loss / self.mom_aux_accum_steps
                loss = loss + self.mom_aux_weight * aux_loss

            logits = None
        else:
            logits = self.lm_head(hidden_states)
            loss = None

        return CausalLMOutputWithPast(loss=loss, logits=logits)

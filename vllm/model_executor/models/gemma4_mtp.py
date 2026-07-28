# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Gemma4 MTP (Multi-Token Prediction) model.

The Gemma4 assistant model is a lightweight decoder that shares KV cache
with the target (backbone) model.  All assistant decoder layers are
KV-shared: they only have Q projections (no K/V projections or norms),
and read K/V from the target model's cache at runtime.

Checkpoint layout (``gemma4_assistant``)::

    model.embed_tokens.*          -- token embeddings
    model.layers.{i}.*            -- decoder layers (Q-only attention + MLP)
    model.norm.*                  -- final RMSNorm
    pre_projection.*              -- Linear(2 * backbone_hidden_size, hidden_size)
    post_projection.*             -- Linear(hidden_size, backbone_hidden_size)
    lm_head.*                     -- language model head (tied to embed_tokens)
    masked_embedding.centroids.*  -- centroid projection (when use_ordered_embeddings)
    masked_embedding.token_ordering -- token-to-centroid mapping buffer
"""

from collections.abc import Iterable

import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import tl, triton

from .gemma4 import Gemma4MLP, _get_text_config
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    get_draft_quant_config,
    maybe_prefix,
)

logger = init_logger(__name__)

# csrc/moe/moe_align_sum_kernels.cu asserts padded_num_experts < 1024.
_MOE_ALIGN_MAX_EXPERTS = 1024


@triton.jit
def _decode_top_token_kernel(
    logits_ptr,
    topk_ids_ptr,
    token_ordering_ptr,
    out_ptr,
    num_selected,
    TOP_K: tl.constexpr,
    VOCAB_PER_CENTROID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    vals = tl.load(
        logits_ptr + token * num_selected + offs,
        mask=offs < num_selected,
        other=-float("inf"),
    )
    best = tl.argmax(vals, axis=0)
    centroid = tl.load(topk_ids_ptr + token * TOP_K + best // VOCAB_PER_CENTROID)
    vocab_id = tl.load(
        token_ordering_ptr
        + centroid.to(tl.int64) * VOCAB_PER_CENTROID
        + best % VOCAB_PER_CENTROID
    )
    tl.store(out_ptr + token, vocab_id)


def _decode_top_token(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    token_ordering: torch.Tensor,
    top_k: int,
    vocab_size_per_centroid: int,
) -> torch.Tensor:
    """Fused argmax over sparse logits + decode to vocab ids.

    The argmax gives a position in the selected candidate set, not a vocab
    id: position p means centroid slot p // vocab_size_per_centroid and
    offset p % vocab_size_per_centroid within it. Doing that eagerly costs
    five launches (argmax, floordiv, mod, gather, index) of almost no work
    each, ~13us at T=1 against a ~4us GEMM. Fusing them costs ~2us.
    """
    num_tokens = logits.shape[0]
    out = torch.empty(num_tokens, dtype=token_ordering.dtype, device=logits.device)
    _decode_top_token_kernel[(num_tokens,)](
        logits,
        topk_ids,
        token_ordering,
        out,
        top_k * vocab_size_per_centroid,
        top_k,
        vocab_size_per_centroid,
        BLOCK=triton.next_power_of_2(top_k * vocab_size_per_centroid),
    )
    return out


def _grouped_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    num_experts: int,
    config_override: dict | None = None,
) -> torch.Tensor:
    """Single grouped GEMM: (T, K) x (E, N, K)[topk_ids] -> (T, top_k, N).

    Borrows the first half of ``fused_experts``: no activation, no second
    GEMM, no routing-weight reduction. The kernel writes rows at the
    original flat (token, k) offset, so no unpermute is needed.

    The only place in this file that touches a private fused_moe API.
    """
    import triton.language as tl

    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _prepare_expert_assignment,
        dispatch_fused_moe_kernel,
    )

    compute_type = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }[x.dtype]

    # Tuned by sweeping BLOCK_SIZE_N x num_warps x num_stages. Block shape
    # swings the GEMM itself by ~4x but the whole sparse-head path by only
    # ~4%, since the GEMM is a small share of it; anything in the
    # BLOCK_SIZE_N 16-64 range with >=2 warps is within noise of this.
    config = config_override or {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    }
    if num_experts >= _MOE_ALIGN_MAX_EXPERTS:
        # moe_align_block_size's CUDA kernel asserts padded_num_experts <
        # 1024, so the token-sorted path is unreachable here. Fall back to
        # naive block assignment (one block per (token, k) pair), which is
        # what _prepare_expert_assignment picks for small batches anyway.
        # Note this forfeits cross-token weight dedup.
        sorted_ids = None
        expert_ids = topk_ids.view(-1)
        num_tokens_post_padded = torch.full(
            (1,),
            topk_ids.numel() * config["BLOCK_SIZE_M"],
            dtype=torch.int32,
            device=topk_ids.device,
        )
    else:
        sorted_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
            topk_ids,
            config,
            x.shape[0],
            top_k,
            num_experts,
            None,
        )
    out = torch.empty(
        (x.shape[0], top_k, weight.shape[1]),
        dtype=x.dtype,
        device=x.device,
    )
    dispatch_fused_moe_kernel(
        x,
        weight,
        out,
        None,  # A_scale
        None,  # B_scale
        None,  # B_zp
        None,  # topk_weights (mul_routed_weight=False)
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        False,  # mul_routed_weight
        top_k,
        config,
        compute_type,
        False,  # use_fp8_w8a8
        False,  # use_int8_w8a8
        False,  # use_int8_w8a16
        False,  # use_int4_w4a16
        False,  # per_channel_quant
    )
    return out


class Gemma4MTPMaskedEmbedder(nn.Module):
    """Sparse logit computation via centroid-based vocabulary masking.

    Instead of computing logits against the full vocabulary, projects
    hidden states to centroid scores, selects top-K centroids, and
    computes logits only for the ~top_k * (vocab_size / num_centroids)
    tokens belonging to those centroids.

    Two backends, selected by ``VLLM_GEMMA4_MTP_SPARSE_HEAD_BACKEND``:

    - ``gather``: scattered row gather + einsum.
    - ``moe``: the same computation expressed as a top-k routed grouped
      GEMM (centroids are experts, the LM head rows for a centroid are
      that expert's weight). Requires ``build_centroid_weight()``.
    """

    token_ordering: torch.Tensor

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_centroids: int,
        centroid_intermediate_top_k: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_centroids = num_centroids
        self.centroid_intermediate_top_k = centroid_intermediate_top_k
        self.vocab_size_per_centroid = vocab_size // num_centroids
        self.num_selected = centroid_intermediate_top_k * self.vocab_size_per_centroid

        # out_dtype=None keeps the ReplicatedLinear fallback, matching the
        # previous nn.Linear numerics exactly. Every specialized GateLinear
        # tier requires fp32 output; call set_out_dtype(torch.float32) to
        # opt into them (changes centroid selection, so measure).
        self.centroids = GateLinear(
            hidden_size,
            num_centroids,
            bias=False,
            prefix=f"{prefix}.centroids",
        )
        self.register_buffer(
            "token_ordering",
            torch.empty(vocab_size, dtype=torch.long),
        )

        self.use_moe_backend = envs.VLLM_GEMMA4_MTP_SPARSE_HEAD_BACKEND == "moe"
        # (num_centroids, vocab_size_per_centroid, hidden_size), built by
        # build_centroid_weight() once the LM head weights are loaded.
        self.centroid_weight: torch.Tensor | None = None

    def build_centroid_weight(self, lm_head_weight: torch.Tensor) -> None:
        """Permute the LM head into centroid-major (E, N, K) layout.

        Turns the per-token scatter of ``num_selected`` rows into
        ``top_k`` contiguous tiles, and is the weight layout the grouped
        GEMM expects. Idempotent; no-op unless the moe backend is active.
        """
        if not self.use_moe_backend or self.centroid_weight is not None:
            return
        self.centroid_weight = (
            lm_head_weight[self.token_ordering]
            .view(self.num_centroids, self.vocab_size_per_centroid, -1)
            .contiguous()
        )

    def _route(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Top-k centroid selection. Returns (num_tokens, top_k) ids."""
        centroid_scores, _ = self.centroids(hidden_states)
        # sorted=False measures 27% faster (12.7us vs 17.3us at T=1) and
        # selects the same centroid set; only the ordering differs, which
        # affects argmax tie-breaking among equal logits. Compare backends
        # by token-set agreement rather than exact token equality.
        _, top_k_indices = torch.topk(
            centroid_scores,
            k=self.centroid_intermediate_top_k,
            dim=-1,
            sorted=False,
        )
        return top_k_indices

    def _select_and_score(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Centroid selection + sparse dot product.

        Returns:
            logits: (num_tokens, num_selected) sparse logits.
            indices: (num_tokens, num_selected) corresponding vocab indices.
        """
        num_tokens = hidden_states.shape[0]
        top_k_indices = self._route(hidden_states)
        clusters = self.token_ordering.view(
            self.num_centroids,
            self.vocab_size_per_centroid,
        )
        selected = clusters[top_k_indices]
        embeddings = lm_head_weight[selected.reshape(-1)].view(
            num_tokens,
            self.num_selected,
            self.hidden_size,
        )
        logits = torch.einsum("td,tsd->ts", hidden_states, embeddings)
        return logits, selected.view(num_tokens, -1)

    def _select_and_score_moe(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Grouped-GEMM equivalent of ``_select_and_score``.

        Returns:
            logits: (num_tokens, top_k, vocab_size_per_centroid).
            top_k_indices: (num_tokens, top_k) selected centroid ids.

        Logits are left unflattened so callers that only need an argmax
        can decode vocab ids arithmetically instead of materializing the
        (num_tokens, num_selected) index tensor.
        """
        assert self.centroid_weight is not None, (
            "build_centroid_weight() must be called before the moe backend is used."
        )
        top_k_indices = self._route(hidden_states).to(torch.int32)
        logits = _grouped_gemm(
            hidden_states,
            self.centroid_weight,
            top_k_indices,
            self.centroid_intermediate_top_k,
            self.num_centroids,
        )
        return logits, top_k_indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Full-vocab logits with non-selected positions masked to -inf."""
        num_tokens = hidden_states.shape[0]
        if self.use_moe_backend:
            logits, top_k_indices = self._select_and_score_moe(hidden_states)
            logits = logits.view(num_tokens, -1)
            clusters = self.token_ordering.view(
                self.num_centroids,
                self.vocab_size_per_centroid,
            )
            indices = clusters[top_k_indices.long()].view(num_tokens, -1)
        else:
            logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        output = torch.full(
            (num_tokens, self.vocab_size),
            fill_value=torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return output.scatter_(-1, indices, logits)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse argmax — returns vocab token IDs without full-vocab tensor."""
        if self.use_moe_backend:
            logits, top_k_indices = self._select_and_score_moe(hidden_states)
            return _decode_top_token(
                logits.view(hidden_states.shape[0], -1),
                top_k_indices,
                self.token_ordering,
                self.centroid_intermediate_top_k,
                self.vocab_size_per_centroid,
            )
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        return indices.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)


class Gemma4MTPAttention(nn.Module):
    """Q-only attention for Gemma4 MTP layers.

    K/V come from the target model's KV cache via
    ``kv_sharing_target_layer_name`` (set by the proposer after
    model construction).
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        attn_logits_soft_cap: float | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.scaling = 1.0

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        sliding_window = config.sliding_window if self.is_sliding else None

        if layer_type in config.rope_parameters:
            rope_parameters = dict(config.rope_parameters[layer_type])
        else:
            rope_parameters = dict(config.rope_parameters.copy())
            if self.is_sliding:
                rope_parameters["rope_theta"] = getattr(
                    config, "rope_local_base_freq", 10000.0
                )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        # kv_sharing_target_layer_name is set after model construction
        # by Gemma4Proposer._setup_gemma4_kv_sharing().
        self.is_kv_shared_layer = True
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=attn_logits_soft_cap,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        q, _ = self.rotary_emb(positions, q, None)

        # Attention reads K/V from the target's cache via KV sharing;
        # these dummy tensors are never consumed but required by the API.
        num_tokens = q.shape[0]
        kv_dummy = torch.empty(
            num_tokens,
            self.num_kv_heads * self.head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        attn_output = self.attn(q, kv_dummy, kv_dummy)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma4MTPDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        is_full_attention = layer_type == "full_attention"
        head_dim = (
            getattr(config, "global_head_dim", config.head_dim)
            if is_full_attention
            else config.head_dim
        )

        use_k_eq_v = is_full_attention and getattr(config, "attention_k_eq_v", False)
        if use_k_eq_v:
            num_kv_heads = getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
        else:
            num_kv_heads = config.num_key_value_heads

        self.self_attn = Gemma4MTPAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=getattr(config, "attn_logit_softcapping", None),
            prefix=f"{prefix}.self_attn",
        )

        text_config = _get_text_config(config)
        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=text_config.intermediate_size,
            hidden_activation=text_config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None


class Gemma4MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.speculative_config.draft_model_config.hf_config
        text_config = _get_text_config(config)
        quant_config = get_draft_quant_config(vllm_config)
        self.config = text_config
        self.quant_config = quant_config

        self.hidden_size = text_config.hidden_size
        self.backbone_hidden_size = getattr(
            config, "backbone_hidden_size", self.hidden_size
        )
        self.vocab_size = text_config.vocab_size
        self.num_mtp_layers = text_config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.pre_projection = ColumnParallelLinear(
            2 * self.backbone_hidden_size,
            self.hidden_size,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.pre_projection",
        )

        self.post_projection = RowParallelLinear(
            self.hidden_size,
            self.backbone_hidden_size,
            bias=False,
            input_is_parallel=False,
            quant_config=quant_config,
            prefix=f"{prefix}.post_projection",
        )

        self.layers = nn.ModuleList(
            Gemma4MTPDecoderLayer(
                text_config,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )

        self.norm = RMSNorm(self.hidden_size, eps=text_config.rms_norm_eps)

        # After embedding sharing, embed_tokens is replaced with the
        # target model's backbone-dim embedding.  Scale by
        # sqrt(backbone_hidden_size) to match the target's convention.
        self.register_buffer(
            "normalizer",
            torch.tensor(self.backbone_hidden_size**0.5),
            persistent=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (draft_hidden_states, backbone_hidden_states).

        draft_hidden_states: draft-dim, used by compute_logits via lm_head.
        backbone_hidden_states: backbone-dim, stored in the proposer's
            hidden-state buffer and fed back as input to the next step.
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        combined = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states, _ = self.pre_projection(combined)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        draft_hidden_states = self.norm(hidden_states)

        backbone_hidden_states, _ = self.post_projection(draft_hidden_states)
        return draft_hidden_states, backbone_hidden_states


@support_torch_compile
class Gemma4MTP(nn.Module):
    """Gemma4 Multi-Token Prediction model for speculative decoding.

    forward() returns (draft_hidden_states, backbone_hidden_states).
    The proposer uses draft_hidden_states for compute_logits (via
    the draft-dim lm_head) and backbone_hidden_states for the
    hidden-state feedback buffer.
    """

    has_own_lm_head = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "pre_projection.": "model.pre_projection.",
            "post_projection.": "model.post_projection.",
        },
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        text_config = _get_text_config(config)
        self.quant_config = get_draft_quant_config(vllm_config)
        self.config = config
        self.vocab_size = text_config.vocab_size
        self._stable_full_lm_head_weight: torch.Tensor | None = None

        self.model = Gemma4MultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "draft_model"),
        )

        # lm_head operates in draft-dim.  Tied to embed_tokens at init
        # so load_weights populates both from a single checkpoint entry.
        # After embedding sharing, lm_head.weight still references the
        # original draft-dim tensor.
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(
            text_config.vocab_size,
            soft_cap=getattr(text_config, "final_logit_softcapping", None),
        )

        if getattr(config, "use_ordered_embeddings", False):
            num_centroids = getattr(config, "num_centroids", 2048)
            top_k = getattr(config, "centroid_intermediate_top_k", 32)
            self.masked_embedding = Gemma4MTPMaskedEmbedder(
                hidden_size=text_config.hidden_size,
                vocab_size=text_config.vocab_size,
                num_centroids=num_centroids,
                centroid_intermediate_top_k=top_k,
                prefix=maybe_prefix(prefix, "masked_embedding"),
            )
            logger.info(
                "Gemma4 MTP: centroids masking enabled "
                "(num_centroids=%d, top_k=%d, active_tokens=%d/%d, "
                "backend=%s).",
                num_centroids,
                top_k,
                top_k * (text_config.vocab_size // num_centroids),
                text_config.vocab_size,
                envs.VLLM_GEMMA4_MTP_SPARSE_HEAD_BACKEND,
            )
        else:
            self.masked_embedding = None

        draft_cfg = vllm_config.speculative_config.draft_model_config
        gen_cfg = draft_cfg.try_get_generation_config()
        self._suppress_token_ids = gen_cfg.get("suppress_tokens") if gen_cfg else None
        # Materialized on first use: indexing with the Python list rebuilds
        # an index tensor and copies H2D on every call, and blocks graph
        # capture.
        self._suppress_mask: torch.Tensor | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx,
        )

    def _get_full_lm_head_weight(self) -> torch.Tensor:
        if self._stable_full_lm_head_weight is not None:
            return self._stable_full_lm_head_weight
        lm_head_weight = self.lm_head.weight
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            lm_head_weight = tensor_model_parallel_all_gather(
                lm_head_weight,
                dim=0,
            )
            lm_head_weight = lm_head_weight[
                : self.masked_embedding.vocab_size
            ].contiguous()
        else:
            lm_head_weight = lm_head_weight[: self.masked_embedding.vocab_size]
        self._stable_full_lm_head_weight = lm_head_weight
        return lm_head_weight

    def _ensure_centroid_weight(self) -> None:
        """Build the centroid-major weight if it is not ready yet.

        Normally built at the end of load_weights, so the allocation is
        counted by memory profiling. This is the fallback for any path
        that reaches compute_logits/get_top_tokens first.
        """
        masked = self.masked_embedding
        if masked is not None:
            masked.build_centroid_weight(self._get_full_lm_head_weight())

    def _get_suppress_mask(self, device: torch.device) -> torch.Tensor | None:
        if not self._suppress_token_ids:
            return None
        if self._suppress_mask is None:
            mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            mask[list(self._suppress_token_ids)] = True
            self._suppress_mask = mask.to(device)
        return self._suppress_mask

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        if self.masked_embedding is not None:
            self._ensure_centroid_weight()
            logits = self.masked_embedding(
                hidden_states,
                self._get_full_lm_head_weight(),
            )
        else:
            logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None:
            suppress_mask = self._get_suppress_mask(logits.device)
            if suppress_mask is not None:
                logits.masked_fill_(suppress_mask, -float("inf"))
        return logits

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse argmax via centroids masking. Returns token IDs directly."""
        self._ensure_centroid_weight()
        return self.masked_embedding.get_top_tokens(
            hidden_states,
            self._get_full_lm_head_weight(),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self._stable_full_lm_head_weight = None
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        # Build here so the permuted copy is allocated before memory
        # profiling and before any dummy run touches the sparse head.
        self._ensure_centroid_weight()
        return loaded

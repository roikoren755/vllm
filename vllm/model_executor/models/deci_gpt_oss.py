# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from copy import copy

import torch
from torch import nn
from transformers import GptOssConfig, PretrainedConfig

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import gpt_oss
from vllm.utils import cdiv

from .utils import (
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class DeciGptOssAttention(gpt_oss.OAIAttention):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ):
        """
        Ideally we would just call OAIAttention.__init__
        after building the GptOssConfig,
        but we have to fix the decision about which layers use sliding window.
        """
        nn.Module.__init__(self)
        layer_idx = extract_layer_index(prefix)
        attention_config = config.block_configs[layer_idx].attention
        config = GptOssConfig(
            head_dim=getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            ),
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads
            // attention_config.n_heads_in_group,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            sliding_window=attention_config.window_length,
        )
        sliding_window = attention_config.window_length

        ################################################################
        ### Original vLLM code from here on out, except for commenting out this line:
        ### sliding_window = (
        ###     config.sliding_window if self.layer_idx % 2 == 0 else None
        ### )
        ################################################################
        self.layer_idx = extract_layer_index(prefix)
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            dtype=torch.float32,
            rope_scaling={
                "rope_type": "yarn",
                "factor": config.rope_scaling["factor"],
                "original_max_position_embeddings": config.rope_scaling[
                    "original_max_position_embeddings"
                ],
                "beta_fast": config.rope_scaling["beta_fast"],
                "beta_slow": config.rope_scaling["beta_slow"],
                "truncate": config.rope_scaling["truncate"],
            },
            is_neox_style=True,
        )

        tp_size = get_tensor_model_parallel_world_size()

        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads // tp_size, requires_grad=False)
        )

        self.q_size = self.num_attention_heads * self.head_dim // tp_size
        self.kv_size = self.num_key_value_heads * self.head_dim // tp_size
        self.scaling = self.head_dim**-0.5
        self.rope_theta = config.rope_theta

        self.qkv = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_attention_heads,
            total_num_kv_heads=self.num_key_value_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.num_local_attention_heads = config.num_attention_heads // tp_size
        self.num_local_key_value_heads = config.num_key_value_heads // tp_size

        ## Deci: we decide whether to apply sliding window based on the attention config
        # Original vllm code: apply sliding window to every other layer
        # sliding_window = (config.sliding_window if self.layer_idx %
        #                   2 == 0 else None)
        self.attn = Attention(
            self.num_local_attention_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_local_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
        )


class DeciGptOssMLPBlock(gpt_oss.MLPBlock):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str = "",
    ):
        deci_config = vllm_config.model_config.hf_config
        ffn_config = deci_config.block_configs[layer_idx].ffn
        vanilla_config = GptOssConfig(
            num_local_experts=ffn_config.moe.num_local_experts,
            num_experts_per_tok=ffn_config.moe.num_experts_per_tok,
            hidden_size=deci_config.hidden_size,
            intermediate_size=ffn_config.moe.expert_intermediate_dim,
        )

        # shallow copy to avoid OOM since vllm_config contains weights
        vllm_config = copy(vllm_config)
        vllm_config.model_config = copy(vllm_config.model_config)
        vllm_config.model_config.hf_config = vanilla_config

        super().__init__(vllm_config, layer_idx, prefix)


class DeciGptOssTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config

        self.layer_idx = extract_layer_index(prefix)

        block_config = config.block_configs[self.layer_idx]
        self.attention_no_op = block_config.attention.no_op
        self.ffn_no_op = block_config.ffn.no_op

        if not self.attention_no_op:
            self.attn = DeciGptOssAttention(
                config, prefix=f"{prefix}.attn", cache_config=cache_config
            )
            self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-5)

        if not self.ffn_no_op:
            self.mlp = DeciGptOssMLPBlock(
                vllm_config, self.layer_idx, prefix=f"{prefix}.mlp"
            )
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.attention_no_op:
            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.attn(hidden_states, positions)
        elif residual is None:
            residual = hidden_states

        if not self.ffn_no_op:
            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            output = self.mlp(hidden_states)
        else:
            output = hidden_states

        return output, residual


@support_torch_compile
class DeciGptOssModel(gpt_oss.GptOssModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.parallel_config = vllm_config.parallel_config
        self.config.hidden_size = self.config.hidden_size
        self.embedding = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: DeciGptOssTransformerBlock(
                vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=1e-5)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )
        self.aux_hidden_state_layers = tuple[int, ...]()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        We override load_weights to accommodate per-layer weight sizes:
        num experts, expert intermediate size.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv", ".q_proj", "q"),
            (".qkv", ".k_proj", "k"),
            (".qkv", ".v_proj", "v"),
        ]

        quant_method = (
            self.config.quantization_config["quant_method"]
            if hasattr(self.config, "quantization_config")
            else None
        )
        if quant_method == "mxfp4":
            return self._load_weights_mxfp4(weights, stacked_params_mapping)
        else:
            return self._load_weights_other(weights, stacked_params_mapping)

    def _get_layer_config(self, layer_idx: int) -> dict:
        block_config = self.config.block_configs[layer_idx]

        layer_config = {}

        if not block_config.ffn.no_op:
            layer_config["num_local_experts"] = block_config.ffn.moe.num_local_experts
            layer_config["intermediate_size"] = (
                block_config.ffn.moe.expert_intermediate_dim
            )

        return layer_config

    def _load_weights_mxfp4(
        self,
        weights,
        stacked_params_mapping: list[tuple[str, ...]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        mxfp4_block = 32
        use_ep = self.parallel_config.enable_expert_parallel

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        ep_size = get_ep_group().world_size
        ep_rank = get_ep_group().rank

        for name, weight in weights:
            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue

            # FIXME(woosuk): Remove this after testing.
            weight = weight.cuda()

            try:
                layer_idx = extract_layer_index(name)
            except AssertionError:  # no layer index e.g. enbeddings layer
                layer_idx = None

            if layer_idx is not None:
                layer_config = self._get_layer_config(layer_idx)

                if "num_local_experts" in layer_config:
                    num_experts = layer_config["num_local_experts"]
                    experts_per_rank = num_experts // ep_size
                    ep_rank_start = ep_rank * experts_per_rank
                    ep_rank_end = (ep_rank + 1) * experts_per_rank

                if "intermediate_size" in layer_config:
                    intermediate_size = layer_config["intermediate_size"]
                    intermediate_size_block = intermediate_size // mxfp4_block
                    per_rank_intermediate_size_block = cdiv(
                        intermediate_size_block, tp_size
                    )
                    per_rank_intermediate_size = (
                        per_rank_intermediate_size_block * mxfp4_block
                    )

                    # Calculate common slicing bounds for current rank
                    tp_rank_start = tp_rank * per_rank_intermediate_size
                    tp_rank_end = min(
                        (tp_rank + 1) * per_rank_intermediate_size, intermediate_size
                    )

            if ".w13_weight_scale" in name:
                # Handle MLP gate and up projection weights scale
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w2_weight_scale" in name:
                # Handle MLP down projection weights
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[
                        ..., tp_rank_start // mxfp4_block : tp_rank_end // mxfp4_block
                    ]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w13_weight" in name:
                # Handle MLP gate and up projection weights
                # flat weight from (E, 2 * N, block_size, entry_per_block)
                # to (E, 2 * N, -1), shouldn't trigger copy for contiguous
                weight = weight.view(
                    num_experts, 2 * intermediate_size, -1
                ).contiguous()

                # Extract gate and up projection parts
                # since the weight is shuffled, we can slice directly
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w2_weight" in name:
                # Handle MLP down projection weights
                # same flatten here, but since 2 mx4 value are packed in 1
                # uint8, divide by 2
                weight = weight.view(
                    num_experts, -1, intermediate_size // 2
                ).contiguous()
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[..., tp_rank_start // 2 : tp_rank_end // 2]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w13_bias" in name:
                # Handle MLP gate and up projection biases
                # Extract gate and up projection bias parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w2_bias" in name:
                # Handle MLP down projection bias
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if use_ep:
                    weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    # (only load on rank 0 to avoid duplication)
                    if tp_rank != 0:
                        weight.zero_()
                weight_loader(
                    param, weight, weight_name=name, shard_id=None, expert_id=None
                )
                loaded_params.add(name)
                continue
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, weight)
                else:
                    weight_loader(param, weight, shard_id)
                break
            else:
                # Handle all other weights with potential renaming
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
            loaded_params.add(name)
        return loaded_params

    def _load_weights_other(
        self,
        weights,
        stacked_params_mapping: list[tuple[str, ...]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        use_ep = self.parallel_config.enable_expert_parallel

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        ep_size = get_ep_group().world_size
        ep_rank = get_ep_group().rank

        for name, weight in weights:
            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue

            try:
                layer_idx = extract_layer_index(name)
            except AssertionError:  # no layer index e.g. enbeddings layer
                layer_idx = None

            if layer_idx is not None:
                layer_config = self._get_layer_config(layer_idx)

                if "num_local_experts" in layer_config:
                    num_experts = layer_config["num_local_experts"]
                    experts_per_rank = num_experts // ep_size
                    ep_rank_start = ep_rank * experts_per_rank
                    ep_rank_end = (ep_rank + 1) * experts_per_rank

                if "intermediate_size" in layer_config:
                    intermediate_size = layer_config["intermediate_size"]
                    per_rank_intermediate_size = cdiv(intermediate_size, tp_size)
                    # Calculate common slicing bounds for current rank
                    tp_rank_start = tp_rank * per_rank_intermediate_size
                    tp_rank_end = min(
                        (tp_rank + 1) * per_rank_intermediate_size, intermediate_size
                    )

            if ".w13_weight" in name:
                # Handle MLP gate and up projection weights
                # Extract gate and up projection parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, :, 2 * tp_rank_start : 2 * tp_rank_end]

                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[name]

                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w2_weight" in name:
                # Handle MLP down projection weights
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, tp_rank_start:tp_rank_end, :]
                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[name]

                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w13_bias" in name:
                # Handle MLP gate and up projection biases
                # Extract gate and up projection bias parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end]

                param = params_dict[name]
                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w2_bias" in name:
                # Handle MLP down projection bias
                if use_ep:
                    weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    # (only load on rank 0 to avoid duplication)
                    if tp_rank != 0:
                        weight.zero_()
                param = params_dict[name]
                param.copy_(weight)
                loaded_params.add(name)
                continue
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, weight)
                else:
                    weight_loader(param, weight, shard_id)
                break
            else:
                # Handle all other weights with potential renaming
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
            loaded_params.add(name)
        return loaded_params


class DeciGptOssForCausalLM(gpt_oss.GptOssForCausalLM):
    allow_patterns_overrides = ["subblocks_safetensors/*.safetensors", "*.safetensors"]

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        self.model = DeciGptOssModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

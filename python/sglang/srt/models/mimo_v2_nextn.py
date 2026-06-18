# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.mimo_v2 import (
    MiMoV2Attention,
    MiMoV2ForCausalLM,
    MiMoV2MLP,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix


def _mimo_norm_eps(config: PretrainedConfig) -> float:
    return getattr(config, "layernorm_epsilon", getattr(config, "rms_norm_eps", 1e-6))

MiMoV2Config = None

logger = logging.getLogger(__name__)


class _DeferredSharedLMHead(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("deferred shared lm_head was used before being attached")


class MiMoV2MTPLayer(nn.Module):
    def __init__(
        self,
        config: MiMoV2Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if (
            isinstance(rope_scaling, dict)
            and rope_scaling.get("rope_type") == "default"
        ):
            rope_scaling = None
        max_position_embeddings = getattr(
            config,
            "context_len",
            getattr(config, "max_position_embeddings", 32768),
        )

        self.self_attn = MiMoV2Attention(
            hidden_size=self.hidden_size,
            num_heads=getattr(config, "swa_num_attention_heads", config.num_attention_heads),
            num_kv_heads=getattr(
                config, "swa_num_key_value_heads", config.num_key_value_heads
            ),
            head_dim=getattr(config, "swa_head_dim", config.head_dim),
            v_head_dim=getattr(
                config, "swa_v_head_dim", getattr(config, "v_head_dim", None)
            ),
            v_scale=getattr(config, "attention_value_scale", None),
            sliding_window_size=getattr(config, "sliding_window_size", None),
            attention_bias=getattr(config, "attention_bias", False),
            attention_sink_bias=getattr(config, "add_swa_attention_sink_bias", False),
            layer_id=layer_id,
            rope_theta=getattr(config, "swa_rope_theta", rope_theta),
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            disable_o_proj_quant=True,
            partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
            prefix=add_prefix("self_attn", prefix),
        )
        self.is_layer_sparse = False
        is_previous_layer_sparse = True
        is_next_layer_sparse = False

        if enable_moe_dense_fully_dp():
            mlp_tp_rank, mlp_tp_size = 0, 1
        else:
            mlp_tp_rank, mlp_tp_size = None, None
        self.mlp = MiMoV2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            tp_rank=mlp_tp_rank,
            tp_size=mlp_tp_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=_mimo_norm_eps(config))
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=_mimo_norm_eps(config)
        )
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        with get_global_expert_distribution_recorder().disable_this_region():
            hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class MiMoV2ModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        draft_model_idx: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            use_attn_tp_group=is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.enorm = RMSNorm(config.hidden_size, eps=_mimo_norm_eps(config))
        self.hnorm = RMSNorm(config.hidden_size, eps=_mimo_norm_eps(config))

        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        self.mtp_block = MiMoV2MTPLayer(
            config,
            0,
            quant_config=quant_config,
            prefix=add_prefix(f"mtp.layers.{draft_model_idx or 0}", prefix),
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=_mimo_norm_eps(config))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        if hidden_states.shape[0] > 0:
            token_norm = self.enorm(hidden_states)
            hidden_norm = self.hnorm(forward_batch.spec_info.hidden_states)
            hidden_states = self.eh_proj(
                torch.cat((token_norm, hidden_norm), dim=-1)
            )
        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=None,
        )
        before_final_norm = hidden_states if residual is None else hidden_states + residual
        hidden_states_before_norm = None
        if not forward_batch.forward_mode.is_idle():
            if forward_batch.return_hidden_states_before_norm:
                hidden_states_before_norm = before_final_norm
            if residual is not None:
                hidden_states, _ = self.final_layernorm(hidden_states, residual)
            else:
                hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, hidden_states_before_norm


class MiMoV2MTP(MiMoV2ForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        draft_model_idx: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.draft_model_idx = draft_model_idx or 0
        self.defer_shared_lm_head = draft_model_idx is not None

        self.model = MiMoV2ModelNextN(
            config,
            quant_config,
            draft_model_idx=self.draft_model_idx,
            prefix=add_prefix("model", prefix),
        )
        if self.defer_shared_lm_head:
            self.lm_head = _DeferredSharedLMHead()
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states, hidden_states_before_norm = self.model(
            input_ids, positions, forward_batch
        )
        logits_output = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            hidden_states_before_norm=hidden_states_before_norm,
        )
        return logits_output

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        modules_dict = dict(self.named_modules())

        def qkv_module_name(name: str) -> str:
            for suffix in (
                ".weight_scale_inv",
                ".weight_scale_2",
                ".weight_scale",
                ".input_scale",
                ".weight",
                ".bias",
            ):
                if name.endswith(suffix):
                    return name[: -len(suffix)]
            return name

        def is_mxfp8_mtp_qkv(name: str) -> bool:
            if "self_attn.qkv_proj" not in name:
                return False

            module_name = qkv_module_name(name)
            module = modules_dict.get(module_name)
            quant_method = getattr(module, "quant_method", None)
            quant_method_config = getattr(quant_method, "quant_config", None)
            if getattr(quant_method, "use_mxfp8", False) or getattr(
                quant_method_config, "use_mxfp8", False
            ):
                return True

            quantized_layers = getattr(self.quant_config, "quantized_layers", None)
            if not isinstance(quantized_layers, dict):
                return False

            candidates = [module_name]
            if "model.mtp_block." in module_name:
                tail = module_name.split("model.mtp_block.", 1)[1]
                candidates.append(f"model.mtp.layers.{self.draft_model_idx}.{tail}")
            elif "model.decoder." in module_name:
                tail = module_name.split("model.decoder.", 1)[1]
                candidates.append(f"model.mtp.layers.{self.draft_model_idx}.{tail}")

            resolve_quant_algo = getattr(self.quant_config, "_resolve_quant_algo", None)
            for candidate in candidates:
                if callable(resolve_quant_algo):
                    quant_algo = resolve_quant_algo(candidate)
                    if isinstance(quant_algo, str) and quant_algo.upper() == "MXFP8":
                        return True
                layer_info = quantized_layers.get(candidate)
                if (
                    isinstance(layer_info, dict)
                    and layer_info.get("quant_algo", "").upper() == "MXFP8"
                ):
                    return True
            return False

        def load_pre_sharded_mtp_qkv_if_needed(
            name: str, param: torch.Tensor, loaded_weight: torch.Tensor
        ) -> bool:
            if "self_attn.qkv_proj" not in name:
                return False

            if is_mxfp8_mtp_qkv(name):
                # Converted MXFP8 MTP qkv tensors use canonical fused
                # [Q_all, K_all, V_all] layout with UE8M0 block scales.
                # Let QKVParallelLinear split Q/K/V and TP shards.
                return False

            attn_tp_size = get_attention_tp_size()
            if (
                loaded_weight.dim() == 0
                or loaded_weight.shape[0] != param.shape[0] * attn_tp_size
            ):
                return False

            # MiMo V2 MTP fused qkv tensors are stored as TP-local
            # [Q, K, V] chunks concatenated on dim 0. This is true for the
            # official FP8 MTP tensors and for dequantized BF16 conversions of
            # those tensors; dtype alone is not a reliable layout signal.
            shard = loaded_weight.chunk(attn_tp_size, dim=0)[get_attention_tp_rank()]
            default_weight_loader(param, shard)
            return True

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if self.defer_shared_lm_head and "lm_head" in name:
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            name = self.map_model_name_to_mtp_param_name(name)

            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    if load_pre_sharded_mtp_qkv_if_needed(
                        name, param, loaded_weight
                    ):
                        continue
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:

                if f".{weight_name}." not in name:
                    continue
                if "mtp_block" not in name:
                    break
                name = name.replace(f".{weight_name}.", f".{param_name}.")
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if "mtp_block" not in name and (
                    "embed_tokens" not in name
                    and "lm_head" not in name
                    and "enorm" not in name
                    and "hnorm" not in name
                    and "eh_proj" not in name
                    and "final_layernorm" not in name
                ):
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    if "attention_sink_bias" in name:
                        start = get_attention_tp_rank() * param.numel()
                        param.data.copy_(loaded_weight[start : start + param.numel()])
                    else:
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def map_model_name_to_mtp_param_name(self, name: str) -> str:
        import re

        if "pre_mlp_layernorm" in name:
            name = name.replace("pre_mlp_layernorm", "post_attention_layernorm")

        name_without_prefix = [
            "enorm",
            "hnorm",
            "eh_proj",
            "final_layernorm",
        ]
        pattern = r"model.mtp.layers.(\d+)."
        group = re.match(pattern, name)
        if group is not None:
            for sub_name in name_without_prefix:
                if sub_name in name:
                    name = name.replace(group.group(), "model.")
                    return name
            name = name.replace(group.group(), "model.mtp_block.")
        return name

    def get_embed_and_head(self):
        if not hasattr(self.lm_head, "weight"):
            raise RuntimeError("deferred shared lm_head has not been attached")
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        if hasattr(self.lm_head, "weight"):
            del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = MiMoV2MTP

# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# All rights reserved.
#
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

from typing import Optional

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Qwen2MoeConfig(PreTrainedConfig):
    r"""
    Configuration class for Qwen2MoE-style models.

    Notes (practical, since transformers loves landmines):
    - We MUST consume (pop) rope_theta/rope_scaling from kwargs or they may leak into PreTrainedConfig and crash.
    - We MUST define layer_types before calling standardize_rope_params(), because it may expand rope params per layer type.
    - We MUST ensure pad_token_id/bos_token_id/eos_token_id attributes exist, even if None, because downstream code accesses them.

    MoE Router Policy (optional experimentation hook):
        routing_policy (`str`, *optional*, defaults to `"topk"`):
            - "topk", "top1", "soft", "hash", "topk_noisy"
        router_noise_epsilon (`float`, *optional*, defaults to `1e-2`):
            noise stddev for "topk_noisy" (if you actually use it)
    """

    model_type = "qwen2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "local_rowwise",
        "layers.*.mlp.experts.down_proj": "local_rowwise",
        "layers.*.mlp.experts": "gather",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: Optional[int] = 151936,
        hidden_size: Optional[int] = 2048,
        intermediate_size: Optional[int] = 6144,
        num_hidden_layers: Optional[int] = 24,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = 4,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 32768,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 1e-6,
        use_cache: Optional[bool] = True,
        tie_word_embeddings: Optional[bool] = False,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        attention_bias: Optional[bool] = False,
        use_sliding_window: Optional[bool] = False,
        sliding_window: Optional[int] = 4096,
        max_window_layers: Optional[int] = 28,
        layer_types: Optional[list[str]] = None,
        attention_dropout: Optional[float] = 0.0,
        decoder_sparse_step: Optional[int] = 1,
        moe_intermediate_size: Optional[int] = 768,
        num_experts_per_tok: Optional[int] = 8,
        num_experts: Optional[int] = 128,
        norm_topk_prob: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        router_aux_loss_coef: Optional[float] = 0.001,
        mlp_only_layers: Optional[list[int]] = None,
        routing_policy: Optional[str] = "topk",
        router_noise_epsilon: Optional[float] = 1e-2,
        **kwargs,
    ):
      
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.use_sliding_window = bool(use_sliding_window)
        self.sliding_window = int(sliding_window) if self.use_sliding_window else 0
        self.max_window_layers = int(max_window_layers) if max_window_layers is not None else 28

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if (self.use_sliding_window and i < self.max_window_layers and ((i + 1) % 2 == 1))
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        bos_token_id = kwargs.pop("bos_token_id", None)
        eos_token_id = kwargs.pop("eos_token_id", None)
        pad_token_id = kwargs.pop("pad_token_id", None)

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        if self.pad_token_id is None and self.eos_token_id is not None:
            self.pad_token_id = self.eos_token_id

        self.rope_theta = float(kwargs.pop("rope_theta", 10000.0))

        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters

        standardize_rope_params(self, rope_theta=self.rope_theta)
        rope_config_validation(self)

  
        self.decoder_sparse_step = int(decoder_sparse_step)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.num_experts = int(num_experts)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.output_router_logits = bool(output_router_logits)
        self.router_aux_loss_coef = float(router_aux_loss_coef)
        self.mlp_only_layers = [] if mlp_only_layers is None else list(mlp_only_layers)

        self.routing_policy = routing_policy
        self.router_noise_epsilon = float(router_noise_epsilon)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            **kwargs,
        )


__all__ = ["Qwen2MoeConfig"]

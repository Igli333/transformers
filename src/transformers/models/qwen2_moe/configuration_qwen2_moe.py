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
"""Qwen3MoE model configuration"""

from typing import Optional

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Qwen2MoeConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3MoeModel`].
    ...

    MoE Router Policy (NEW):
        routing_policy (`str`, *optional*, defaults to `"topk"`):
            Routing strategy for selecting experts. Supported:
            - `"topk"`: softmax then top-k (default)
            - `"top1"` / `"switch"`: select 1 expert (repeated to K internally if needed)
            - `"soft"`: softmax then top-k (same as topk, but kept for experimentation hooks)
            - `"hash"`: deterministic hashing-based routing from hidden state sign pattern
            - `"topk_noisy"`: adds Gaussian noise to router logits during training, then top-k

        router_noise_epsilon (`float`, *optional*, defaults to `1e-2`):
            Stddev for Gaussian noise added to router logits in `"topk_noisy"` routing (training only).
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
        rms_norm_eps: Optional[int] = 1e-6,
        use_cache: Optional[bool] = True,
        tie_word_embeddings: Optional[bool] = False,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        attention_bias: Optional[bool] = False,
        use_sliding_window: Optional[bool] = False,
        sliding_window: Optional[int] = 4096,
        attention_dropout: Optional[float] = 0.0,
        decoder_sparse_step: Optional[int] = 1,
        moe_intermediate_size: Optional[int] = 768,
        num_experts_per_tok: Optional[int] = 8,
        num_experts: Optional[int] = 128,
        norm_topk_prob: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        router_aux_loss_coef: Optional[float] = 0.001,
        mlp_only_layers: Optional[bool] = None,
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
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters

        # Validate RoPE params
        rope_theta = kwargs.get("rope_theta", 10000.0)
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers
       
        self.routing_policy = routing_policy
        self.router_noise_epsilon = router_noise_epsilon
        # --- Standard special-token ids: must exist as attributes ---
        # Some Qwen configs don't define pad_token_id in JSON, but code expects the attribute to exist.
        bos_token_id = kwargs.pop("bos_token_id", 151643)
        eos_token_id = kwargs.pop("eos_token_id", 151643)

        # Prefer explicit pad_token_id if present, else fall back to tokenizer's pad later.
        pad_token_id = kwargs.pop("pad_token_id", None)

        # Ensure attributes exist even if None
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )



__all__ = ["Qwen2MoeConfig"]

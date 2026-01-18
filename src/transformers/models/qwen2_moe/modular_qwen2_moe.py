# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
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

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs, logging
from transformers.models.llama.modeling_llama import (
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaRMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralModel, load_balancing_loss_func
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig



from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.generic import OutputRecorder, check_model_inputs
from transformers.gemma.modeling_gemma import GemmaMLP
from transformers.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding
from transformers.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaRMSNorm
from transformers.mixtral.modeling_mixtral import (
    MixtralExperts,
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
)
from .configuration_qwen2_moe import Qwen2MoeConfig


class Qwen2MoeRMSNorm(LlamaRMSNorm):
    pass


class Qwen2MoeRotaryEmbedding(Gemma2RotaryEmbedding):
    pass


class Qwen2MoeMLP(GemmaMLP):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]


class Qwen2MoeAttention(LlamaAttention):
    def __init__(self, config: Qwen2MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if self.config.layer_types[layer_idx] == "sliding_attention":
            self.sliding_window = config.sliding_window

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)


class Qwen2MoeExperts(MixtralExperts):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts
        self.intermediate_dim = config.moe_intermediate_size


class Qwen2MoeTopKRouter(nn.Module):
    """
    Upgraded router supporting multiple routing policies while keeping the original
    output contract expected by MixtralExperts:

      returns: (router_scores, selected_experts)

    router_scores: [T, E] dense scores with zeros except at selected experts
    selected_experts: [T, K] expert indices per token

    Policies (config.routing_policy):
      - "topk"        : softmax then top-k
      - "top1"/"switch": select 1 expert then repeat to K (uniform weights across K)
      - "soft"        : alias of topk (hook for experimentation)
      - "hash"        : deterministic routing based on sign pattern
      - "topk_noisy"  : add Gaussian noise to logits during training, then top-k
    """

    def __init__(self, config: Qwen2MoeConfig):
        super().__init__()
        self.top_k = int(config.num_experts_per_tok)
        self.num_experts = int(config.num_experts)
        self.norm_topk_prob = bool(config.norm_topk_prob)
        self.hidden_dim = int(config.hidden_size)

        # Router weight (like original)
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

        # Policy controls (safe defaults)
        self.routing_policy = getattr(config, "routing_policy", "topk")
        self.router_noise_epsilon = float(getattr(config, "router_noise_epsilon", 1e-2))

    # ---- runtime setters (so wrappers can change behavior post-load) ----
    def set_policy(self, routing_policy: str):
        self.routing_policy = (routing_policy or "").lower().strip()

    def set_top_k(self, top_k: int):
        self.top_k = int(top_k)

    def set_noise_epsilon(self, eps: float):
        self.router_noise_epsilon = float(eps)

    # ---- helpers ----
    def _normalize_topk(self, w: torch.Tensor) -> torch.Tensor:
        if self.norm_topk_prob:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
        return w

    def _ensure_k(self, idx: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ensure outputs are [T, K]. If we have K=1 (top1), repeat to self.top_k.
        """
        if idx.size(1) == self.top_k:
            return idx, w
        if idx.size(1) == 1:
            idx = idx.repeat(1, self.top_k)
            # uniform weights across repeated experts
            w = torch.full((w.size(0), self.top_k), 1.0 / self.top_k, device=w.device, dtype=w.dtype)
            return idx, w
        raise ValueError(f"Router produced K={idx.size(1)} but config expects K={self.top_k}")

    def _route_topk(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w, idx = torch.topk(probs, self.top_k, dim=-1)
        w = self._normalize_topk(w)
        return idx.to(torch.int64), w

    def _route_top1(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w, idx = torch.topk(probs, 1, dim=-1)
        idx, w = self._ensure_k(idx.to(torch.int64), w.to(probs.dtype))
        return idx, w

    def _route_hash(self, hidden_states_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # deterministic hash based on sign pattern
        T = hidden_states_2d.size(0)
        hv = (hidden_states_2d > 0).to(torch.int32).sum(dim=-1)  # [T]
        expert = (hv % self.num_experts).to(torch.int64)         # [T]
        idx = expert.unsqueeze(1).repeat(1, self.top_k)          # [T,K]
        w = torch.full((T, self.top_k), 1.0 / self.top_k, device=hidden_states_2d.device, dtype=hidden_states_2d.dtype)
        return idx, w

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)  # [T,H]
        router_logits = F.linear(hidden_states, self.weight)        # [T,E] (pre-softmax)

        policy = (getattr(self, "routing_policy", "topk") or "topk").lower().strip()

        # Policy: hash uses hidden_states directly, no softmax needed
        if policy == "hash":
            router_indices, router_top_value = self._route_hash(hidden_states)
            router_top_value = router_top_value.to(router_logits.dtype)
            router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
            return router_scores, router_indices

        # Policy: topk_noisy adds noise during training only
        if policy == "topk_noisy" and self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise_epsilon

        # Convert to probs for selection (keep float stability, cast back later)
        probs = torch.softmax(router_logits, dim=-1, dtype=torch.float).to(router_logits.dtype)

        if policy in ("topk", "soft"):
            router_indices, router_top_value = self._route_topk(probs)
        elif policy in ("top1", "switch"):
            router_indices, router_top_value = self._route_top1(probs)
        else:
            raise ValueError(f"Unknown routing_policy: {policy}")

        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Qwen2MoeTopKRouter(config)
        self.experts = Qwen2MoeExperts(config)
        self.shared_expert = Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        shared_expert_output = self.shared_expert(hidden_states_reshaped)

        routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output


class Qwen2MoeDecoderLayer(LlamaDecoderLayer, nn.Module):
    def __init__(self, config: Qwen2MoeConfig, layer_idx: int):
        nn.Module.__init__()
        self.self_attn = Qwen2MoeAttention(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen2MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen2MoeMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size


@auto_docstring
class Qwen2MoePreTrainedModel(MixtralPreTrainedModel):
    _can_record_outputs = {
        # Note: OutputRecorder name kept for BC even though it records "router_scores" in practice.
        "router_logits": OutputRecorder(Qwen2MoeTopKRouter, index=0),
        "hidden_states": Qwen2MoeDecoderLayer,
        "attentions": Qwen2MoeAttention,
    }


@auto_docstring
class Qwen2MoeModel(MixtralModel):
    def __init__(self, config: Qwen2MoeConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2MoeRotaryEmbedding(config=config)

    # ---- runtime routing API (used by wrappers / eval harnesses) ----
    def set_moe_routing_policy(self, routing_policy: str):
        routing_policy = (routing_policy or "").lower().strip()
        self.config.routing_policy = routing_policy
        for layer in self.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp.gate, "set_policy"):
                mlp.gate.set_policy(routing_policy)

    def set_moe_num_selects(self, top_k: int):
        top_k = int(top_k)
        self.config.num_experts_per_tok = top_k
        for layer in self.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp.gate, "set_top_k"):
                mlp.gate.set_top_k(top_k)

    def set_router_noise_epsilon(self, eps: float):
        eps = float(eps)
        self.config.router_noise_epsilon = eps
        for layer in self.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp.gate, "set_noise_epsilon"):
                mlp.gate.set_noise_epsilon(eps)

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen2MoeForCausalLM(MixtralForCausalLM, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts
        self.model = Qwen2MoeModel(config)

    # Pass-through runtime APIs (so your wrapper can call model.set_* safely)
    def set_moe_routing_policy(self, routing_policy: str):
        self.model.set_moe_routing_policy(routing_policy)

    def set_moe_num_selects(self, top_k: int):
        self.model.set_moe_num_selects(top_k)

    def set_router_noise_epsilon(self, eps: float):
        self.model.set_router_noise_epsilon(eps)


class Qwen2MoeForSequenceClassification(GenericForSequenceClassification, Qwen2MoePreTrainedModel): ...


class Qwen2MoeForTokenClassification(GenericForTokenClassification, Qwen2MoePreTrainedModel): ...


class Qwen2MoeForQuestionAnswering(GenericForQuestionAnswering, Qwen2MoePreTrainedModel): ...


__all__ = [
    "Qwen2MoeForCausalLM",
    "Qwen2MoeForQuestionAnswering",
    "Qwen2MoeModel",
    "Qwen2MoePreTrainedModel",
    "Qwen2MoeForSequenceClassification",
    "Qwen2MoeForTokenClassification",
]

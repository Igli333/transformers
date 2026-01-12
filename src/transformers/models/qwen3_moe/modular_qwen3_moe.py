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

from ...activations import ACT2FN
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import LossKwargs, logging
from ..llama.modeling_llama import (
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaRMSNorm,
)
from ..mixtral.modeling_mixtral import MixtralForCausalLM, MixtralModel, load_balancing_loss_func
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer
from ..qwen3.modeling_qwen3 import Qwen3Attention
from .configuration_qwen3_moe import Qwen3MoeConfig


logger = logging.get_logger(__name__)


class Qwen3MoeAttention(Qwen3Attention):  # main diff with qwen2MoE
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.sliding_window = getattr(config, "sliding_window", None)


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeRouter(nn.Module):
    """
    Multi-policy router.

    Returns:
      router_logits: [T, E] (float) pre-softmax logits (for aux loss/debug)
      selected_experts: [T, K] (int64)
      selected_weights: [T, K] (float, same dtype as input hidden)
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # router linear layer
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # policy controls
        self.routing_policy = getattr(config, "routing_policy", "topk")  # topk|top1|switch|soft|hash|topk_noisy
        self.router_noise_epsilon = getattr(config, "router_noise_epsilon", 1e-2)

    def _normalize_topk(self, w: torch.Tensor) -> torch.Tensor:
        if self.norm_topk_prob:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
        return w

    def _ensure_k(self, idx: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ensure [T,K]. If policy yields K=1 but model expects K>1, repeat.
        """
        if idx.size(1) == self.top_k:
            return idx, w
        if idx.size(1) == 1:
            idx = idx.repeat(1, self.top_k)
            w = torch.full((w.size(0), self.top_k), 1.0 / self.top_k, device=w.device, dtype=w.dtype)
            return idx, w
        raise ValueError(f"Router produced K={idx.size(1)} but config expects K={self.top_k}")

    def _route_topk(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w, idx = torch.topk(probs, self.top_k, dim=-1)
        w = self._normalize_topk(w)
        return idx.to(torch.int64), w

    def _route_top1(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w, idx = torch.topk(probs, 1, dim=-1)
        w = w.to(probs.dtype)
        idx, w = self._ensure_k(idx.to(torch.int64), w)
        return idx, w

    def _route_soft(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._route_topk(probs)

    def _route_hash(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # deterministic hash based on sign pattern
        T = hidden_states.size(0)
        hv = (hidden_states > 0).to(torch.int32).sum(dim=-1)  # [T]
        expert = (hv % self.num_experts).to(torch.int64)      # [T]
        idx = expert.unsqueeze(1).repeat(1, self.top_k)       # [T,K]
        w = torch.full((T, self.top_k), 1.0 / self.top_k, device=hidden_states.device, dtype=hidden_states.dtype)
        return idx, w

    def _route_topk_noisy(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            logits = logits + torch.randn_like(logits) * self.router_noise_epsilon
        probs = torch.softmax(logits, dim=-1, dtype=torch.float).to(logits.dtype)
        return self._route_topk(probs)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden_states: [T,H]
        router_logits = self.gate(hidden_states)  # [T,E]
        policy = getattr(self, "routing_policy", "topk")

        if policy == "hash":
            idx, w = self._route_hash(hidden_states)
            return router_logits, idx, w

        if policy == "topk_noisy":
            idx, w = self._route_topk_noisy(router_logits)
            return router_logits, idx, w

        probs = torch.softmax(router_logits, dim=-1, dtype=torch.float).to(router_logits.dtype)

        if policy == "topk":
            idx, w = self._route_topk(probs)
        elif policy in ("top1", "switch"):
            idx, w = self._route_top1(probs)
        elif policy == "soft":
            idx, w = self._route_soft(probs)
        else:
            raise ValueError(f"Unknown routing_policy: {policy}")

        return router_logits, idx, w


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.top_k = config.num_experts_per_tok

        self.router = Qwen3MoeRouter(config)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            final_hidden_states: [B,S,H]
            router_logits: [B*S, E] (for aux loss)
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_2d = hidden_states.view(-1, hidden_dim)  # [T,H], T=B*S

        router_logits, selected_experts, selected_weights = self.router(hidden_states_2d)  # logits [T,E], idx [T,K]

        # Safety: avoid CUDA device-side asserts when experimenting
        if selected_experts.dtype != torch.int64:
            selected_experts = selected_experts.to(torch.int64)

        if selected_experts.numel() > 0:
            min_idx = int(selected_experts.min())
            max_idx = int(selected_experts.max())
            if min_idx < 0 or max_idx >= self.num_experts:
                raise RuntimeError(
                    f"Expert index out of range: min={min_idx}, max={max_idx}, num_experts={self.num_experts}"
                )

        final_hidden_states = torch.zeros_like(hidden_states_2d)

        # Build expert mask: [E, K, T]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Find which experts get at least one token
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

        for expert_row in expert_hit:
            expert_idx = int(expert_row[0])
            expert_layer = self.experts[expert_idx]

            k_pos, token_pos = torch.where(expert_mask[expert_idx])  # k_pos: [N], token_pos: [N]

            current_state = hidden_states_2d[token_pos]  # [N,H]
            current_out = expert_layer(current_state)     # [N,H]
            current_out = current_out * selected_weights[token_pos, k_pos].unsqueeze(-1)

            final_hidden_states.index_add_(0, token_pos, current_out.to(final_hidden_states.dtype))

        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class Qwen3MoeRMSNorm(LlamaRMSNorm):
    pass

class Qwen3MoeDecoderLayer(Qwen2MoeDecoderLayer, nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3MoeAttention(config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class Qwen3MoeModel(MixtralModel):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

    # NEW: runtime routing policy API
    def set_moe_routing_policy(self, routing_policy: str):
        self.config.routing_policy = routing_policy
        for layer in self.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "router"):
                mlp.router.routing_policy = routing_policy


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs):
    ...

class Qwen3MoeForCausalLM(MixtralForCausalLM):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.model = Qwen3MoeModel(config)
        self.num_experts = config.num_experts

    # NEW: pass-through routing policy API
    def set_moe_routing_policy(self, routing_policy: str):
        self.model.set_moe_routing_policy(routing_policy)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> MoeCausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

class Qwen3MoeForSequenceClassification(LlamaForSequenceClassification):
    pass


class Qwen3MoeForTokenClassification(LlamaForTokenClassification):
    pass


class Qwen3MoeForQuestionAnswering(LlamaForQuestionAnswering):
    pass


__all__ = [
    "Qwen3MoeForCausalLM",
    "Qwen3MoeForQuestionAnswering",
    "Qwen3MoeModel",
    "Qwen3MoePreTrainedModel",  # noqa: F822
    "Qwen3MoeForSequenceClassification",
    "Qwen3MoeForTokenClassification",
]

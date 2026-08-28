from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import (
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)
from freetoken.layers.base import BaseOP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen2MoeSharedExpert(BaseOP):
    def __init__(self, config: ModelConfig):
        width = config.shared_expert_intermediate_size
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [width, width], has_bias=False
        )
        self.down_proj = LinearRowParallel(width, config.hidden_size, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(hidden_states)))


class Qwen2MoeMLP(BaseOP):
    """Qwen2-MoE routed experts plus its sigmoid-gated shared SwiGLU expert."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            weight_format="bf16",
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = Qwen2MoeSharedExpert(config)
        self.shared_expert_gate = LinearReplicated(
            config.hidden_size, 1, has_bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        shared *= torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
        routed = self.experts.forward(hidden_states, router_logits)
        return (routed + shared).view(num_tokens, hidden_dim)


__all__ = ["Qwen2MoeMLP", "Qwen2MoeSharedExpert"]

from __future__ import annotations

from typing import Any

from freetoken.models.config import ModelConfig, RotaryConfig


def parse_config(hf_config: Any) -> ModelConfig:
    sparse_step = int(getattr(hf_config, "decoder_sparse_step", 1))
    mlp_only = tuple(getattr(hf_config, "mlp_only_layers", ()) or ())
    if sparse_step != 1 or mlp_only:
        raise ValueError(
            "Qwen2-MoE currently requires every decoder layer to be sparse "
            f"(decoder_sparse_step=1, mlp_only_layers=[]), got {sparse_step=} {mlp_only=}"
        )
    layer_types = tuple(getattr(hf_config, "layer_types", ()) or ())
    uses_sliding = bool(getattr(hf_config, "use_sliding_window", False)) or any(
        layer_type != "full_attention" for layer_type in layer_types
    )
    if uses_sliding:
        raise ValueError("Qwen2-MoE sliding-window attention is not supported yet")
    hidden_act = str(getattr(hf_config, "hidden_act", ""))
    if hidden_act != "silu":
        raise ValueError(f"Qwen2-MoE currently supports hidden_act='silu', got {hidden_act!r}")
    num_experts = int(hf_config.num_experts)
    top_k = int(hf_config.num_experts_per_tok)
    moe_intermediate = int(hf_config.moe_intermediate_size)
    shared_intermediate = int(hf_config.shared_expert_intermediate_size)
    if num_experts <= 0 or not 0 < top_k <= num_experts:
        raise ValueError(f"Invalid Qwen2-MoE expert geometry: {num_experts=} {top_k=}")
    if moe_intermediate <= 0 or shared_intermediate <= 0:
        raise ValueError(
            "Qwen2-MoE expert widths must be positive: "
            f"{moe_intermediate=} {shared_intermediate=}"
        )
    num_kv_heads = getattr(
        hf_config, "num_key_value_heads", hf_config.num_attention_heads
    )
    head_dim = (
        getattr(hf_config, "head_dim", None)
        or hf_config.hidden_size // hf_config.num_attention_heads
    )
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is None:
        rope_parameters = getattr(hf_config, "rope_parameters", None) or {}
        rope_theta = rope_parameters.get("rope_theta")
    if rope_theta is None:
        rope_theta = 10_000.0

    return ModelConfig(
        num_layers=hf_config.num_hidden_layers,
        num_qo_heads=hf_config.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        intermediate_size=hf_config.intermediate_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        rotary_config=RotaryConfig(
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=hf_config.max_position_embeddings,
            base=rope_theta,
            scaling=rope_scaling,
        ),
        hidden_act=hidden_act,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_intermediate,
        norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", False)),
        model_type=getattr(hf_config, "model_type", "qwen2_moe"),
        architectures=getattr(hf_config, "architectures", ["Qwen2MoeForCausalLM"]),
        shared_expert_intermediate_size=shared_intermediate,
        has_attn_bias=bool(getattr(hf_config, "qkv_bias", True)),
    )


__all__ = ["parse_config"]

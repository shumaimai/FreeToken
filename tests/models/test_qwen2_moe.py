from __future__ import annotations

from types import SimpleNamespace

import torch


def _ensure_tp1() -> None:
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    else:
        assert try_get_tp_info().size == 1


def _hf_config(**overrides):
    values = dict(
        architectures=["Qwen2MoeForCausalLM"],
        model_type="qwen2_moe",
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=1_000_000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        num_experts=3,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=128,
        norm_topk_prob=False,
        qkv_bias=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen2_moe_config_maps_offload_geometry():
    from freetoken.models.qwen2_moe.config import parse_config

    config = parse_config(_hf_config())

    assert config.is_moe
    assert config.num_layers == 2
    assert config.num_experts == 3
    assert config.num_experts_per_tok == 2
    assert config.moe_intermediate_size == 16
    assert config.shared_expert_intermediate_size == 128
    assert config.has_attn_bias
    assert config.norm_topk_prob is False
    assert config.rotary_config.base == 1_000_000.0


def test_qwen2_moe_rejects_mixed_dense_layers():
    import pytest
    from freetoken.models.qwen2_moe.config import parse_config

    with pytest.raises(ValueError, match="every decoder layer"):
        parse_config(_hf_config(decoder_sparse_step=2))


def test_qwen2_moe_rejects_unimplemented_model_variants():
    import pytest
    from freetoken.models.qwen2_moe.config import parse_config

    with pytest.raises(ValueError, match="sliding-window"):
        parse_config(_hf_config(use_sliding_window=True))
    with pytest.raises(ValueError, match="hidden_act"):
        parse_config(_hf_config(hidden_act="gelu"))
    with pytest.raises(ValueError, match="expert geometry"):
        parse_config(_hf_config(num_experts_per_tok=4, num_experts=3))


def test_qwen2_moe_offload_model_state_excludes_routed_experts(monkeypatch):
    from freetoken.models import create_model
    from freetoken.models.qwen2_moe.config import parse_config
    from freetoken.layers.rotary import get_rope, set_rope_device

    _ensure_tp1()
    set_rope_device(torch.device("cpu"))
    get_rope.cache_clear()
    config = parse_config(_hf_config())
    object.__setattr__(config, "moe_backend", "offload")

    with torch.device("meta"):
        state = create_model(config).state_dict()

    assert not any(".experts.gate_up_proj" in key for key in state)
    assert state["model.layers.0.self_attn.qkv_proj.weight"].shape == (192, 64)
    assert state["model.layers.0.self_attn.qkv_proj.bias"].shape == (192,)
    assert state["model.layers.0.mlp.gate.weight"].shape == (3, 64)
    assert state["model.layers.0.mlp.shared_expert.gate_up_proj.weight"].shape == (256, 64)
    assert state["model.layers.0.mlp.shared_expert.down_proj.weight"].shape == (64, 128)
    assert state["model.layers.0.mlp.shared_expert_gate.weight"].shape == (1, 64)


def test_qwen2_moe_weight_stream_fuses_qkv_shared_and_experts():
    from freetoken.models.qwen2_moe.weight import _finish_experts

    tensors = [
        ("model.layers.0.self_attn.q_proj.weight", torch.full((2, 2), 1.0)),
        ("model.layers.0.self_attn.k_proj.weight", torch.full((2, 2), 2.0)),
        ("model.layers.0.self_attn.v_proj.weight", torch.full((2, 2), 3.0)),
        ("model.layers.0.mlp.shared_expert.gate_proj.weight", torch.full((2, 2), 4.0)),
        ("model.layers.0.mlp.shared_expert.up_proj.weight", torch.full((2, 2), 5.0)),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.full((2, 2), 10.0)),
        ("model.layers.0.mlp.experts.0.up_proj.weight", torch.full((2, 2), 11.0)),
        ("model.layers.0.mlp.experts.1.gate_proj.weight", torch.full((2, 2), 20.0)),
        ("model.layers.0.mlp.experts.1.up_proj.weight", torch.full((2, 2), 21.0)),
        ("model.layers.0.mlp.experts.0.down_proj.weight", torch.full((2, 2), 12.0)),
        ("model.layers.0.mlp.experts.1.down_proj.weight", torch.full((2, 2), 22.0)),
    ]

    packed = dict(_finish_experts(iter(tensors), num_experts=2))

    assert packed["model.layers.0.self_attn.qkv_proj.weight"][:, 0].tolist() == [
        1.0, 1.0, 2.0, 2.0, 3.0, 3.0
    ]
    assert packed["model.layers.0.mlp.shared_expert.gate_up_proj.weight"][:, 0].tolist() == [
        4.0, 4.0, 5.0, 5.0
    ]
    assert packed["model.layers.0.mlp.experts.gate_up_proj"].shape == (2, 4, 2)
    assert packed["model.layers.0.mlp.experts.down_proj"].shape == (2, 2, 2)


def test_qwen2_moe_accepts_current_hf_packed_experts():
    from freetoken.models.qwen2_moe.weight import _finish_experts, _normalize_and_shard

    gate_up = torch.arange(2 * 8 * 4).reshape(2, 8, 4)
    down = torch.arange(2 * 4 * 4).reshape(2, 4, 4)
    name_gu, local_gu = _normalize_and_shard(
        "model.layers.0.mlp.experts.gate_up_proj",
        gate_up,
        rank=0,
        world_size=1,
        num_kv_heads=1,
    )
    name_dn, local_dn = _normalize_and_shard(
        "model.layers.0.mlp.experts.down_proj.weight",
        down,
        rank=0,
        world_size=1,
        num_kv_heads=1,
    )

    packed = dict(_finish_experts(iter(((name_gu, local_gu), (name_dn, local_dn))), 2))
    torch.testing.assert_close(packed[name_gu], gate_up)
    torch.testing.assert_close(packed[name_dn], down)


def test_qwen2_moe_shards_current_hf_packed_experts_by_projection():
    from freetoken.models.qwen2_moe.weight import _normalize_and_shard

    gate_up = torch.arange(2 * 8 * 4).reshape(2, 8, 4)
    down = torch.arange(2 * 4 * 8).reshape(2, 4, 8)
    _, gu_rank1 = _normalize_and_shard(
        "model.layers.0.mlp.experts.gate_up_proj",
        gate_up,
        rank=1,
        world_size=2,
        num_kv_heads=1,
    )
    _, dn_rank1 = _normalize_and_shard(
        "model.layers.0.mlp.experts.down_proj",
        down,
        rank=1,
        world_size=2,
        num_kv_heads=1,
    )
    gate, up = gate_up.chunk(2, dim=1)
    expected_gu = torch.cat((gate.chunk(2, dim=1)[1], up.chunk(2, dim=1)[1]), dim=1)
    torch.testing.assert_close(gu_rank1, expected_gu)
    torch.testing.assert_close(dn_rank1, down.chunk(2, dim=2)[1])


def test_qwen2_moe_combines_routed_and_gated_shared(monkeypatch):
    from freetoken.models.qwen2_moe.config import parse_config
    from freetoken.models.qwen2_moe.moe import Qwen2MoeMLP

    _ensure_tp1()
    config = parse_config(_hf_config())
    object.__setattr__(config, "moe_backend", "offload")
    with torch.device("meta"):
        mlp = Qwen2MoeMLP(config, layer_id=0)

    hidden = torch.zeros(2, 64)
    mlp.gate.forward = lambda _x: torch.zeros(2, 3)
    mlp.shared_expert.forward = lambda _x: torch.full((2, 64), 4.0)
    mlp.shared_expert_gate.forward = lambda _x: torch.zeros(2, 1)
    mlp.experts.forward = lambda _x, _router: torch.full((2, 64), 3.0)

    torch.testing.assert_close(mlp.forward(hidden), torch.full((2, 64), 5.0))


def test_qwen2_moe_triton_router_matches_full_softmax_topk():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("requires a CUDA or ROCm GPU")
    from freetoken.moe.fused import _triton_fused_topk

    # Unique, exactly representable values avoid backend-specific tie ordering.
    base = torch.arange(60, device="cuda", dtype=torch.float32) / 16
    logits = torch.stack([torch.roll(base, shifts=7 * i) for i in range(5)]).to(
        torch.bfloat16
    )
    weights, ids = _triton_fused_topk(logits, 4, False, None)
    probs = torch.softmax(logits.float(), dim=-1)
    ref_weights, ref_ids = torch.topk(probs, 4, dim=-1)

    torch.testing.assert_close(ids, ref_ids.to(torch.int32))
    torch.testing.assert_close(weights, ref_weights, rtol=2e-5, atol=2e-6)


def test_qwen2_moe_triton_router_renormalizes_and_masks_padding():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("requires a CUDA or ROCm GPU")
    from freetoken.moe.fused import _triton_fused_topk

    logits = torch.arange(4 * 128, device="cuda", dtype=torch.float32).reshape(4, 128)
    logits = (logits % 127 - logits // 128).to(torch.bfloat16)
    limit = torch.tensor(2, dtype=torch.int32, device="cuda")
    weights, ids = _triton_fused_topk(logits, 8, True, limit)
    probs = torch.softmax(logits[:2].float(), dim=-1)
    ref_weights, ref_ids = torch.topk(probs, 8, dim=-1)
    ref_weights /= ref_weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(ids[:2], ref_ids.to(torch.int32))
    torch.testing.assert_close(weights[:2], ref_weights, rtol=2e-5, atol=2e-6)
    assert ids[2:].eq(-1).all()
    assert weights[2:].eq(0).all()


def test_qwen2_moe_triton_router_graph_replay_reads_new_logits():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("requires a CUDA or ROCm GPU")
    from freetoken.moe.fused import _triton_fused_topk

    logits = torch.arange(60, device="cuda", dtype=torch.float32).view(1, 60).to(
        torch.bfloat16
    )
    out_weights = torch.empty(1, 4, dtype=torch.float32, device="cuda")
    out_ids = torch.empty(1, 4, dtype=torch.int32, device="cuda")

    def call() -> None:
        weights, ids = _triton_fused_topk(logits, 4, False, None)
        out_weights.copy_(weights)
        out_ids.copy_(ids)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()

    logits.copy_(torch.flip(logits, dims=(1,)))
    graph.replay()
    torch.cuda.synchronize()
    ref_weights, ref_ids = torch.topk(torch.softmax(logits.float(), dim=-1), 4, dim=-1)

    torch.testing.assert_close(out_ids, ref_ids.to(torch.int32))
    torch.testing.assert_close(out_weights, ref_weights, rtol=2e-5, atol=2e-6)

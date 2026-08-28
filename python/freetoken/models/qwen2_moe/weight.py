from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import (
    MergeRule,
    iter_merged_tensors,
    iter_stacked_experts,
    iter_weight_files,
    shard_tensor,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_EXPERT_PATTERN = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$"
)
_PACKED_EXPERT_PATTERN = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<name>gate_up_proj|down_proj)(?:\.weight)?$"
)
_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}


def _is_expert_weight(name: str) -> bool:
    return _EXPERT_PATTERN.match(name) is not None or _PACKED_EXPERT_PATTERN.match(name) is not None


def _normalize_and_shard(
    name: str,
    tensor: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_kv_heads: int,
) -> tuple[str, torch.Tensor]:
    """Normalize old per-expert and current HF packed-expert checkpoint layouts."""
    packed = _PACKED_EXPERT_PATTERN.match(name)
    if packed is None:
        return name, shard_tensor(
            name,
            tensor,
            rank=rank,
            world_size=world_size,
            num_kv_heads=num_kv_heads,
        )

    normalized = f"{packed.group('prefix')}.{packed.group('name')}"
    if world_size == 1:
        return normalized, tensor
    if packed.group("name") == "gate_up_proj":
        if tensor.dim() != 3 or tensor.shape[1] % 2:
            raise ValueError(f"Unexpected packed gate_up shape: {tuple(tensor.shape)}")
        gate, up = tensor.chunk(2, dim=1)
        gate_local = gate.chunk(world_size, dim=1)[rank]
        up_local = up.chunk(world_size, dim=1)[rank]
        return normalized, torch.cat((gate_local, up_local), dim=1).clone()
    if tensor.dim() != 3:
        raise ValueError(f"Unexpected packed down shape: {tuple(tensor.shape)}")
    return normalized, tensor.chunk(world_size, dim=2)[rank].clone()


def _finish_experts(
    tensors: Iterator[tuple[str, torch.Tensor]], num_experts: int
) -> Iterator[tuple[str, torch.Tensor]]:
    merged = iter_merged_tensors(tensors, _MERGE_RULES, model_name="qwen2_moe")
    yield from iter_stacked_experts(
        merged,
        num_experts=num_experts,
        model_name="qwen2_moe",
        expert_pattern=_EXPERT_PATTERN,
    )


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()

    def sharded_tensors() -> Iterator[tuple[str, torch.Tensor]]:
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading weights",
            disable=not tp_info.is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    name = raw_name.removeprefix("language_model.")
                    is_expert = _is_expert_weight(name)
                    if is_expert and not include_moe_experts:
                        continue
                    if not is_expert and not include_non_moe:
                        continue
                    raw = f.get_tensor(raw_name)
                    name, tensor = _normalize_and_shard(
                        name,
                        raw,
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_kv_heads=config.num_kv_heads,
                    )
                    del raw
                    yield name, tensor

    yield from _finish_experts(sharded_tensors(), config.num_experts)


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert include_moe_experts and not include_non_moe
    from freetoken.models.weight import iter_expert_tensors_parallel

    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()

    def is_expert(raw_name: str) -> bool:
        return _is_expert_weight(raw_name.removeprefix("language_model."))

    def raw_experts() -> Iterator[tuple[str, torch.Tensor]]:
        for raw_name, raw in iter_expert_tensors_parallel(
            model_path, is_expert, workers=workers, chunk=chunk
        ):
            name = raw_name.removeprefix("language_model.")
            name, tensor = _normalize_and_shard(
                name,
                raw,
                rank=tp_info.rank,
                world_size=tp_info.size,
                num_kv_heads=config.num_kv_heads,
            )
            yield name, tensor

    yield from _finish_experts(raw_experts(), config.num_experts)


__all__ = ["iter_weights", "iter_weights_parallel"]

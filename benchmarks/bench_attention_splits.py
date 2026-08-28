"""Sweep split-K decode attention on the current CUDA or ROCm GPU."""

from __future__ import annotations

import argparse

import torch

from freetoken.benchmark.perf import perf_cuda
from freetoken.kernel.triton.attention import (
    _gfx1101_decode_split_policy,
    decode_paged_attention,
)
from freetoken.utils.arch import get_rocm_gfx_arch


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(item) for item in value.split(",") if item.strip()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lengths", type=_csv_ints, default=(128, 1024, 4096, 16384))
    parser.add_argument("--splits", type=_csv_ints, default=(1, 2, 4, 8, 16))
    parser.add_argument(
        "--max-splits",
        type=int,
        default=16,
        help="fixed launch/scratch split capacity (default 16)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="apply the production gfx1101 active-split policy instead of a raw sweep",
    )
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    assert args.q_heads % args.kv_heads == 0
    assert torch.cuda.is_available(), "a CUDA or ROCm GPU is required"
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = args.batch_size
    max_len = max(args.lengths)
    q = torch.randn(batch, args.q_heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(batch * max_len, args.kv_heads, args.head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    q_positions = torch.empty(batch, dtype=torch.int64, device=device)
    out = torch.empty_like(q)

    print(
        f"gpu={torch.cuda.get_device_name(device)} qh={args.q_heads} "
        f"kvh={args.kv_heads} d={args.head_dim} batch={batch} graph=yes"
    )
    print("length requested capacity effective latency_us")
    split_limit, short_cap = _gfx1101_decode_split_policy(
        batch, args.q_heads, args.kv_heads, args.head_dim
    )
    for length in args.lengths:
        indptr = torch.arange(
            0, (batch + 1) * length, length, dtype=torch.int32, device=device
        )
        case_indices = torch.cat(
            [
                torch.arange(i * max_len, i * max_len + length, dtype=torch.int32, device=device)
                for i in range(batch)
            ]
        )
        q_positions.fill_(length - 1)
        for splits in args.splits:
            max_splits = args.max_splits or splits
            if splits > max_splits:
                continue
            mid_o = torch.empty(
                batch, args.q_heads, max_splits, args.head_dim, dtype=torch.float32, device=device
            )
            mid_lse = torch.empty(
                batch, args.q_heads, max_splits, dtype=torch.float32, device=device
            )
            split_counts = torch.full((batch,), splits, dtype=torch.int32, device=device)

            def call() -> None:
                decode_paged_attention(
                    q,
                    k,
                    v,
                    indptr,
                    case_indices,
                    q_positions,
                    mid_o,
                    mid_lse,
                    split_counts,
                    max_splits,
                    args.head_dim ** -0.5,
                    out=out,
                    enable_gfx1101_adaptive_splits=args.adaptive,
                )

            latency_ms = perf_cuda(
                call,
                repetitions=args.repetitions,
                cuda_graph_repetitions=10,
            )
            effective = splits
            if (
                args.adaptive
                and get_rocm_gfx_arch() == "gfx1101"
                and max_splits > 8
                and length < split_limit
            ):
                effective = min(effective, short_cap)
            print(
                f"{length:6d} {splits:9d} {max_splits:8d} "
                f"{effective:9d} {latency_ms * 1000:10.2f}"
            )


if __name__ == "__main__":
    main()

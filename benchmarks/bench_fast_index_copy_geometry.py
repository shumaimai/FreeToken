"""Sweep fused expert-copy launch geometry on the current GPU.

Run after ``ft bench bw`` has verified the basic pinned-memory path::

    python benchmarks/bench_fast_index_copy_geometry.py --format nvfp4
"""

from __future__ import annotations

import argparse
import statistics

import torch

from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
from freetoken.moe.benchbw import DTYPE_WORKLOADS, _build_gather_rig, _expert_bytes


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(item) for item in value.split(",") if item.strip()))


def _time_copy(cache, *, num_threads: int, blocks_per_bank: int, repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def launch() -> None:
        fast_index_copy_multi_jit(
            cache._copy_dst_ptrs,
            cache._copy_src_ptrs[0],
            cache._copy_feat_bytes,
            cache.evict_slots,
            cache.src_indices,
            cache.num_indices,
            num_threads=num_threads,
            blocks_per_bank=blocks_per_bank,
        )

    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=tuple(DTYPE_WORKLOADS), default="nvfp4")
    parser.add_argument("--threads", type=_csv_ints, default=(128, 256, 512, 1024))
    parser.add_argument("--blocks", type=_csv_ints, default=(2, 4, 8, 16))
    parser.add_argument("--misses", type=_csv_ints, default=(0, 1, 4, 8, 16))
    parser.add_argument("--repeat", type=int, default=21)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "a CUDA or ROCm GPU is required"
    device = torch.device("cuda")
    workload = DTYPE_WORKLOADS[args.format]
    cache, _ = _build_gather_rig(args.format, workload, device)
    expert_bytes = _expert_bytes(args.format, workload.hidden, workload.inter)
    misses = tuple(min(cache.num_experts, value) for value in args.misses)

    print(
        f"gpu={torch.cuda.get_device_name(device)} format={args.format} "
        f"banks={len(cache.banks)} expert={expert_bytes / 2**20:.2f} MiB"
    )
    print("threads blocks misses median_us effective_GBps")
    for threads in args.threads:
        for blocks in args.blocks:
            for count in misses:
                cache.num_indices.fill_(count)
                ms = _time_copy(
                    cache,
                    num_threads=threads,
                    blocks_per_bank=blocks,
                    repeat=args.repeat,
                )
                gbs = count * expert_bytes / (ms * 1e6) if count else 0.0
                print(f"{threads:7d} {blocks:6d} {count:6d} {ms * 1000:9.2f} {gbs:14.2f}")


if __name__ == "__main__":
    main()

"""Sweep GPT-OSS MXFP4 split-K decode geometry on the current GPU."""

from __future__ import annotations

import argparse

import torch

from freetoken.benchmark.perf import perf_cuda
from freetoken.kernel import mxfp4_splitk_gemv_triton


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(item) for item in value.split(",") if item.strip()))


def _weights(experts: int, n: int, k: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    blocks = torch.randint(
        0, 256, (experts, k // 2, n), dtype=torch.uint8, device="cuda", generator=gen
    )
    scales = torch.full((experts, k // 32, n), 127, dtype=torch.uint8, device="cuda")
    bias = torch.randn(experts, n, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.01
    return blocks, scales, bias


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gate", "down"), default="gate")
    parser.add_argument("--splits", type=_csv_ints, default=(8, 12, 15, 18, 24, 30, 36, 45, 60, 90))
    parser.add_argument("--block-n", type=_csv_ints, default=(32, 64, 128))
    parser.add_argument("--warps", type=_csv_ints, default=(1, 2, 4))
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "a CUDA or ROCm GPU is required"
    experts, routes, hidden, intermediate = 32, 4, 2880, 2880
    n = 2 * intermediate if args.stage == "gate" else hidden
    k = hidden if args.stage == "gate" else intermediate
    blocks, scales, bias = _weights(experts, n, k, seed=31 if args.stage == "gate" else 32)
    x = torch.randn(routes, k, dtype=torch.bfloat16, device="cuda") * 0.1
    expert_ids = torch.tensor([0, 7, 19, 31], dtype=torch.int64, device="cuda")
    expert_wts = (
        None
        if args.stage == "gate"
        else torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32, device="cuda")
    )

    reference = mxfp4_splitk_gemv_triton(
        x,
        blocks,
        scales,
        bias,
        expert_ids,
        N=n,
        K=k,
        stride_xe=x.stride(0),
        num_splits=45 if args.stage == "gate" else 18,
        expert_wts=expert_wts,
    ).clone()
    torch.cuda.synchronize()

    print(
        f"gpu={torch.cuda.get_device_name()} stage={args.stage} routes={routes} "
        f"N={n} K={k}"
    )
    print("splits block_n warps latency_us max_abs_diff")
    for splits in args.splits:
        for block_n in args.block_n:
            for warps in args.warps:
                partial = torch.empty(
                    routes * splits, n, dtype=torch.float32, device="cuda"
                )
                out = torch.empty(routes, n, dtype=torch.bfloat16, device="cuda")

                def call() -> None:
                    mxfp4_splitk_gemv_triton(
                        x,
                        blocks,
                        scales,
                        bias,
                        expert_ids,
                        N=n,
                        K=k,
                        stride_xe=x.stride(0),
                        num_splits=splits,
                        partial=partial,
                        out=out,
                        expert_wts=expert_wts,
                        block_n=block_n,
                        num_warps=warps,
                    )

                call()
                torch.cuda.synchronize()
                max_diff = (out.float() - reference.float()).abs().max().item()
                torch.testing.assert_close(out.float(), reference.float(), rtol=6e-2, atol=6e-2)
                latency_ms = perf_cuda(
                    call,
                    repetitions=args.repetitions,
                    cuda_graph_repetitions=10,
                )
                print(
                    f"{splits:6d} {block_n:7d} {warps:5d} "
                    f"{latency_ms * 1000:10.2f} {max_diff:12.5f}"
                )


if __name__ == "__main__":
    main()

# RX 7800 XT ROCm 10 results

Validated stack:

- Radeon RX 7800 XT (`gfx1101`)
- Ryzen 7 5800X3D
- 32 GB physical RAM; WSL configured for 29 GB RAM and 16 GB swap
- ROCm package 10.0.0, HIP ABI 7.15.26333
- PyTorch 2.13.0+rocm10.0.0
- AMD Triton 3.8.0+git4cff872c.rocm10.0.0
- Ubuntu 24.04 under WSL2

The ROCm 10 stack was installed in a separate virtual environment. System ROCm
7.2.4 and its previous FreeToken environment were not upgraded or removed.

## Correctness

The test suite was completed in three groups to isolate GPU JIT compilation:

- Core, engine, models, cache, scheduler, server, tokenizer: 1,084 passed, 4 skipped
- Kernels: 214 passed, 20 skipped
- MoE: 109 passed, 12 skipped
- Total: 1,407 passed, 36 skipped

ROCm 10's TheRock SDK does not ship `rocm_agent_enumerator`. FreeToken now passes
the architecture it already detects from PyTorch to TVM-FFI, so JIT compilation
works without manually setting `TVM_FFI_ROCM_ARCH_LIST`.

AMD Triton 3.8 could spend indefinitely compiling several grouped MoE kernels,
and the DeepSeek FP4 grouped kernel was sequence-dependent after a long-context
attention launch. On ROCm 10/gfx1101, BF16, DeepSeek FP4, MXFP4, and NVFP4
prefill therefore use their graph-safe route kernels in 256-token chunks. CUDA,
other GPUs, and older ROCm/Triton stacks keep their grouped paths. The
long-context-attention then DeepSeek-FP4 sequence test passes with the fallback.

## Kernel tuning

All times are HIP graph replay latency on the same RX 7800 XT.

- Grouped decode attention, 16 Q heads, 2 KV heads, head dimension 256,
  128-token context: 29.06 us with the old 4-split cap, 15.66 us with all 16
  splits. The ROCm 10 policy is 1.86x faster for this shape.
- GPT-OSS MXFP4 production decode path: 107.13 us with the old geometry and
  94.08 us with the ROCm 10 geometry, a 12.2% latency reduction.
- DSV4 sparse attention retained one pipeline stage: 167.71 us versus 194.39 us
  with two stages.
- DSV4 retained separate window/compressed gather loops: 165.05 us versus
  192.12 us for the unified gather.
- Expert copy retained 1,024 threads x 8 blocks per bank. The best sweep result
  was within about 2% and did not justify changing the stable default.

## GPT-OSS-20B

Measured through the real OpenAI-compatible streaming server path, greedy,
single stream, 96 prompt tokens, 127 completion tokens. The server exposes
visible text deltas rather than token-indexed timestamps, so the rate below is
visible SSE events per second, not exact token/s:

- Visible event rate: 80.45 event/s
- First visible text: 128.5 ms
- VRAM: 13.66 GiB
- Output SHA-1 prefix: `5f2f92156c65`

One forced `TORCH_BLAS_PREFER_HIPBLASLT=1` run produced the same output at
78.69 visible event/s and 218.2 ms to first visible text. This single-run
difference is not enough to establish a general performance regression. The
launch profiles do not set it globally; evaluate it with repeated workload-
specific measurements.

## Bandwidth profile

The ROCm 10 `ft bench bw --dtype bf16,mxfp4` result:

- CPU STREAM read: 23.89 GB/s
- PCIe linear H2D: 24.62 GB/s
- PCIe linear D2H: 22.90 GB/s
- BF16 CPU-MoE / PCIe gather: 22.5 / 20.2 GB/s, recommends offload
- MXFP4 CPU-MoE / PCIe gather: 20.9 / 20.2 GB/s, recommends offload

Profiles now include Torch, Triton, and ROCm package versions. A profile from a
different stack is ignored until `ft bench bw` is rerun.

## Native Windows ROCm 10

A separate native Windows ROCm 10 workspace was also checked. Its optimized HIP
extension and PyTorch smoke tests passed, but native Windows ROCm does not ship
Triton. FreeToken therefore remains a WSL/Linux runtime. The native measurements
also showed no general benefit from globally forcing hipBLASLt and confirmed the
value of 16-byte vectorized copies, which FreeToken's expert-copy kernel already
uses through `uint4` loads and stores.

## Docker safety

The 28.63 GB Qwen1.5-MoE host-offload test created enough WSL memory pressure to
restart Docker Desktop indirectly, even though no Docker or WSL stop command was
issued. The Qwen launch profile now refuses to start while a Docker container is
running unless `FREETOKEN_ALLOW_DOCKER_MEMORY_PRESSURE=1` is explicitly set.

Machine-readable result summaries, including visible event counts and timing
windows, are in
[`benchmarks/results/rx7800xt-rocm10.json`](../benchmarks/results/rx7800xt-rocm10.json).

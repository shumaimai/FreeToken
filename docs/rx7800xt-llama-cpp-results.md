# RX 7800 XT external llama.cpp experiments

These measurements are included as hardware feasibility records. They use a
separate `llama.cpp` build and do not measure FreeToken or a FreeToken code
path. Model weights are not stored in this repository.

Hardware and runtime:

- Radeon RX 7800 XT, 16 GB (`gfx1101`)
- Ryzen 7 5800X3D
- 32 GB physical RAM; WSL configured for 29 GB RAM and 16 GB swap
- ROCm 7.2.4
- `llama.cpp` commit `511f9c1379a52516f328859af86daa124eddc717`
- `HSA_ENABLE_DXG_DETECTION=1`

## Qwen3.8-Flash-Next

The `AtomicChat/Qwen3.8-Flash-Next-GGUF` 3.84 bpw build contains 84.93 GB of
GGUF shards. Its isolated 38.40 GB PLE n-gram shard remained SSD-backed through
lazy mmap, allowing the model to run with 14 layers on the GPU and a 512-token
context.

Observed over three uncached streaming runs:

- Median TTFT: 15.10 seconds
- Median prompt rate: 1.58 token/s
- Median decode rate: 1.09 token/s
- Decode range: 0.36 to 1.23 token/s
- Process RAM high-water mark: 25.64 GiB
- VRAM use: 14.26 GiB
- Process swap: 0

The model produced a valid chat response, but SSD page-in caused large latency
variance. The full raw record is
[`benchmarks/results/rx7800xt-qwen38-flash-next-ad384.json`](../benchmarks/results/rx7800xt-qwen38-flash-next-ad384.json).

## Qwen3.8-27B

Two local GGUF configurations were measured:

- A 12.60 GB low-bit local artifact ran fully offloaded at 23.40 token/s decode,
  595.56 token/s `pp512`, and 0.679-second median TTFT. Its SHA-256 is recorded,
  but its original distribution repository was not retained; treat the result
  as a hardware datapoint rather than a reproducible model recommendation.
- `unsloth/Qwen3.8-27B-GGUF` `UD-Q8_K_XL`, 31.46 GB, ran with partial offload at
  1.21 token/s median decode and 3.74-second median TTFT.

The raw record is
[`benchmarks/results/rx7800xt-qwen38-27b-gguf.json`](../benchmarks/results/rx7800xt-qwen38-27b-gguf.json).

## Scope

These records show what the hardware and ROCm/WSL memory path could load. They
must not be compared directly with the FreeToken Qwen1.5-MoE result because the
models, quantization, runtime, context, and placement policy differ.

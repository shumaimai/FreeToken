# RX 7800 XT unquantized offload results (ROCm 7.2.4 historical baseline)

For the current ROCm 10 stack and resident GPT-OSS measurement, see
[`rx7800xt-rocm10-results.md`](rx7800xt-rocm10-results.md).

Hardware and runtime:

- Radeon RX 7800 XT, 16 GB (`gfx1101`, 60 CUs, wave32)
- Ryzen 7 5800X3D, 8 cores / 16 threads
- 32 GB physical RAM; WSL configured for 29 GB RAM and 16 GB swap
- ROCm 7.2.4, PyTorch 2.11, AMD Triton 3.7

Qwen3.8-27B BF16 was rejected before download/serve testing as an impossible
placement. The checkpoint is 55.56 GB decimal (51.75 GiB), has no routed
experts, and therefore cannot use FreeToken's expert offload. The text weights
alone exceed host RAM plus VRAM before KV, graph, and runtime allocations.

The practical official unquantized target was Qwen1.5-MoE-A2.7B-Chat:

- Checkpoint: 28.63 GB decimal (26.67 GiB)
- Routed BF16 expert bank in host memory: 23.20 GiB
- Dense/shared BF16 trunk on GPU: 3.46 GiB
- Final placement: 10 CPU MoE layers, 14 GPU-offload MoE layers
- Unified GPU expert cache: 525 slots; 17,301,504 bytes per slot
- KV cache: 2,048 tokens
- API-reported PyTorch-reserved VRAM: 13,430,161,408 bytes (12.51 GiB)
- Host use during serving: about 27 GiB plus about 2.4 GiB swap
- HIP graph: batch size 1

Measurement method:

- OpenAI streaming API, greedy generation, `ignore_eos=true`
- A 512-token completion budget was requested; API usage and visible SSE events both reported 511 tokens
- Decode rate excludes TTFT and uses the 510 first-to-last visible-token intervals
- Three repeated warm runs with identical output hashes

Results:

- Decode: 14.47, 14.64, and 14.65 token/s
- Median decode: 14.64 token/s
- Tail after the first 64 tokens: 14.75 token/s median
- Tail after the first 128 tokens: 14.73 token/s median
- Warm TTFT for a 41-token prompt: 2.10 seconds median
- Startup prefill warmup: about 118 seconds

The conservative automatic WSL pin budget used 13 CPU / 11 GPU-offload layers
and stabilized around 12.5 token/s. A 14 GiB pin budget improved steady decode
to 14.64 token/s. Pin budgets of 15 and 16 GiB failed `cudaHostRegister` on this
WDDM/ROCDXG system.

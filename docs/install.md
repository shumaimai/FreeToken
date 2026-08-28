# Install

## Requirements

- Linux x86_64 with either:
  - NVIDIA GPU, driver r580+ (CUDA 13), or
  - AMD RDNA3/RDNA4 GPU (`gfx1100`-`gfx1103`, `gfx1200`, or `gfx1201`) with ROCm 7.2.4 or 7.14
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Method 1: Install from PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH.

### AMD ROCm source install (experimental)

Use an official ROCm PyTorch image whose PyTorch version satisfies the project's
`torch>=2.11,<2.12` constraint. For RDNA4, the matching ROCm 7.14 image is:

```bash
VIDEO_GID="$(getent group video | cut -d: -f3)"
RENDER_GID="$(getent group render | cut -d: -f3)"
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add="$VIDEO_GID" --group-add="$RENDER_GID" --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e PYTORCH_ROCM_ARCH=gfx1201 -e FREETOKEN_ROCM_ARCH=gfx1201 \
  -v "$PWD:/workspace/FreeToken" -w /workspace/FreeToken \
  rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0 bash
```

Inside the container, preserve the ROCm-enabled PyTorch already supplied by the
image and disable build isolation so it is also used to compile the extensions:

```bash
python -m pip install --no-build-isolation -e .
```

Set both architecture variables to `gfx1200` for RX 9060 family GPUs, or to the
actual target reported by `rocminfo`.

The optional native GGUF kernels also require Thrust headers. Install the generic
headers before the first GGUF kernel JIT build:

```bash
apt-get update && apt-get install -y --no-install-recommends libthrust-dev
```

#### RX 7800 XT on WSL2

The RX 7800 XT is `gfx1101`. ROCm 7.2.x under WSL also needs the official
ROCDXG bridge. Install `rocdxg-roct` from the
[librocdxg releases](https://github.com/ROCm/librocdxg/releases), then verify the
GPU before installing FreeToken:

```bash
export HSA_ENABLE_DXG_DETECTION=1
rocminfo | grep -E 'Name:.*gfx1101|Marketing Name:.*RX 7800 XT'
```

FreeToken sets this legacy HSA switch automatically when `/dev/dxg` and
`librocdxg.so` are present. It does not override an explicit user setting.

Use Python 3.12 and the ROCm 7.2.4 PyTorch and AMD Triton wheels that match the
system runtime; the direct wheel URLs below are CPython 3.12 builds. Do not
install the CUDA `accel` extra:

```bash
python3 -m venv ~/.venv-freetoken-rocm
source ~/.venv-freetoken-rocm/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torch-2.11.0%2Brocm7.2.4.lw.git5fbd98f3-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/triton-3.7.0%2Brocm7.2.4.gitb4e20bbe-cp312-cp312-linux_x86_64.whl'
sudo apt-get install -y --no-install-recommends \
  build-essential python3-dev cmake ninja-build libthrust-dev

export PYTORCH_ROCM_ARCH=gfx1101
export FREETOKEN_ROCM_ARCH=gfx1101
export TVM_FFI_ROCM_ARCH_LIST=gfx1101
export TORCH_EXTENSIONS_DIR="$HOME/.cache/freetoken/torch-ext-gfx1101"
export TRITON_CACHE_DIR="$HOME/.cache/freetoken/triton-gfx1101"
python -m pip install --no-build-isolation -e .
```

Calibrate the actual CPU, RAM, and PCIe path once. The profile is used by
`--moe-backend auto` on later runs:

```bash
ft bench bw --dtype all
```

For latency-focused single-stream serving on this 16 GiB card, retain HIP graph
capture at batch size 1:

```bash
ft serve --model /path/to/model --gpu 0 \
  --max-running-requests 1 --cuda-graph-max-bs 1
```

The exact `openai/gpt-oss-20b` MXFP4 checkpoint fits resident in 16 GiB. The
repository includes a latency-focused configuration that explicitly selects
the resident `fused` backend:

```bash
bash scripts/serve-gpt-oss-20b-rx7800xt.sh /path/to/gpt-oss-20b
```

Keep `auto` for larger MoE checkpoints that cannot fit all experts in VRAM.

#### Unquantized Qwen1.5-MoE on RX 7800 XT

The official Qwen1.5-MoE-A2.7B checkpoint is 28.63 GB in BF16. Its routed
experts occupy 23.20 GiB of host RAM, while the 3.46 GiB dense/shared trunk,
expert cache, KV cache, and graph state use the GPU. Configure WSL before
starting Docker or FreeToken:

```ini
[wsl2]
memory=29GB
swap=16GB
processors=16
vmIdleTimeout=-1
```

After applying that setting during a normal machine/Docker restart, use:

```bash
bash scripts/serve-qwen15-moe-rx7800xt.sh /path/to/Qwen1.5-MoE-A2.7B-Chat
```

The measured WDDM pin ceiling on this machine is 14 expert layers (about
13.54 GiB). The script uses a 14 GiB pin budget, leaving 10 MoE layers on the
CPU and 14 layers on GPU offload. A 15 GiB budget failed `cudaHostRegister`, so
raising the value is not recommended.

Do not force `PYTORCH_ALLOC_CONF=expandable_segments:True` on ROCm. PyTorch 2.11
can fail allocations after HIP graph capture with that CUDA-oriented allocator
mode; FreeToken leaves ROCm on its default allocator unless the user explicitly
sets an allocator configuration.

## Method 2: Install from source

```bash
git clone https://github.com/shumaimai/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## Verify

```bash
source .venv/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).

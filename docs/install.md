# Install

## Requirements

- Linux x86_64 with either:
  - NVIDIA GPU, driver r580+ (CUDA 13), or
  - AMD RDNA3/RDNA4 GPU (`gfx1100`-`gfx1103`, `gfx1200`, or `gfx1201`) with ROCm 10.0.0
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Method 1: Install from PyPI

The PyPI package is the upstream release, not this ROCm fork. For the ROCm 10
changes documented here, use the source installation below.

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH.

### AMD ROCm source install (experimental)

Use an official ROCm PyTorch environment whose PyTorch version satisfies the
project's `torch>=2.13,<2.14` constraint. The validated ROCm 10 stack is
PyTorch 2.13 and AMD Triton 3.8.

ROCm 10 native Windows PyTorch does not include Triton, so it cannot run
FreeToken's Triton attention and MoE kernels. Use WSL2 or Linux for FreeToken.
Native Windows remains useful for non-Triton PyTorch workloads.

For a containerized Linux installation, use AMD's matching ROCm 10 image and
set the target to the architecture reported by `rocminfo`:

```bash
group_args=()
getent group video >/dev/null && group_args+=(--group-add="$(getent group video | cut -d: -f3)")
getent group render >/dev/null && group_args+=(--group-add="$(getent group render | cut -d: -f3)")
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  "${group_args[@]}" --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e PYTORCH_ROCM_ARCH=gfx1201 -e FREETOKEN_ROCM_ARCH=gfx1201 \
  -v "$PWD:/workspace/FreeToken" -w /workspace/FreeToken \
  rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0@sha256:bbdaba66029f905321be3bbc95206a2d9a56a2bc5300877d146ad24d40db9a12 bash
```

The container uses ptrace and an unconfined seccomp profile for development;
run only trusted images and source trees.

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

The RX 7800 XT is `gfx1101`. Install the system prerequisites first:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3.12 python3.12-venv python3.12-dev \
  build-essential cmake ninja-build libthrust-dev
```

ROCm 10 normally detects WSL automatically. The RX 7800 XT launch profiles
default the legacy `HSA_ENABLE_DXG_DETECTION` switch to `1` and preserve an
explicit user setting.

Use Python 3.12 and AMD's ROCm 10 wheel stack in a new dedicated virtual
environment. `--index-url` applies to the whole Torch installation transaction,
not only the named packages; do not use that index for the later FreeToken
dependency transaction. Do not install the CUDA `accel` extra:

```bash
python3.12 -m venv ~/.venv-freetoken-rocm10
source ~/.venv-freetoken-rocm10/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  'torch[device-gfx1101]==2.13.0+rocm10.0.0' \
  'triton==3.8.0+git4cff872c.rocm10.0.0'

export HSA_ENABLE_DXG_DETECTION=1
rocminfo | grep -E 'Name:.*gfx1101|Marketing Name:.*RX 7800 XT'

export PYTORCH_ROCM_ARCH=gfx1101
export FREETOKEN_ROCM_ARCH=gfx1101
export TVM_FFI_ROCM_ARCH_LIST=gfx1101
export TORCH_EXTENSIONS_DIR="$HOME/.cache/freetoken/torch-ext-rocm10-gfx1101"
export TRITON_CACHE_DIR="$HOME/.cache/freetoken/triton-rocm10-gfx1101"
python -m pip install --no-build-isolation -e .
python -m pip check
```

Calibrate the actual CPU, RAM, and PCIe path once. The profile is used by
`--moe-backend auto` on later runs:

```bash
ft bench bw --dtype all
```

Profiles include the Torch, Triton, and ROCm package versions. A stack upgrade
invalidates the old profile and requires this command to be rerun.

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

Do not set `TORCH_BLAS_PREFER_HIPBLASLT=1` globally. One same-output GPT-OSS
comparison favored the default, but the single-run difference was too small to
generalize; test it repeatedly per workload instead.

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

The script refuses to load this 28.63 GB checkpoint while Docker containers are
running unless `FREETOKEN_ALLOW_DOCKER_MEMORY_PRESSURE=1` is explicitly set.
The combined host-memory pressure can otherwise restart Docker Desktop.

The measured WDDM pin ceiling on this machine is 14 expert layers (about
13.54 GiB). The script uses a 14 GiB pin budget, leaving 10 MoE layers on the
CPU and 14 layers on GPU offload. A 15 GiB budget failed `cudaHostRegister`, so
raising the value is not recommended.

Do not force `PYTORCH_ALLOC_CONF=expandable_segments:True` on ROCm. FreeToken
keeps ROCm on its default allocator for HIP graph compatibility unless the user
explicitly sets an allocator configuration.

## Method 2: Install this fork from source

After creating the ROCm 10 environment in the RX 7800 XT procedure above:

```bash
git clone https://github.com/shumaimai/FreeToken.git && cd FreeToken
source ~/.venv-freetoken-rocm10/bin/activate
python -m pip install --no-build-isolation -e .
```

## Verify

```bash
source ~/.venv-freetoken-rocm10/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).

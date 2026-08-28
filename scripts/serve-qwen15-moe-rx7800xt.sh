#!/usr/bin/env bash
set -euo pipefail

model="${1:-$HOME/qwen15-moe}"
if [[ $# -gt 0 ]]; then
  shift
fi

python_bin="${FREETOKEN_PYTHON:-$HOME/.venv-freetoken-rocm/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "FreeToken ROCm Python not found: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$model/model.safetensors.index.json" ]]; then
  echo "Qwen1.5-MoE checkpoint not found: $model" >&2
  exit 1
fi

mem_kib="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
swap_free_kib="$(awk '/SwapFree:/ {print $2}' /proc/meminfo)"
if (( mem_kib < 28 * 1024 * 1024 )); then
  echo "Qwen1.5-MoE BF16 needs at least 28 GiB visible WSL RAM; found $((mem_kib / 1024 / 1024)) GiB." >&2
  echo "Set memory=29GB and swap=16GB in %USERPROFILE%/.wslconfig before starting it." >&2
  exit 1
fi
if (( available_kib < 24 * 1024 * 1024 || available_kib + swap_free_kib < 30 * 1024 * 1024 )); then
  echo "Insufficient free backing memory for the 23.20 GiB expert bank." >&2
  echo "Need at least 24 GiB MemAvailable and 30 GiB MemAvailable+SwapFree; found $((available_kib / 1024 / 1024)) GiB + $((swap_free_kib / 1024 / 1024)) GiB swap." >&2
  exit 1
fi

export FREETOKEN_PIN_BUDGET_GB="${FREETOKEN_PIN_BUDGET_GB:-14}"
export PYTORCH_ROCM_ARCH=gfx1101
export FREETOKEN_ROCM_ARCH=gfx1101
export TVM_FFI_ROCM_ARCH_LIST=gfx1101
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/freetoken/torch-ext-gfx1101}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/freetoken/triton-gfx1101}"

exec "$python_bin" -m freetoken.cli serve \
  --model "$model" \
  --gpu 0 \
  --moe-backend offload \
  --expert-load serial \
  --max-running-requests 1 \
  --max-seq-len-override 2048 \
  --num-tokens 2048 \
  --cuda-graph-max-bs 1 \
  --memory-ratio 0.9 \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

model="${1:-$HOME/gpt-oss-20b}"
if [[ $# -gt 0 ]]; then
  shift
fi

python_bin="${FREETOKEN_PYTHON:-$HOME/.venv-freetoken-rocm/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "FreeToken ROCm Python not found: $python_bin" >&2
  exit 1
fi
if [[ ! -d "$model" ]]; then
  echo "GPT-OSS checkpoint directory not found: $model" >&2
  exit 1
fi

export PYTORCH_ROCM_ARCH=gfx1101
export FREETOKEN_ROCM_ARCH=gfx1101
export TVM_FFI_ROCM_ARCH_LIST=gfx1101
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/freetoken/torch-ext-gfx1101}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/freetoken/triton-gfx1101}"

exec "$python_bin" -m freetoken.cli serve \
  --model "$model" \
  --gpu 0 \
  --moe-backend fused \
  --max-running-requests 1 \
  --max-seq-len-override 4096 \
  --num-tokens 4096 \
  --cuda-graph-max-bs 1 \
  --memory-ratio 0.9 \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

model="${1:-$HOME/qwen15-moe}"
if [[ $# -gt 0 ]]; then
  shift
fi

python_bin="${FREETOKEN_PYTHON:-$HOME/.venv-freetoken-rocm10/bin/python}"
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
allow_docker_pressure=false
case "${FREETOKEN_ALLOW_DOCKER_MEMORY_PRESSURE:-}" in
  1|true|TRUE|yes|YES|on|ON) allow_docker_pressure=true ;;
esac
if [[ "$allow_docker_pressure" == false ]]; then
  docker_running=false
  docker_client_found=false
  docker_query_failed=false
  # In WSL, the Windows client with the explicit desktop-linux context is the
  # authoritative Docker Desktop target. Fall back to the native client only
  # when docker.exe is unavailable (ordinary Linux hosts).
  if command -v docker.exe >/dev/null; then
    docker_client_found=true
    if docker_output="$(docker.exe --context desktop-linux ps -q 2>/dev/null)"; then
      [[ -n "$docker_output" ]] && docker_running=true
    else
      docker_query_failed=true
    fi
  elif command -v docker >/dev/null; then
    docker_client_found=true
    if docker_output="$(docker ps -q 2>/dev/null)"; then
      [[ -n "$docker_output" ]] && docker_running=true
    else
      docker_query_failed=true
    fi
  fi
  if [[ "$docker_running" == true || ("$docker_client_found" == true && "$docker_query_failed" == true) ]]; then
    echo "Refusing to load the 28.63 GB checkpoint while Docker may be active." >&2
    echo "Docker is running or could not be queried safely; this workload can restart Docker Desktop." >&2
    echo "Stop the containers yourself, or set FREETOKEN_ALLOW_DOCKER_MEMORY_PRESSURE=1." >&2
    exit 1
  fi
fi

export FREETOKEN_PIN_BUDGET_GB="${FREETOKEN_PIN_BUDGET_GB:-14}"
export PYTORCH_ROCM_ARCH=gfx1101
export FREETOKEN_ROCM_ARCH=gfx1101
export TVM_FFI_ROCM_ARCH_LIST=gfx1101
export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/freetoken/torch-ext-rocm10-gfx1101}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/freetoken/triton-rocm10-gfx1101}"

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

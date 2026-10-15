#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${1:-${MODEL_PATH:-}}"
output_mode="${MING_OUTPUT_MODE:-text}"
port="${MING_PORT:-8091}"

if [[ -z "$model_path" ]]; then
  echo "Usage: MODEL_PATH=/path/to/model $0 [model_path]" >&2
  exit 2
fi

if [[ ! -d "$model_path" ]]; then
  echo "Model directory does not exist: $model_path" >&2
  exit 2
fi

case "$output_mode" in
  text)
    deploy_config="$repo_dir/deploy/ming3_audio_flash_thinker_only.yaml"
    ;;
  audio)
    deploy_config="$repo_dir/deploy/ming3_audio_flash_with_talker.yaml"
    if [[ ! -d "$model_path/talker" ]]; then
      echo "Speech output requires the talker directory: $model_path/talker" >&2
      exit 2
    fi
    ;;
  *)
    echo "MING_OUTPUT_MODE must be 'text' or 'audio', got: $output_mode" >&2
    exit 2
    ;;
esac

export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0
export VLLM_USE_NCCL_SYMM_MEM=0

exec vllm serve "$model_path" \
  --omni \
  --deploy-config "$deploy_config" \
  --init-timeout "${MING_INIT_TIMEOUT:-1800}" \
  --stage-init-timeout "${MING_STAGE_INIT_TIMEOUT:-1800}" \
  --port "$port"

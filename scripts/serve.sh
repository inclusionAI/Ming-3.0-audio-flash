#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${1:-${MODEL_PATH:-}}"
output_mode="${MING_OUTPUT_MODE:-text}"
model_variant="${MING_MODEL_VARIANT:-}"
port="${MING_PORT:-8091}"

if [[ -z "$model_path" ]]; then
  echo "Usage: MODEL_PATH=/path/to/model $0 [model_path]" >&2
  exit 2
fi

if [[ ! -d "$model_path" ]]; then
  echo "Model directory does not exist: $model_path" >&2
  exit 2
fi

if [[ -z "$model_variant" ]]; then
  case "$(basename "$model_path")" in
    *tiny*) model_variant="tiny" ;;
    *) model_variant="flash" ;;
  esac
fi

case "$model_variant" in
  flash|tiny) ;;
  *)
    echo "MING_MODEL_VARIANT must be 'flash' or 'tiny', got: $model_variant" >&2
    exit 2
    ;;
esac

case "$model_variant:$output_mode" in
  flash:text)
    deploy_config="$repo_dir/deploy/ming3_audio_flash_thinker_only.yaml"
    ;;
  flash:asr)
    deploy_config="$repo_dir/deploy/ming3_audio_flash_asr.yaml"
    ;;
  flash:audio)
    deploy_config="$repo_dir/deploy/ming3_audio_flash_with_talker.yaml"
    ;;
  tiny:text)
    deploy_config="$repo_dir/deploy/ming3_audio_tiny_thinker_only.yaml"
    ;;
  tiny:asr)
    deploy_config="$repo_dir/deploy/ming3_audio_tiny_asr.yaml"
    ;;
  tiny:audio)
    deploy_config="$repo_dir/deploy/ming3_audio_tiny.yaml"
    ;;
  *)
    echo "MING_OUTPUT_MODE must be 'text', 'asr', or 'audio', got: $output_mode" >&2
    exit 2
    ;;
esac

case "$output_mode" in
  text)
    ;;
  asr|audio)
    if [[ ! -d "$model_path/talker" ]]; then
      echo "The selected profile requires the talker directory: $model_path/talker" >&2
      exit 2
    fi
    ;;
esac

export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0
export VLLM_USE_NCCL_SYMM_MEM=0
export VLLM_MAX_AUDIO_DECODE_DURATION_S="${VLLM_MAX_AUDIO_DECODE_DURATION_S:-3600}"

exec vllm serve "$model_path" \
  --omni \
  --deploy-config "$deploy_config" \
  --init-timeout "${MING_INIT_TIMEOUT:-1800}" \
  --stage-init-timeout "${MING_STAGE_INIT_TIMEOUT:-1800}" \
  --port "$port"

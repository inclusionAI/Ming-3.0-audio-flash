#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-${MODEL_PATH:-}}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ -z "$model_path" ]]; then
  echo "Usage: MODEL_PATH=/path/to/model $0 [MODEL_PATH] [extra arguments...]" >&2
  exit 2
fi
if [[ ! -d "$model_path" ]]; then
  echo "Model directory not found: $model_path" >&2
  exit 2
fi

python_bin="${SGLANG_PYTHON:-python3}"
if ! "$python_bin" -c "import sglang" >/dev/null 2>&1; then
  echo "SGLang is not installed for: $python_bin" >&2
  echo "Install it with: pip install -r requirements-sglang.txt" >&2
  exit 2
fi

variant="${MING_MODEL_VARIANT:-}"
if [[ -z "$variant" ]]; then
  if [[ "${model_path,,}" == *tiny* ]]; then
    variant="tiny"
  else
    variant="flash"
  fi
fi
if [[ "$variant" != "flash" && "$variant" != "tiny" ]]; then
  echo "MING_MODEL_VARIANT must be 'flash' or 'tiny', got: $variant" >&2
  exit 2
fi

host="${SGLANG_HOST:-127.0.0.1}"
port="${SGLANG_PORT:-8008}"
if [[ "$variant" == "flash" ]]; then
  served_model_name="${SGLANG_SERVED_MODEL_NAME:-audio-rope-test}"
  max_audio_items="${SGLANG_MAX_AUDIO_ITEMS:-99}"
else
  served_model_name="${SGLANG_SERVED_MODEL_NAME:-auto}"
  max_audio_items="${SGLANG_MAX_AUDIO_ITEMS:-1}"
fi

export PYTHONPATH="$model_path${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONFAULTHANDLER=1

common_args=(
  --model-path "$model_path"
  --served-model-name "$served_model_name"
  --host "$host"
  --port "$port"
  --trust-remote-code
  --enable-multimodal
  --limit-mm-data-per-request "{\"image\":0,\"video\":0,\"audio\":${max_audio_items}}"
  --dtype bfloat16
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.80}"
  --skip-server-warmup
)

if [[ "$variant" == "flash" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  audio_process_workers="${SGLANG_BAILING_AUDIO_PROCESS_WORKERS:-4}"
  export SGLANG_BAILING_AUDIO_PROCESS_WORKERS="$audio_process_workers"
  export SGLANG_BAILING_MAX_AUDIO_ITEMS="${SGLANG_BAILING_MAX_AUDIO_ITEMS:-99}"
  variant_args=(
    --tp-size "${SGLANG_TP_SIZE:-4}"
    --attention-backend "${SGLANG_ATTENTION_BACKEND:-fa3}"
    --max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-1}"
    --chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
    --max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS:-32768}"
    --context-length "${SGLANG_CONTEXT_LENGTH:-131072}"
    --disable-cuda-graph
    --weight-loader-drop-cache-after-load
  )
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export SGLANG_IO_WORKERS="${SGLANG_IO_WORKERS:-8}"
  variant_args=(
    --tp-size "${SGLANG_TP_SIZE:-1}"
    --pp-size 1
    --nnodes 1
    --context-length "${SGLANG_CONTEXT_LENGTH:-8192}"
    --max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-64}"
    --max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS:-16384}"
    --chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
    --tokenizer-worker-num "${SGLANG_TOKENIZER_WORKER_NUM:-1}"
    --enable-custom-logit-processor
  )
fi

printf 'Model:       %s\n' "$model_path"
printf 'Variant:     %s\n' "$variant"
printf 'Endpoint:    http://%s:%s/v1\n' "$host" "$port"
printf 'GPUs:        %s\n' "$CUDA_VISIBLE_DEVICES"
printf 'Model name:  %s\n' "$served_model_name"

exec "$python_bin" -u -m sglang.launch_server \
  "${common_args[@]}" \
  "${variant_args[@]}" \
  "$@"

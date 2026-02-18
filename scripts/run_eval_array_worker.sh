#!/usr/bin/env bash
set -euo pipefail

# AWS Batch/ECS-friendly worker wrapper.
# Runs three Ollama model families in parallel:
#   llama, mistral, qwen
# then merges outputs into a single JSON.
#
# Expected env vars (AWS Batch):
#   AWS_BATCH_JOB_ARRAY_INDEX, AWS_BATCH_JOB_ARRAY_SIZE
# Optional env vars:
#   EVAL_GPU_ID (defaults to 0 if unset)
#
# Script-specific options:
#   --combined-output <path>   merged output path (default: results_array_ollama.json)
#   --python-bin <path>        python executable (default: python3)
#   --keep-temporary           keep intermediate per-model json files
# All other args are forwarded to cpu_hotpot_qasper_grid_eval.py.

JOB_INDEX="${AWS_BATCH_JOB_ARRAY_INDEX:-0}"
JOB_COUNT="${AWS_BATCH_JOB_ARRAY_SIZE:-1}"
GPU_ID="${EVAL_GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COMBINED_OUTPUT="${COMBINED_OUTPUT:-results_array_ollama.json}"
KEEP_TEMPORARY=0
FORWARD_ARGS=()

while (($#)); do
  case "$1" in
    --combined-output)
      COMBINED_OUTPUT="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --keep-temporary)
      KEEP_TEMPORARY=1
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p "$(dirname "${COMBINED_OUTPUT}")"

LLAMA_TMP="$(mktemp)"
MISTRAL_TMP="$(mktemp)"
QWEN_TMP="$(mktemp)"

cleanup() {
  if [[ "${KEEP_TEMPORARY}" -eq 0 ]]; then
    rm -f "${LLAMA_TMP}" "${MISTRAL_TMP}" "${QWEN_TMP}"
  fi
}
trap cleanup EXIT

run_one() {
  local family="$1"
  local out="$2"
  "${PYTHON_BIN}" cpu_hotpot_qasper_grid_eval.py \
    --job-index "${JOB_INDEX}" \
    --job-count "${JOB_COUNT}" \
    --gpu-id "${GPU_ID}" \
    --answer-provider ollama \
    --hotpot-answer-model "${family}" \
    --output "${out}" \
    "${FORWARD_ARGS[@]}"
}

echo "Starting parallel Ollama runs: llama, mistral, qwen ..."
run_one "llama" "${LLAMA_TMP}" &
PID_LLAMA=$!
run_one "mistral" "${MISTRAL_TMP}" &
PID_MISTRAL=$!
run_one "qwen" "${QWEN_TMP}" &
PID_QWEN=$!

wait "${PID_LLAMA}"
wait "${PID_MISTRAL}"
wait "${PID_QWEN}"

echo "Merging outputs into ${COMBINED_OUTPUT}..."
"${PYTHON_BIN}" - "${LLAMA_TMP}" "${MISTRAL_TMP}" "${QWEN_TMP}" "${COMBINED_OUTPUT}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

llama_path = pathlib.Path(sys.argv[1])
mistral_path = pathlib.Path(sys.argv[2])
qwen_path = pathlib.Path(sys.argv[3])
out_path = pathlib.Path(sys.argv[4])

with llama_path.open("r", encoding="utf-8") as f:
    llama = json.load(f)
with mistral_path.open("r", encoding="utf-8") as f:
    mistral = json.load(f)
with qwen_path.open("r", encoding="utf-8") as f:
    qwen = json.load(f)

payload = {
    "config": {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {"answer_provider": "ollama", "model_family": "llama", "config": llama.get("config", {})},
            {"answer_provider": "ollama", "model_family": "mistral", "config": mistral.get("config", {})},
            {"answer_provider": "ollama", "model_family": "qwen", "config": qwen.get("config", {})},
        ],
    },
    "results": (
        list(llama.get("results", []))
        + list(mistral.get("results", []))
        + list(qwen.get("results", []))
    ),
}

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY

echo "Saved merged results: ${COMBINED_OUTPUT}"
if [[ "${KEEP_TEMPORARY}" -eq 1 ]]; then
  echo "Kept temporary files:"
  echo "  llama=${LLAMA_TMP}"
  echo "  mistral=${MISTRAL_TMP}"
  echo "  qwen=${QWEN_TMP}"
fi

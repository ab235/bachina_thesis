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
GPU_ID="${EVAL_GPU_ID:--1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COMBINED_OUTPUT="${COMBINED_OUTPUT:-results_array_ollama.json}"
KEEP_TEMPORARY=0
FORWARD_ARGS=()
AUTO_MERGE_ALL_SHARDS="${AUTO_MERGE_ALL_SHARDS:-1}"
MERGE_WAIT_SECONDS="${MERGE_WAIT_SECONDS:-1800}"
MERGE_POLL_SECONDS="${MERGE_POLL_SECONDS:-10}"
MODEL_FAMILY_PARALLELISM="${MODEL_FAMILY_PARALLELISM:-1}"

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

resolve_sharded_output() {
  local out="$1"
  "${PYTHON_BIN}" - "${out}" "${JOB_INDEX}" "${JOB_COUNT}" <<'PY'
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
job_index = int(sys.argv[2])
job_count = int(sys.argv[3])

if job_count > 1:
    out = out.with_name(f"{out.stem}.job{job_index:03d}-of-{job_count:03d}{out.suffix}")
print(out)
PY
}

list_expected_shards() {
  local out="$1"
  "${PYTHON_BIN}" - "${out}" "${JOB_COUNT}" <<'PY'
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
job_count = int(sys.argv[2])

for idx in range(job_count):
    shard = out.with_name(f"{out.stem}.job{idx:03d}-of-{job_count:03d}{out.suffix}")
    print(shard)
PY
}

LLAMA_OUT="$(resolve_sharded_output "${LLAMA_TMP}")"
MISTRAL_OUT="$(resolve_sharded_output "${MISTRAL_TMP}")"
QWEN_OUT="$(resolve_sharded_output "${QWEN_TMP}")"
COMBINED_SHARD_OUTPUT="$(resolve_sharded_output "${COMBINED_OUTPUT}")"

cleanup() {
  if [[ "${KEEP_TEMPORARY}" -eq 0 ]]; then
    rm -f "${LLAMA_TMP}" "${MISTRAL_TMP}" "${QWEN_TMP}" "${LLAMA_OUT}" "${MISTRAL_OUT}" "${QWEN_OUT}"
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

if [[ "${MODEL_FAMILY_PARALLELISM}" -ge 3 ]]; then
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
else
  echo "Starting serial Ollama runs (MODEL_FAMILY_PARALLELISM=${MODEL_FAMILY_PARALLELISM})..."
  run_one "llama" "${LLAMA_TMP}"
  run_one "mistral" "${MISTRAL_TMP}"
  run_one "qwen" "${QWEN_TMP}"
fi

echo "Merging model-family outputs into ${COMBINED_SHARD_OUTPUT}..."
"${PYTHON_BIN}" - "${LLAMA_OUT}" "${MISTRAL_OUT}" "${QWEN_OUT}" "${COMBINED_SHARD_OUTPUT}" <<'PY'
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

echo "Saved merged results: ${COMBINED_SHARD_OUTPUT}"

if [[ "${JOB_COUNT}" -gt 1 && "${AUTO_MERGE_ALL_SHARDS}" == "1" && "${JOB_INDEX}" -eq 0 ]]; then
  echo "Job 0 waiting for all shard files to produce final merge at ${COMBINED_OUTPUT}..."
  mapfile -t EXPECTED_SHARDS < <(list_expected_shards "${COMBINED_OUTPUT}")

  START_TS="$(date +%s)"
  while true; do
    MISSING=0
    for shard in "${EXPECTED_SHARDS[@]}"; do
      if [[ ! -s "${shard}" ]]; then
        MISSING=1
        break
      fi
    done
    if [[ "${MISSING}" -eq 0 ]]; then
      break
    fi

    NOW_TS="$(date +%s)"
    if (( NOW_TS - START_TS >= MERGE_WAIT_SECONDS )); then
      echo "Timed out waiting for all shard files after ${MERGE_WAIT_SECONDS}s; skipping final merge."
      break
    fi
    sleep "${MERGE_POLL_SECONDS}"
  done

  if [[ "${MISSING}" -eq 0 ]]; then
    "${PYTHON_BIN}" - "${COMBINED_OUTPUT}" "${EXPECTED_SHARDS[@]}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

out_path = pathlib.Path(sys.argv[1])
inputs = [pathlib.Path(p) for p in sys.argv[2:]]

all_results = []
runs = []
for path in inputs:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cfg = payload.get("config", {})
    runs.extend(list(cfg.get("runs", [])))
    all_results.extend(list(payload.get("results", [])))

final_payload = {
    "config": {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "merged_from_shards": [str(p) for p in inputs],
        "num_shards": len(inputs),
        "runs": runs,
    },
    "results": all_results,
}

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(final_payload, f, indent=2)
PY
    echo "Saved final merged results (all shards): ${COMBINED_OUTPUT}"
  fi
fi
if [[ "${KEEP_TEMPORARY}" -eq 1 ]]; then
  echo "Kept temporary files:"
  echo "  llama=${LLAMA_TMP} (resolved output: ${LLAMA_OUT})"
  echo "  mistral=${MISTRAL_TMP} (resolved output: ${MISTRAL_OUT})"
  echo "  qwen=${QWEN_TMP} (resolved output: ${QWEN_OUT})"
  echo "  shard_merged=${COMBINED_SHARD_OUTPUT}"
fi

#!/usr/bin/env bash
set -euo pipefail

# Run cpu_hotpot_qasper_grid_eval.py sequentially for Bedrock model families:
#   llama, mistral, qwen
# and merge results into a single JSON file.
# This script explicitly includes:
#   chunkers: token, sentence, recursive, semantic, late_token_pool
#   retrievers: sbert, e5, bm25s
#
# Usage:
#   bash scripts/run_hotpot_qasper_early_hierarchical.sh [grid-eval args...]
#
# Script-specific options:
#   --combined-output <path>   Output JSON path for merged results
#   --python-bin <path>        Python executable to use
#   --keep-temporary           Keep intermediate llama/mistral/qwen JSON files
#
# All other args are forwarded to cpu_hotpot_qasper_grid_eval.py.

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
COMBINED_OUTPUT="${COMBINED_OUTPUT:-results/hotpot_qasper_cpu_grid_combined.json}"
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

echo "Running Bedrock llama grid..."
"${PYTHON_BIN}" cpu_hotpot_qasper_grid_eval.py \
  --mode 2 \
  --chunkers token sentence recursive semantic late_token_pool \
  --retrievers sbert e5 bm25s \
  --answer-provider bedrock \
  --hotpot-answer-model llama \
  --output "${LLAMA_TMP}" \
  "${FORWARD_ARGS[@]}"

echo "Running Bedrock mistral grid..."
"${PYTHON_BIN}" cpu_hotpot_qasper_grid_eval.py \
  --mode 2 \
  --chunkers token sentence recursive semantic late_token_pool \
  --retrievers sbert e5 bm25s \
  --answer-provider bedrock \
  --hotpot-answer-model mistral \
  --output "${MISTRAL_TMP}" \
  "${FORWARD_ARGS[@]}"

echo "Running Bedrock qwen grid..."
"${PYTHON_BIN}" cpu_hotpot_qasper_grid_eval.py \
  --mode 2 \
  --chunkers token sentence recursive semantic late_token_pool \
  --retrievers sbert e5 bm25s \
  --answer-provider bedrock \
  --hotpot-answer-model qwen \
  --output "${QWEN_TMP}" \
  "${FORWARD_ARGS[@]}"

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

merged_results = (
    list(llama.get("results", []))
    + list(mistral.get("results", []))
    + list(qwen.get("results", []))
)
payload = {
    "config": {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {"answer_provider": "bedrock", "model_family": "llama", "config": llama.get("config", {})},
            {"answer_provider": "bedrock", "model_family": "mistral", "config": mistral.get("config", {})},
            {"answer_provider": "bedrock", "model_family": "qwen", "config": qwen.get("config", {})},
        ],
    },
    "results": merged_results,
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

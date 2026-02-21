#!/usr/bin/env bash
set -euo pipefail

# Small mode-4 smoke check:
# - max_queries=5
# - one chunker (token)
# - one retriever (sbert)
# - one generator model (llama)

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="logs"
OUT_FILE="results/mode4_smoke_check.json"
LOG_FILE="${LOG_DIR}/mode4_smoke_check.log"

mkdir -p "${LOG_DIR}" "$(dirname "${OUT_FILE}")"

EVAL_GPU_ID=0 \
OLLAMA_KEEP_ALIVE=0 \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_NUM_PARALLEL=1 \
"${PYTHON_BIN}" cpu_hotpot_qasper_grid_eval.py \
  --mode 4 \
  --dataset-path-mode3 datasets/hotpot_dev_fullwiki_v1.json \
  --wiki-corpus-path datasets/wiki_raw \
  --answer-provider ollama \
  --hotpot-answer-model llama \
  --retrievers sbert \
  --chunkers token \
  --k 5 \
  --max-queries 10 \
  --output "${OUT_FILE}" \
  > "${LOG_FILE}" 2>&1

echo "Smoke check complete."
echo "Output: ${OUT_FILE}"
echo "Log: ${LOG_FILE}"

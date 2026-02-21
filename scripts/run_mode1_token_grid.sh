#!/usr/bin/env bash
set -euo pipefail

# Small mode-1 token grid:
# token_size in {64,128,256,512}
# overlap in {16,32,48}
#
# Defaults:
# - chunkers: token
# - retrievers: sbert
# - generator model: llama

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="logs"
RESULTS_DIR="results"
COMBINED_OUTPUT="${RESULTS_DIR}/mode1_token_grid_combined.json"
CHUNKERS="${CHUNKERS:-token}"

TOKEN_SIZES=(64 128 256 512)
TOKEN_OVERLAPS=(16 32 48)

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

OUT_FILES=()
FAILED=0

for token_size in "${TOKEN_SIZES[@]}"; do
  for overlap in "${TOKEN_OVERLAPS[@]}"; do
    OUT_FILE="${RESULTS_DIR}/mode1_token_grid_t${token_size}_o${overlap}.json"
    LOG_FILE="${LOG_DIR}/mode1_token_grid_t${token_size}_o${overlap}.log"
    OUT_FILES+=("${OUT_FILE}")

    echo "Running token_size=${token_size}, overlap=${overlap} ..."
    if ! "${PYTHON_BIN}" cpu_hotpot_qasper_grid_eval.py \
      --mode 1 \
      --dataset-path-mode1 datasets/dev-v1.1.json \
      --answer-provider ollama \
      --hotpot-answer-model llama \
      --retrievers sbert \
      --chunkers ${CHUNKERS} \
      --token-size "${token_size}" \
      --overlap "${overlap}" \
      --k 5 \
      --max-queries 200 \
      --output "${OUT_FILE}" \
      > "${LOG_FILE}" 2>&1; then
      echo "FAILED: token_size=${token_size}, overlap=${overlap}. See ${LOG_FILE}"
      FAILED=1
    fi
  done
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "One or more runs failed. Fix errors in logs before merging."
  exit 1
fi

echo "Merging grid outputs into ${COMBINED_OUTPUT} ..."
"${PYTHON_BIN}" - "${COMBINED_OUTPUT}" "${OUT_FILES[@]}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

out_path = pathlib.Path(sys.argv[1])
input_paths = [pathlib.Path(p) for p in sys.argv[2:]]

results = []
configs = []
for p in input_paths:
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    configs.append({"file": str(p), "config": payload.get("config", {})})
    results.extend(list(payload.get("results", [])))

merged = {
    "config": {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": configs,
    },
    "results": results,
}

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)
PY

echo "Saved combined output: ${COMBINED_OUTPUT}"

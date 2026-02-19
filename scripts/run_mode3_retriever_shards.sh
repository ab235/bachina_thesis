#!/usr/bin/env bash
set -euo pipefail

# Run mode-3 evaluation as three parallel retriever shards:
#   sbert, e5, bm25s
# Each shard runs all chunkers and all answer generator models
# (via --all-hotpot-answer-models), then outputs are merged.

PYTHON_BIN="${PYTHON_BIN:-python3}"
COMBINED_OUTPUT="results/mode3_retriever_shards_combined.json"
LOG_DIR="logs"
KEEP_TEMPORARY=0

mkdir -p "${LOG_DIR}" "$(dirname "${COMBINED_OUTPUT}")"

SBERT_OUT="$(mktemp)"
E5_OUT="$(mktemp)"
BM25S_OUT="$(mktemp)"

cleanup() {
  if [[ "${KEEP_TEMPORARY}" -eq 0 ]]; then
    rm -f "${SBERT_OUT}" "${E5_OUT}" "${BM25S_OUT}"
  fi
}
trap cleanup EXIT

run_one() {
  local retriever="$1"
  local out="$2"
  local log_file="$3"
  "${PYTHON_BIN}" cpu_hotpot_qasper_grid_eval.py \
    --mode 3 \
    --dataset-path-mode3 datasets/hotpot_dev_fullwiki_v1.json \
    --wiki-corpus-path datasets/wiki_raw \
    --answer-provider ollama \
    --retrievers "${retriever}" \
    --chunkers token sentence recursive semantic late_token_pool \
    --all-hotpot-answer-models \
    --k 5 \
    --max-queries 200 \
    --output "${out}" \
    > "${log_file}" 2>&1
}

echo "Starting mode3 retriever shards in parallel: sbert, e5, bm25s ..."
run_one "sbert" "${SBERT_OUT}" "${LOG_DIR}/mode3_shard_sbert.log" &
PID_SBERT=$!
run_one "e5" "${E5_OUT}" "${LOG_DIR}/mode3_shard_e5.log" &
PID_E5=$!
run_one "bm25s" "${BM25S_OUT}" "${LOG_DIR}/mode3_shard_bm25s.log" &
PID_BM25S=$!

FAILED=0
wait "${PID_SBERT}" || FAILED=1
wait "${PID_E5}" || FAILED=1
wait "${PID_BM25S}" || FAILED=1

if [[ "${FAILED}" -ne 0 ]]; then
  echo "One or more retriever shards failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "Merging shard outputs into ${COMBINED_OUTPUT} ..."
"${PYTHON_BIN}" - "${SBERT_OUT}" "${E5_OUT}" "${BM25S_OUT}" "${COMBINED_OUTPUT}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

sbert_path = pathlib.Path(sys.argv[1])
e5_path = pathlib.Path(sys.argv[2])
bm25s_path = pathlib.Path(sys.argv[3])
out_path = pathlib.Path(sys.argv[4])

with sbert_path.open("r", encoding="utf-8") as f:
    sbert = json.load(f)
with e5_path.open("r", encoding="utf-8") as f:
    e5 = json.load(f)
with bm25s_path.open("r", encoding="utf-8") as f:
    bm25s = json.load(f)

payload = {
    "config": {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shards": [
            {"retriever": "sbert", "config": sbert.get("config", {})},
            {"retriever": "e5", "config": e5.get("config", {})},
            {"retriever": "bm25s", "config": bm25s.get("config", {})},
        ],
    },
    "results": (
        list(sbert.get("results", []))
        + list(e5.get("results", []))
        + list(bm25s.get("results", []))
    ),
}

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY

echo "Saved combined output: ${COMBINED_OUTPUT}"
if [[ "${KEEP_TEMPORARY}" -eq 1 ]]; then
  echo "Kept temporary outputs:"
  echo "  sbert=${SBERT_OUT}"
  echo "  e5=${E5_OUT}"
  echo "  bm25s=${BM25S_OUT}"
fi

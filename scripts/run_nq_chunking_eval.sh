#!/usr/bin/env bash
set -euo pipefail

# Evaluate retrieval + chunking on BEIR NQ.
# Runs a no-chunk baseline first, then several chunking configs.
#
# Usage:
#   bash scripts/run_nq_chunking_eval.sh
#
# Optional env overrides:
#   BACKEND=sbert
#   SBERT_MODEL=sentence-transformers/msmarco-MiniLM-L6-v3
#   BATCH_SIZE=64
#   SPLIT=test
#   K_VALUES="1 3 5 10 100"
#   ENABLE_SEMANTIC=0
#   OUT_DIR=results/nq_eval
#   ENCODINGS_DIR=encodings/nq_eval
#   LIVE_OUTPUT=1

DATASET="${DATASET:-nq}"
SPLIT="${SPLIT:-test}"
BACKEND="${BACKEND:-sbert}"
SBERT_MODEL="${SBERT_MODEL:-sentence-transformers/msmarco-MiniLM-L6-v3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
K_VALUES="${K_VALUES:-1 3 5 10 100}"
ENABLE_SEMANTIC="${ENABLE_SEMANTIC:-0}"
LIVE_OUTPUT="${LIVE_OUTPUT:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/nq_eval}"
ENCODINGS_DIR="${ENCODINGS_DIR:-${REPO_ROOT}/encodings/nq_eval}"
mkdir -p "${OUT_DIR}" "${ENCODINGS_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUT_DIR}/${DATASET}_${SPLIT}_${BACKEND}_${STAMP}.log"

print_header() {
  {
    echo "=== NQ CHUNKING EVAL START ==="
    echo "dataset=${DATASET} split=${SPLIT}"
    echo "backend=${BACKEND} model=${SBERT_MODEL} batch_size=${BATCH_SIZE}"
    echo "k_values=${K_VALUES}"
    echo "encodings_dir=${ENCODINGS_DIR}"
    echo "enable_semantic=${ENABLE_SEMANTIC}"
    echo "live_output=${LIVE_OUTPUT}"
    echo
  } | tee -a "${LOG_FILE}"
}

run_eval() {
  local label="$1"
  shift

  echo "--- ${label} ---" | tee -a "${LOG_FILE}"

  local tmp_run_log
  tmp_run_log="$(mktemp)"

  local -a cmd=(
    python "${REPO_ROOT}/beir_eval.py"
    --dataset "${DATASET}"
    --split "${SPLIT}"
    --backend "${BACKEND}"
    --sbert-model "${SBERT_MODEL}"
    --batch-size "${BATCH_SIZE}"
    --normalize
    --k-values ${K_VALUES}
    --encodings-dir "${ENCODINGS_DIR}"
    "$@"
  )

  local cmd_status
  set +e
  if [[ "${LIVE_OUTPUT}" == "1" ]]; then
    "${cmd[@]}" 2>&1 | tee "${tmp_run_log}"
    cmd_status=${PIPESTATUS[0]}
  else
    "${cmd[@]}" >"${tmp_run_log}" 2>&1
    cmd_status=$?
  fi
  set -e

  if [[ ${cmd_status} -ne 0 ]]; then
    {
      echo "status=FAILED exit_code=${cmd_status}"
      echo "error_tail:"
      tail -n 30 "${tmp_run_log}"
      echo
    } | tee -a "${LOG_FILE}"
  else
    {
      echo "status=OK"
      grep -E "Loaded dataset=|Chunked retrieval depth:|Collapse scoring:|Time taken to retrieve:|NDCG@|MAP@|Recall@|P@|MRR@|R_cap@|Hole@|Extra metrics" "${tmp_run_log}" || true
      echo
    } | tee -a "${LOG_FILE}"
  fi

  rm -f "${tmp_run_log}"
}

print_header

# Baseline: no chunking.
run_eval "baseline chunker=none" --chunker none

# Token chunking sweeps.
run_eval "token ts=128 ov=25" --chunker token --token-size 128 --overlap 25
run_eval "token ts=200 ov=50" --chunker token --token-size 200 --overlap 50
run_eval "token ts=256 ov=50" --chunker token --token-size 256 --overlap 50

# Other chunkers.
run_eval "sentence chunker" --chunker sentence
run_eval "recursive mc=1200 ov=200 mn=200" \
  --chunker recursive \
  --max-chars 1200 \
  --overlap 200 \
  --min-chars 200

if [[ "${ENABLE_SEMANTIC}" == "1" ]]; then
  run_eval "semantic mc=1200 ov=200 st=0.8" \
    --chunker semantic \
    --max-chars 1200 \
    --overlap 200 \
    --similarity-threshold 0.8
fi

echo "=== NQ CHUNKING EVAL END ===" | tee -a "${LOG_FILE}"
echo "Saved log: ${LOG_FILE}"

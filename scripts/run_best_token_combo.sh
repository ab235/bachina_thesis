#!/usr/bin/env bash
set -euo pipefail

# Run a fixed best chunking config across selected BEIR datasets.
# Usage:
#   bash scripts/run_best_token_combo.sh
# Optional env overrides:
#   BACKEND=sbert SBERT_MODEL=sentence-transformers/msmarco-MiniLM-L6-v3 BATCH_SIZE=64
#   TOKEN_SIZE=256 OVERLAP=25
#   NFP_SPLIT=dev HOTPOT_SPLIT=test TREC_COVID_SPLIT=test
#   DATASETS="nfcorpus hotpotqa trec-covid"

BACKEND="${BACKEND:-sbert}"
SBERT_MODEL="${SBERT_MODEL:-sentence-transformers/msmarco-MiniLM-L6-v3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TOKEN_SIZE="${TOKEN_SIZE:-256}"
OVERLAP="${OVERLAP:-25}"

NFCORPUS_SPLIT="${NFCORPUS_SPLIT:-dev}"
HOTPOT_SPLIT="${HOTPOT_SPLIT:-test}"
TREC_COVID_SPLIT="${TREC_COVID_SPLIT:-test}"

DATASET_LIST="${DATASETS:-nfcorpus hotpotqa trec-covid}"
read -r -a DATASET_ARR <<< "${DATASET_LIST}"

OUT_DIR="results/dev_tuning"
mkdir -p "${OUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUT_DIR}/best_combo_${BACKEND}_${STAMP}.log"

{
  echo "=== BEST TOKEN COMBO RUN START ==="
  echo "datasets=${DATASET_ARR[*]}"
  echo "backend=${BACKEND} model=${SBERT_MODEL} batch_size=${BATCH_SIZE}"
  echo "chunker=token token_size=${TOKEN_SIZE} overlap=${OVERLAP}"
  echo
} | tee -a "${LOG_FILE}"

for DATASET in "${DATASET_ARR[@]}"; do
  RUN_SPLIT="${NFCORPUS_SPLIT}"
  if [[ "${DATASET}" == "hotpotqa" ]]; then
    RUN_SPLIT="${HOTPOT_SPLIT}"
  elif [[ "${DATASET}" == "trec-covid" ]]; then
    RUN_SPLIT="${TREC_COVID_SPLIT}"
  fi

  echo "--- dataset=${DATASET} split=${RUN_SPLIT} token_size=${TOKEN_SIZE} overlap=${OVERLAP} ---" | tee -a "${LOG_FILE}"

  TMP_RUN_LOG="$(mktemp)"

  set +e
  python beir_eval.py \
    --dataset "${DATASET}" \
    --split "${RUN_SPLIT}" \
    --backend "${BACKEND}" \
    --sbert-model "${SBERT_MODEL}" \
    --chunker token \
    --token-size "${TOKEN_SIZE}" \
    --overlap "${OVERLAP}" \
    --batch-size "${BATCH_SIZE}" \
    --normalize \
    --k-values 1 3 5 10 >"${TMP_RUN_LOG}" 2>&1
  CMD_STATUS=$?
  set -e

  if [[ ${CMD_STATUS} -ne 0 ]]; then
    {
      echo "status=FAILED exit_code=${CMD_STATUS}"
      echo "error_tail:"
      tail -n 20 "${TMP_RUN_LOG}"
      echo
    } | tee -a "${LOG_FILE}"
  else
    {
      echo "status=OK"
      grep -E "Time taken to retrieve:|NDCG@|MAP@|Recall@|P@|MRR@" "${TMP_RUN_LOG}" || true
      echo
    } | tee -a "${LOG_FILE}"
  fi

  rm -f "${TMP_RUN_LOG}"
done

echo "=== BEST TOKEN COMBO RUN END ===" | tee -a "${LOG_FILE}"
echo "Saved log: ${LOG_FILE}"

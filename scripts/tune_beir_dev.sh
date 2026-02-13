#!/usr/bin/env bash
set -euo pipefail

# Grid search token chunking settings on BEIR dev split.
# Usage:
#   bash scripts/tune_beir_dev.sh
# Optional env overrides:
#   DATASET=nfcorpus BACKEND=sbert SBERT_MODEL=sentence-transformers/msmarco-MiniLM-L6-v3 BATCH_SIZE=64
#   DATASETS="nfcorpus hotpotqa"   # preferred for multi-dataset tuning

# Default split for datasets that provide dev qrels.
SPLIT="${SPLIT:-dev}"
# trec-covid in BEIR commonly provides test qrels, not dev.
TREC_COVID_SPLIT="${TREC_COVID_SPLIT:-test}"
# hotpotqa is commonly evaluated on test in BEIR releases.
HOTPOTQA_SPLIT="${HOTPOTQA_SPLIT:-test}"
BACKEND="${BACKEND:-sbert}"
SBERT_MODEL="${SBERT_MODEL:-sentence-transformers/msmarco-MiniLM-L6-v3}"
BATCH_SIZE="${BATCH_SIZE:-64}"

TOKEN_SIZES=(128 200 256 320)
OVERLAPS=(0 25 50 75 100)

OUT_DIR="results/dev_tuning"
mkdir -p "${OUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ -n "${DATASETS:-}" ]]; then
  DATASET_LIST="${DATASETS}"
elif [[ -n "${DATASET:-}" ]]; then
  DATASET_LIST="${DATASET}"
else
  # Default experiment pair: control + chunking-friendly QA corpus.
  DATASET_LIST="nfcorpus hotpotqa"
fi

read -r -a DATASET_ARR <<< "${DATASET_LIST}"
LOG_FILE="${OUT_DIR}/dev_tuning_${SPLIT}_${BACKEND}_${STAMP}.log"

{
  echo "=== BEIR DEV TUNING START ==="
  echo "datasets=${DATASET_ARR[*]}"
  echo "default_split=${SPLIT} trec_covid_split=${TREC_COVID_SPLIT} hotpotqa_split=${HOTPOTQA_SPLIT} backend=${BACKEND} model=${SBERT_MODEL} batch_size=${BATCH_SIZE}"
  echo "token_sizes=${TOKEN_SIZES[*]} overlaps=${OVERLAPS[*]}"
  echo
} | tee -a "${LOG_FILE}"

for DATASET in "${DATASET_ARR[@]}"; do
  RUN_SPLIT="${SPLIT}"
  if [[ "${DATASET}" == "trec-covid" ]]; then
    RUN_SPLIT="${TREC_COVID_SPLIT}"
  elif [[ "${DATASET}" == "hotpotqa" ]]; then
    RUN_SPLIT="${HOTPOTQA_SPLIT}"
  fi

  {
    echo "=== DATASET: ${DATASET} ==="
    echo "split=${RUN_SPLIT}"
    echo
  } | tee -a "${LOG_FILE}"

  for size in "${TOKEN_SIZES[@]}"; do
    for overlap in "${OVERLAPS[@]}"; do
      echo "--- token_size=${size} overlap=${overlap} split=${RUN_SPLIT} ---" | tee -a "${LOG_FILE}"

      TMP_RUN_LOG="$(mktemp)"

      set +e
      python beir_eval.py \
        --dataset "${DATASET}" \
        --split "${RUN_SPLIT}" \
        --backend "${BACKEND}" \
        --sbert-model "${SBERT_MODEL}" \
        --chunker token \
        --token-size "${size}" \
        --overlap "${overlap}" \
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
  done
done

echo "=== BEIR DEV TUNING END ===" | tee -a "${LOG_FILE}"
echo "Saved log: ${LOG_FILE}"
echo "Completed datasets: ${DATASET_ARR[*]}"

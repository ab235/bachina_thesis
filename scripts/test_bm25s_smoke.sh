#!/usr/bin/env bash
set -euo pipefail

# Small smoke test for local BM25S retrieval in cpu_hotpot_qasper_grid_eval.py.
#
# Usage:
#   bash scripts/test_bm25s_smoke.sh
#
# Optional env overrides:
#   PYTHON_BIN=.venv/bin/python
#   OUT=results/hotpot_qasper_bm25s_smoke.json
#   MAX_QUERIES=5

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUT="${OUT:-results/hotpot_qasper_bm25s_smoke.json}"
MAX_QUERIES="${MAX_QUERIES:-5}"

mkdir -p "$(dirname "${OUT}")"

"${PYTHON_BIN}" cpu_hotpot_qasper_grid_eval.py \
  --datasets hotpotqa_distractor \
  --chunking-mode early \
  --chunkers token \
  --retrievers bm25s \
  --token-size 128 \
  --overlap 32 \
  --max-queries "${MAX_QUERIES}" \
  --k-values 1 3 5 10 \
  --output "${OUT}"

echo "Smoke test complete: ${OUT}"

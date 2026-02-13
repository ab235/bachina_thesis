#!/usr/bin/env bash
set -euo pipefail

# Evaluate the new late_token_pool chunker for both modes and retrievers.
# Produces one combined JSON file with early + hierarchical runs.
#
# Usage:
#   bash scripts/run_late_token_pool_eval.sh [extra cpu_hotpot_qasper_grid_eval args...]
#
# Optional env overrides:
#   PYTHON_BIN=.venv/bin/python
#   OUT=results/hotpot_qasper_late_token_pool_combined.json
#   DATASETS="hotpotqa_distractor"
#   MAX_QUERIES=100
#   TOKEN_SIZE=200
#   OVERLAP=100
#   TOP_DOCS=20

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUT="${OUT:-results/hotpot_qasper_late_token_pool_combined.json}"
DATASETS="${DATASETS:-hotpotqa_distractor}"
MAX_QUERIES="${MAX_QUERIES:-100}"
TOKEN_SIZE="${TOKEN_SIZE:-200}"
OVERLAP="${OVERLAP:-100}"
TOP_DOCS="${TOP_DOCS:-20}"

EXTRA_ARGS=("$@")

mkdir -p "$(dirname "${OUT}")"

EARLY_TMP="$(mktemp)"
HIER_TMP="$(mktemp)"

cleanup() {
  rm -f "${EARLY_TMP}" "${HIER_TMP}"
}
trap cleanup EXIT

echo "Running late_token_pool (early)..."
"${PYTHON_BIN}" cpu_hotpot_qasper_grid_eval.py \
  --datasets ${DATASETS} \
  --chunking-mode early \
  --chunkers late_token_pool \
  --retrievers sbert bm25 \
  --max-queries "${MAX_QUERIES}" \
  --token-size "${TOKEN_SIZE}" \
  --overlap "${OVERLAP}" \
  --output "${EARLY_TMP}" \
  "${EXTRA_ARGS[@]}"

echo "Running late_token_pool (hierarchical)..."
"${PYTHON_BIN}" cpu_hotpot_qasper_grid_eval.py \
  --datasets ${DATASETS} \
  --chunking-mode hierarchical \
  --hierarchical-top-docs "${TOP_DOCS}" \
  --chunkers late_token_pool \
  --retrievers sbert bm25 \
  --max-queries "${MAX_QUERIES}" \
  --token-size "${TOKEN_SIZE}" \
  --overlap "${OVERLAP}" \
  --output "${HIER_TMP}" \
  "${EXTRA_ARGS[@]}"

echo "Merging outputs..."
"${PYTHON_BIN}" - "${EARLY_TMP}" "${HIER_TMP}" "${OUT}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

early_path = pathlib.Path(sys.argv[1])
hier_path = pathlib.Path(sys.argv[2])
out_path = pathlib.Path(sys.argv[3])

with early_path.open("r", encoding="utf-8") as f:
    early = json.load(f)
with hier_path.open("r", encoding="utf-8") as f:
    hierarchical = json.load(f)

payload = {
    "config": {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {"chunking_mode": "early", "config": early.get("config", {})},
            {"chunking_mode": "hierarchical", "config": hierarchical.get("config", {})},
        ],
    },
    "results": list(early.get("results", [])) + list(hierarchical.get("results", [])),
}

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY

echo "Saved: ${OUT}"

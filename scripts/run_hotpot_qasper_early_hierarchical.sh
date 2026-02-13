#!/usr/bin/env bash
set -euo pipefail

# Run cpu_hotpot_qasper_grid_eval.py for both chunking modes (early + hierarchical)
# and merge results into a single JSON file.
# This script explicitly includes:
#   chunkers: token, sentence, recursive, semantic, late_token_pool
#   retrievers: sbert, e5, bm25
#
# Usage:
#   bash scripts/run_hotpot_qasper_early_hierarchical.sh [grid-eval args...]
#
# Script-specific options:
#   --combined-output <path>   Output JSON path for merged results
#   --python-bin <path>        Python executable to use
#   --keep-temporary           Keep intermediate early/hierarchical JSON files
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

EARLY_TMP="$(mktemp)"
HIERARCHICAL_TMP="$(mktemp)"

cleanup() {
  if [[ "${KEEP_TEMPORARY}" -eq 0 ]]; then
    rm -f "${EARLY_TMP}" "${HIERARCHICAL_TMP}"
  fi
}
trap cleanup EXIT

echo "Running early chunking grid..."
"${PYTHON_BIN}" cpu_hotpot_qasper_grid_eval.py \
  --chunking-mode early \
  --chunkers token sentence recursive semantic late_token_pool \
  --retrievers sbert e5 bm25 \
  --bm25-hostname localhost \
  --bm25-port 9200 \
  --bm25-username elastic \
  --bm25-password "${ES_LOCAL_PASSWORD}" \
  --output "${EARLY_TMP}" \
  "${FORWARD_ARGS[@]}"

echo "Running hierarchical chunking grid..."
"${PYTHON_BIN}" cpu_hotpot_qasper_grid_eval.py \
  --chunking-mode hierarchical \
  --chunkers token sentence recursive semantic late_token_pool \
  --retrievers sbert e5 bm25 \
  --bm25-hostname localhost \
  --bm25-port 9200 \
  --bm25-username elastic \
  --bm25-password "${ES_LOCAL_PASSWORD}" \
  --output "${HIERARCHICAL_TMP}" \
  "${FORWARD_ARGS[@]}"

echo "Merging outputs into ${COMBINED_OUTPUT}..."
"${PYTHON_BIN}" - "${EARLY_TMP}" "${HIERARCHICAL_TMP}" "${COMBINED_OUTPUT}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

early_path = pathlib.Path(sys.argv[1])
hierarchical_path = pathlib.Path(sys.argv[2])
out_path = pathlib.Path(sys.argv[3])

with early_path.open("r", encoding="utf-8") as f:
    early = json.load(f)
with hierarchical_path.open("r", encoding="utf-8") as f:
    hierarchical = json.load(f)

merged_results = list(early.get("results", [])) + list(hierarchical.get("results", []))
payload = {
    "config": {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {"chunking_mode": "early", "config": early.get("config", {})},
            {"chunking_mode": "hierarchical", "config": hierarchical.get("config", {})},
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
  echo "  early=${EARLY_TMP}"
  echo "  hierarchical=${HIERARCHICAL_TMP}"
fi

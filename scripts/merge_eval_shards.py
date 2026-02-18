#!/usr/bin/env python3
import argparse
import glob
import json
import pathlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


JOB_RE = re.compile(r"\.job(\d+)-of-(\d+)")


def _job_key(path: pathlib.Path) -> Tuple[int, int, str]:
    match = JOB_RE.search(path.name)
    if not match:
        return (10**9, 10**9, path.name)
    return (int(match.group(2)), int(match.group(1)), path.name)


def _load_payload(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge cpu_hotpot_qasper_grid_eval shard JSON files into one payload."
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help="Glob pattern for shard files, e.g. 'results/mode2_ollama.job*-of-015.json'",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="Output merged JSON path.",
    )
    args = parser.parse_args()

    matched = sorted(
        [pathlib.Path(p) for p in glob.glob(args.pattern)],
        key=_job_key,
    )
    if not matched:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    payloads: List[Dict[str, Any]] = [_load_payload(path) for path in matched]
    merged_results: List[Dict[str, Any]] = []
    for payload in payloads:
        merged_results.extend(list(payload.get("results", [])))

    merged: Dict[str, Any] = {
        "config": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "merged_from_pattern": args.pattern,
            "input_files": [str(p) for p in matched],
            "num_files": len(matched),
        },
        "results": merged_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(matched)} files -> {args.output}")
    print(f"Total results rows: {len(merged_results)}")


if __name__ == "__main__":
    main()

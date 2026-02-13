#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List


def metric_value(metrics: Dict[str, float], name: str, k: int) -> float:
    return float(metrics.get(f"{name}@{k}", 0.0))


def build_rows(results: List[Dict[str, object]], k: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r in results:
        recall = metric_value(r.get("recall", {}), "Recall", k)
        mrr = metric_value(r.get("mrr", {}), "MRR", k)
        ndcg = metric_value(r.get("ndcg", {}), "NDCG", k)
        rows.append(
            {
                "dataset": r.get("dataset", ""),
                "chunker": r.get("chunker", ""),
                "retriever": r.get("retriever", ""),
                "started_at": r.get("started_at", ""),
                "queries": int(r.get("num_queries", 0)),
                "docs": int(r.get("num_docs", 0)),
                "chunks": int(r.get("num_chunks", 0)),
                "duration_seconds": float(r.get("duration_seconds", 0.0)),
                "Recall": recall,
                "MRR": mrr,
                "NDCG": ndcg,
            }
        )
    return rows


def to_fixed_table(rows: List[Dict[str, object]], k: int) -> str:
    headers = [
        "dataset",
        "chunker",
        "retriever",
        "started_at",
        "queries",
        "docs",
        "chunks",
        "duration_s",
        f"Recall@{k}",
        f"MRR@{k}",
        f"NDCG@{k}",
    ]
    table_rows = []
    for r in rows:
        table_rows.append(
            [
                str(r["dataset"]),
                str(r["chunker"]),
                str(r["retriever"]),
                str(r["started_at"]),
                str(r["queries"]),
                str(r["docs"]),
                str(r["chunks"]),
                f'{r["duration_seconds"]:.3f}',
                f'{r["Recall"]:.4f}',
                f'{r["MRR"]:.4f}',
                f'{r["NDCG"]:.4f}',
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(vals: List[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(r) for r in table_rows)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render eval JSON results as a Markdown table.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/hotpot_qasper_cpu_grid.json"),
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--sort-by",
        choices=["ndcg", "mrr", "recall"],
        default="ndcg",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = build_rows(payload.get("results", []), k=args.k)
    sort_key = {"ndcg": "NDCG", "mrr": "MRR", "recall": "Recall"}[args.sort_by]
    rows.sort(key=lambda x: (str(x["dataset"]), float(x[sort_key])), reverse=True)
    print(to_fixed_table(rows, k=args.k))


if __name__ == "__main__":
    main()

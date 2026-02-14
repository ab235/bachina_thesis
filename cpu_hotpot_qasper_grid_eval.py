import json
import logging
import pathlib
from typing import Dict, List

from preprocessing import load_hotpot_distractor, parse_args
from run import evaluate_one


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    if args.mode == 1:
        raise NotImplementedError("Mode 1 is not implemented yet.")
    if args.mode == 3:
        raise NotImplementedError("Mode 3 is not implemented yet.")
    if args.mode != 2:
        raise ValueError(f"Unsupported mode: {args.mode}")

    dataset_name = "hotpotqa_distractor"
    hotpot_path = args.dataset_path_mode2
    if not hotpot_path.exists():
        raise FileNotFoundError(
            f"HotpotQA distractor file not found: {hotpot_path}. "
            "Download hotpot_dev_distractor_v1.json and pass --dataset-path-mode2."
        )
    corpus, queries, hotpot_gold_facts, hotpot_doc_sentences, hotpot_answers = load_hotpot_distractor(
        hotpot_path=hotpot_path,
        max_queries=args.max_queries,
        seed=args.seed,
    )
    logging.info(
        "Loaded %s: docs=%d queries=%d",
        dataset_name,
        len(corpus),
        len(queries),
    )

    rows: List[Dict[str, object]] = []
    for chunker in args.chunkers:
        for retriever in args.retrievers:
            logging.info("Run: dataset=%s chunker=%s retriever=%s", dataset_name, chunker, retriever)
            row = evaluate_one(
                dataset_name=dataset_name,
                corpus=corpus,
                queries=queries,
                chunker=chunker,
                retriever=retriever,
                args=args,
                hotpot_gold_facts=hotpot_gold_facts,
                hotpot_doc_sentences=hotpot_doc_sentences,
                hotpot_answers=hotpot_answers,
            )
            rows.append(row)
            print(
                json.dumps(
                    {
                        "dataset": row["dataset"],
                        "chunker": row["chunker"],
                        "retriever": row["retriever"],
                        "chunk_recall": row.get("chunk_recall", {}),
                        "hotpot_official_emf1": row.get("hotpot_official_emf1", {}),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        k: str(v) if isinstance(v, pathlib.Path) else v
        for k, v in vars(args).items()
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"config": config, "results": rows}, f, indent=2)
    logging.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()

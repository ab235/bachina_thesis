import json
import logging
import os
import pathlib
from datetime import datetime, timezone
from time import perf_counter
from typing import Dict, List

from preprocessing import load_hotpot_distractor, load_hotpot_fullwiki, load_squad_v11, parse_args
from run import (
    build_result_row,
    compute_metrics_bundle,
    evaluate_one,
    init_retriever_context,
    run_early_mode,
    run_hierarchical_mode,
)


def main() -> None:
    args = parse_args()
    gpu_id = int(getattr(args, "gpu_id", -1))
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        # Make --gpu-id < 0 a true CPU mode for torch/sentence-transformers.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    if args.mode == 1:
        dataset_name = "squad_v11"
        squad_path = args.dataset_path_mode1
        if not squad_path.exists():
            raise FileNotFoundError(
                f"SQuAD v1.1 file not found: {squad_path}. "
                "Pass --dataset-path-mode1 with an official SQuAD v1.1 JSON file."
            )
        corpus, queries, hotpot_answers = load_squad_v11(
            squad_path=squad_path,
            max_queries=args.max_queries,
            seed=args.seed,
            keep_all_docs=bool(getattr(args, "squad_keep_all_docs", False)),
        )
        hotpot_gold_facts = {}
        hotpot_doc_sentences = {}
    elif args.mode == 2:
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
    elif args.mode == 3:
        dataset_name = "hotpotqa_fullwiki"
        hotpot_path = args.dataset_path_mode3
        wiki_path = args.wiki_corpus_path
        if not hotpot_path.exists():
            raise FileNotFoundError(
                f"HotpotQA fullwiki question file not found: {hotpot_path}. "
                "Pass --dataset-path-mode3 with a Hotpot fullwiki JSON split."
            )
        if not wiki_path.exists():
            raise FileNotFoundError(
                f"Global wiki corpus path not found: {wiki_path}. "
                "Pass --wiki-corpus-path with a corpus file or shard directory."
            )
        corpus, queries, hotpot_gold_facts, hotpot_doc_sentences, hotpot_answers = load_hotpot_fullwiki(
            hotpot_path=hotpot_path,
            wiki_path=wiki_path,
            max_queries=args.max_queries,
            seed=args.seed,
            cache_enabled=bool(getattr(args, "mode3_cache_enabled", True)),
            cache_dir=pathlib.Path(getattr(args, "mode3_cache_dir", pathlib.Path(".cache/mode3_fullwiki"))),
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    logging.info(
        "Loaded %s: docs=%d queries=%d",
        dataset_name,
        len(corpus),
        len(queries),
    )

    combo_list = [(chunker, retriever) for chunker in args.chunkers for retriever in args.retrievers]
    job_count = max(1, int(getattr(args, "job_count", 1)))
    job_index = int(getattr(args, "job_index", 0))
    if job_index < 0 or job_index >= job_count:
        raise ValueError(f"Invalid --job-index={job_index} for --job-count={job_count}.")
    selected_combos = [
        combo
        for idx, combo in enumerate(combo_list)
        if idx % job_count == job_index
    ]
    if not selected_combos:
        raise ValueError(
            f"No (chunker,retriever) configs assigned to job-index={job_index} with job-count={job_count}."
        )
    logging.info(
        "Combo shard: job %d/%d processing %d of %d configs",
        job_index,
        job_count,
        len(selected_combos),
        len(combo_list),
    )

    output_path = args.output
    if job_count > 1:
        output_path = output_path.with_name(
            f"{output_path.stem}.job{job_index:03d}-of-{job_count:03d}{output_path.suffix}"
        )

    rows: List[Dict[str, object]] = []
    all_model_families = ["llama", "mistral", "qwen"]
    for chunker, retriever in selected_combos:
        logging.info("Run: dataset=%s chunker=%s retriever=%s", dataset_name, chunker, retriever)
        if bool(getattr(args, "all_hotpot_answer_models", False)):
            started_at = datetime.now(timezone.utc)
            t0 = perf_counter()
            ctx = init_retriever_context(
                dataset_name=dataset_name,
                chunker=chunker,
                retriever=retriever,
                args=args,
            )
            if args.chunking_mode == "early":
                artifacts = run_early_mode(
                    corpus=corpus,
                    queries=queries,
                    chunker=chunker,
                    retriever=retriever,
                    args=args,
                    ctx=ctx,
                )
            else:
                artifacts = run_hierarchical_mode(
                    corpus=corpus,
                    queries=queries,
                    chunker=chunker,
                    retriever=retriever,
                    args=args,
                    ctx=ctx,
                )

            original_model_family = str(getattr(args, "hotpot_answer_model", "llama"))
            families = all_model_families if args.answer_provider == "ollama" else [original_model_family]
            for family in families:
                args.hotpot_answer_model = family
                metrics = compute_metrics_bundle(
                    dataset_name=dataset_name,
                    corpus=corpus,
                    queries=queries,
                    artifacts=artifacts,
                    args=args,
                    ctx=ctx,
                    hotpot_gold_facts=hotpot_gold_facts,
                    hotpot_doc_sentences=hotpot_doc_sentences,
                    hotpot_answers=hotpot_answers,
                )
                finished_at = datetime.now(timezone.utc)
                row = build_result_row(
                    dataset_name=dataset_name,
                    chunking_mode=args.chunking_mode,
                    chunker=chunker,
                    retriever=retriever,
                    num_queries=len(queries),
                    num_docs=len(corpus),
                    num_chunks=artifacts.num_chunks,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=perf_counter() - t0,
                    metrics=metrics,
                    answer_provider=str(getattr(args, "answer_provider", "")),
                    generator_model_family=family,
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "dataset": row["dataset"],
                            "chunker": row["chunker"],
                            "retriever": row["retriever"],
                            "answer_provider": row.get("answer_provider", ""),
                            "generator_model_family": row.get("generator_model_family", ""),
                            "chunk_recall": row.get("chunk_recall", {}),
                            "hotpot_official_emf1": row.get("hotpot_official_emf1", {}),
                            "squad_rag_generated_emf1": row.get("squad_rag_generated_emf1", {}),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            args.hotpot_answer_model = original_model_family
        else:
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
                        "answer_provider": row.get("answer_provider", ""),
                        "generator_model_family": row.get("generator_model_family", ""),
                        "chunk_recall": row.get("chunk_recall", {}),
                        "hotpot_official_emf1": row.get("hotpot_official_emf1", {}),
                        "squad_rag_generated_emf1": row.get("squad_rag_generated_emf1", {}),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        k: str(v) if isinstance(v, pathlib.Path) else v
        for k, v in vars(args).items()
    }
    config["resolved_output"] = str(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"config": config, "results": rows}, f, indent=2)
    logging.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()

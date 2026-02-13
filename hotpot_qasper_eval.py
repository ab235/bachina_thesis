import argparse
import importlib.util  # Ensure importlib.util is available for BEIR dense search import path.
import json
import logging
import pathlib
from time import time
from typing import Dict, List, Tuple

from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from beir_eval import (
    E5_EMBED_MODEL,
    E5Embedder,
    OPENAI_EMBED_MODEL,
    OpenAIEmbedder,
    SBERTEmbedder,
    collapse_results_to_docs,
    download_beir_dataset,
)
from chunking import recursive_chunk, sentence_chunk, token_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal retrieval eval for HotpotQA/QASPER with optional chunking."
    )
    parser.add_argument("--dataset", type=str, default="hotpotqa", choices=["hotpotqa", "qasper"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("datasets"))
    parser.add_argument("--backend", type=str, default="sbert", choices=["sbert", "e5", "openai"])
    parser.add_argument(
        "--sbert-model",
        type=str,
        default="sentence-transformers/msmarco-MiniLM-L6-v3",
    )
    parser.add_argument("--e5-model", type=str, default=E5_EMBED_MODEL)
    parser.add_argument("--model", type=str, default=OPENAI_EMBED_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--chunker", type=str, default="none", choices=["none", "token", "sentence", "recursive"])
    parser.add_argument("--token-size", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--chunk-retrieve-multiplier", type=int, default=5)

    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10, 100])
    return parser.parse_args()


def chunk_corpus(
    corpus: Dict[str, Dict[str, str]],
    chunker: str,
    token_size: int,
    max_chars: int,
    overlap: int,
    min_chars: int,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    chunked: Dict[str, Dict[str, str]] = {}
    chunk_to_doc: Dict[str, str] = {}
    for doc_id, doc in corpus.items():
        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""
        joined = "\n\n".join([part for part in (title, text) if part]).strip()
        if not joined:
            continue
        if chunker == "token":
            chunks = token_chunk(joined, target_size=token_size, overlap=overlap)
        elif chunker == "sentence":
            chunks = sentence_chunk(joined)
        elif chunker == "recursive":
            chunks = recursive_chunk(joined, min_chars=min_chars, max_chars=max_chars, overlap=overlap)
        else:
            raise ValueError(f"Unsupported chunker: {chunker}")

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}#chunk{idx}"
            chunked[chunk_id] = {"title": "", "text": chunk}
            chunk_to_doc[chunk_id] = doc_id
    return chunked, chunk_to_doc


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    eval_k = sorted(set(args.k_values))
    retrieval_k = list(eval_k)

    data_path = download_beir_dataset(args.dataset, args.data_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=args.split)
    original_doc_count = len(corpus)
    logging.info(
        "Loaded dataset=%s split=%s docs=%d queries=%d qrels=%d",
        args.dataset,
        args.split,
        len(corpus),
        len(queries),
        len(qrels),
    )

    chunk_to_doc: Dict[str, str] = {}
    if args.chunker != "none":
        corpus, chunk_to_doc = chunk_corpus(
            corpus,
            chunker=args.chunker,
            token_size=args.token_size,
            max_chars=args.max_chars,
            overlap=args.overlap,
            min_chars=args.min_chars,
        )
        avg_chunks_per_doc = len(corpus) / max(1, original_doc_count)
        expanded_k = int(max(eval_k) * max(1, args.chunk_retrieve_multiplier) * max(1.0, avg_chunks_per_doc))
        retrieval_k = sorted(set(retrieval_k + [expanded_k]))
        logging.info(
            "Chunking enabled (%s): chunked_docs=%d avg_chunks_per_doc=%.2f retrieve_k=%s",
            args.chunker,
            len(corpus),
            avg_chunks_per_doc,
            retrieval_k,
        )

    if args.backend == "sbert":
        embedder = SBERTEmbedder(
            model_name=args.sbert_model,
            batch_size=args.batch_size,
            normalize=args.normalize,
        )
    elif args.backend == "e5":
        embedder = E5Embedder(
            model_name=args.e5_model,
            batch_size=args.batch_size,
            normalize=args.normalize,
        )
    else:
        embedder = OpenAIEmbedder(
            model_name=args.model,
            batch_size=args.batch_size,
            normalize=args.normalize,
        )
    retriever = EvaluateRetrieval(
        DRES(embedder, batch_size=args.batch_size),
        score_function="cos_sim",
        k_values=retrieval_k,
    )

    t0 = time()
    results = retriever.retrieve(corpus, queries)
    logging.info("Retrieval finished in %.2fs", time() - t0)

    if args.chunker != "none":
        results = collapse_results_to_docs(
            results,
            chunk_to_doc,
            top_k=max(eval_k),
            score_mode="max",
        )

    ndcg, _map, recall, _precision = retriever.evaluate(qrels, results, eval_k)
    mrr = retriever.evaluate_custom(qrels, results, eval_k, metric="mrr")

    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "backend": args.backend,
        "chunker": args.chunker,
        "k_values": eval_k,
        "ndcg": ndcg,
        "recall": recall,
        "mrr": mrr,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

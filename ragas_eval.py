import argparse
import asyncio
import logging
import os
import pathlib
import random
from time import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.lexical import BM25Search
from datasets import Dataset
from openai import OpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from tqdm import tqdm

from chunking import recursive_chunk, semantic_chunking, sentence_chunk, token_chunk
from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_EMBED_MODEL
from embeddings import embed_texts


def batched(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


class OpenAIEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32, normalize: bool = True) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        for batch in batched(texts, self.batch_size):
            vectors.extend(embed_texts(batch, model_name=self.model_name))
        arr = np.asarray(vectors, dtype=np.float32)
        if self.normalize and len(arr):
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            arr = arr / norms
        return arr

    def encode_queries(self, queries, **kwargs):
        if isinstance(queries, dict):
            texts = [queries[qid] for qid in queries]
        else:
            texts = list(queries)
        return self._embed(texts)

    def encode_corpus(self, corpus, **kwargs):
        if isinstance(corpus, dict):
            items = list(corpus.values())
        else:
            items = list(corpus)
        texts: List[str] = []
        for doc in items:
            if isinstance(doc, str):
                texts.append(doc)
                continue
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            joined = "\n\n".join([part for part in (title, text) if part]).strip()
            texts.append(joined)
        return self._embed(texts)


class SBERTEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32, normalize: bool = True) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the SBERT backend. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.normalize = normalize

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

    def encode_queries(self, queries, **kwargs):
        if isinstance(queries, dict):
            texts = [queries[qid] for qid in queries]
        else:
            texts = list(queries)
        return self._encode(texts)

    def encode_corpus(self, corpus, **kwargs):
        if isinstance(corpus, dict):
            items = list(corpus.values())
        else:
            items = list(corpus)
        texts: List[str] = []
        for doc in items:
            if isinstance(doc, str):
                texts.append(doc)
                continue
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            joined = "\n\n".join([part for part in (title, text) if part]).strip()
            texts.append(joined)
        return self._encode(texts)


class OpenAIEmbeddingsAdapter(RagasOpenAIEmbeddings):
    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await self.aembed_text(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.aembed_texts(texts)


def _chunk_text(
    text: str,
    chunker: str,
    token_size: int,
    max_chars: int,
    overlap: int,
    min_chars: int,
    similarity_threshold: float,
    sentence_embed_fn,
) -> List[str]:
    if chunker == "token":
        return token_chunk(text, target_size=token_size, overlap=overlap)
    if chunker == "sentence":
        return sentence_chunk(text)
    if chunker == "recursive":
        return recursive_chunk(text, min_chars=min_chars, max_chars=max_chars, overlap=overlap)
    if chunker == "semantic":
        return semantic_chunking(
            text,
            max_chars=max_chars,
            overlap=overlap,
            similarity_threshold=similarity_threshold,
            embed_fn=sentence_embed_fn,
            show_progress=False,
        )
    raise ValueError(f"Unknown chunker: {chunker}")


def chunk_corpus(
    corpus: Dict[str, Dict[str, str]],
    chunker: str,
    token_size: int,
    max_chars: int,
    overlap: int,
    min_chars: int,
    similarity_threshold: float,
    sentence_embed_fn,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    chunked: Dict[str, Dict[str, str]] = {}
    chunk_to_doc: Dict[str, str] = {}
    for doc_id, doc in tqdm(corpus.items(), total=len(corpus), desc="Chunking corpus"):
        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""
        joined = "\n\n".join([part for part in (title, text) if part]).strip()
        if not joined:
            continue
        chunks = _chunk_text(
            joined,
            chunker=chunker,
            token_size=token_size,
            max_chars=max_chars,
            overlap=overlap,
            min_chars=min_chars,
            similarity_threshold=similarity_threshold,
            sentence_embed_fn=sentence_embed_fn,
        )
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}#chunk{idx}"
            chunked[chunk_id] = {"title": "", "text": chunk}
            chunk_to_doc[chunk_id] = doc_id
    return chunked, chunk_to_doc


def build_sentence_embed_fn(args: argparse.Namespace):
    if args.backend == "openai":
        return lambda texts: embed_texts(texts, model_name=args.model)
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentence-transformers is required for semantic chunking with this backend."
        ) from exc
    model = SentenceTransformer(args.sbert_model)
    return lambda texts: model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=args.normalize,
        show_progress_bar=False,
    ).tolist()


def collapse_results_to_docs(
    results: Dict[str, Dict[str, float]],
    chunk_to_doc: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    collapsed: Dict[str, Dict[str, float]] = {}
    for qid, ranking in results.items():
        doc_scores: Dict[str, float] = {}
        for chunk_id, score in ranking.items():
            doc_id = chunk_to_doc.get(chunk_id)
            if not doc_id:
                continue
            prev = doc_scores.get(doc_id)
            if prev is None or score > prev:
                doc_scores[doc_id] = score
        collapsed[qid] = doc_scores
    return collapsed


def build_ground_truth(qrels: Dict[str, Dict[str, int]], corpus: Dict[str, Dict[str, str]], qid: str) -> str:
    rel_docs = qrels.get(qid, {})
    if not rel_docs:
        return ""
    best_doc_id = max(rel_docs.items(), key=lambda kv: kv[1])[0]
    doc = corpus.get(best_doc_id, {})
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    return "\n\n".join([part for part in (title, text) if part]).strip()


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() or text[:max_chars]


def trim_contexts(contexts: List[str], max_chars_per_context: int, max_total_chars: int) -> List[str]:
    trimmed: List[str] = []
    total = 0
    for context in contexts:
        clipped = truncate_text(context, max_chars_per_context)
        if not clipped:
            continue
        if max_total_chars > 0 and total + len(clipped) > max_total_chars:
            remaining = max_total_chars - total
            if remaining <= 0:
                break
            clipped = truncate_text(clipped, remaining)
            if not clipped:
                break
        trimmed.append(clipped)
        total += len(clipped)
        if max_total_chars > 0 and total >= max_total_chars:
            break
    return trimmed


def build_prompt(question: str, contexts: List[str]) -> List[dict]:
    system = (
        "You are a retrieval QA assistant. Answer strictly using the provided context. "
        "If the answer is not present, say you do not know. "
        "For each factual claim you make, include one exact supporting quote from the context in double quotes. "
        "Do not use outside knowledge."
    )
    joined = "\n\n---\n\n".join(contexts)
    user = f"Question: {question}\n\nContext:\n{joined}\n\nAnswer:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def answer_with_context(client: OpenAI, question: str, contexts: List[str]) -> str:
    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        temperature=0.2,
        messages=build_prompt(question, contexts),
    )
    return response.choices[0].message.content.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on BEIR queries with your retrieval pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="nfcorpus", help="BEIR dataset name.")
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("datasets"),
        help="Where to store BEIR datasets.",
    )
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate (test/dev/train).")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K contexts to retrieve.")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=50,
        help="Limit number of queries for eval.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "sbert", "bm25"],
        default="openai",
        help="Retrieval backend to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OPENAI_EMBED_MODEL,
        help="Embedding model name (OpenAI).",
    )
    parser.add_argument(
        "--sbert-model",
        type=str,
        default="sentence-transformers/msmarco-MiniLM-L6-v3",
        help="Sentence-Transformers model name.",
    )
    parser.add_argument(
        "--chunker",
        type=str,
        choices=["none", "token", "sentence", "recursive", "semantic"],
        default="none",
        help="Chunking strategy to apply to the corpus.",
    )
    parser.add_argument("--token-size", type=int, default=200, help="Token chunk size (token chunker).")
    parser.add_argument("--max-chars", type=int, default=1200, help="Max chars per chunk.")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks.")
    parser.add_argument("--min-chars", type=int, default=200, help="Min chars per chunk (recursive).")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for semantic chunking.",
    )
    parser.add_argument(
        "--bm25-index",
        type=str,
        help="Elasticsearch index name for BM25 (defaults to dataset name).",
    )
    parser.add_argument(
        "--bm25-hostname",
        type=str,
        default="localhost",
        help="Elasticsearch hostname for BM25.",
    )
    parser.add_argument(
        "--bm25-init",
        action="store_true",
        help="Initialize (create) the BM25 index if needed.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["context_precision", "context_recall", "faithfulness", "answer_relevancy"],
        help="RAGAS metrics to compute.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens for RAGAS LLM responses.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=900,
        help="Max characters kept for each retrieved context.",
    )
    parser.add_argument(
        "--max-total-context-chars",
        type=int,
        default=2200,
        help="Max total characters across all contexts per query.",
    )
    parser.add_argument(
        "--max-ground-truth-chars",
        type=int,
        default=1400,
        help="Max characters kept for each ground-truth passage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set.")

    dataset = args.dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, str(args.data_dir))

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=args.split)
    original_corpus = corpus

    if args.max_queries is not None and args.max_queries < len(queries):
        rng = random.Random(args.seed)
        qids = list(queries.keys())
        rng.shuffle(qids)
        qids = qids[: args.max_queries]
        queries = {qid: queries[qid] for qid in qids}
    logging.info(f"Evaluating {len(queries)} queries after sampling.")

    if args.chunker != "none":
        logging.info(f"Applying chunker '{args.chunker}' to corpus...")
        sentence_embed_fn = build_sentence_embed_fn(args) if args.chunker == "semantic" else None
        corpus, _chunk_to_doc = chunk_corpus(
            corpus,
            chunker=args.chunker,
            token_size=args.token_size,
            max_chars=args.max_chars,
            overlap=args.overlap,
            min_chars=args.min_chars,
            similarity_threshold=args.similarity_threshold,
            sentence_embed_fn=sentence_embed_fn,
        )

    if args.backend == "bm25":
        index_name = args.bm25_index or dataset
        try:
            model = BM25Search(index_name=index_name, hostname=args.bm25_hostname, initialize=args.bm25_init)
        except TypeError:
            model = BM25Search()
        retriever = EvaluateRetrieval(model, score_function="bm25", k_values=[args.top_k])
    elif args.backend == "sbert":
        embedder = SBERTEmbedder(
            model_name=args.sbert_model,
            batch_size=args.batch_size,
            normalize=args.normalize,
        )
        model = DRES(embedder, batch_size=args.batch_size)
        retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=[args.top_k])
    else:
        embedder = OpenAIEmbedder(
            model_name=args.model,
            batch_size=args.batch_size,
            normalize=args.normalize,
        )
        model = DRES(embedder, batch_size=args.batch_size)
        retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=[args.top_k])

    logging.info("Starting retrieval...")
    start_time = time()
    if args.backend == "bm25":
        results = retriever.retrieve(corpus, queries)
    else:
        results = retriever.encode_and_retrieve(corpus, queries)
    logging.info(f"Time taken to retrieve: {time() - start_time:.2f} seconds")

    client = OpenAI(api_key=OPENAI_API_KEY)

    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    logging.info("Building RAGAS rows (retrieval contexts + generated answers)...")
    for qid, question in tqdm(queries.items(), total=len(queries), desc="Preparing eval rows"):
        ranked = results.get(qid, {})
        if not ranked:
            continue
        top = sorted(ranked.items(), key=lambda kv: kv[1], reverse=True)[: args.top_k]
        contexts = [
            "\n\n".join([part for part in (corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")) if part]).strip()
            for doc_id, _ in top
            if doc_id in corpus
        ]
        contexts = [c for c in contexts if c]
        contexts = trim_contexts(
            contexts,
            max_chars_per_context=args.max_context_chars,
            max_total_chars=args.max_total_context_chars,
        )
        if not contexts:
            continue

        ground_truth = build_ground_truth(qrels, original_corpus, qid)
        ground_truth = truncate_text(ground_truth, args.max_ground_truth_chars)
        if not ground_truth:
            continue

        answer = answer_with_context(client, question, contexts)

        rows["question"].append(question)
        rows["answer"].append(answer)
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(ground_truth)

    logging.info(f"Built {len(rows['question'])} rows for RAGAS evaluation.")
    dataset = Dataset.from_dict(rows)

    metric_map = {
        "context_precision": context_precision,
        "context_recall": context_recall,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
    }
    selected_metrics = []
    for name in args.metrics:
        metric = metric_map.get(name)
        if metric is None:
            raise ValueError(f"Unknown metric: {name}. Valid: {', '.join(metric_map)}")
        selected_metrics.append(metric)

    llm = llm_factory(OPENAI_CHAT_MODEL, client=client, max_tokens=args.max_tokens)
    embeddings = OpenAIEmbeddingsAdapter(client=client, model=OPENAI_EMBED_MODEL)
    logging.info("Starting RAGAS metric evaluation...")
    result = evaluate(dataset, metrics=selected_metrics, llm=llm, embeddings=embeddings)

    print(result)


if __name__ == "__main__":
    main()

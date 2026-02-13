import argparse
import json
import logging
import os
import pathlib
import zipfile
import urllib.error
import urllib.request
from time import time
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.lexical import BM25Search
from tqdm import tqdm

from config import OPENAI_EMBED_MODEL
from embeddings import embed_texts
from chunking import recursive_chunk, semantic_chunking, sentence_chunk, token_chunk

E5_EMBED_MODEL = "intfloat/e5-base-v2"


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
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for batch in tqdm(
            batched(texts, self.batch_size),
            total=total_batches,
            desc="Retrieval: embedding batches (OpenAI)",
            leave=False,
        ):
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
            show_progress_bar=True,
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


class E5Embedder:
    def __init__(self, model_name: str, batch_size: int = 32, normalize: bool = True) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the E5 backend. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.normalize = normalize

    @staticmethod
    def _format_query(text: str) -> str:
        return f"query: {text}"

    @staticmethod
    def _format_passage(text: str) -> str:
        return f"passage: {text}"

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )

    def encode_queries(self, queries, **kwargs):
        if isinstance(queries, dict):
            texts = [self._format_query(queries[qid]) for qid in queries]
        else:
            texts = [self._format_query(q) for q in queries]
        return self._encode(texts)

    def encode_corpus(self, corpus, **kwargs):
        if isinstance(corpus, dict):
            items = list(corpus.values())
        else:
            items = list(corpus)
        texts: List[str] = []
        for doc in items:
            if isinstance(doc, str):
                texts.append(self._format_passage(doc))
                continue
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            joined = "\n\n".join([part for part in (title, text) if part]).strip()
            texts.append(self._format_passage(joined))
        return self._encode(texts)


def _chunk_text(
    text: str,
    chunker: str,
    token_size: int,
    max_chars: int,
    overlap: int,
    min_chars: int,
    similarity_threshold: float,
    sentence_embed_fn,
    token_tokenize_fn: Optional[Callable[[str], Sequence[Any]]] = None,
    token_detokenize_fn: Optional[Callable[[Sequence[Any]], str]] = None,
) -> List[str]:
    if chunker == "token":
        return token_chunk(
            text,
            target_size=token_size,
            overlap=overlap,
            tokenize_fn=token_tokenize_fn,
            detokenize_fn=token_detokenize_fn,
        )
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
            show_progress=True,
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
    token_tokenize_fn: Optional[Callable[[str], Sequence[Any]]] = None,
    token_detokenize_fn: Optional[Callable[[Sequence[Any]], str]] = None,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    chunked: Dict[str, Dict[str, str]] = {}
    chunk_to_doc: Dict[str, str] = {}
    for doc_id, doc in tqdm(
        corpus.items(),
        total=len(corpus),
        desc=f"Chunking corpus ({chunker})",
    ):
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
            token_tokenize_fn=token_tokenize_fn,
            token_detokenize_fn=token_detokenize_fn,
        )
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}#chunk{idx}"
            chunked[chunk_id] = {"title": "", "text": chunk}
            chunk_to_doc[chunk_id] = doc_id
    return chunked, chunk_to_doc


def build_sentence_embed_fn(args: argparse.Namespace):
    if args.backend == "openai":
        return lambda texts: embed_texts(texts, model_name=args.model)
    if args.backend == "e5":
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sentence-transformers is required for semantic chunking with this backend."
            ) from exc
        model = SentenceTransformer(args.e5_model)
        return lambda texts: model.encode(
            [f"passage: {t}" for t in texts],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=args.normalize,
            show_progress_bar=True,
        ).tolist()
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
        show_progress_bar=True,
    ).tolist()


def build_token_chunk_tokenizer(
    args: argparse.Namespace,
) -> Tuple[Optional[Callable[[str], Sequence[Any]]], Optional[Callable[[Sequence[Any]], str]], str]:
    if args.chunker != "token":
        return None, None, "not-applicable"

    preference = args.tokenizer
    candidates: List[str]
    if preference == "auto":
        if args.backend == "sbert":
            candidates = ["sbert", "tiktoken", "nltk"]
        elif args.backend == "e5":
            candidates = ["sbert", "tiktoken", "nltk"]
        elif args.backend == "openai":
            candidates = ["tiktoken", "sbert", "nltk"]
        else:
            candidates = ["nltk"]
    else:
        candidates = [preference]

    for choice in candidates:
        if choice == "sbert":
            try:
                from transformers import AutoTokenizer
            except ModuleNotFoundError:
                continue
            model_name = args.sbert_model if args.backend == "sbert" else args.e5_model
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            # We only use this tokenizer to split text into chunk-sized token windows.
            # Avoid noisy max-length warnings on raw full-document tokenization.
            try:
                tokenizer.deprecation_warnings[
                    "sequence-length-is-longer-than-the-specified-maximum"
                ] = True
            except Exception:
                pass
            tokenizer.model_max_length = int(1e30)
            if getattr(tokenizer, "is_fast", False) and hasattr(tokenizer, "backend_tokenizer"):
                return (
                    lambda text: tokenizer.backend_tokenizer.encode(
                        text, add_special_tokens=False
                    ).ids,
                    lambda ids: tokenizer.decode(
                        list(ids),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    ),
                    "sbert",
                )
            return (
                lambda text: tokenizer.encode(text, add_special_tokens=False),
                lambda ids: tokenizer.decode(
                    list(ids),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ),
                "sbert",
            )
        if choice == "tiktoken":
            try:
                import tiktoken
            except ModuleNotFoundError:
                continue
            try:
                encoding = tiktoken.encoding_for_model(args.model)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")
            return (
                lambda text: encoding.encode(text),
                lambda ids: encoding.decode(list(ids)),
                "tiktoken",
            )
        if choice == "nltk":
            return None, None, "nltk"

    return None, None, "nltk-fallback"


def build_encoding_suffix(args: argparse.Namespace) -> str:
    parts = [args.backend]
    if args.backend == "openai":
        parts.append(args.model.replace("/", "_"))
    elif args.backend == "sbert":
        parts.append(args.sbert_model.replace("/", "_"))
    elif args.backend == "e5":
        parts.append(args.e5_model.replace("/", "_"))
    parts.append(args.chunker)
    if args.chunker == "token":
        parts.extend([f"ts{args.token_size}", f"ov{args.overlap}", f"tok{args.tokenizer}"])
    elif args.chunker == "sentence":
        parts.append("sent")
    elif args.chunker in {"recursive", "semantic"}:
        parts.extend(
            [
                f"mc{args.max_chars}",
                f"ov{args.overlap}",
                f"mn{args.min_chars}",
                f"st{str(args.similarity_threshold).replace('.', '_')}",
            ]
        )
    return "_".join(parts)


def collapse_results_to_docs(
    results: Dict[str, Dict[str, float]],
    chunk_to_doc: Dict[str, str],
    top_k: Optional[int] = None,
    score_mode: str = "max",
    sum_weight: float = 0.2,
    max_chunks_per_doc: int = 3,
) -> Dict[str, Dict[str, float]]:
    collapsed: Dict[str, Dict[str, float]] = {}
    for qid, ranking in results.items():
        doc_chunks: Dict[str, List[float]] = {}
        for chunk_id, score in ranking.items():
            doc_id = chunk_to_doc.get(chunk_id)
            if not doc_id:
                continue
            doc_chunks.setdefault(doc_id, []).append(score)

        doc_scores: Dict[str, float] = {}
        for doc_id, scores in doc_chunks.items():
            ordered = sorted(scores, reverse=True)
            if score_mode == "max":
                doc_scores[doc_id] = ordered[0]
            elif score_mode == "max_plus_sum":
                use_scores = ordered[: max(1, max_chunks_per_doc)]
                head = use_scores[0]
                tail = sum(use_scores[1:])
                doc_scores[doc_id] = head + (sum_weight * tail)
            else:
                raise ValueError(f"Unknown collapse score mode: {score_mode}")
        if top_k is not None:
            ordered = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
            collapsed[qid] = dict(ordered[:top_k])
        else:
            collapsed[qid] = doc_scores
    return collapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval on a BEIR dataset using OpenAI embeddings with optional chunking.",
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
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings.")
    parser.add_argument(
        "--model",
        type=str,
        default=OPENAI_EMBED_MODEL,
        help="Embedding model name (OpenAI).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "sbert", "e5", "bm25"],
        default="openai",
        help="Embedding backend to use.",
    )
    parser.add_argument(
        "--sbert-model",
        type=str,
        default="sentence-transformers/msmarco-MiniLM-L6-v3",
        help="Sentence-Transformers model name.",
    )
    parser.add_argument(
        "--e5-model",
        type=str,
        default=E5_EMBED_MODEL,
        help="E5 model name.",
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
        "--chunker",
        type=str,
        choices=["none", "token", "sentence", "recursive", "semantic"],
        default="none",
        help="Chunking strategy to apply to the corpus.",
    )
    parser.add_argument("--token-size", type=int, default=200, help="Token chunk size (token chunker).")
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["auto", "sbert", "tiktoken", "nltk"],
        default="auto",
        help="Tokenizer used by token chunking.",
    )
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
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 100],
        help="K values for evaluation metrics.",
    )
    parser.add_argument(
        "--chunk-retrieve-multiplier",
        type=int,
        default=5,
        help="When chunking is enabled, retrieve this multiple of max(k) before collapsing to docs.",
    )
    parser.add_argument(
        "--collapse-score-mode",
        type=str,
        choices=["max", "max_plus_sum"],
        default="max_plus_sum",
        help="How to aggregate chunk scores into a document score after collapse.",
    )
    parser.add_argument(
        "--collapse-sum-weight",
        type=float,
        default=0.2,
        help="Weight on additional chunk scores when --collapse-score-mode=max_plus_sum.",
    )
    parser.add_argument(
        "--collapse-max-chunks-per-doc",
        type=int,
        default=3,
        help="Max number of top chunk scores per doc used for collapse aggregation.",
    )
    parser.add_argument(
        "--encodings-dir",
        type=pathlib.Path,
        default=pathlib.Path("encodings"),
        help="Where to store cached encodings.",
    )
    return parser.parse_args()


def check_elasticsearch(hostname: str) -> None:
    url = f"http://{hostname}:9200/_cluster/health?wait_for_status=yellow&timeout=2s"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Elasticsearch health check failed: HTTP {resp.status}")
            payload = json.loads(resp.read().decode("utf-8"))
            status = payload.get("status")
            if status not in {"yellow", "green"}:
                raise RuntimeError(f"Elasticsearch unhealthy: status={status!r}")
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(
            f"Elasticsearch not reachable at {hostname}:9200. "
            "Start Elasticsearch and try again."
        ) from exc


def download_beir_dataset(dataset: str, data_dir: pathlib.Path) -> str:
    """Download a BEIR dataset, with robust04 alias fallback and zip corruption recovery."""
    # Some mirrors use an alternate slug for robust04.
    candidates = [dataset]
    if dataset == "robust04":
        candidates.append("trec-robust04")

    errors: List[str] = []
    for candidate in candidates:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{candidate}.zip"
        zip_path = data_dir / f"{candidate}.zip"
        for attempt in (1, 2):
            try:
                return util.download_and_unzip(url, str(data_dir))
            except zipfile.BadZipFile:
                if attempt == 2:
                    errors.append(f"{candidate}: BadZipFile")
                    break
                logging.warning(
                    "Cached %s appears corrupted (BadZipFile). Removing and re-downloading...",
                    zip_path,
                )
                try:
                    zip_path.unlink(missing_ok=True)
                except OSError:
                    pass
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                break

    joined = "; ".join(errors) if errors else "unknown download failure"
    raise RuntimeError(
        f"Failed to download dataset '{dataset}'. Checked candidates: {', '.join(candidates)}. "
        f"Details: {joined}"
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    dataset = args.dataset
    eval_k_values = sorted(set(args.k_values))
    retrieval_k_values = list(eval_k_values)
    data_path = download_beir_dataset(dataset, args.data_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=args.split)
    original_doc_count = len(corpus)
    logging.info(
        "Loaded dataset=%s split=%s (docs=%d, queries=%d, qrels=%d)",
        dataset,
        args.split,
        len(corpus),
        len(queries),
        len(qrels),
    )
    chunk_to_doc: Dict[str, str] = {}
    if args.chunker != "none":
        logging.info("Stage: chunking started (chunker=%s)", args.chunker)
        sentence_embed_fn = build_sentence_embed_fn(args) if args.chunker == "semantic" else None
        token_tokenize_fn, token_detokenize_fn, tokenizer_name = build_token_chunk_tokenizer(args)
        if args.chunker == "token":
            logging.info("Token chunking tokenizer: %s", tokenizer_name)
        corpus, chunk_to_doc = chunk_corpus(
            corpus,
            chunker=args.chunker,
            token_size=args.token_size,
            max_chars=args.max_chars,
            overlap=args.overlap,
            min_chars=args.min_chars,
            similarity_threshold=args.similarity_threshold,
            sentence_embed_fn=sentence_embed_fn,
            token_tokenize_fn=token_tokenize_fn,
            token_detokenize_fn=token_detokenize_fn,
        )
        logging.info("Stage: chunking finished (chunked_docs=%d)", len(corpus))
        avg_chunks_per_doc = len(corpus) / max(1, original_doc_count)
        expanded_k = int(
            max(eval_k_values)
            * max(1, args.chunk_retrieve_multiplier)
            * max(1.0, avg_chunks_per_doc)
        )
        if expanded_k not in retrieval_k_values:
            retrieval_k_values.append(expanded_k)
        retrieval_k_values = sorted(set(retrieval_k_values))
        logging.info(
            "Chunked retrieval depth: eval_k=%s -> retrieve_k=%s (avg_chunks_per_doc=%.2f)",
            eval_k_values,
            retrieval_k_values,
            avg_chunks_per_doc,
        )

    if args.backend == "bm25":
        check_elasticsearch(args.bm25_hostname)
        index_name = args.bm25_index or dataset
        try:
            model = BM25Search(index_name=index_name, hostname=args.bm25_hostname, initialize=args.bm25_init)
        except TypeError:
            model = BM25Search()
        retriever = EvaluateRetrieval(model, score_function="bm25", k_values=retrieval_k_values)
    else:
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
        model = DRES(embedder, batch_size=args.batch_size)
        retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=retrieval_k_values)

    start_time = time()
    logging.info("Stage: retrieval started (backend=%s)", args.backend)
    if args.backend == "bm25":
        results = retriever.retrieve(corpus, queries)
    else:
        args.encodings_dir.mkdir(parents=True, exist_ok=True)
        suffix = build_encoding_suffix(args)
        save_encodings_path = os.path.join(args.encodings_dir, f"{dataset}_{suffix}")
        results = retriever.encode_and_retrieve(
            corpus,
            queries,
            encode_output_path=save_encodings_path,
        )
    end_time = time()
    logging.info("Stage: retrieval finished")
    logging.info(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")

    if args.chunker != "none":
        logging.info("Stage: collapsing chunk-level results to document-level scores")
        logging.info(
            "Collapse scoring: mode=%s sum_weight=%.3f max_chunks_per_doc=%d",
            args.collapse_score_mode,
            args.collapse_sum_weight,
            args.collapse_max_chunks_per_doc,
        )
        results = collapse_results_to_docs(
            results,
            chunk_to_doc,
            top_k=max(eval_k_values),
            score_mode=args.collapse_score_mode,
            sum_weight=args.collapse_sum_weight,
            max_chunks_per_doc=args.collapse_max_chunks_per_doc,
        )
        if not any(ranking for ranking in results.values()):
            raise RuntimeError(
                "Collapsed retrieval results are empty. Encodings likely came from a different "
                "chunking configuration. Use a fresh encodings path or delete stale cache files."
            )

    logging.info("Stage: evaluation started")
    logging.info(f"Retriever evaluation for k in: {eval_k_values}")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, eval_k_values)
    mrr = retriever.evaluate_custom(qrels, results, eval_k_values, metric="mrr")
    recall_cap = retriever.evaluate_custom(qrels, results, eval_k_values, metric="recall_cap")
    hole = retriever.evaluate_custom(qrels, results, eval_k_values, metric="hole")
    for k in eval_k_values:
        logging.info(
            "Extra metrics @%d: Recall_cap=%.4f Hole=%.4f",
            k,
            recall_cap.get(f"R_cap@{k}", float("nan")),
            hole.get(f"Hole@{k}", float("nan")),
        )
    logging.info("Stage: evaluation finished")

    results_dir = pathlib.Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    util.save_runfile(str(results_dir / f"{dataset}.run.trec"), results)
    util.save_results(
        str(results_dir / f"{dataset}.json"),
        ndcg,
        _map,
        recall,
        precision,
        mrr,
        recall_cap=recall_cap,
        hole=hole,
    )


if __name__ == "__main__":
    main()

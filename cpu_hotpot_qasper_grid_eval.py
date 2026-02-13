import argparse
import json
import logging
import math
import os
import pathlib
import random
import re
import uuid
from datetime import datetime, timezone
from time import perf_counter
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers

from beir_eval import E5Embedder, SBERTEmbedder, build_sentence_embed_fn, download_beir_dataset
from chunking import recursive_chunk, semantic_chunking, sentence_chunk, token_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU grid eval: hotpotqa distractor + qasper with chunking x retriever."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa_distractor", "qasper"],
        choices=["hotpotqa_distractor", "qasper"],
    )
    parser.add_argument(
        "--hotpot-distractor-file",
        type=pathlib.Path,
        default=pathlib.Path("datasets/hotpot_dev_distractor_v1.json"),
        help="Path to original HotpotQA distractor JSON.",
    )
    parser.add_argument("--qasper-split", type=str, default="test")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("datasets"))
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--max-corpus-docs", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--sbert-model",
        type=str,
        default="sentence-transformers/msmarco-MiniLM-L6-v3",
    )
    parser.add_argument("--token-size", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument(
        "--chunking-mode",
        type=str,
        choices=["early", "hierarchical"],
        default="early",
        help="early: chunk full corpus first; hierarchical: retrieve docs first, then chunk top docs per query.",
    )
    parser.add_argument(
        "--hierarchical-top-docs",
        type=int,
        default=20,
        help="Top docs per query to chunk in hierarchical mode.",
    )
    parser.add_argument(
        "--chunkers",
        nargs="+",
        default=["token", "sentence", "recursive", "semantic"],
        choices=["token", "sentence", "recursive", "semantic", "late_token_pool"],
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["sbert", "bm25"],
        choices=["sbert", "e5", "bm25"],
    )
    parser.add_argument(
        "--e5-model",
        type=str,
        default="intfloat/e5-base-v2",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results/hotpot_qasper_cpu_grid.json"),
    )
    parser.add_argument("--bm25-hostname", type=str, default="localhost")
    parser.add_argument("--bm25-port", type=int, default=9200)
    parser.add_argument("--bm25-username", type=str, default="elastic")
    parser.add_argument(
        "--bm25-password",
        type=str,
        default=None,
        help="Elasticsearch password. If omitted, reads ES_LOCAL_PASSWORD env var when available.",
    )
    parser.add_argument(
        "--bm25-api-key",
        type=str,
        default=None,
        help="Optional Elasticsearch API key; overrides username/password when set.",
    )
    parser.add_argument("--bm25-index-prefix", type=str, default="cpu-grid")
    parser.add_argument(
        "--bm25-keep-indices",
        action="store_true",
        help="Keep temporary Elasticsearch BM25 indices after run.",
    )
    return parser.parse_args()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class BM25Index:
    doc_ids: List[str]
    doc_lens: np.ndarray
    avgdl: float
    tf_maps: List[Dict[str, int]]
    df: Dict[str, int]
    N: int
    k1: float = 1.2
    b: float = 0.75

    @classmethod
    def build(cls, docs: Dict[str, str]) -> "BM25Index":
        doc_ids = list(docs.keys())
        tf_maps: List[Dict[str, int]] = []
        df: Dict[str, int] = defaultdict(int)
        lens: List[int] = []
        for did in doc_ids:
            toks = _tokenize(docs[did])
            tf = Counter(toks)
            tf_maps.append(dict(tf))
            lens.append(len(toks))
            for term in tf:
                df[term] += 1
        N = len(doc_ids)
        avgdl = float(np.mean(lens)) if lens else 0.0
        return cls(
            doc_ids=doc_ids,
            doc_lens=np.asarray(lens, dtype=np.float32),
            avgdl=avgdl,
            tf_maps=tf_maps,
            df=df,
            N=N,
        )

    def score_query(self, query: str) -> np.ndarray:
        q_terms = _tokenize(query)
        if self.N == 0:
            return np.zeros((0,), dtype=np.float32)
        scores = np.zeros((self.N,), dtype=np.float32)
        for term in q_terms:
            n_qi = self.df.get(term, 0)
            if n_qi == 0:
                continue
            idf = math.log(1 + (self.N - n_qi + 0.5) / (n_qi + 0.5))
            for i, tf in enumerate(self.tf_maps):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * (self.doc_lens[i] / max(self.avgdl, 1e-9)))
                scores[i] += idf * ((f * (self.k1 + 1)) / denom)
        return scores


def _join_doc(doc: Dict[str, str]) -> str:
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    return "\n\n".join([x for x in (title, text) if x]).strip()


def chunk_text(
    text: str,
    chunker: str,
    args: argparse.Namespace,
    sentence_embed_fn: Optional[object] = None,
) -> List[str]:
    if chunker == "token":
        return token_chunk(text, target_size=args.token_size, overlap=args.overlap)
    if chunker == "sentence":
        return sentence_chunk(text)
    if chunker == "recursive":
        return recursive_chunk(
            text,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            overlap=args.overlap,
        )
    if chunker == "semantic":
        return semantic_chunking(
            text,
            max_chars=args.max_chars,
            overlap=args.overlap,
            similarity_threshold=args.similarity_threshold,
            embed_fn=sentence_embed_fn,
            show_progress=False,
        )
    raise ValueError(f"Unsupported chunker: {chunker}")


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0.0:
        return v
    return v / n


@dataclass
class LateChunkData:
    chunk_texts: Dict[str, str]
    chunk_to_doc: Dict[str, str]
    chunk_vectors: Dict[str, np.ndarray]
    truncated_docs: int


class LateTokenPoolEncoder:
    def __init__(self, model_name: str, max_tokens: int = 512, use_e5_format: bool = False) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers and torch are required for late_token_pool chunking."
            ) from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # We intentionally tokenize full sequences (truncation=False) and handle windowing ourselves.
        # Avoid tokenizer max-length warning noise for this path.
        try:
            self.tokenizer.deprecation_warnings[
                "sequence-length-is-longer-than-the-specified-maximum"
            ] = True
        except Exception:
            pass
        self.tokenizer.model_max_length = int(1e30)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.use_e5_format = use_e5_format
        self._passage_prefix_ids: List[int] = []
        if self.use_e5_format:
            self._passage_prefix_ids = self.tokenizer.encode(
                "passage: ",
                add_special_tokens=False,
                truncation=False,
            )
        model_max = int(getattr(self.tokenizer, "model_max_length", 512))
        if model_max <= 0 or model_max > 100000:
            model_max = 512
        self.max_tokens = max(8, min(max_tokens, model_max))
        self.num_special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        self.max_content_tokens = max(1, self.max_tokens - self.num_special_tokens)
        # Overlap between model forward windows when docs exceed max token length.
        self.window_overlap = min(50, max(0, self.max_content_tokens - 1))

    def _forward_token_ids(self, ids: List[int]) -> np.ndarray:
        if not ids:
            hidden = int(getattr(self.model.config, "hidden_size", 384))
            return np.zeros((0, hidden), dtype=np.float32)
        model_input_ids = self.tokenizer.build_inputs_with_special_tokens(ids)
        special_mask = self.tokenizer.get_special_tokens_mask(
            model_input_ids,
            already_has_special_tokens=True,
        )
        input_ids = self.torch.tensor([model_input_ids], dtype=self.torch.long, device=self.device)
        attention_mask = self.torch.ones_like(input_ids, dtype=self.torch.long, device=self.device)
        with self.torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_vecs = out.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)
        special_mask_arr = np.asarray(special_mask, dtype=np.int64)
        content_idx = np.where(special_mask_arr == 0)[0]
        if content_idx.size == 0:
            hidden = int(getattr(self.model.config, "hidden_size", 384))
            return np.zeros((0, hidden), dtype=np.float32)
        return token_vecs[content_idx]

    def _encode_ids(self, ids: List[int]) -> np.ndarray:
        token_vecs = self._forward_token_ids(ids)
        if len(token_vecs) == 0:
            hidden = int(getattr(self.model.config, "hidden_size", 384))
            return np.zeros((hidden,), dtype=np.float32)
        return _l2_normalize(token_vecs.mean(axis=0).astype(np.float32))

    def _encode_token_sequence(self, ids: List[int]) -> np.ndarray:
        if not ids:
            hidden = int(getattr(self.model.config, "hidden_size", 384))
            return np.zeros((0, hidden), dtype=np.float32)
        if len(ids) <= self.max_content_tokens:
            return self._forward_token_ids(ids)

        hidden = int(getattr(self.model.config, "hidden_size", 384))
        n = len(ids)
        accum = np.zeros((n, hidden), dtype=np.float32)
        counts = np.zeros((n, 1), dtype=np.float32)
        step = max(1, self.max_content_tokens - self.window_overlap)
        starts = list(range(0, n, step))
        if starts[-1] + self.max_content_tokens < n:
            starts.append(n - self.max_content_tokens)

        for start in starts:
            end = min(start + self.max_content_tokens, n)
            sub_ids = ids[start:end]
            vecs = self._forward_token_ids(sub_ids)
            if len(vecs) != len(sub_ids):
                # Defensive guard: keep alignment stable even if tokenizer/model edge-cases occur.
                vecs = vecs[: len(sub_ids)]
                if len(vecs) < len(sub_ids):
                    pad = np.zeros((len(sub_ids) - len(vecs), hidden), dtype=np.float32)
                    vecs = np.vstack([vecs, pad])
            accum[start:end] += vecs
            counts[start:end] += 1.0

        counts[counts == 0.0] = 1.0
        return accum / counts

    def encode_query(self, text: str) -> np.ndarray:
        if self.use_e5_format:
            text = f"query: {text}"
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_content_tokens,
        )
        return self._encode_ids(ids)

    def build_doc_chunks(self, text: str, target_size: int, overlap: int) -> Tuple[List[str], List[np.ndarray], bool]:
        model_text = f"passage: {text}" if self.use_e5_format else text
        ids = self.tokenizer.encode(model_text, add_special_tokens=False, truncation=False)
        over_model_max = len(ids) > self.max_content_tokens
        if not ids:
            return [], [], over_model_max
        if target_size <= 0:
            raise ValueError("target_size must be > 0")
        if overlap >= target_size:
            raise ValueError("overlap must be < target_size")
        token_vecs = self._encode_token_sequence(ids)
        prefix_len = min(len(self._passage_prefix_ids), len(ids)) if self.use_e5_format else 0
        prefix_sum = token_vecs[:prefix_len].sum(axis=0) if prefix_len > 0 else None

        chunks: List[str] = []
        vecs: List[np.ndarray] = []
        i = 0
        n = len(ids)
        while i < n:
            j = min(i + target_size, n)
            chunk_ids = ids[i:j]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
            if self.use_e5_format and chunk_text.startswith("passage:"):
                chunk_text = chunk_text[len("passage:"):].strip()
            if chunk_text:
                chunks.append(chunk_text)
                span_start = max(i, prefix_len) if self.use_e5_format else i
                span_vecs = token_vecs[span_start:j]
                if self.use_e5_format and prefix_len > 0 and prefix_sum is not None:
                    if len(span_vecs):
                        pooled = (span_vecs.sum(axis=0) + prefix_sum) / float(len(span_vecs) + prefix_len)
                    else:
                        pooled = prefix_sum / float(prefix_len)
                else:
                    pooled = token_vecs[i:j].mean(axis=0)
                pooled = pooled.astype(np.float32)
                vecs.append(_l2_normalize(pooled))
            if j == n:
                break
            i = j - overlap
        return chunks, vecs, over_model_max


def build_late_token_pool_chunks(
    doc_texts: Dict[str, str],
    encoder: LateTokenPoolEncoder,
    args: argparse.Namespace,
    desc: str,
) -> LateChunkData:
    chunk_texts: Dict[str, str] = {}
    chunk_to_doc: Dict[str, str] = {}
    chunk_vectors: Dict[str, np.ndarray] = {}
    truncated_docs = 0
    for doc_id, joined in tqdm(doc_texts.items(), desc=desc, leave=False):
        if not joined:
            continue
        chunks, vecs, truncated = encoder.build_doc_chunks(
            joined, target_size=args.token_size, overlap=args.overlap
        )
        if truncated:
            truncated_docs += 1
        for idx, (chunk_text_value, vec) in enumerate(zip(chunks, vecs)):
            cid = f"{doc_id}#chunk{idx}"
            chunk_texts[cid] = chunk_text_value
            chunk_to_doc[cid] = doc_id
            chunk_vectors[cid] = vec
    return LateChunkData(
        chunk_texts=chunk_texts,
        chunk_to_doc=chunk_to_doc,
        chunk_vectors=chunk_vectors,
        truncated_docs=truncated_docs,
    )


def chunk_corpus(
    corpus: Dict[str, Dict[str, str]],
    chunker: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    chunk_texts: Dict[str, str] = {}
    chunk_to_doc: Dict[str, str] = {}
    sentence_embed_fn = build_sentence_embed_fn(args) if chunker == "semantic" else None
    for doc_id, doc in tqdm(corpus.items(), desc=f"Chunking ({chunker})", leave=False):
        joined = _join_doc(doc)
        if not joined:
            continue
        chunks = chunk_text(joined, chunker=chunker, args=args, sentence_embed_fn=sentence_embed_fn)
        for i, c in enumerate(chunks):
            cid = f"{doc_id}#chunk{i}"
            chunk_texts[cid] = c
            chunk_to_doc[cid] = doc_id
    return chunk_texts, chunk_to_doc


def collapse_scores(score_by_chunk: Dict[str, float], chunk_to_doc: Dict[str, str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cid, s in score_by_chunk.items():
        did = chunk_to_doc[cid]
        prev = out.get(did)
        if prev is None or s > prev:
            out[did] = s
    return out


def topk_rank(scores: Dict[str, float], k: int) -> List[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def _es_client(hostname: str, port: int) -> Elasticsearch:
    if hasattr(_es_client, "_args"):
        args = _es_client._args  # type: ignore[attr-defined]
    else:
        args = None
    if args and args.bm25_api_key:
        client = Elasticsearch(
            f"http://{hostname}:{port}",
            request_timeout=60,
            api_key=args.bm25_api_key,
        )
    elif args:
        password = args.bm25_password or os.getenv("ES_LOCAL_PASSWORD")
        if password:
            client = Elasticsearch(
                f"http://{hostname}:{port}",
                request_timeout=60,
                basic_auth=(args.bm25_username, password),
            )
        else:
            client = Elasticsearch(f"http://{hostname}:{port}", request_timeout=60)
    else:
        client = Elasticsearch(f"http://{hostname}:{port}", request_timeout=60)
    if not client.ping():
        raise RuntimeError(
            f"Elasticsearch not reachable at {hostname}:{port}. Start it and retry."
        )
    return client


def _es_create_index(client: Elasticsearch, index_name: str) -> None:
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    client.indices.create(
        index=index_name,
        mappings={
            "properties": {
                "text": {"type": "text"},
                "doc_id": {"type": "keyword"},
            }
        },
    )


def _es_index_docs(
    client: Elasticsearch,
    index_name: str,
    chunk_texts: Dict[str, str],
    chunk_to_doc: Dict[str, str],
) -> None:
    actions = []
    for chunk_id, text in chunk_texts.items():
        actions.append(
            {
                "_index": index_name,
                "_id": chunk_id,
                "_source": {
                    "text": text,
                    "doc_id": chunk_to_doc.get(chunk_id, chunk_id),
                },
            }
        )
    if actions:
        helpers.bulk(client, actions, refresh="wait_for", request_timeout=120)


def _es_query(
    client: Elasticsearch,
    index_name: str,
    query_text: str,
    size: int,
    allowed_doc_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    must_clause = (
        {"match": {"text": {"query": query_text}}}
        if query_text.strip()
        else {"match_all": {}}
    )
    bool_query: Dict[str, object] = {"must": [must_clause]}
    if allowed_doc_ids:
        bool_query["filter"] = [{"terms": {"doc_id": allowed_doc_ids}}]
    body = {"query": {"bool": bool_query}}
    resp = client.search(index=index_name, body=body, size=size)
    hits = resp.get("hits", {}).get("hits", [])
    return {hit["_id"]: float(hit.get("_score", 0.0)) for hit in hits}


def compute_metrics(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Dict[str, Dict[str, float]]:
    recalls = {}
    mrrs = {}
    ndcgs = {}
    valid_qids = [q for q, rels in qrels.items() if any(v > 0 for v in rels.values())]
    for k in k_values:
        rec_sum = 0.0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        for qid in valid_qids:
            rel_set = {d for d, v in qrels[qid].items() if v > 0}
            ranked = topk_rank(results.get(qid, {}), k)
            hit_count = sum(1 for d in ranked if d in rel_set)
            rec_sum += hit_count / max(1, len(rel_set))

            rr = 0.0
            for i, d in enumerate(ranked, start=1):
                if d in rel_set:
                    rr = 1.0 / i
                    break
            mrr_sum += rr

            dcg = 0.0
            for i, d in enumerate(ranked, start=1):
                rel = 1.0 if d in rel_set else 0.0
                if rel > 0:
                    dcg += rel / math.log2(i + 1)
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel_set), k) + 1))
            ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0

        n = max(1, len(valid_qids))
        recalls[f"Recall@{k}"] = rec_sum / n
        mrrs[f"MRR@{k}"] = mrr_sum / n
        ndcgs[f"NDCG@{k}"] = ndcg_sum / n
    return {"recall": recalls, "mrr": mrrs, "ndcg": ndcgs}


def retrieve_dense(
    queries: Dict[str, str],
    chunk_texts: Dict[str, str],
    chunk_to_doc: Dict[str, str],
    sbert_model: str,
    batch_size: int,
    embedder: Optional[SBERTEmbedder] = None,
) -> Dict[str, Dict[str, float]]:
    if embedder is None:
        embedder = SBERTEmbedder(model_name=sbert_model, batch_size=batch_size, normalize=True)
    cids = list(chunk_texts.keys())
    ctexts = [chunk_texts[c] for c in cids]
    doc_emb = embedder.encode_corpus(ctexts)

    qids = list(queries.keys())
    qtexts = [queries[q] for q in qids]
    q_emb = embedder.encode_queries(qtexts)

    results: Dict[str, Dict[str, float]] = {}
    for i, qid in enumerate(qids):
        scores_vec = np.dot(doc_emb, q_emb[i])
        by_chunk = {cids[j]: float(scores_vec[j]) for j in range(len(cids))}
        results[qid] = collapse_scores(by_chunk, chunk_to_doc)
    return results


def retrieve_bm25(
    queries: Dict[str, str],
    chunk_texts: Dict[str, str],
    chunk_to_doc: Dict[str, str],
    args: argparse.Namespace,
    allowed_docs_by_qid: Optional[Dict[str, List[str]]] = None,
    size: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    client = _es_client(args.bm25_hostname, args.bm25_port)
    index_name = (
        f"{args.bm25_index_prefix}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    _es_create_index(client, index_name)
    _es_index_docs(client, index_name, chunk_texts, chunk_to_doc)

    retrieve_size = size if size is not None else max(args.k_values)
    results: Dict[str, Dict[str, float]] = {}
    try:
        for qid, qtext in queries.items():
            allowed = allowed_docs_by_qid.get(qid) if allowed_docs_by_qid else None
            by_chunk = _es_query(
                client=client,
                index_name=index_name,
                query_text=qtext,
                size=retrieve_size,
                allowed_doc_ids=allowed,
            )
            results[qid] = collapse_scores(by_chunk, chunk_to_doc)
    finally:
        if not args.bm25_keep_indices:
            try:
                client.indices.delete(index=index_name)
            except Exception:
                pass
    return results


def retrieve_dense_pooled(
    queries: Dict[str, str],
    chunk_vectors: Dict[str, np.ndarray],
    chunk_to_doc: Dict[str, str],
    encoder: LateTokenPoolEncoder,
    allowed_docs_by_qid: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, float]]:
    cids = list(chunk_vectors.keys())
    if not cids:
        return {qid: {} for qid in queries}
    mat = np.vstack([chunk_vectors[cid] for cid in cids]).astype(np.float32)
    doc_ids = [chunk_to_doc[cid] for cid in cids]
    qids = list(queries.keys())
    qvecs = np.vstack([encoder.encode_query(queries[qid]) for qid in qids]).astype(np.float32)

    results: Dict[str, Dict[str, float]] = {}
    for i, qid in enumerate(qids):
        scores_vec = mat @ qvecs[i]
        allowed = set(allowed_docs_by_qid.get(qid, [])) if allowed_docs_by_qid else None
        if allowed is not None:
            by_chunk = {
                cids[j]: float(scores_vec[j])
                for j in range(len(cids))
                if doc_ids[j] in allowed
            }
        else:
            by_chunk = {cids[j]: float(scores_vec[j]) for j in range(len(cids))}
        results[qid] = collapse_scores(by_chunk, chunk_to_doc)
    return results


def sample_qids(all_qids: List[str], max_queries: int, seed: int) -> List[str]:
    if len(all_qids) <= max_queries:
        return all_qids
    rng = random.Random(seed)
    return rng.sample(all_qids, max_queries)


def load_hotpot_distractor(
    hotpot_path: pathlib.Path,
    max_queries: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    with hotpot_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    qids = [r["_id"] for r in rows]
    keep = set(sample_qids(qids, max_queries=max_queries, seed=seed))

    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}
    for r in rows:
        qid = r["_id"]
        if qid not in keep:
            continue
        queries[qid] = r["question"]
        supporting_titles = {t for t, _ in r.get("supporting_facts", [])}
        rels: Dict[str, int] = {}
        for i, ctx in enumerate(r.get("context", [])):
            title = ctx[0]
            sents = ctx[1] if len(ctx) > 1 else []
            text = " ".join(sents).strip()
            did = f"{qid}::d{i}"
            corpus[did] = {"title": title, "text": text}
            if title in supporting_titles:
                rels[did] = 1
        qrels[qid] = rels
    return corpus, queries, qrels


def load_qasper_beir(
    data_dir: pathlib.Path,
    split: str,
    max_queries: int,
    max_corpus_docs: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    local_qasper = data_dir / "qasper"
    if local_qasper.exists() and local_qasper.is_dir():
        data_path = str(local_qasper)
    else:
        try:
            data_path = download_beir_dataset("qasper", data_dir)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load qasper via BEIR downloader. "
                "Your cached datasets/qasper.zip appears to be invalid (often a 404 HTML page), "
                "or the current BEIR mirror does not host qasper at that URL. "
                "Either provide an extracted local BEIR-format folder at datasets/qasper "
                "(corpus.jsonl, queries.jsonl, qrels/*.tsv), or run without qasper."
            ) from exc
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    qids = sample_qids(list(queries.keys()), max_queries=max_queries, seed=seed)
    queries = {qid: queries[qid] for qid in qids}
    qrels = {qid: qrels[qid] for qid in qids if qid in qrels}

    rel_doc_ids = set()
    for qid in qrels:
        for did, rel in qrels[qid].items():
            if rel > 0:
                rel_doc_ids.add(did)

    all_doc_ids = list(corpus.keys())
    rng = random.Random(seed)
    neg_pool = [d for d in all_doc_ids if d not in rel_doc_ids]
    target_neg = max(0, max_corpus_docs - len(rel_doc_ids))
    if len(neg_pool) > target_neg:
        neg_pool = rng.sample(neg_pool, target_neg)
    keep_docs = rel_doc_ids.union(neg_pool)
    corpus = {did: corpus[did] for did in keep_docs if did in corpus}
    return corpus, queries, qrels


def evaluate_one(
    dataset_name: str,
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    chunker: str,
    retriever: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    started_at = datetime.now(timezone.utc)
    t0 = perf_counter()
    shared_embedder = None
    dense_model_name = args.sbert_model
    if retriever == "sbert":
        shared_embedder = SBERTEmbedder(
            model_name=args.sbert_model,
            batch_size=args.batch_size,
            normalize=True,
        )
    elif retriever == "e5":
        dense_model_name = args.e5_model
        shared_embedder = E5Embedder(
            model_name=args.e5_model,
            batch_size=args.batch_size,
            normalize=True,
        )
    late_pool_encoder: Optional[LateTokenPoolEncoder] = None
    if chunker == "late_token_pool":
        late_pool_encoder = LateTokenPoolEncoder(
            model_name=dense_model_name,
            use_e5_format=(retriever == "e5"),
        )

    if args.chunking_mode == "early":
        if retriever in {"sbert", "e5"}:
            if chunker == "late_token_pool":
                if late_pool_encoder is None:
                    raise RuntimeError("late_token_pool encoder was not initialized.")
                doc_texts = {doc_id: _join_doc(doc) for doc_id, doc in corpus.items()}
                late_data = build_late_token_pool_chunks(
                    doc_texts=doc_texts,
                    encoder=late_pool_encoder,
                    args=args,
                    desc="Late token pool build (early)",
                )
                if late_data.truncated_docs:
                    logging.info(
                        "late_token_pool docs above model max length (windowed): %d",
                        late_data.truncated_docs,
                    )
                results = retrieve_dense_pooled(
                    queries=queries,
                    chunk_vectors=late_data.chunk_vectors,
                    chunk_to_doc=late_data.chunk_to_doc,
                    encoder=late_pool_encoder,
                )
                num_chunks = len(late_data.chunk_texts)
            else:
                chunk_texts, chunk_to_doc = chunk_corpus(corpus, chunker=chunker, args=args)
                results = retrieve_dense(
                    queries=queries,
                    chunk_texts=chunk_texts,
                    chunk_to_doc=chunk_to_doc,
                    sbert_model=dense_model_name,
                    batch_size=args.batch_size,
                    embedder=shared_embedder,
                )
                num_chunks = len(chunk_texts)
        elif retriever == "bm25":
            if chunker == "late_token_pool":
                if late_pool_encoder is None:
                    raise RuntimeError("late_token_pool encoder was not initialized.")
                doc_texts = {doc_id: _join_doc(doc) for doc_id, doc in corpus.items()}
                late_data = build_late_token_pool_chunks(
                    doc_texts=doc_texts,
                    encoder=late_pool_encoder,
                    args=args,
                    desc="Late token pool build (early)",
                )
                if late_data.truncated_docs:
                    logging.info(
                        "late_token_pool docs above model max length (windowed): %d",
                        late_data.truncated_docs,
                    )
                results = retrieve_bm25(
                    queries=queries,
                    chunk_texts=late_data.chunk_texts,
                    chunk_to_doc=late_data.chunk_to_doc,
                    args=args,
                    size=max(args.k_values),
                )
                num_chunks = len(late_data.chunk_texts)
            else:
                chunk_texts, chunk_to_doc = chunk_corpus(corpus, chunker=chunker, args=args)
                results = retrieve_bm25(
                    queries=queries,
                    chunk_texts=chunk_texts,
                    chunk_to_doc=chunk_to_doc,
                    args=args,
                    size=max(args.k_values),
                )
                num_chunks = len(chunk_texts)
        else:
            raise ValueError(f"Unsupported retriever: {retriever}")
    else:
        # Hierarchical chunking: retrieve full docs first, then chunk/rerank top docs per query.
        doc_texts = {doc_id: _join_doc(doc) for doc_id, doc in corpus.items()}
        doc_id_map = {doc_id: doc_id for doc_id in doc_texts}
        if retriever in {"sbert", "e5"}:
            doc_results = retrieve_dense(
                queries=queries,
                chunk_texts=doc_texts,
                chunk_to_doc=doc_id_map,
                sbert_model=dense_model_name,
                batch_size=args.batch_size,
                embedder=shared_embedder,
            )
        elif retriever == "bm25":
            doc_results = retrieve_bm25(
                queries=queries,
                chunk_texts=doc_texts,
                chunk_to_doc=doc_id_map,
                args=args,
                size=args.hierarchical_top_docs,
            )
        else:
            raise ValueError(f"Unsupported retriever: {retriever}")

        top_docs_by_qid: Dict[str, List[str]] = {
            qid: topk_rank(doc_results.get(qid, {}), args.hierarchical_top_docs)
            for qid in queries
        }
        needed_docs = {doc_id for docs in top_docs_by_qid.values() for doc_id in docs}

        late_chunk_texts: Dict[str, str] = {}
        late_chunk_to_doc: Dict[str, str] = {}
        late_chunk_vectors: Dict[str, np.ndarray] = {}
        if chunker == "late_token_pool":
            if late_pool_encoder is None:
                raise RuntimeError("late_token_pool encoder was not initialized.")
            sub_doc_texts = {doc_id: doc_texts.get(doc_id, "") for doc_id in needed_docs}
            late_data = build_late_token_pool_chunks(
                doc_texts=sub_doc_texts,
                encoder=late_pool_encoder,
                args=args,
                desc="Late token pool build (hierarchical)",
            )
            if late_data.truncated_docs:
                logging.info(
                    "late_token_pool docs above model max length (windowed): %d",
                    late_data.truncated_docs,
                )
            late_chunk_texts = late_data.chunk_texts
            late_chunk_to_doc = late_data.chunk_to_doc
            late_chunk_vectors = late_data.chunk_vectors
        else:
            sentence_embed_fn = build_sentence_embed_fn(args) if chunker == "semantic" else None
            for doc_id in tqdm(
                needed_docs,
                total=len(needed_docs),
                desc=f"Hierarchical chunk build ({chunker})",
                leave=False,
            ):
                joined = doc_texts.get(doc_id, "")
                if not joined:
                    continue
                chunks = chunk_text(
                    joined,
                    chunker=chunker,
                    args=args,
                    sentence_embed_fn=sentence_embed_fn,
                )
                for idx, chunk in enumerate(chunks):
                    cid = f"{doc_id}#chunk{idx}"
                    late_chunk_texts[cid] = chunk
                    late_chunk_to_doc[cid] = doc_id

        if retriever in {"sbert", "e5"}:
            if chunker == "late_token_pool":
                if late_pool_encoder is None:
                    raise RuntimeError("late_token_pool encoder was not initialized.")
                results = retrieve_dense_pooled(
                    queries=queries,
                    chunk_vectors=late_chunk_vectors,
                    chunk_to_doc=late_chunk_to_doc,
                    encoder=late_pool_encoder,
                    allowed_docs_by_qid=top_docs_by_qid,
                )
            else:
                results = {}
                for qid, qtext in tqdm(
                    queries.items(),
                    total=len(queries),
                    desc=f"Hierarchical rerank ({chunker},{retriever})",
                    leave=False,
                ):
                    allowed = set(top_docs_by_qid.get(qid, []))
                    sub_chunk_texts = {
                        cid: text
                        for cid, text in late_chunk_texts.items()
                        if late_chunk_to_doc[cid] in allowed
                    }
                    sub_chunk_to_doc = {
                        cid: did
                        for cid, did in late_chunk_to_doc.items()
                        if did in allowed
                    }
                    if not sub_chunk_texts:
                        results[qid] = {}
                        continue
                    sub_results = retrieve_dense(
                        queries={qid: qtext},
                        chunk_texts=sub_chunk_texts,
                        chunk_to_doc=sub_chunk_to_doc,
                        sbert_model=dense_model_name,
                        batch_size=args.batch_size,
                        embedder=shared_embedder,
                    )
                    results[qid] = sub_results.get(qid, {})
        else:
            results = retrieve_bm25(
                queries=queries,
                chunk_texts=late_chunk_texts,
                chunk_to_doc=late_chunk_to_doc,
                args=args,
                allowed_docs_by_qid=top_docs_by_qid,
                size=max(args.k_values) * max(2, args.hierarchical_top_docs),
            )
        num_chunks = len(late_chunk_texts)

    metrics = compute_metrics(qrels=qrels, results=results, k_values=sorted(set(args.k_values)))
    finished_at = datetime.now(timezone.utc)
    row: Dict[str, object] = {
        "dataset": dataset_name,
        "chunking_mode": args.chunking_mode,
        "chunker": chunker,
        "retriever": retriever,
        "num_queries": len(queries),
        "num_docs": len(corpus),
        "num_chunks": num_chunks,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round(perf_counter() - t0, 3),
    }
    row.update(metrics)
    return row


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    args.backend = "sbert"
    args.model = ""
    args.normalize = True
    _es_client._args = args  # type: ignore[attr-defined]

    datasets_data: Dict[str, Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]] = {}

    if "hotpotqa_distractor" in args.datasets:
        if not args.hotpot_distractor_file.exists():
            raise FileNotFoundError(
                f"HotpotQA distractor file not found: {args.hotpot_distractor_file}. "
                "Download hotpot_dev_distractor_v1.json and pass --hotpot-distractor-file."
            )
        datasets_data["hotpotqa_distractor"] = load_hotpot_distractor(
            hotpot_path=args.hotpot_distractor_file,
            max_queries=args.max_queries,
            seed=args.seed,
        )
        logging.info(
            "Loaded hotpotqa_distractor: docs=%d queries=%d",
            len(datasets_data["hotpotqa_distractor"][0]),
            len(datasets_data["hotpotqa_distractor"][1]),
        )

    if "qasper" in args.datasets:
        datasets_data["qasper"] = load_qasper_beir(
            data_dir=args.data_dir,
            split=args.qasper_split,
            max_queries=args.max_queries,
            max_corpus_docs=args.max_corpus_docs,
            seed=args.seed,
        )
        logging.info(
            "Loaded qasper: docs=%d queries=%d",
            len(datasets_data["qasper"][0]),
            len(datasets_data["qasper"][1]),
        )

    rows: List[Dict[str, object]] = []
    for dname in args.datasets:
        if dname not in datasets_data:
            continue
        corpus, queries, qrels = datasets_data[dname]
        for chunker in args.chunkers:
            for retriever in args.retrievers:
                logging.info("Run: dataset=%s chunker=%s retriever=%s", dname, chunker, retriever)
                row = evaluate_one(
                    dataset_name=dname,
                    corpus=corpus,
                    queries=queries,
                    qrels=qrels,
                    chunker=chunker,
                    retriever=retriever,
                    args=args,
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "dataset": row["dataset"],
                            "chunker": row["chunker"],
                            "retriever": row["retriever"],
                            "ndcg": row["ndcg"],
                            "recall": row["recall"],
                            "mrr": row["mrr"],
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

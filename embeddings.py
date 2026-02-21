import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm
import bm25s
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_EMBED_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
    for attempt in range(5):
        try:
            model = model_name or OPENAI_EMBED_MODEL
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            if attempt == 4:
                raise
            sleep_time = 1.5 * (2 ** attempt)
            time.sleep(sleep_time)

    raise RuntimeError("embed_texts gone horribly wrong??")


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


def build_sentence_embed_fn(args: object):
    backend = getattr(args, "backend", "sbert")
    if backend == "openai":
        return lambda texts: embed_texts(texts, model_name=args.model)
    if backend == "e5":
        model = E5Embedder(
            model_name=args.e5_model,
            batch_size=args.batch_size,
            normalize=getattr(args, "normalize", True),
        )
        return lambda texts: model._encode([model._format_passage(t) for t in texts]).tolist()
    model = SBERTEmbedder(
        model_name=args.sbert_model,
        batch_size=args.batch_size,
        normalize=getattr(args, "normalize", True),
    )
    return lambda texts: model._encode(texts).tolist()


def _collapse_scores(score_by_chunk: Dict[str, float], chunk_to_doc: Dict[str, str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cid, s in score_by_chunk.items():
        did = chunk_to_doc[cid]
        prev = out.get(did)
        if prev is None or s > prev:
            out[did] = s
    return out


def _bm25_scores_bm25s(
    queries: Dict[str, str],
    chunk_texts: Dict[str, str],
    chunk_to_doc: Dict[str, str],
    allowed_docs_by_qid: Optional[Dict[str, List[str]]] = None,
    size: Optional[int] = None,
    k_values: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    cids = list(chunk_texts.keys())
    corpus = [chunk_texts[cid] for cid in cids]
    max_k = max(k_values) if k_values else 10
    retrieve_size = size if size is not None else max_k

    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    results: Dict[str, Dict[str, float]] = {}
    for qid, qtext in queries.items():
        allowed = set(allowed_docs_by_qid.get(qid, [])) if allowed_docs_by_qid else None
        query_tokens = bm25s.tokenize([qtext])
        hits, scores = retriever.retrieve(query_tokens, k=max(1, retrieve_size))
        by_chunk: Dict[str, float] = {}
        for idx, score in zip(hits[0], scores[0]):
            cid = cids[int(idx)]
            did = chunk_to_doc.get(cid, cid)
            if allowed is not None and did not in allowed:
                continue
            by_chunk[cid] = float(score)
        results[qid] = _collapse_scores(by_chunk, chunk_to_doc)
    return results


def retrieve_bm25(
    queries: Dict[str, str],
    chunk_texts: Dict[str, str],
    chunk_to_doc: Dict[str, str],
    k_values: List[int],
    allowed_docs_by_qid: Optional[Dict[str, List[str]]] = None,
    size: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    return _bm25_scores_bm25s(
        queries=queries,
        chunk_texts=chunk_texts,
        chunk_to_doc=chunk_to_doc,
        allowed_docs_by_qid=allowed_docs_by_qid,
        size=size,
        k_values=k_values,
    )


def _bm25_scores_bm25s_chunks(
    queries: Dict[str, str],
    chunk_texts: Dict[str, str],
    chunk_to_doc: Dict[str, str],
    allowed_docs_by_qid: Optional[Dict[str, List[str]]] = None,
    top_n: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    cids = list(chunk_texts.keys())
    corpus = [chunk_texts[cid] for cid in cids]
    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    results: Dict[str, Dict[str, float]] = {}
    retrieve_n = max(1, len(cids)) if top_n is None else max(1, min(int(top_n), len(cids)))
    for qid, qtext in queries.items():
        allowed = set(allowed_docs_by_qid.get(qid, [])) if allowed_docs_by_qid else None
        query_tokens = bm25s.tokenize([qtext])
        hits, scores = retriever.retrieve(query_tokens, k=retrieve_n)
        by_chunk: Dict[str, float] = {}
        for idx, score in zip(hits[0], scores[0]):
            cid = cids[int(idx)]
            did = chunk_to_doc.get(cid, cid)
            if allowed is not None and did not in allowed:
                continue
            by_chunk[cid] = float(score)
        results[qid] = by_chunk
    return results


def retrieve_bm25_chunks(
    queries: Dict[str, str],
    chunk_texts: Dict[str, str],
    chunk_to_doc: Dict[str, str],
    allowed_docs_by_qid: Optional[Dict[str, List[str]]] = None,
    top_n: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    return _bm25_scores_bm25s_chunks(
        queries=queries,
        chunk_texts=chunk_texts,
        chunk_to_doc=chunk_to_doc,
        allowed_docs_by_qid=allowed_docs_by_qid,
        top_n=top_n,
    )


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0.0:
        return v
    return v / n


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

    def build_doc_chunks(
        self,
        text: str,
        target_size: int,
        overlap: int,
        min_size: int = 0,
    ) -> Tuple[List[str], List[np.ndarray], List[Tuple[int, int]], bool]:
        model_text = f"passage: {text}" if self.use_e5_format else text
        ids = self.tokenizer.encode(model_text, add_special_tokens=False, truncation=False)
        over_model_max = len(ids) > self.max_content_tokens
        if not ids:
            return [], [], [], over_model_max
        if target_size <= 0:
            raise ValueError("target_size must be > 0")
        if overlap >= target_size:
            raise ValueError("overlap must be < target_size")
        if min_size < 0:
            raise ValueError("min_size must be >= 0")
        token_vecs = self._encode_token_sequence(ids)
        prefix_len = min(len(self._passage_prefix_ids), len(ids)) if self.use_e5_format else 0
        prefix_sum = token_vecs[:prefix_len].sum(axis=0) if prefix_len > 0 else None

        chunks: List[str] = []
        vecs: List[np.ndarray] = []
        spans: List[Tuple[int, int]] = []
        offsets = self.tokenizer(model_text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        i = 0
        n = len(ids)
        while i < n:
            j = min(i + target_size, n)
            if j < n and (n - j) < min_size:
                j = n
            chunk_ids = ids[i:j]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
            if self.use_e5_format and chunk_text.startswith("passage:"):
                chunk_text = chunk_text[len("passage:"):].strip()
            if chunk_text:
                chunks.append(chunk_text)
                # Map token-window to original text offsets.
                span_start = int(offsets[i][0]) if i < len(offsets) else 0
                span_end = int(offsets[j - 1][1]) if (j - 1) < len(offsets) else span_start
                # Remove e5 "passage: " prefix from offsets.
                if self.use_e5_format:
                    prefix_chars = len("passage: ")
                    span_start = max(0, span_start - prefix_chars)
                    span_end = max(0, span_end - prefix_chars)
                spans.append((span_start, span_end))
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
        return chunks, vecs, spans, over_model_max

from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text
from db import SessionLocal
from embeddings import SBERTEmbedder, embed_texts
from config import TOP_K

def retrieve_similar(question: str, top_k: int = TOP_K) -> List[Tuple[int, str, int, float]]:
    [q_vec] = embed_texts([question])
    sql = text("""
        SELECT c.document_id, c.content, c.ordinal, (c.embedding <-> CAST(:qvec AS vector)) AS score
        FROM chunks c
        ORDER BY c.embedding <-> CAST(:qvec AS vector)
        LIMIT :k
    """)
    with SessionLocal() as session:
        rows = session.execute(sql, {"qvec": q_vec, "k": top_k}).fetchall()
    return [(r[0], r[1], r[2], float(r[3])) for r in rows]


def _collapse_scores(score_by_chunk: Dict[str, float], chunk_to_doc: Dict[str, str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cid, score in score_by_chunk.items():
        did = chunk_to_doc[cid]
        prev = out.get(did)
        if prev is None or score > prev:
            out[did] = score
    return out


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
    if not cids:
        return {qid: {} for qid in queries}
    ctexts = [chunk_texts[c] for c in cids]
    doc_emb = np.asarray(embedder.encode_corpus(ctexts), dtype=np.float32)

    qids = list(queries.keys())
    qtexts = [queries[q] for q in qids]
    q_emb = np.asarray(embedder.encode_queries(qtexts), dtype=np.float32)

    results: Dict[str, Dict[str, float]] = {}
    use_cuda = False
    torch = None
    try:
        import torch as _torch  # type: ignore

        torch = _torch
        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False

    if use_cuda and torch is not None:
        device = torch.device("cuda")
        doc_emb_t = torch.from_numpy(doc_emb).to(device=device, dtype=torch.float32)
        q_batch = max(1, min(len(qids), batch_size * 4))
        for start in range(0, len(qids), q_batch):
            end = min(len(qids), start + q_batch)
            q_emb_t = torch.from_numpy(q_emb[start:end]).to(device=device, dtype=torch.float32)
            scores_block = (q_emb_t @ doc_emb_t.T).detach().cpu().numpy()
            for row_idx, qid in enumerate(qids[start:end]):
                scores_vec = scores_block[row_idx]
                by_chunk = {cids[j]: float(scores_vec[j]) for j in range(len(cids))}
                results[qid] = _collapse_scores(by_chunk, chunk_to_doc)
    else:
        for i, qid in enumerate(qids):
            scores_vec = np.dot(doc_emb, q_emb[i])
            by_chunk = {cids[j]: float(scores_vec[j]) for j in range(len(cids))}
            results[qid] = _collapse_scores(by_chunk, chunk_to_doc)
    return results


def retrieve_dense_chunks(
    queries: Dict[str, str],
    chunk_texts: Dict[str, str],
    chunk_to_doc: Dict[str, str],
    sbert_model: str,
    batch_size: int,
    embedder: Optional[SBERTEmbedder] = None,
    allowed_docs_by_qid: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, float]]:
    if embedder is None:
        embedder = SBERTEmbedder(model_name=sbert_model, batch_size=batch_size, normalize=True)
    cids = list(chunk_texts.keys())
    if not cids:
        return {qid: {} for qid in queries}
    ctexts = [chunk_texts[c] for c in cids]
    doc_emb = np.asarray(embedder.encode_corpus(ctexts), dtype=np.float32)

    qids = list(queries.keys())
    qtexts = [queries[q] for q in qids]
    q_emb = np.asarray(embedder.encode_queries(qtexts), dtype=np.float32)

    results: Dict[str, Dict[str, float]] = {}
    for i, qid in enumerate(qids):
        scores_vec = np.dot(doc_emb, q_emb[i])
        allowed = set(allowed_docs_by_qid.get(qid, [])) if allowed_docs_by_qid else None
        if allowed is None:
            results[qid] = {cids[j]: float(scores_vec[j]) for j in range(len(cids))}
        else:
            results[qid] = {
                cids[j]: float(scores_vec[j])
                for j in range(len(cids))
                if chunk_to_doc.get(cids[j], cids[j]) in allowed
            }
    return results


def retrieve_dense_pooled(
    queries: Dict[str, str],
    chunk_vectors: Dict[str, np.ndarray],
    chunk_to_doc: Dict[str, str],
    encoder: object,
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
        results[qid] = _collapse_scores(by_chunk, chunk_to_doc)
    return results


def retrieve_dense_pooled_chunks(
    queries: Dict[str, str],
    chunk_vectors: Dict[str, np.ndarray],
    chunk_to_doc: Dict[str, str],
    encoder: object,
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
        if allowed is None:
            results[qid] = {cids[j]: float(scores_vec[j]) for j in range(len(cids))}
        else:
            results[qid] = {
                cids[j]: float(scores_vec[j])
                for j in range(len(cids))
                if doc_ids[j] in allowed
            }
    return results

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from preprocessing.text import tokenize_text


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def select_docs_with_answer(
    corpus: Dict[str, Dict[str, str]],
    answers_by_qid: Dict[str, str],
    candidate_doc_ids_by_qid: Optional[Dict[str, Iterable[str]]] = None,
    enforce_unique: bool = True,
    drop_boolean_answers: bool = True,
) -> Dict[str, Set[str]]:
    """
    Build qid -> set(doc_id) where the answer string appears in document text/title.
    If candidate_doc_ids_by_qid is provided, search only those docs per query.
    If enforce_unique=True, keep only qids where answer appears exactly once overall
    (single total match and single matching doc).
    """
    out: Dict[str, Set[str]] = {}
    for qid, answer in answers_by_qid.items():
        answer_norm = _normalize_text(answer)
        if not answer_norm:
            out[qid] = set()
            continue
        if drop_boolean_answers and answer_norm in {"yes", "no", "noanswer"}:
            out[qid] = set()
            continue

        if candidate_doc_ids_by_qid is None:
            doc_ids = [did for did in corpus.keys() if did.startswith(f"{qid}::")]
        else:
            doc_ids = list(candidate_doc_ids_by_qid.get(qid, []))

        hits: Set[str] = set()
        total_matches = 0
        pattern = re.compile(re.escape(answer_norm))
        for did in doc_ids:
            doc = corpus.get(did)
            if not doc:
                continue
            joined = " ".join([doc.get("title", "") or "", doc.get("text", "") or ""])
            norm_joined = _normalize_text(joined)
            match_count = len(pattern.findall(norm_joined))
            if match_count > 0:
                hits.add(did)
                total_matches += match_count

        if enforce_unique and (len(hits) != 1 or total_matches != 1):
            out[qid] = set()
            continue
        out[qid] = hits
    return out


def recall_at_k_from_top_chunks(
    raw_chunk_results: Dict[str, Dict[str, float]],
    chunk_to_doc: Dict[str, str],
    relevant_docs_by_qid: Dict[str, Set[str]],
    k: int = 5,
) -> Dict[str, object]:
    """
    Recall@k over top-k chunks:
    For each qid, map top-k chunks to docs and compute doc recall against relevant_docs_by_qid[qid].
    """
    qids = [qid for qid, rel in relevant_docs_by_qid.items() if rel]
    if not qids:
        return {"Recall@5_chunks": 0.0, "num_answerable": 0}

    rec_sum = 0.0
    for qid in qids:
        ranked_chunks = sorted(
            raw_chunk_results.get(qid, {}).items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[: max(1, k)]
        retrieved_docs = {chunk_to_doc[cid] for cid, _ in ranked_chunks if cid in chunk_to_doc}
        rel_docs = relevant_docs_by_qid[qid]
        rec_sum += len(retrieved_docs & rel_docs) / float(len(rel_docs))

    return {
        f"Recall@{k}_chunks": rec_sum / float(len(qids)),
        "num_answerable": len(qids),
    }


def _sentence_hit_indices(chunk_text: str, sentences: List[str], min_token_recall: float = 0.9) -> Set[int]:
    out: Set[int] = set()
    norm_chunk = " ".join(tokenize_text(chunk_text))
    if not norm_chunk:
        return out
    chunk_tokens = set(norm_chunk.split())
    for idx, sent in enumerate(sentences):
        norm_sent = " ".join(tokenize_text(sent))
        if not norm_sent:
            continue
        if norm_sent in norm_chunk:
            out.add(idx)
            continue
        sent_tokens = set(norm_sent.split())
        if not sent_tokens:
            continue
        overlap = len(sent_tokens & chunk_tokens) / float(len(sent_tokens))
        if overlap >= min_token_recall:
            out.add(idx)
    return out


def _build_chunk_to_sent_idxs(
    chunk_texts: Dict[str, str],
    chunk_to_doc: Dict[str, str],
    hotpot_doc_sentences: Dict[str, List[str]],
) -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = {}
    for cid, chunk_text in chunk_texts.items():
        did = chunk_to_doc.get(cid)
        if not did:
            continue
        sents = hotpot_doc_sentences.get(did)
        if not sents:
            continue
        out[cid] = _sentence_hit_indices(chunk_text, sents)
    return out


def compute_hotpot_support_fact_coverage(
    raw_chunk_results: Dict[str, Dict[str, float]],
    chunk_texts: Dict[str, str],
    chunk_to_doc: Dict[str, str],
    k_values: List[int],
    hotpot_gold_facts: Dict[str, Set[Tuple[str, int]]],
    hotpot_doc_sentences: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    chunk_to_sent_idxs = _build_chunk_to_sent_idxs(chunk_texts, chunk_to_doc, hotpot_doc_sentences)

    qids = [qid for qid, gold in hotpot_gold_facts.items() if gold and qid in raw_chunk_results]
    n = max(1, len(qids))
    all_cov: Dict[str, float] = {}
    recall_cov: Dict[str, float] = {}
    for k in sorted(set(k_values)):
        all_hits = 0
        recall_sum = 0.0
        for qid in qids:
            ranked = sorted(raw_chunk_results.get(qid, {}).items(), key=lambda kv: kv[1], reverse=True)[:k]
            covered: Set[Tuple[str, int]] = set()
            for cid, _ in ranked:
                did = chunk_to_doc.get(cid)
                if not did:
                    continue
                for sent_idx in chunk_to_sent_idxs.get(cid, set()):
                    covered.add((did, sent_idx))
            gold = hotpot_gold_facts.get(qid, set())
            if gold and gold.issubset(covered):
                all_hits += 1
            if gold:
                recall_sum += len(gold & covered) / float(len(gold))
        all_cov[f"SupportFactAll@{k}"] = all_hits / n
        recall_cov[f"SupportFactRecall@{k}"] = recall_sum / n
    return {"support_fact_all": all_cov, "support_fact_recall": recall_cov}


def build_predicted_supporting_facts(
    raw_chunk_results: Dict[str, Dict[str, float]],
    chunk_texts: Dict[str, str],
    chunk_to_doc: Dict[str, str],
    hotpot_doc_sentences: Dict[str, List[str]],
    qids: Iterable[str],
    top_k: int,
    max_facts: int,
) -> Dict[str, Set[Tuple[str, int]]]:
    chunk_to_sent_idxs = _build_chunk_to_sent_idxs(chunk_texts, chunk_to_doc, hotpot_doc_sentences)
    out: Dict[str, Set[Tuple[str, int]]] = {}
    for qid in qids:
        ranked_chunks = sorted(
            raw_chunk_results.get(qid, {}).items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[: max(1, int(top_k))]
        predicted_sp: List[Tuple[str, int]] = []
        seen_sp: Set[Tuple[str, int]] = set()
        for cid, _ in ranked_chunks:
            did = chunk_to_doc.get(cid, "")
            for sent_idx in sorted(chunk_to_sent_idxs.get(cid, set())):
                sf = (did, sent_idx)
                if sf in seen_sp:
                    continue
                seen_sp.add(sf)
                predicted_sp.append(sf)
                if len(predicted_sp) >= max(1, int(max_facts)):
                    break
            if len(predicted_sp) >= max(1, int(max_facts)):
                break
        out[qid] = set(predicted_sp)
    return out

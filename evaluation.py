import re
import string
from typing import Dict, Iterable, List, Optional, Set, Tuple

from preprocessing.text import tokenize_text


def _normalize_for_answer_match(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def select_docs_with_answer(
    corpus: Dict[str, Dict[str, str]],
    answers_by_qid: Dict[str, List[str]],
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
    for qid, answer_aliases in answers_by_qid.items():
        aliases = [a for a in answer_aliases if str(a or "").strip()]
        answer_norm = _normalize_for_answer_match(aliases[0] if aliases else "")
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
            norm_joined = _normalize_for_answer_match(joined)
            match_count = len(pattern.findall(norm_joined))
            if match_count > 0:
                hits.add(did)
                total_matches += match_count

        if enforce_unique and (len(hits) != 1 or total_matches != 1):
            out[qid] = set()
            continue
        out[qid] = hits
    return out


def _has_answer_match(
    chunk_text: str,
    answer_aliases: List[str],
    min_answer_tokens: int = 2,
) -> bool:
    normalized_chunk = _normalize_for_answer_match(chunk_text)
    if not normalized_chunk:
        return False
    chunk_tokens = normalized_chunk.split()
    if not chunk_tokens:
        return False
    chunk_token_set = set(chunk_tokens)
    padded_chunk = f" {normalized_chunk} "

    for alias in answer_aliases:
        normalized_alias = _normalize_for_answer_match(alias)
        if not normalized_alias:
            continue
        answer_tokens = normalized_alias.split()
        if not answer_tokens:
            continue

        if len(answer_tokens) >= max(1, int(min_answer_tokens)):
            if f" {normalized_alias} " in padded_chunk:
                return True
            continue

        # Guard short single-token answers to avoid noisy matches (e.g., "an", "us").
        if len(answer_tokens) == 1 and len(answer_tokens[0]) >= 3 and answer_tokens[0] in chunk_token_set:
            return True
    return False


def recall_at_k_from_top_chunks(
    raw_chunk_results: Dict[str, Dict[str, float]],
    chunk_texts: Dict[str, str],
    answers_by_qid: Dict[str, List[str]],
    k: int = 5,
    min_answer_tokens: int = 2,
) -> Dict[str, object]:
    """
    Recall@k over top-k chunks:
    For each qid, score 1 if any normalized gold-answer alias is present in any top-k chunk.
    """
    qids = [
        qid
        for qid, aliases in answers_by_qid.items()
        if any(_normalize_for_answer_match(alias) for alias in aliases)
    ]
    if not qids:
        return {"Recall@5_chunks": 0.0, "num_answerable": 0}

    hit_count = 0
    for qid in qids:
        ranked_chunks = sorted(
            raw_chunk_results.get(qid, {}).items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[: max(1, k)]
        aliases = answers_by_qid.get(qid, [])
        if any(
            _has_answer_match(
                chunk_text=chunk_texts.get(cid, ""),
                answer_aliases=aliases,
                min_answer_tokens=min_answer_tokens,
            )
            for cid, _ in ranked_chunks
        ):
            hit_count += 1

    return {
        f"Recall@{k}_chunks": hit_count / float(len(qids)),
        "num_answerable": len(qids),
    }


def _sentence_hit_indices(chunk_text: str, sentences: List[str], min_token_recall: float = 0.7) -> Set[int]:
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


_HPDOC_SENT_TAG_RE = re.compile(r"\[\[HPDOC:(?P<did>.+?)::S:(?P<sent_idx>\d+)\]\]")


def _extract_tagged_support_facts(chunk_text: str) -> Set[Tuple[str, int]]:
    out: Set[Tuple[str, int]] = set()
    if not chunk_text:
        return out
    for m in _HPDOC_SENT_TAG_RE.finditer(chunk_text):
        did = str(m.group("did")).strip()
        sent_idx_str = str(m.group("sent_idx")).strip()
        if not did:
            continue
        try:
            sent_idx = int(sent_idx_str)
        except ValueError:
            continue
        out.add((did, sent_idx))
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
    chunk_to_tagged_support: Dict[str, Set[Tuple[str, int]]] = {
        cid: _extract_tagged_support_facts(chunk_text)
        for cid, chunk_text in chunk_texts.items()
    }
    use_tagged_support = any(bool(v) for v in chunk_to_tagged_support.values())
    chunk_to_sent_idxs = (
        {}
        if use_tagged_support
        else _build_chunk_to_sent_idxs(chunk_texts, chunk_to_doc, hotpot_doc_sentences)
    )

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
                if use_tagged_support:
                    covered.update(chunk_to_tagged_support.get(cid, set()))
                    continue
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
    chunk_to_tagged_support: Dict[str, Set[Tuple[str, int]]] = {
        cid: _extract_tagged_support_facts(chunk_text)
        for cid, chunk_text in chunk_texts.items()
    }
    use_tagged_support = any(bool(v) for v in chunk_to_tagged_support.values())
    chunk_to_sent_idxs = (
        {}
        if use_tagged_support
        else _build_chunk_to_sent_idxs(chunk_texts, chunk_to_doc, hotpot_doc_sentences)
    )
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
            if use_tagged_support:
                support_facts = sorted(chunk_to_tagged_support.get(cid, set()))
            else:
                did = chunk_to_doc.get(cid, "")
                support_facts = [(did, sent_idx) for sent_idx in sorted(chunk_to_sent_idxs.get(cid, set()))]
            for sf in support_facts:
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

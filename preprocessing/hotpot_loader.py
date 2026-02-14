import json
import pathlib
from typing import Dict, List, Set, Tuple

from .sampling import sample_qids


def load_hotpot_distractor(
    hotpot_path: pathlib.Path,
    max_queries: int,
    seed: int,
) -> Tuple[
    Dict[str, Dict[str, str]],
    Dict[str, str],
    Dict[str, Set[Tuple[str, int]]],
    Dict[str, List[str]],
    Dict[str, str],
]:
    hotpot_gold_facts: Dict[str, Set[Tuple[str, int]]] = {}
    hotpot_doc_sentences: Dict[str, List[str]] = {}
    with hotpot_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    qids = [r["_id"] for r in rows]
    keep = set(sample_qids(qids, max_queries=max_queries, seed=seed))

    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    answers: Dict[str, str] = {}
    for r in rows:
        qid = r["_id"]
        if qid not in keep:
            continue
        queries[qid] = r["question"]
        answers[qid] = str(r.get("answer", ""))
        supporting_facts = r.get("supporting_facts", [])
        title_to_did: Dict[str, str] = {}
        for i, ctx in enumerate(r.get("context", [])):
            title = ctx[0]
            sents = ctx[1] if len(ctx) > 1 else []
            text = " ".join(sents).strip()
            did = f"{qid}::d{i}"
            corpus[did] = {"title": title, "text": text}
            hotpot_doc_sentences[did] = sents
            title_to_did.setdefault(title, did)
        gold_facts: Set[Tuple[str, int]] = set()
        for title, sent_idx in supporting_facts:
            did = title_to_did.get(title)
            if did is None:
                continue
            sents = hotpot_doc_sentences.get(did, [])
            if isinstance(sent_idx, int) and 0 <= sent_idx < len(sents):
                gold_facts.add((did, sent_idx))
        hotpot_gold_facts[qid] = gold_facts
    return corpus, queries, hotpot_gold_facts, hotpot_doc_sentences, answers

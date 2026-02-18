import json
import pathlib
import bz2
from typing import Dict, List, Optional, Set, Tuple

from .sampling import sample_qids


def _normalize_title(title: str) -> str:
    return " ".join(str(title).strip().split()).lower()


def _coerce_sentence_list(value: object) -> List[str]:
    if isinstance(value, list):
        out = [str(x).strip() for x in value if str(x).strip()]
        return out
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _extract_page(entry: object) -> Optional[Tuple[str, List[str]]]:
    # Format: {"title": "...", "sentences": [...]} (or text variants)
    if isinstance(entry, dict):
        title = str(entry.get("title", "")).strip()
        if not title:
            return None
        if "sentences" in entry:
            sents = _coerce_sentence_list(entry.get("sentences"))
        elif "text" in entry:
            text_val = entry.get("text")
            if isinstance(text_val, list):
                sents = _coerce_sentence_list(text_val)
            else:
                sents = _coerce_sentence_list(str(text_val or ""))
        else:
            sents = []
        return title, sents

    # Format: ["Title", ["sent1", "sent2", ...]]
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        title = str(entry[0]).strip()
        if not title:
            return None
        sents = _coerce_sentence_list(entry[1])
        return title, sents

    return None


def _build_global_wiki_corpus(
    wiki_path: pathlib.Path,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]], Dict[str, str]]:
    corpus: Dict[str, Dict[str, str]] = {}
    doc_sentences: Dict[str, List[str]] = {}
    title_to_did: Dict[str, str] = {}
    used_doc_ids: Set[str] = set()

    for item in _iter_wiki_items(wiki_path):
        parsed = _extract_page(item)
        if parsed is None:
            continue
        title, sents = parsed
        normalized = _normalize_title(title)
        if not normalized:
            continue

        base_did = f"wiki::{normalized}"
        did = base_did
        idx = 2
        while did in used_doc_ids:
            did = f"{base_did}#{idx}"
            idx += 1
        used_doc_ids.add(did)

        text = " ".join(sents).strip()
        corpus[did] = {"title": title, "text": text}
        doc_sentences[did] = sents
        title_to_did.setdefault(normalized, did)

    return corpus, doc_sentences, title_to_did


def _iter_wiki_items(wiki_path: pathlib.Path):
    if wiki_path.is_dir():
        files = sorted(
            p for p in wiki_path.rglob("*")
            if p.is_file() and _is_wiki_file(p)
        )
        for p in files:
            yield from _iter_wiki_items_from_file(p)
        return
    yield from _iter_wiki_items_from_file(wiki_path)


def _is_wiki_file(path: pathlib.Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".json")
        or name.endswith(".jsonl")
        or name.endswith(".jsonl.bz2")
        or name.endswith(".bz2")
    )


def _iter_wiki_items_from_file(path: pathlib.Path):
    name = path.name.lower()
    if name.endswith(".jsonl") or name.endswith(".jsonl.bz2") or name.endswith(".bz2"):
        opener = bz2.open if name.endswith(".bz2") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL in {path} at line {i}: {exc}"
                    ) from exc
                yield obj
        return

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        for title, value in raw.items():
            yield {"title": title, "sentences": value}
        return
    if isinstance(raw, list):
        for item in raw:
            yield item
        return
    raise ValueError(f"Unsupported wiki corpus format in: {path}")


def load_hotpot_fullwiki(
    hotpot_path: pathlib.Path,
    wiki_path: pathlib.Path,
    max_queries: int,
    seed: int,
) -> Tuple[
    Dict[str, Dict[str, str]],
    Dict[str, str],
    Dict[str, Set[Tuple[str, int]]],
    Dict[str, List[str]],
    Dict[str, List[str]],
]:
    corpus, hotpot_doc_sentences, title_to_did = _build_global_wiki_corpus(wiki_path)

    with hotpot_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    qids = [str(r.get("_id", "")) for r in rows if str(r.get("_id", "")).strip()]
    keep = set(sample_qids(qids, max_queries=max_queries, seed=seed))

    queries: Dict[str, str] = {}
    answers: Dict[str, List[str]] = {}
    hotpot_gold_facts: Dict[str, Set[Tuple[str, int]]] = {}
    for r in rows:
        qid = str(r.get("_id", "")).strip()
        if not qid or qid not in keep:
            continue
        queries[qid] = str(r.get("question", "")).strip()
        answers[qid] = [str(r.get("answer", ""))]

        gold_facts: Set[Tuple[str, int]] = set()
        for title, sent_idx in r.get("supporting_facts", []):
            did = title_to_did.get(_normalize_title(title))
            if did is None:
                continue
            sents = hotpot_doc_sentences.get(did, [])
            if isinstance(sent_idx, int) and 0 <= sent_idx < len(sents):
                gold_facts.add((did, sent_idx))
        hotpot_gold_facts[qid] = gold_facts

    return corpus, queries, hotpot_gold_facts, hotpot_doc_sentences, answers

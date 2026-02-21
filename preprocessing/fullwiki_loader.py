import json
import pathlib
import bz2
import logging
import random
from typing import Dict, List, Optional, Set, Tuple

from .sampling import sample_qids

FULLWIKI_DEFAULT_DOC_SAMPLE_SIZE = 4000


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

    page_count = 0
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
        page_count += 1
        if page_count % 50000 == 0:
            logging.info(
                "Fullwiki ingest progress: pages=%d docs=%d",
                page_count,
                len(corpus),
            )

    logging.info(
        "Fullwiki ingest complete: pages=%d docs=%d unique_titles=%d",
        page_count,
        len(corpus),
        len(title_to_did),
    )

    return corpus, doc_sentences, title_to_did


def _iter_wiki_items(wiki_path: pathlib.Path):
    if wiki_path.is_dir():
        files = sorted(
            p for p in wiki_path.rglob("*")
            if p.is_file() and _is_wiki_file(p)
        )
        logging.info(
            "Fullwiki ingest scanning directory: %s (files=%d)",
            wiki_path,
            len(files),
        )
        for p in files:
            yield from _iter_wiki_items_from_file(p)
        return
    logging.info("Fullwiki ingest reading file: %s", wiki_path)
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
    logging.info("Fullwiki ingest file start: %s", path)
    if name.endswith(".jsonl") or name.endswith(".jsonl.bz2") or name.endswith(".bz2"):
        opener = bz2.open if name.endswith(".bz2") else open
        with opener(path, "rt", encoding="utf-8") as f:
            line_count = 0
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
                line_count = i
                if i % 200000 == 0:
                    logging.info("Fullwiki ingest file progress: %s line=%d", path, i)
                yield obj
            logging.info("Fullwiki ingest file done: %s lines=%d", path, line_count)
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
    cache_enabled: bool = True,
    cache_dir: Optional[pathlib.Path] = None,
) -> Tuple[
    Dict[str, Dict[str, str]],
    Dict[str, str],
    Dict[str, Set[Tuple[str, int]]],
    Dict[str, List[str]],
    Dict[str, List[str]],
]:
    # Cache intentionally disabled: always build from raw wiki source.
    _ = cache_enabled
    _ = cache_dir
    corpus, hotpot_doc_sentences, title_to_did = _build_global_wiki_corpus(wiki_path)

    with hotpot_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    # Default mode-3 speed path:
    # sample a fixed subset of docs, then keep only questions whose supporting
    # facts are fully contained in that sampled doc set.
    candidate_dids: Set[str] = set()
    for r in rows:
        for title, _sent_idx in r.get("supporting_facts", []):
            did = title_to_did.get(_normalize_title(title))
            if did is not None:
                candidate_dids.add(did)
    all_candidate = sorted(candidate_dids)
    rng = random.Random(seed)
    if len(all_candidate) > FULLWIKI_DEFAULT_DOC_SAMPLE_SIZE:
        sampled_dids = set(rng.sample(all_candidate, FULLWIKI_DEFAULT_DOC_SAMPLE_SIZE))
    else:
        sampled_dids = set(all_candidate)

    sampled_norm_titles = {
        _normalize_title(corpus[did].get("title", ""))
        for did in sampled_dids
        if did in corpus
    }
    corpus = {did: doc for did, doc in corpus.items() if did in sampled_dids}
    hotpot_doc_sentences = {did: s for did, s in hotpot_doc_sentences.items() if did in sampled_dids}
    title_to_did = {t: did for t, did in title_to_did.items() if t in sampled_norm_titles}
    logging.info(
        "Mode3 default doc sampling: selected_docs=%d candidate_docs=%d",
        len(corpus),
        len(all_candidate),
    )

    filtered_rows = []
    for r in rows:
        sf = r.get("supporting_facts", [])
        if not sf:
            continue
        sf_dids: Set[str] = set()
        missing = False
        for title, _sent_idx in sf:
            did = title_to_did.get(_normalize_title(title))
            if did is None:
                missing = True
                break
            sf_dids.add(did)
        if missing or not sf_dids:
            continue
        if sf_dids.issubset(sampled_dids):
            filtered_rows.append(r)

    qids = [str(r.get("_id", "")) for r in filtered_rows if str(r.get("_id", "")).strip()]
    keep = set(sample_qids(qids, max_queries=max_queries, seed=seed))

    queries: Dict[str, str] = {}
    answers: Dict[str, List[str]] = {}
    hotpot_gold_facts: Dict[str, Set[Tuple[str, int]]] = {}
    for r in filtered_rows:
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

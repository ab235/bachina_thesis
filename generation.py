import os
import json
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm



def _build_contexts_for_qid(
    qid: str,
    raw_chunk_results: Dict[str, Dict[str, float]],
    chunk_texts: Dict[str, str],
    top_k: int,
) -> List[str]:
    ranked = sorted(
        raw_chunk_results.get(qid, {}).items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[: max(1, int(top_k))]
    contexts: List[str] = []
    for cid, score in ranked:
        chunk_text = chunk_texts.get(cid, "")
        if not chunk_text:
            continue
        contexts.append(f"[chunk={cid} score={score:.4f}]\n{chunk_text}")
    return contexts


def _generate_one_answer(
    ollama_url: str,
    model: str,
    question: str,
    contexts: List[str],
    seed: int
) -> str:
    joined = "\n\n---\n\n".join(contexts)
    system = (
        "You are answering a HotpotQA question using only provided evidence.\n"
        "STRICT OUTPUT RULES:\n"
        "1) Output exactly one line.\n"
        "2) Output only the final answer text, maximum 6 words.\n"
        "3) Do not output any explanation, reasoning, punctuation-only text, labels, or quotes.\n"
        "4) If evidence is insufficient, output exactly: unknown"
    )
    user = (
        f"Question: {question}\n\nEvidence:\n{joined}\n\n"
        "Return only the final answer text (one line, max 6 words)."
    )
    payload = {
        "model": model,
        "system": system,
        "prompt": user,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": int(seed),
            "num_predict": 64,
        },
    }
    req = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to call Ollama at {ollama_url}. Is `ollama serve` running?"
        ) from exc
    content = data.get("response", "")
    return str(content).strip()


def generate_answers_from_top_chunks(
    queries: Dict[str, str],
    raw_chunk_results: Dict[str, Dict[str, float]],
    chunk_texts: Dict[str, str],
    model: str,
    top_k: int,
    seed: int,
    qids: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    ollama_base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    ollama_url = f"{ollama_base}/api/generate"

    target_qids = list(qids) if qids is not None else list(queries.keys())
    out: Dict[str, str] = {}
    for qid in tqdm(target_qids, desc="Hotpot answer generation", leave=False):
        contexts = _build_contexts_for_qid(
            qid=qid,
            raw_chunk_results=raw_chunk_results,
            chunk_texts=chunk_texts,
            top_k=top_k,
        )
        out[qid] = _generate_one_answer(
            ollama_url=ollama_url,
            model=model,
            question=queries.get(qid, ""),
            contexts=contexts,
            seed=seed,
        )
    return out

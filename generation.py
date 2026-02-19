import os
import json
import socket
import time
import logging
import http.client
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm
from config import (
    BEDROCK_LLAMA_MODEL_ID,
    BEDROCK_MISTRAL_MODEL_ID,
    BEDROCK_QWEN_MODEL_ID,
    BEDROCK_REGION,
    OLLAMA_MAX_RETRIES,
    OLLAMA_RETRY_BACKOFF_SECONDS,
    OLLAMA_TIMEOUT_SECONDS,
)

OLLAMA_MODELS: Dict[str, str] = {
    "llama": "llama3.1",
    "mistral": "mistral",
    "qwen": "qwen2.5",
}

BEDROCK_MODEL_IDS: Dict[str, str] = {
    "llama": BEDROCK_LLAMA_MODEL_ID,
    "mistral": BEDROCK_MISTRAL_MODEL_ID,
    "qwen": BEDROCK_QWEN_MODEL_ID,
}



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


def _build_hotpot_prompt(question: str, contexts: List[str]) -> Dict[str, str]:
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
    return {"system": system, "user": user}


def _build_squad_prompt(question: str, contexts: List[str]) -> Dict[str, str]:
    joined = "\n\n---\n\n".join(contexts)
    system = (
        "You are answering a SQuAD-style question using only provided evidence.\n"
        "STRICT OUTPUT RULES:\n"
        "1) Output exactly one line.\n"
        "2) Output only the final answer text, no explanation.\n"
        "3) Prefer the shortest exact span from evidence.\n"
        "4) If evidence is insufficient, output exactly: unknown"
    )
    user = (
        f"Question: {question}\n\nEvidence:\n{joined}\n\n"
        "Return only the final answer text (one line)."
    )
    return {"system": system, "user": user}


def _build_prompt(question: str, contexts: List[str], prompt_style: str) -> Dict[str, str]:
    if prompt_style == "squad":
        return _build_squad_prompt(question=question, contexts=contexts)
    return _build_hotpot_prompt(question=question, contexts=contexts)


def _generate_one_answer(
    ollama_url: str,
    model: str,
    question: str,
    contexts: List[str],
    seed: int,
    prompt_style: str,
) -> str:
    prompt = _build_prompt(question=question, contexts=contexts, prompt_style=prompt_style)
    payload = {
        "model": model,
        "system": prompt["system"],
        "prompt": prompt["user"],
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
    timeout_s = int(OLLAMA_TIMEOUT_SECONDS)
    max_retries = int(OLLAMA_MAX_RETRIES)
    retry_backoff_s = float(OLLAMA_RETRY_BACKOFF_SECONDS)
    attempts = max(1, max_retries + 1)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data.get("response", "")
            return str(content).strip()
        except (
            urllib.error.URLError,
            socket.timeout,
            TimeoutError,
            json.JSONDecodeError,
            http.client.RemoteDisconnected,
            ConnectionError,
        ) as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            time.sleep(retry_backoff_s * attempt)
    raise RuntimeError(
        f"Failed to call Ollama at {ollama_url} after {attempts} attempts "
        f"(timeout={timeout_s}s)."
    ) from last_exc


def _resolve_ollama_model(model_family: str) -> str:
    return OLLAMA_MODELS.get(model_family, model_family)


def _resolve_bedrock_model_id(model_family: str, explicit_model_id: str) -> str:
    if explicit_model_id:
        return explicit_model_id
    model_id = BEDROCK_MODEL_IDS.get(model_family, "")
    if model_id:
        return model_id
    raise RuntimeError(
        "Bedrock model ID not configured. Set --bedrock-model-id or define the "
        f"matching model id in config.py/.env for --hotpot-answer-model={model_family}."
    )


def _get_bedrock_client(region: str):
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "boto3 is required for Bedrock answer generation. Install it with: pip install boto3"
        ) from exc
    kwargs = {}
    if region:
        kwargs["region_name"] = region
    return boto3.client("bedrock-runtime", **kwargs)


def _generate_one_answer_bedrock(
    client: object,
    model_id: str,
    question: str,
    contexts: List[str],
    prompt_style: str,
) -> str:
    prompt = _build_prompt(question=question, contexts=contexts, prompt_style=prompt_style)
    response = client.converse(
        modelId=model_id,
        system=[{"text": prompt["system"]}],
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt["user"]}],
            }
        ],
        inferenceConfig={
            "temperature": 0.0,
            "topP": 1.0,
            "maxTokens": 64,
        },
    )
    content = response.get("output", {}).get("message", {}).get("content", [])
    if not content:
        return "unknown"
    text = content[0].get("text", "")
    return str(text).strip() or "unknown"


def generate_answers_from_top_chunks(
    queries: Dict[str, str],
    raw_chunk_results: Dict[str, Dict[str, float]],
    chunk_texts: Dict[str, str],
    provider: str,
    model_family: str,
    top_k: int,
    seed: int,
    qids: Optional[Iterable[str]] = None,
    bedrock_model_id: str = "",
    bedrock_region: str = "",
    prompt_style: str = "hotpot",
) -> Dict[str, str]:
    ollama_base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    ollama_url = f"{ollama_base}/api/generate"
    model = ""
    bedrock_client = None
    if provider == "ollama":
        model = _resolve_ollama_model(model_family)
    elif provider == "bedrock":
        model = _resolve_bedrock_model_id(model_family=model_family, explicit_model_id=bedrock_model_id)
        region = bedrock_region or BEDROCK_REGION
        bedrock_client = _get_bedrock_client(region=region)
    else:
        raise ValueError(f"Unsupported answer provider: {provider}")

    target_qids = list(qids) if qids is not None else list(queries.keys())
    out: Dict[str, str] = {}
    for qid in tqdm(target_qids, desc="Hotpot answer generation", leave=False):
        contexts = _build_contexts_for_qid(
            qid=qid,
            raw_chunk_results=raw_chunk_results,
            chunk_texts=chunk_texts,
            top_k=top_k,
        )
        if provider == "bedrock":
            out[qid] = _generate_one_answer_bedrock(
                client=bedrock_client,
                model_id=model,
                question=queries.get(qid, ""),
                contexts=contexts,
                prompt_style=prompt_style,
            )
        else:
            try:
                out[qid] = _generate_one_answer(
                    ollama_url=ollama_url,
                    model=model,
                    question=queries.get(qid, ""),
                    contexts=contexts,
                    seed=seed,
                    prompt_style=prompt_style,
                )
            except RuntimeError as exc:
                # Avoid failing an entire shard for a single timed-out answer request.
                logging.warning("Ollama answer generation failed for qid=%s: %s", qid, exc)
                out[qid] = "unknown"
    return out
